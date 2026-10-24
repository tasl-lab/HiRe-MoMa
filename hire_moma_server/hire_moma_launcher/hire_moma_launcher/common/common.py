import functools
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax import struct

from hire_moma_launcher.common.typing import Params, PRNGKey

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)
default_init = nn.initializers.xavier_uniform


class ModuleDict(nn.Module):
    """Holds several modules under one `params` tree; call one of them with `name=...`.

    Sub-modules may share parameters (e.g. an image encoder shared by actor and critic).
    Init: `model_def.init(rng, actor=[obs], critic=[obs, act])`.
    Apply: `model_def.apply({"params": params}, obs, name="actor")`.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(f"kwargs keys {kwargs.keys()} must match module keys {self.modules.keys()}")
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out
        return self.modules[name](*args, **kwargs)


class JaxRLTrainState(struct.PyTreeNode):
    """TrainState with target params, an internal rng and one optimizer per loss.

    `apply_loss_fns` takes a dict of loss functions (same keys as `txs`), differentiates each
    one w.r.t. all params, passes each gradient through its own optimizer and sums the updates.
    """

    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: Params
    target_params: Params
    txs: Any = struct.field(pytree_node=False)
    opt_states: Any
    rng: PRNGKey
    epsilon: float = 0.0

    @staticmethod
    def _tx_tree_map(*args, **kwargs):
        return jax.tree.map(*args, is_leaf=lambda x: isinstance(x, optax.GradientTransformation), **kwargs)

    def target_update(self, tau: float) -> "JaxRLTrainState":
        """Polyak averaging: target <- tau * params + (1 - tau) * target."""
        new_target_params = jax.tree.map(lambda p, tp: p * tau + tp * (1 - tau), self.params, self.target_params)
        return self.replace(target_params=new_target_params)

    def apply_gradients(self, *, grads: Any) -> "JaxRLTrainState":
        updates_and_new_states = self._tx_tree_map(
            lambda tx, opt_state, grad: tx.update(grad, opt_state, self.params), self.txs, self.opt_states, grads)
        updates = self._tx_tree_map(lambda _, x: x[0], self.txs, updates_and_new_states)
        new_opt_states = self._tx_tree_map(lambda _, x: x[1], self.txs, updates_and_new_states)

        updates_flat = []
        self._tx_tree_map(lambda _, update: updates_flat.append(update), self.txs, updates)
        updates_acc = jax.tree.map(lambda *xs: jnp.sum(jnp.array(xs), axis=0), *updates_flat)
        new_params = optax.apply_updates(self.params, updates_acc)
        return self.replace(step=self.step + 1, params=new_params, opt_states=new_opt_states)

    def apply_loss_fns(self, loss_fns: Any, pmap_axis: str = None,
                       has_aux: bool = False) -> Union["JaxRLTrainState", Tuple["JaxRLTrainState", Any]]:
        """Each loss fn has signature `loss_fn(params, rng) -> loss` (or `(loss, aux)` if has_aux)."""
        treedef = jax.tree_util.tree_structure(loss_fns)
        new_rng, *rngs = jax.random.split(self.rng, treedef.num_leaves + 1)
        rngs = jax.tree_util.tree_unflatten(treedef, rngs)
        grads_and_aux = jax.tree.map(
            lambda loss_fn, rng: jax.grad(loss_fn, has_aux=has_aux)(self.params, rng), loss_fns, rngs)
        self = self.replace(rng=new_rng)
        if pmap_axis is not None:
            grads_and_aux = jax.lax.pmean(grads_and_aux, axis_name=pmap_axis)
        if has_aux:
            grads = jax.tree.map(lambda _, x: x[0], loss_fns, grads_and_aux)
            aux = jax.tree.map(lambda _, x: x[1], loss_fns, grads_and_aux)
            return self.apply_gradients(grads=grads), aux
        return self.apply_gradients(grads=grads_and_aux)

    @classmethod
    def create(cls, *, apply_fn, params, txs, target_params=None, rng=jax.random.PRNGKey(0), epsilon=0.0):
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            target_params=target_params,
            txs=txs,
            opt_states=cls._tx_tree_map(lambda tx: tx.init(params), txs),
            rng=rng,
            epsilon=epsilon,
        )
