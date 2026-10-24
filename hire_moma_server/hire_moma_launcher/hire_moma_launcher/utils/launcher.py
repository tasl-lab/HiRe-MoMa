"""Factories for agents, the actor/learner connection and logging."""
import flax.linen as nn
import jax

from hire_moma_launcher.vision.data_augmentations import make_batch_augmentation_func

_MLP = {"activations": nn.tanh, "use_layer_norm": True, "hidden_dims": [256, 256]}


def _policy_kwargs(residual_bound):
    return {
        "tanh_squash_distribution": True,
        "std_parameterization": "exp",
        "action_scale": 0.2,          # scales the pre-tanh mean
        "std_clip": residual_bound,   # residual in [-eps, eps]
        "std_min": 1e-5,
        "std_max": 1.0,
    }


def make_agent(cfg, sample_obs, sample_action, seed: int = 42, full_base_action: bool = False):
    """Create the agent of `cfg.method` for `cfg.task` (`cfg`: tiago_server.exe_space.experiments.config.Config).

    The methods in `config.METHODS` map onto three agents:
      hire_moma (Full), hire_moma_no_align (+Hierarchical Decoupling), decomposed (+Decomposition Only)
          -> HiReMoMaAgent, differing in `conditioning` (Eq. 3) and `lambda_imi` (Eq. 4)
      flat_residual -> FlatResidualAgent;  hg_dagger -> BCAgent

    full_base_action: HiRe-MoMa heads see the whole base action instead of their own dimensions.

    Target entropy = cfg.train.target_entropy_per_dim * dim, with dim = the head's action dims
    (Flat Residual: all action dims including grippers). See TrainConfig.target_entropy_per_dim.
    """
    t, m, tr = cfg.task, cfg.method, cfg.train
    image_keys = (t.image_key,)
    rng = jax.random.PRNGKey(seed)
    common = dict(
        image_keys=image_keys,
        discount=tr.discount,
        critic_network_kwargs=_MLP,
        policy_network_kwargs=_MLP,
        policy_kwargs=_policy_kwargs(tr.residual_bound),
        temperature_init=0.1,
        critic_ensemble_size=10,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )
    if m.agent == "hire_moma":
        from hire_moma_launcher.agents.continuous.hire_moma import HiReMoMaAgent
        return HiReMoMaAgent.create_pixels(rng, sample_obs, sample_action, base_group=t.base_group,
                                           arm_group=t.arm_group, conditioning=m.conditioning,
                                           lambda_imi=m.lambda_imi, full_base_action=full_base_action,
                                           target_entropy_per_dim=tr.target_entropy_per_dim, **common)
    if m.agent == "flat_residual":
        from hire_moma_launcher.agents.continuous.flat_residual import FlatResidualAgent
        return FlatResidualAgent.create_pixels(rng, sample_obs, sample_action, grip_dims=t.grip_dims,
                                               grasp_critic_network_kwargs=_MLP,
                                               target_entropy=tr.target_entropy_per_dim * t.action_dim, **common)
    if m.agent == "hg_dagger":
        from hire_moma_launcher.agents.continuous.hgdagger_bc import BCAgent
        return BCAgent.create(
            rng, sample_obs, sample_action, grip_dims=t.grip_dims, image_keys=image_keys, loss_type="mse",
            network_kwargs={"activations": nn.tanh, "use_layer_norm": True, "hidden_dims": [512, 512, 512],
                            "dropout_rate": 0.25},
            policy_kwargs={"tanh_squash_distribution": False, "std_parameterization": "fixed", "fixed_std": 0.0,
                           "std_min": 1e-5, "std_max": 1.0, "action_scale": 0.2, "std_clip": 0.3},
            augmentation_function=make_batch_augmentation_func(image_keys))
    raise ValueError(f"Unknown agent {m.agent}")


def networks_to_update(agent):
    """(critic-only networks, all networks) for the UTD loop of the learner."""
    from hire_moma_launcher.agents.continuous.flat_residual import FlatResidualAgent

    if isinstance(agent, FlatResidualAgent):
        return frozenset({"critic", "grasp_critic"}), frozenset({"critic", "grasp_critic", "actor", "temperature"})
    return frozenset({"critic"}), frozenset({"critic", "actor", "temperature"})


def make_trainer_config(port_number: int = 5588, broadcast_port: int = 5589):
    from agentlace.trainer import TrainerConfig

    return TrainerConfig(port_number=port_number, broadcast_port=broadcast_port, request_types=["send-stats"])


def make_wandb_logger(project: str, name: str, debug: bool = False):
    from hire_moma_launcher.common.wandb import WandBLogger

    return WandBLogger(project=project, name=name, disabled=debug)
