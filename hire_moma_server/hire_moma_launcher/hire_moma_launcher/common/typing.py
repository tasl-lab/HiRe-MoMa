from typing import Any, Dict, Union

import flax
import jax.numpy as jnp
import numpy as np

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]
Array = Union[np.ndarray, jnp.ndarray]
Data = Union[Array, Dict[str, "Data"]]
Batch = Dict[str, Data]
