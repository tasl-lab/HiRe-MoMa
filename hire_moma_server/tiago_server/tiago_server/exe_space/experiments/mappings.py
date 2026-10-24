from tiago_server.exe_space.experiments.config import METHODS, Config
from tiago_server.exe_space.experiments.pick_place.config import TASK as PICK_PLACE
from tiago_server.exe_space.experiments.place_only.config import TASK as PLACE_ONLY
from tiago_server.exe_space.experiments.sweep_litter.config import TASK as SWEEP_LITTER

TASK_MAPPING = {t.name: t for t in (PLACE_ONLY, SWEEP_LITTER, PICK_PLACE)}


def get_config(task: str, method: str) -> Config:
    assert task in TASK_MAPPING, f"Unknown task {task}. Choose from {list(TASK_MAPPING)}."
    assert method in METHODS, f"Unknown method {method}. Choose from {list(METHODS)}."
    return Config(task=TASK_MAPPING[task], method=METHODS[method])
