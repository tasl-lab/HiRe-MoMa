def _flatten_dict(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten_dict(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = v
    return out


class WandBLogger:
    def __init__(self, project: str, name: str, config: dict = None, disabled: bool = False):
        import wandb

        self._wandb = wandb
        wandb.init(project=project, name=name, config=config or {}, mode="disabled" if disabled else "online")

    def log(self, data: dict, step: int = None):
        self._wandb.log(_flatten_dict(data), step=step)
