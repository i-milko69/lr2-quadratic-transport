import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from datetime import datetime
import numpy as np


def _to_jsonable(obj):
    """Recursively convert numpy types and dataclasses to JSON-serializable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class Logger:
    """Saves experiment results to JSON in results/logs/."""

    def __init__(self, out_dir: str | Path = "results/logs",
                 run_name: str | None = None):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(self, payload: dict, filename: str | None = None) -> Path:
        name = filename or f"{self.run_name}.json"
        path = self.out_dir / name
        path.write_text(json.dumps(_to_jsonable(payload), indent=2))
        return path

    def save_history(self, history, config: dict,
                     oracle=None, extra: dict | None = None,
                     filename: str | None = None) -> Path:
        from .metrics import summarize
        payload = {
            "config": config,
            "f": history.f,
            "grad_norm": history.grad_norm,
            "step_size": history.step_size,
            "summary": summarize(history, oracle),
        }
        if extra:
            payload["extra"] = extra
        return self.save(payload, filename)