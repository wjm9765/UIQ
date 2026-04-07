from .base import BaseClapModel
from .laion import LaionClapModel
from .mga import MGAClapModel
from .m2d import M2DClapModel

def get_model(name: str, hf_model_id: str, device: str) -> BaseClapModel:
    if name.lower() == "laion":
        return LaionClapModel(name, hf_model_id, device)
    elif name.lower() == "mga":
        return MGAClapModel(name, hf_model_id, device)
    elif name.lower() == "m2d":
        return M2DClapModel(name, hf_model_id, device)
    else:
        raise ValueError(f"Unknown model name: {name}")
