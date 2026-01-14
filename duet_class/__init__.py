import importlib

__all__ = [
    "preprocess",
    "train",
    "evaluate",
    "predict",
    "timefeatures",
    "model",
    "components",
    "DUETConfig",
]

_MODULES = {
    "preprocess": "duet_class.pipeline.preprocess",
    "train": "duet_class.pipeline.train",
    "evaluate": "duet_class.pipeline.evaluate",
    "predict": "duet_class.pipeline.predict",
    "timefeatures": "duet_class.pipeline.timefeatures",
    "model": "duet_class.duet.model",
    "components": "duet_class.duet.components",
}


def __getattr__(name: str):
    if name in _MODULES:
        return importlib.import_module(_MODULES[name])
    if name == "DUETConfig":
        return importlib.import_module("duet_class.pipeline.config").DUETConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
