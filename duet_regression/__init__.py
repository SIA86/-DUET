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
    "preprocess": "duet_regression.pipeline.preprocess",
    "train": "duet_regression.pipeline.train",
    "evaluate": "duet_regression.pipeline.evaluate",
    "predict": "duet_regression.pipeline.predict",
    "timefeatures": "duet_regression.pipeline.timefeatures",
    "model": "duet_regression.duet.model",
    "components": "duet_regression.duet.components",
}


def __getattr__(name: str):
    if name in _MODULES:
        return importlib.import_module(_MODULES[name])
    if name == "DUETConfig":
        return importlib.import_module("duet_regression.pipeline.config").DUETConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
