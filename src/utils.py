import importlib
import copy

def init_module(config, **kwargs):
    module_str = config['module']

    import_str, _, module_name = module_str.rpartition(".")

    py_module = importlib.import_module(import_str)

    if type(config) != dict:
        config = config.to_dict()

    config = copy.deepcopy(config)
    config.pop("module")
    config.update(kwargs)

    return getattr(py_module, module_name)(**config)

def get_module(module_str):
    import_str, _, module_name = module_str.rpartition(".")

    py_module = importlib.import_module(import_str)

    return getattr(py_module, module_name)
