import importlib
import os


def load_all_schedulers():
    schedulers = {}
    schedulers_params = {}
    package_dir = os.path.dirname(__file__)

    for filename in os.listdir(package_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"{__name__}.{filename[:-3]}"
            module = importlib.import_module(module_name)

            if hasattr(module, "scheduler_entry"):
                key, label = module.scheduler_entry
                schedulers[key] = label

            if hasattr(module, "schedulers_params"):
                schedulers_params.update(module.schedulers_params)

    return schedulers, schedulers_params
