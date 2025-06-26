import tomllib
import platform
import collections

values = {}
_reload_handlers = []


# https://stackoverflow.com/questions/7204805/deep-merge-dictionaries-of-dictionaries-in-python/7205107#7205107
def merge(a: dict, b: dict, path=[]):
    for key in b:
        if (key in a) and (isinstance(a[key], dict) and isinstance(b[key], dict)):
            merge(a[key], b[key], path + [str(key)])
        else:
            a[key] = b[key]
    return a


def reload():
    global values
    with open("config.toml", "rb") as f:
        values = tomllib.load(f)
    is_on_pc = platform.system() == "Windows"
    with open("windows.toml" if is_on_pc else "linux.toml", "rb") as f:
        values = merge(values, tomllib.load(f))

    for handler in _reload_handlers:
        handler()


def add_reload_handler(handler):
    global _reload_handlers
    _reload_handlers.append(handler)


reload()
