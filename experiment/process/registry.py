from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pkgutil import iter_modules
from typing import Callable


ProcessorFn = Callable[[], None]


@dataclass(frozen=True)
class Postprocessor:
    exp_name: str
    order: int
    name: str
    func: ProcessorFn
    module: str


_REGISTERED: dict[str, list[Postprocessor]] = {}
_DISCOVERY_ERRORS: dict[str, ModuleNotFoundError] = {}
_DISCOVERED = False
_SUBPACKAGES = ("tables", "plots")


def _default_exp_name(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1].replace("_", "-")


def register_postprocessor(exp_name: str | None = None, *, order: int = 0) -> Callable[[ProcessorFn], ProcessorFn]:
    def decorator(func: ProcessorFn) -> ProcessorFn:
        target_exp = exp_name or _default_exp_name(func.__module__)
        spec = Postprocessor(
            exp_name=target_exp,
            order=order,
            name=func.__name__,
            func=func,
            module=func.__module__,
        )
        specs = _REGISTERED.setdefault(target_exp, [])
        specs.append(spec)
        specs.sort(key=lambda item: (item.order, item.module, item.name))
        return func

    return decorator


def _discover_subpackage(package_name: str) -> None:
    package = import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return

    for module_info in iter_modules(package_path):
        if module_info.name.startswith("_"):
            continue
        module_name = f"{package_name}.{module_info.name}"
        try:
            import_module(module_name)
        except ModuleNotFoundError as exc:
            _DISCOVERY_ERRORS.setdefault(_default_exp_name(module_name), exc)


def discover_postprocessors() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    base_package = __name__.rsplit(".", 1)[0]
    for subpackage in _SUBPACKAGES:
        _discover_subpackage(f"{base_package}.{subpackage}")
    _DISCOVERED = True


def get_postprocessors(exp_name: str) -> list[Postprocessor]:
    discover_postprocessors()
    if exp_name in _DISCOVERY_ERRORS and exp_name not in _REGISTERED:
        exc = _DISCOVERY_ERRORS[exp_name]
        raise RuntimeError(f"Failed to load postprocessors for '{exp_name}': missing dependency '{exc.name}'") from exc
    return list(_REGISTERED.get(exp_name, []))


def registered_experiments() -> list[str]:
    discover_postprocessors()
    return sorted(set(_REGISTERED) | set(_DISCOVERY_ERRORS))
