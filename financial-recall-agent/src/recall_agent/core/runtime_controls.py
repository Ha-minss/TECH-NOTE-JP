"""Runtime safety controls for demo and evaluation execution."""


def path_override_allowed(dev_mode: bool) -> bool:
    """Return whether runtime path overrides are allowed.

    In normal/evaluation mode, approved bundle paths should not be overridden.
    Path overrides are only allowed in explicit development mode.
    """
    return bool(dev_mode)
