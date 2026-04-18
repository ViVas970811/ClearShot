# ---------------------------------------------------------------------------
# torchvision / basicsr compatibility shim
#
# basicsr (a transitive dep of realesrgan) does
#   `from torchvision.transforms.functional_tensor import rgb_to_grayscale`
# but that module was removed in torchvision >= 0.17. This shim aliases the
# old path to the new one so any basicsr import downstream works.
#
# Placed in src/__init__.py so it runs on ANY src import, before any
# consumer module can trigger a basicsr import.
# ---------------------------------------------------------------------------
import sys as _sys
import types as _types

if "torchvision.transforms.functional_tensor" not in _sys.modules:
    try:
        import torchvision.transforms.functional as _tvf
        _shim = _types.ModuleType("torchvision.transforms.functional_tensor")
        _shim.rgb_to_grayscale = _tvf.rgb_to_grayscale
        _sys.modules["torchvision.transforms.functional_tensor"] = _shim
    except ImportError:
        # torchvision not installed; let real import errors surface downstream
        pass
