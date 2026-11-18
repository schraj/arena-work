"""Restore a stub `torchvision.io.VideoReader` for `datasets` 3.x on macOS.

torchvision >= 0.29 removed the video API (moved to torchcodec), but
datasets 3.6 unconditionally does `from torchvision.io import VideoReader`
in its torch/np/jax/tf formatters whenever torchvision is in sys.modules --
which transformer_lens guarantees. datasets only uses the symbol for an
isinstance() check, so a never-instantiated stub is sufficient.

Upgrading datasets to 4.x is the upstream fix, but sae-vis pins <4.0.0.
Re-run this after any `pip install`/`uv sync` that reinstalls torchvision.
"""

import torchvision.io as tvio

MARKER = "# --- ARENA macOS compat: VideoReader stub ---"
SHIM = f'''

{MARKER}
class VideoReader:  # noqa: D101 - stub for datasets<4 isinstance checks
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "torchvision.io.VideoReader was removed in torchvision 0.29; "
            "use torchcodec for video decoding."
        )


__all__.append("VideoReader")
'''

path = tvio.__file__
src = open(path).read()
if MARKER in src:
    print(f"already patched: {path}")
else:
    open(path, "a").write(SHIM)
    print(f"patched: {path}")
