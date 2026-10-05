"""Compat shim for the released C-JEPA checkpoints.

The checkpoints were pickled when the world models and the predictor lived in a
top-level ``custom_models`` package. They now live under ``src/``, so alias the
old names to the current modules (same module objects, so isinstance checks
still hold). Third-party packages are pickled by their top-level name too, so
``src/third_party`` has to be importable as well.
"""

import sys
from pathlib import Path

_third_party = str(Path(__file__).resolve().parent.parent / "src" / "third_party")
if _third_party not in sys.path:
    sys.path.insert(0, _third_party)

from src import cjepa_predictor  # noqa: E402
from src.world_models import dinowm_causal, dinowm_causal_AP_node  # noqa: E402

for _name, _module in [
    ("cjepa_predictor", cjepa_predictor),
    ("dinowm_causal", dinowm_causal),
    ("dinowm_causal_AP_node", dinowm_causal_AP_node),
]:
    sys.modules[f"{__name__}.{_name}"] = _module

try:
    from src.world_models import dinowm_causal_savi

    sys.modules[f"{__name__}.dinowm_causal_savi"] = dinowm_causal_savi
except ImportError:  # SAVi deps are optional
    pass


def _register_dinov2_output_capturing():
    """Re-enable ``output_hidden_states`` for the pickled DINOv2 backbone.

    transformers >= 5 collects hidden states through hooks that are registered
    from ``__init__`` (via ``post_init``). Unpickled modules never run
    ``__init__``, so the registry entry is missing and ``out.hidden_states``
    silently comes back as ``None`` -- which the VideoSAUR encoder indexes.
    Registering the classes up front restores the capture.
    """
    try:
        from transformers.models.dinov2.modeling_dinov2 import Dinov2Encoder, Dinov2Model
        from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
    except ImportError:  # older transformers records hidden states directly
        return

    for cls in (Dinov2Model, Dinov2Encoder):
        _CAN_RECORD_REGISTRY.setdefault(str(cls), cls._can_record_outputs)


_register_dinov2_output_capturing()
