from .conv_gtcrn_small_new_v2 import ConvGTCRN_Small_New_V2
from .tiny_conv_se_v5 import TinyConvSE_v5


MODEL_REGISTRY = {
    "conv_gtcrn_small_new_v2": ConvGTCRN_Small_New_V2,
    "tiny_conv_v5": TinyConvSE_v5,
}


def build_model(config):
    model_name = str(config.get("model_name", "")).lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model_name: {model_name}. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_name](**config.get("network_config", {}))
