from .QwenVL.nodes import NODE_CLASS_MAPPINGS as NODES_CLASS_QwenVL, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_QwenVL
from .Common.option import NODE_CLASS_MAPPINGS as NODES_CLASS_Option, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_Option

NODE_CLASS_MAPPINGS = {**NODES_CLASS_QwenVL,
                       **NODES_CLASS_Option,
                       }

NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_QwenVL,
                              **NODES_DISPLAY_Option,
                              }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]