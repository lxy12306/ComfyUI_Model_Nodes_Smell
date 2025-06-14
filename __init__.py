from .QwenVL.nodes import NODE_CLASS_MAPPINGS as NODES_CLASS_QwenVL, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_QwenVL
from .JoyCaption.node import NODE_CLASS_MAPPINGS as NODES_CLASS_JoyCaption, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_JoyCaption
from .Common.option import NODE_CLASS_MAPPINGS as NODES_CLASS_Option, NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_Option

NODE_CLASS_MAPPINGS = {**NODES_CLASS_QwenVL,
                       **NODES_CLASS_Option,
                       **NODES_CLASS_JoyCaption,
                       }

NODE_DISPLAY_NAME_MAPPINGS = {**NODES_DISPLAY_QwenVL,
                              **NODES_DISPLAY_Option,
                              **NODES_DISPLAY_JoyCaption,
                              }

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]