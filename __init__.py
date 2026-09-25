from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "./web"

try:
    from .nodes_server import (
        NODE_CLASS_MAPPINGS as SERVER_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as SERVER_NODE_DISPLAY_NAME_MAPPINGS,
    )
except Exception as e:  # keep the in-process nodes usable even if the server nodes fail to import
    print(f"[llama-cpp_vlm] Remote llama-server nodes unavailable: {e}")
else:
    NODE_CLASS_MAPPINGS = {**NODE_CLASS_MAPPINGS, **SERVER_NODE_CLASS_MAPPINGS}
    NODE_DISPLAY_NAME_MAPPINGS = {**NODE_DISPLAY_NAME_MAPPINGS, **SERVER_NODE_DISPLAY_NAME_MAPPINGS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
