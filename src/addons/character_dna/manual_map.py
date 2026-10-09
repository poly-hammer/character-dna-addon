from .constants import ToolInfo


DOCUMENTATION_URL = "https://docs.polyhammer.com/character-dna-addon/"


def manual_map() -> tuple[str, tuple[tuple[str, str], ...]]:
    manual_mapping = ((f"bpy.ops.{ToolInfo.NAME}.convert_to_dna", "pro-features/converter/"),)
    return (DOCUMENTATION_URL, manual_mapping)
