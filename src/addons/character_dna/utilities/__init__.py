from ..typing import *


def get_active_rig_instance() -> "RigInstance | None":
    # Avoid circular import
    from ..ui.callbacks import get_active_rig_instance as _get_active_rig_instance

    return _get_active_rig_instance()


from .action import *  # noqa: E402
from .armature import *  # noqa: E402
from .blend_file import *  # noqa: E402
from .material import *  # noqa: E402
from .mesh import *  # noqa: E402

# Preserve the package-level migration API for addon integrations.
from .migration import (  # noqa: E402
    detect_legacy_data as detect_legacy_data,
    detect_runtime_migration as detect_runtime_migration,
    get_raw_scene_data as get_raw_scene_data,
    migrate_by_collection_data as migrate_by_collection_data,
    migrate_legacy_data as migrate_legacy_data,
    migrate_runtime_data as migrate_runtime_data,
    runtime_migration_sources as runtime_migration_sources,
)
from .misc import *  # noqa: E402
from .sentry import *  # noqa: E402
