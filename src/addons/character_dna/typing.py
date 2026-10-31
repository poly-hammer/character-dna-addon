# Type checking utilities for the Character DNA addon.

from typing import TYPE_CHECKING, ClassVar, Literal


if TYPE_CHECKING:
    import bpy

    from bpy.types import bpy_prop_collection, bpy_struct

    from .bindings.windows.x64.py313 import dna, riglogic  # pyright: ignore[reportAttributeAccessIssue] # noqa: TC004
    from .bindings.windows.x64.py313.dna import BinaryStreamReader, BinaryStreamWriter  # noqa: TC004
    from .components.body import CharacterComponentBase, CharacterComponentBody  # noqa: TC004
    from .components.head import CharacterComponentHead  # noqa: TC004
    from .editors.backup_manager.properties import (  # noqa: TC004
        BackupManagerPreferences,
        BackupManagerProperties,
        DnaBackupEntry,
    )
    from .editors.behavior_viewer.properties import BehaviorViewerProperties, PsdCorrectiveListItem  # noqa: TC004
    from .editors.converter.properties import (  # noqa: TC004
        ConverterExtraMeshItem,
        ConverterProperties,
    )
    from .editors.mesh_editor.properties import (  # noqa: TC004
        MeshDataTransferItem,
        MeshEditorProperties,
        ShimDataItem,
    )
    from .editors.raw_control_editor.properties import (  # noqa: TC004
        RawControlEditorPreferences,
        RawControlEditorProperties,
        RawControlListItem,
        TargetMeshItem,
    )
    from .editors.rbf_editor.properties import (  # noqa: TC004
        RBFDrivenBoneSelectionItem,
        RBFDrivenData,
        RBFDriverData,
        RBFEditorPreferences,
        RBFEditorProperties,
        RBFPoseData,
        RBFSolverData,
    )
    from .editors.shape_key_editor.properties import (  # noqa: TC004
        ShapeKeyData,
        ShapeKeyDependencyItem,
        ShapeKeyEditorPreferences,
        ShapeKeyEditorProperties,
    )
    from .editors.shared.rbf_solve import RBFDriverTarget, RBFPoseTarget  # noqa: TC004
    from .editors.swing_twist_editor.properties import SwingTwistEditorPreferences, SwingTwistEditorProperties
    from .operators import BakeAnimationBase, DuplicateRigInstance  # noqa: TC004
    from .properties import (
        CharacterAddonProperties,
        CharacterFaceBoardProperties,  # noqa: TC004
        CharacterImportProperties,  # noqa: TC004
        CharacterOutputProperties,  # noqa: TC004
        CharacterSceneProperties,  # noqa: TC004
        CharacterViewOptionsProperties,  # noqa: TC004
        CharacterWindowManagerProperties as _CharacterWindowManagerProperties,
        ExtraDnaFolder,
        OutputData,  # noqa: TC004
    )
    from .rig_definition import (
        RigDefinition,  # noqa: TC004
        RigJointGroup,  # noqa: TC004
    )
    from .rig_instance import RigInstance as _RigInstanceBase

    ComponentType = Literal["head", "body", "all"]

    # =========================================================================
    # Custom Collections
    # =========================================================================
    class ExtraDnaFolders(bpy_prop_collection[ExtraDnaFolder], bpy_struct):
        def add(self) -> ExtraDnaFolder: ...
        def move(self, src_index: int, dst_index: int) -> None: ...
        def remove(self, index: int) -> None: ...
        def clear(self) -> None: ...

    class DnaBackupEntrys(bpy_prop_collection[DnaBackupEntry], bpy_struct):
        def add(self) -> DnaBackupEntry: ...
        def move(self, src_index: int, dst_index: int) -> None: ...
        def remove(self, index: int) -> None: ...
        def clear(self) -> None: ...

    class RBFSolvers(bpy_prop_collection[RBFSolverData], bpy_struct):
        def add(self) -> RBFSolverData: ...
        def move(self, src_index: int, dst_index: int) -> None: ...
        def remove(self, index: int) -> None: ...
        def clear(self) -> None: ...

    # =========================================================================
    # Extended RigInstance with dynamically assigned editor properties
    # These are added at runtime in properties.py register() function
    # =========================================================================
    class RigInstance(_RigInstanceBase):
        """Extended RigInstance type with dynamically registered properties."""

        backup_manager: BackupManagerProperties
        rbf_editor: RBFEditorProperties
        raw_control_editor: RawControlEditorProperties
        behavior_viewer: BehaviorViewerProperties
        shape_key_editor: ShapeKeyEditorProperties
        swing_twist_editor: SwingTwistEditorProperties
        output: CharacterOutputProperties
        view_options: CharacterViewOptionsProperties

    # =========================================================================
    # Extended CharacterWindowManagerProperties with dynamically assigned editor properties
    # These are added at runtime in properties.py register() function
    # =========================================================================
    class CharacterWindowManagerProperties(_CharacterWindowManagerProperties):
        """Extended WindowManager properties with RBF Editor properties."""

        add_pose_driven_bones: bpy_prop_collection[RBFDrivenBoneSelectionItem]
        add_pose_driven_bones_active_index: int

    # =========================================================================
    # Addon Preferences Types
    # =========================================================================
    class CharacterAddonPreferences(CharacterAddonProperties, bpy.types.AddonPreferences):
        """Typed addon preferences for Character DNA."""

        bl_idname: str
        metrics_collection: bool
        rbf_editor: RBFEditorPreferences
        raw_control_editor: RawControlEditorPreferences
        shape_key_editor: ShapeKeyEditorPreferences
        swing_twist_editor: SwingTwistEditorPreferences
        backup_manager: BackupManagerPreferences
        next_metrics_consent_timestamp: float
        extra_dna_folder_list: ExtraDnaFolders
        extra_dna_folder_list_active_index: int

    class _CharacterAddon(bpy.types.Addon):
        """Typed addon module reference."""

        preferences: CharacterAddonPreferences

    class _CharacterAddons(bpy.types.bpy_prop_collection[bpy.types.Addon]):
        """Typed addons collection with Character DNA addon."""

        def get(self, name: str, default: _CharacterAddon | None = None) -> _CharacterAddon | None: ...
        def __getitem__(self, name: str) -> _CharacterAddon: ...
        def __contains__(self, name: str) -> bool: ...

    # =========================================================================
    # Patch bpy.types.Preferences
    # =========================================================================
    class Preferences(bpy.types.Preferences):
        """Extended Preferences type with typed addons access."""

        addons: _CharacterAddons

    # =========================================================================
    # Patch bpy.types.Scene
    # =========================================================================
    class Scene(bpy.types.Scene):
        """Extended Scene type with Character DNA properties."""

        character_dna: CharacterSceneProperties

    # =========================================================================
    # Patch bpy.types.WindowManager
    # =========================================================================
    class WindowManager(bpy.types.WindowManager):
        """Extended WindowManager type with Character DNA properties."""

        character_dna: CharacterWindowManagerProperties

    # =========================================================================
    # Patch bpy.types.Context
    # =========================================================================
    class Context(bpy.types.Context):
        """Extended Context type with typed properties."""

        window_manager: WindowManager
        preferences: Preferences
        scene: Scene

    __all__ = [
        "BakeAnimationBase",
        "BakeAnimationBase",
        "BehaviorViewerProperties",
        "BinaryStreamReader",
        "BinaryStreamWriter",
        "CharacterAddonPreferences",
        "CharacterComponentBase",
        "CharacterComponentBody",
        "CharacterComponentHead",
        "CharacterFaceBoardProperties",
        "CharacterImportProperties",
        "CharacterOutputProperties",
        "CharacterSceneProperties",
        "CharacterViewOptionsProperties",
        "CharacterWindowManagerProperties",
        "ClassVar",
        "ComponentType",
        "Context",
        "ConverterExtraMeshItem",
        "ConverterProperties",
        "DnaBackupEntry",
        "DuplicateRigInstance",
        "MeshDataTransferItem",
        "MeshEditorProperties",
        "OutputData",
        "Preferences",
        "PsdCorrectiveListItem",
        "RBFDrivenBoneSelectionItem",
        "RBFDrivenData",
        "RBFDriverData",
        "RBFDriverTarget",
        "RBFEditorPreferences",
        "RBFEditorProperties",
        "RBFPoseData",
        "RBFPoseTarget",
        "RBFSolverData",
        "RawControlEditorPreferences",
        "RawControlEditorProperties",
        "RawControlListItem",
        "RigDefinition",
        "RigInstance",
        "RigJointGroup",
        "Scene",
        "ShapeKeyData",
        "ShapeKeyDependencyItem",
        "ShapeKeyEditorPreferences",
        "ShapeKeyEditorProperties",
        "ShimDataItem",
        "TargetMeshItem",
        "WindowManager",
        "dna",
        "riglogic",
    ]
