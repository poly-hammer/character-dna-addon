"""Explicit access to one DNA component, independent of the tool using it.

Component identity scopes SDK indices and caches. Editor selection never changes
which component a runtime binding evaluates.
"""

from dataclasses import dataclass
from typing import Any

import bpy
import numpy as np

from . import shape_names


@dataclass(frozen=True)
class RigComponent:
    instance: Any
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in {"head", "body"}:
            raise ValueError(f"Unknown DNA component: {self.kind}")

    def cache_key(self, descriptor: str) -> str:
        return self.instance.cache_key(self.kind, descriptor)

    @property
    def document_path(self) -> str:
        """The authoritative document, independent of its disposable SDK partition."""
        return self.instance.get("combined_dna_path") or self.dna_file_path

    @property
    def partition_id(self) -> str:
        return self.kind

    @property
    def rig(self) -> bpy.types.Object | None:
        return getattr(self.instance, f"{self.kind}_rig")

    @property
    def mesh(self) -> bpy.types.Object | None:
        return getattr(self.instance, f"{self.kind}_mesh")

    @property
    def dna_file_path(self) -> str:
        return getattr(self.instance, f"{self.kind}_dna_file_path")

    @property
    def dna_reader(self) -> Any:
        return getattr(self.instance, f"{self.kind}_dna_reader")

    @property
    def manager(self) -> Any:
        return getattr(self.instance, f"{self.kind}_manager")

    @property
    def state(self) -> Any:
        return getattr(self.instance, f"{self.kind}_instance")

    @property
    def initialized(self) -> bool:
        return bool(getattr(self.instance, f"{self.kind}_initialized"))

    def initialize(self, **_kwargs: Any) -> None:
        if self.kind == "head":
            self.instance.head_initialize(update_raw_control_list=False)
        else:
            self.instance.body_initialize(update_rbf_solver_list=False)

    def joint_outputs_euler(self) -> list[float]:
        """Return the editor's 9-channel Euler layout regardless of runtime output format."""
        if self.kind == "head":
            return list(self.state.getJointOutputs())
        from .bindings import riglogic
        from .dna_io import release_dna_handle

        manager = riglogic.RigLogic(self.dna_reader, riglogic.Configuration(), None)
        state = None
        try:
            state = riglogic.RigInstance(rigLogic=manager, memRes=None)
            source = self.state
            for index in range(self.dna_reader.getGUIControlCount()):
                state.setGUIControl(index, source.getGUIControl(index))
            for index in range(self.dna_reader.getRawControlCount()):
                state.setRawControl(index, source.getRawControl(index))
            state.setLOD(source.getLOD())
            manager.calculate(state)
            return list(state.getJointOutputs())
        finally:
            release_dna_handle(state)
            release_dna_handle(manager)

    @property
    def driver_rig(self) -> bpy.types.Object | None:
        if self.kind == "head" and self.instance.head_constrained_to_body:
            return self.instance.body_rig
        return self.rig

    @property
    def rest_pose(self) -> dict:
        return getattr(self.instance, f"{self.kind}_rest_pose")

    @property
    def gui_control_plan(self) -> list[tuple[int, str, str]]:
        reader = self.dna_reader
        if reader is None:
            return []
        result = []
        for index in range(reader.getGUIControlCount()):
            name, separator, axis = str(reader.getGUIControlName(index)).rpartition(".")
            if separator and axis.lower() in {"tx", "ty", "tz"}:
                result.append((index, name, axis[-1].lower()))
        return result

    @property
    def raw_quat_plan(self) -> list[tuple[int, str, str]]:
        reader = self.dna_reader
        if reader is None:
            return []
        result = []
        for index in range(reader.getRawControlCount()):
            name, separator, axis = str(reader.getRawControlName(index)).rpartition(".")
            if separator and axis in {"qx", "qy", "qz", "qw"}:
                result.append((index, name, axis[-1]))
        return result

    @property
    def driver_bone_names(self) -> list[str]:
        return list(dict.fromkeys(name for _, name, _ in self.raw_quat_plan))

    def _output_bones(self, category: str) -> list[str]:
        reader = self.dna_reader
        if reader is None:
            return []
        return list(
            dict.fromkeys(
                reader.getJointName(index)
                for setup in range(getattr(reader, f"get{category}Count")())
                for index in getattr(reader, f"get{category}OutputJointIndices")(setup)
            )
        )

    @property
    def swing_bone_names(self) -> list[str]:
        return self._output_bones("Swing")

    @property
    def twist_bone_names(self) -> list[str]:
        return self._output_bones("Twist")

    @property
    def mesh_index_lookup(self) -> dict[int, bpy.types.Object]:
        key = self.cache_key("mesh_index_lookup")
        cached = self.instance.data.get(key)
        if cached:
            return cached
        result = self.meshes_for_reader(self.dna_reader)
        if result:
            self.instance.data[key] = result
        return result

    def meshes_for_reader(self, reader: Any) -> dict[int, bpy.types.Object]:
        """Resolve stable scene IDs even while writeback has released cached readers."""
        if reader is None:
            return {}
        # Imported metadata survives user renames. Names support older blends.
        tagged = {
            obj.get("ca_dna_mesh_index"): obj
            for obj in bpy.data.objects
            if obj.type == "MESH" and obj.get("ca_dna_rig") == self.rig and obj.get("ca_dna_component") == self.kind
        }
        result = {}
        for index in range(reader.getMeshCount()):
            obj = tagged.get(index) or bpy.data.objects.get(f"{self.instance.name}_{reader.getMeshName(index)}")
            if obj is not None:
                result[index] = obj
        return result

    def shape_targets(self, lod: int | None = 0):
        reader = self.dna_reader
        if reader is None:
            return
        key = self.cache_key(f"shape_targets_{lod}")
        plan = self.instance.data.get(key)
        if plan is None:
            plan = []
            indices = range(reader.getMeshCount()) if lod is None else reader.getMeshIndicesForLOD(lod)
            for mesh_index in indices:
                for target in range(reader.getBlendShapeTargetCount(mesh_index)):
                    channel = int(reader.getBlendShapeChannelIndex(mesh_index, target))
                    name = f"{reader.getMeshName(mesh_index)}__{reader.getBlendShapeChannelName(channel)}"
                    plan.append((int(mesh_index), channel, name))
            self.instance.data[key] = plan
        yield from plan

    @property
    def channel_name_to_index_lookup(self) -> dict[str, int]:
        return {name: channel for _, channel, name in self.shape_targets()}

    @property
    def channel_index_to_mesh_index_lookup(self) -> dict[int, int]:
        return {channel: mesh for mesh, channel, _ in self.shape_targets()}

    def get_shape_key_block(self, mesh_index: int, name: str) -> bpy.types.ShapeKey | None:
        obj = self.mesh_index_lookup.get(mesh_index)
        keys = obj.data.shape_keys if obj and isinstance(obj.data, bpy.types.Mesh) else None
        return shape_names.find_key(keys, name)

    @property
    def shape_key_blocks_by_name(self) -> dict[str, bpy.types.ShapeKey]:
        """Resolve logical UI identities once, outside value/preview callbacks."""
        key = self.cache_key("shape_key_blocks_by_name")
        if key not in self.instance.data:
            self.instance.data[key] = {
                name: block
                for mesh, _channel, name in self.shape_targets()
                if (block := self.get_shape_key_block(mesh, name)) is not None
            }
        return self.instance.data[key]

    @property
    def shape_key_blocks(self) -> dict[int, list[bpy.types.ShapeKey]]:
        key = self.cache_key("shape_key_blocks")
        if key in self.instance.data:
            return self.instance.data[key]
        result = {}
        for mesh, channel, name in self.shape_targets():
            block = self.get_shape_key_block(mesh, name)
            if block is not None:
                result.setdefault(channel, []).append(block)
        if result:
            self.instance.data[key] = result
        return result

    @property
    def shape_key_apply_plan(self) -> list:
        key = self.cache_key("shape_key_apply_plan")
        if key not in self.instance.data:
            self.instance.data[key] = self.build_shape_key_apply_plan(self.dna_reader)
        return self.instance.data[key]

    def build_shape_key_apply_plan(self, reader: Any) -> list:
        """Bind real or provisional DNA outputs to numeric Blender key slots."""
        if reader is None:
            return []
        targets = {}
        for mesh in range(reader.getMeshCount()):
            for target in range(reader.getBlendShapeTargetCount(mesh)):
                channel = int(reader.getBlendShapeChannelIndex(mesh, target))
                name = f"{reader.getMeshName(mesh)}__{reader.getBlendShapeChannelName(channel)}"
                targets.setdefault(mesh, []).append((channel, name))
        plan = []
        for mesh, channels_and_names in targets.items():
            obj = self.mesh_index_lookup.get(mesh)
            if obj is None or not obj.data.shape_keys:
                continue
            blocks = obj.data.shape_keys.key_blocks
            indices = {shape_names.logical_name(obj.data.shape_keys, block.name): i for i, block in enumerate(blocks)}
            pairs = [(indices[name], channel) for channel, name in channels_and_names if name in indices]
            if pairs:
                positions, channels = (np.asarray(values, dtype=np.intp) for values in zip(*pairs, strict=True))
                plan.append(
                    (
                        blocks,
                        positions,
                        channels,
                        [blocks[int(i)] for i in positions],
                        np.empty(len(blocks), dtype=np.float32),
                    )
                )
        return plan

    def update_shape_keys(self, collect_values: bool = False, *, state: Any = None) -> list:
        state = state if state is not None else self.state
        if state is None:
            return []
        outputs = np.asarray(state.getBlendShapeOutputs(), dtype=np.float32)
        result = []
        for blocks, positions, channels, driven, buffer in self.shape_key_apply_plan:
            valid = channels < len(outputs)
            blocks.foreach_get("value", buffer)
            buffer[positions[valid]] = outputs[channels[valid]]
            blocks.foreach_set("value", buffer)
            blocks.id_data.update_tag()
            if collect_values:
                result.extend(
                    (block, float(outputs[channel]))
                    for block, channel, keep in zip(driven, channels, valid, strict=True)
                    if keep
                )
        return result

    def zero_shape_keys(self):
        for blocks, positions, _, _, buffer in self.shape_key_apply_plan:
            blocks.foreach_get("value", buffer)
            buffer[positions] = 0
            blocks.foreach_set("value", buffer)
            blocks.id_data.update_tag()

    def solo_shape_key_value(self, shape_key: bpy.types.ShapeKey) -> None:
        self.zero_shape_keys()
        shape_key.value = 1.0

    @property
    def texture_masks_node(self) -> bpy.types.Node | None:
        from .ui import callbacks

        material = getattr(self.instance, f"{self.kind}_material", None)
        resolver = getattr(callbacks, f"get_{self.kind}_texture_logic_node", None)
        return resolver(material) if resolver and material else None

    @property
    def animated_map_plan(self) -> list[tuple[int, str]]:
        reader = self.dna_reader
        if reader is None:
            return []
        node = self.texture_masks_node
        plan = []
        for index in range(reader.getAnimatedMapCount()):
            name = str(reader.getAnimatedMapName(index))
            # Exact names support custom DNA; retain the shipped MetaHuman shader convention.
            if node is not None and node.inputs.get(name) is not None:
                plan.append((index, name))
                continue
            prefix, separator, region = name.partition(".")
            parts = prefix.split("_")
            if separator and len(parts) > 1:
                name = f"{parts[1].lower().replace('cm', 'wm')}.{region}_msk"
            plan.append((index, name))
        return plan

    def update_texture_masks(self, *, state: Any = None) -> list:
        node = self.texture_masks_node
        state = state if state is not None else self.state
        if node is None or state is None:
            return []
        outputs = state.getAnimatedMapOutputs()
        result = []
        for index, name in self.animated_map_plan:
            socket = node.inputs.get(name)
            if socket is not None and index < len(outputs):
                socket.default_value = outputs[index]
                result.append((name, outputs[index]))
        return result

    def update_bone_transforms(self, collect_transforms: bool = False) -> list:
        from .runtime.authoring import bone_transforms

        return bone_transforms(self.instance, self.kind, collect_transforms)


def get_editor_component(instance: Any) -> str:
    value = instance.get("editor_component")
    if value in {"head", "body"}:
        return value
    # Old files may only have the Output enum's stored integer selection.
    output = getattr(instance, "output", None)
    legacy_value = output.get("component") if output is not None else None
    if legacy_value in (0, 1):
        value = ("head", "body")[legacy_value]
        if getattr(instance, f"{value}_rig", None):
            return value
    return "head" if instance.head_rig else "body"


def component_selection_locked(instance: Any) -> bool:
    """Keep every panel on the same component throughout an editing operation."""
    from .runtime.controller import is_suspended

    return any(
        getattr(getattr(instance, name, None), "is_editing", False)
        for name in ("raw_control_editor", "shape_key_editor", "rbf_editor")
    ) or is_suspended(instance)


def set_editor_component(instance: Any, value: int) -> None:
    # Protect the write destination even when scripts bypass the disabled UI.
    if component_selection_locked(instance):
        return
    instance["editor_component"] = ("head", "body")[value]


def refresh_editor_component(instance: Any, context: Any = None) -> None:
    if component_selection_locked(instance):
        return
    target = RigComponent(instance, get_editor_component(instance))
    if target.rig and target.dna_file_path and not target.initialized:
        target.initialize()
    from .ui.callbacks import update_body_output_items, update_head_output_items

    update_head_output_items(instance, context or bpy.context)
    update_body_output_items(instance, context or bpy.context)
    if not instance.is_pro:
        return
    from .editors.raw_control_editor.callbacks import update_head_raw_control_list
    from .editors.rbf_editor.callbacks import update_body_rbf_solver_list

    for editor_name in ("raw_control_editor", "rbf_editor", "shape_key_editor"):
        # Component-local lists are rebuilt; session data must not leak across selections.
        instance.data.pop(f"{instance.name}_{editor_name}_editor", None)
        instance.data.pop(f"{instance.name}_{editor_name}_cache", None)
    instance.raw_control_editor.target_mesh_filter = "ALL"
    update_head_raw_control_list(instance)
    update_body_rbf_solver_list(instance)
    instance.sync_shape_key_list()
