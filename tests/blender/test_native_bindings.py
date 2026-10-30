"""Native-driver ownership must not overwrite or remove external animation."""

from types import SimpleNamespace

import bpy
import pytest

from character_dna.runtime.bindings import Target, install_targets, remove_targets


@pytest.fixture
def native_objects():
    """Provide isolated writable native carrier and target objects."""
    carrier = bpy.data.objects.new("NativeOwnershipCarrier", None)
    target = bpy.data.objects.new("NativeOwnershipTarget", None)
    bpy.context.scene.collection.objects.link(carrier)
    bpy.context.scene.collection.objects.link(target)
    carrier["outputs"] = [0.25]
    carrier["epoch"] = 0.0
    try:
        yield SimpleNamespace(carrier=carrier, target=target)
    finally:
        bpy.data.objects.remove(target, do_unlink=True)
        bpy.data.objects.remove(carrier, do_unlink=True)


def test_native_install_and_remove(native_objects):
    """Removal affects only curves recorded and owned by the carrier."""
    carrier, target = native_objects.carrier, native_objects.target
    target.driver_add("location", 2).driver.expression = "3"
    install_targets(carrier, [Target(target, "location", 0, 0)])
    assert target.animation_data.drivers.find("location", index=0)
    remove_targets(carrier)
    assert not target.animation_data.drivers.find("location", index=0)
    assert target.animation_data.drivers.find("location", index=2).driver.expression == "3"


@pytest.mark.parametrize("owner_kind", ["OBJECT", "SCENE"])
def test_driver_variable_resolves_default_and_nondefault_owner_types(native_objects, owner_kind):
    from character_dna.runtime.bindings import add_variable

    owner = native_objects.carrier if owner_kind == "OBJECT" else bpy.context.scene
    owner["test_native_input"] = 0.375
    try:
        curve = native_objects.target.driver_add("location", 0)
        add_variable(curve.driver, "input_value", owner, '["test_native_input"]')
        curve.driver.expression = "input_value"
        variable = curve.driver.variables[0]
        assert variable.type == "SINGLE_PROP"
        assert variable.targets[0].id_type == owner_kind
        assert variable.targets[0].id == owner
        assert variable.targets[0].data_path == '["test_native_input"]'
        bpy.context.view_layer.update()
        assert native_objects.target.evaluated_get(bpy.context.evaluated_depsgraph_get()).location.x == pytest.approx(
            0.375
        )
    finally:
        del owner["test_native_input"]


def test_native_preserves_user_replacement(native_objects):
    """A user-edited owned channel becomes an external writer."""
    carrier, target = native_objects.carrier, native_objects.target
    install_targets(carrier, [Target(target, "location", 0, 0)])
    target.animation_data.drivers.find("location", index=0).driver.expression = "42"
    remove_targets(carrier)
    assert target.animation_data.drivers.find("location", index=0).driver.expression == "42"


def test_native_conflict_is_transactional(native_objects):
    """A conflict discovered late in the plan cannot install an earlier target."""
    carrier, target = native_objects.carrier, native_objects.target
    target.driver_add("location", 2).driver.expression = "3"
    with pytest.raises(ValueError, match="Existing driver"):
        install_targets(carrier, [Target(target, "location", 0, 0), Target(target, "location", 2, 0)])
    assert not target.animation_data.drivers.find("location", index=0)
    assert "targets" not in carrier


def test_native_duplicate_rejected(native_objects):
    """Aliased output channels require an explicit final-write plan."""
    carrier, target = native_objects.carrier, native_objects.target
    with pytest.raises(ValueError, match="Duplicate"):
        install_targets(carrier, [Target(target, "location", 0, 0)] * 2)
    assert target.animation_data is None


def test_batched_editor_mutes_preserve_unowned_drivers(native_objects):
    from character_dna.runtime import retention

    carrier, target = native_objects.carrier, native_objects.target
    unrelated = target.driver_add("location", 2)
    unrelated.driver.expression = "3"
    unrelated.mute = True
    install_targets(carrier, [Target(target, "location", 0, 0)])
    curves = retention._curves([carrier])
    retention._set_mutes(curves, True)
    assert all(curve.mute for curve in target.animation_data.drivers)
    retention.restore_mutes(curves)
    assert not target.animation_data.drivers.find("location", index=0).mute
    assert unrelated.mute and unrelated.driver.expression == "3"


@pytest.mark.parametrize("change", ["remove", "expression", "owner"])
def test_mute_rechecks_current_ownership_and_array_channels(native_objects, change):
    from character_dna.runtime.bindings import mute_targets

    carrier, target = native_objects.carrier, native_objects.target
    install_targets(carrier, [Target(target, "location", axis, 0) for axis in range(3)])
    mute_targets(carrier, True)
    assert all(curve.mute for curve in target.animation_data.drivers)
    curve = target.animation_data.drivers.find("location", index=1)
    if change == "remove":
        target.driver_remove("location", 1)
    elif change == "expression":
        curve.driver.expression = "42"
    else:
        curve.driver.variables[0].targets[0].id = target
    mute_targets(carrier, False)
    for axis in (0, 2):
        assert not target.animation_data.drivers.find("location", index=axis).mute
    replaced = target.animation_data.drivers.find("location", index=1)
    assert replaced is None if change == "remove" else replaced.mute
    # A later call must discover a newly installed curve, with no cached RNA.
    if change == "remove":
        replacement = target.driver_add("location", 1)
        replacement.driver.expression = "7"
        mute_targets(carrier, True)
        assert not replacement.mute


def test_unchanged_mute_does_not_revalidate_or_cache_ownership(native_objects, monkeypatch):
    from character_dna.runtime import bindings

    carrier, target = native_objects.carrier, native_objects.target
    install_targets(carrier, [Target(target, "location", 0, 0)])
    curve = target.animation_data.drivers.find("location", index=0)
    with monkeypatch.context() as patches:
        patches.setattr(bindings, "owned_curve", lambda *_args: pytest.fail("No mute write needs validation"))
        bindings.mute_targets(carrier, False)
        assert not curve.mute
    # A later requested change still checks the current expression, even when
    # the preceding no-op observed this exact curve.
    curve.driver.expression = "42"
    bindings.mute_targets(carrier, True)
    assert not curve.mute


def test_native_rejects_keyframed_output(native_objects):
    """Output drivers must not hide an existing user's keyed animation."""
    carrier, target = native_objects.carrier, native_objects.target
    target.keyframe_insert("location", index=0, frame=1)
    with pytest.raises(ValueError, match="Keyframed"):
        install_targets(carrier, [Target(target, "location", 0, 0)])
    assert len(target.animation_data.drivers) == 0


@pytest.mark.parametrize("change", ["remove", "expression", "owner"])
def test_validation_rechecks_current_driver_ownership(native_objects, change):
    """The indexed validator must not reuse success or overlook one array channel."""
    from character_dna.runtime import engine

    carrier, target = native_objects.carrier, native_objects.target
    carrier["schema_version"] = engine.SCHEMA_VERSION
    solve = carrier.driver_add('["epoch"]').driver
    solve.type = "SCRIPTED"
    solve.use_self = True
    solve.expression = engine.SOLVE_EXPRESSION
    install_targets(carrier, [Target(target, "location", axis, 0) for axis in range(3)])
    carrier.property_overridable_library_set('["targets"]', True)
    assert engine.carrier_issues(carrier) == []
    curve = target.animation_data.drivers.find("location", index=1)
    if change == "remove":
        target.driver_remove("location", 1)
    elif change == "expression":
        curve.driver.expression = "42"
    else:
        curve.driver.variables[0].targets[0].id = target
    assert len(engine.carrier_issues(carrier)) == 1


@pytest.mark.parametrize("scalar_first", [True, False])
def test_copied_outputs_resolve_scalar_array_and_distinct_owners(native_objects, scalar_first):
    """Copied drivers retain independent variables and the original evaluated values."""
    from character_dna.runtime.bindings import owned_curve

    carrier, target = native_objects.carrier, native_objects.target
    carrier["outputs"] = [0.125, 0.5, 0.875]
    target["native_scalar"] = 0.0
    other = bpy.data.objects.new("NativeCopyOther", None)
    bpy.context.scene.collection.objects.link(other)
    before = {obj.as_pointer() for obj in bpy.data.objects}
    plans = [Target(target, '["native_scalar"]', -1, 0), Target(target, "location", 0, 1)]
    if not scalar_first:
        plans.reverse()
    plans.append(Target(other, "scale", 2, 2))
    try:
        install_targets(carrier, plans)
        assert {obj.as_pointer() for obj in bpy.data.objects} == before
        pointers = set()
        for plan in plans:
            curve = plan.owner.animation_data.drivers.find(plan.path, index=max(0, plan.index))
            assert owned_curve(curve, carrier)
            assert not curve.keyframe_points and not curve.modifiers
            variable = curve.driver.variables["output"]
            assert variable.targets[0].data_path == f'["outputs"][{plan.channel}]'
            pointers.add(variable.as_pointer())
        assert len(pointers) == len(plans)
        bpy.context.view_layer.update()
        graph = bpy.context.evaluated_depsgraph_get()
        assert target.evaluated_get(graph)["native_scalar"] == 0.125
        assert target.evaluated_get(graph).location.x == 0.5
        assert other.evaluated_get(graph).scale.z == 0.875
        remove_targets(carrier)
        assert not target.animation_data.drivers and not other.animation_data.drivers
    finally:
        bpy.data.objects.remove(other, do_unlink=True)


def test_yielded_output_changes_cannot_modify_later_copies(native_objects):
    """The resumable installer must not use a caller's mutable curve as its template."""
    from character_dna.runtime.bindings import DRIVER_EXPRESSION, install_targets_iter, owned_curve

    carrier, target = native_objects.carrier, native_objects.target
    before = {obj.as_pointer() for obj in bpy.data.objects}
    pending = install_targets_iter(carrier, [Target(target, "location", axis, 0) for axis in range(3)])
    _owner, first = next(pending)
    first.driver.expression = "99"
    first.driver.variables["epoch"].targets[0].id = target
    first.mute = True
    copies = list(pending)
    assert {obj.as_pointer() for obj in bpy.data.objects} == before
    for _owner, curve in copies:
        assert curve.driver.expression == DRIVER_EXPRESSION
        assert owned_curve(curve, carrier) and not curve.mute
    bpy.context.view_layer.update()
    evaluated = target.evaluated_get(bpy.context.evaluated_depsgraph_get())
    assert tuple(evaluated.location) == (0.0, 0.25, 0.25)
    remove_targets(carrier)
    assert len(target.animation_data.drivers) == 1
    assert target.animation_data.drivers[0] == first


def test_late_invalid_index_rolls_back_copies_and_template(native_objects):
    """Retain Blender's index validation and remove every partial native write."""
    carrier, target = native_objects.carrier, native_objects.target
    external = target.driver_add("rotation_euler", 2)
    external.driver.expression = "3"
    before = {obj.as_pointer() for obj in bpy.data.objects}
    with pytest.raises((TypeError, RuntimeError), match="index"):
        install_targets(carrier, [Target(target, "location", axis, 0) for axis in (0, 1, 9)])
    assert {obj.as_pointer() for obj in bpy.data.objects} == before
    assert list(target.animation_data.drivers) == [external]
    assert not carrier["targets"]


@pytest.mark.parametrize("append", [True, False])
@pytest.mark.parametrize("steps", [1, 2])
def test_cancelled_install_restores_prior_targets_without_template_leaks(native_objects, append, steps):
    from character_dna.runtime.bindings import install_targets_iter

    carrier, target = native_objects.carrier, native_objects.target
    if append:
        install_targets(carrier, [Target(target, "location", 0, 0)])
    previous = [dict(item) for item in carrier.get("targets", [])]
    before = {obj.as_pointer() for obj in bpy.data.objects}
    pending = install_targets_iter(carrier, [Target(target, "location", axis, 0) for axis in (1, 2)], append=append)
    for _step in range(steps):
        next(pending)
    # A caller can remove an already yielded curve before canceling the rest.
    target.driver_remove("location", 1)
    pending.close()
    assert {obj.as_pointer() for obj in bpy.data.objects} == before
    assert [dict(item) for item in carrier["targets"]] == previous
    assert len(target.animation_data.drivers) == int(append)


@pytest.mark.parametrize("change", ["clear_animation", "remove_owner"])
def test_cancelled_install_tolerates_removed_output_storage(native_objects, change):
    """Cancellation remains safe when a caller has already released the output ID or curves."""
    from character_dna.runtime.bindings import install_targets_iter

    carrier = native_objects.carrier
    before = {obj.as_pointer() for obj in bpy.data.objects}
    owner = bpy.data.objects.new("NativeCancelledOwner", None)
    bpy.context.scene.collection.objects.link(owner)
    pending = install_targets_iter(carrier, [Target(owner, "location", axis, 0) for axis in (0, 1)])
    alive = True
    try:
        next(pending)
        if change == "clear_animation":
            owner.animation_data_clear()
        else:
            bpy.data.objects.remove(owner, do_unlink=True)
            alive = False
        pending.close()
        assert not carrier["targets"]
    finally:
        pending.close()
        if alive:
            bpy.data.objects.remove(owner, do_unlink=True)
    assert {obj.as_pointer() for obj in bpy.data.objects} == before
