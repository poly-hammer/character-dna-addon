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
