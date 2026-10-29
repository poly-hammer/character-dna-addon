# Swing and Twist Editor

The Swing and Twist Editor adjusts the behavior of swing and twist bones in a component's DNA.

Blend values range from 0 to 1 and are independent: they do not need to add up to 1. A valid setup may contain weights such as `0.5, 0.0`.
The displayed bone rotations include local compensation, so a blend value is not an absolute world-space rotation percentage.

**Axis** is expressed in the driver's local Blender coordinate system. When swing and twist share their quaternion inputs,
changing the axis keeps the matching setup's axis in sync to match OpenRigLogic's evaluation. Its blend weights stay unchanged.
Conflicting or unsupported existing setups are
shown with an explanation and are not silently repaired.

Each affected bone also has selection and reset buttons beside its blend value. Resetting a blend restores only that
value to the start of the session; resetting the driver preview returns its rotation offset to **0°**.
Only blend weights and the decomposition axis are authored. **Preview Rotation** is a preview tool:
changing it alone does not change the DNA on Commit, and both Commit and Revert restore the original driver pose.
