# Converter

!!! info "Pro Feature"
    The Converter is part of the **Pro** edition and only appears when Pro is installed.

Re-fit a base MetaHuman DNA onto your own head and body meshes, producing a new `.dna` calibrated to your geometry.

<video autoplay loop muted playsinline class="rounded-image" style="width:100%" poster="../../images/converter/auto_fitting_algorithm.png">
  <source src="../../images/converter/auto_fitting_algorithm.webm" type="video/webm">
</video>

This uses the UV layout of your wrapped mesh to relocate the DNA's bones and shapes to match.

## Basic Setup

| Field | Purpose |
| ----- | ------- |
| Base DNA | The source `.dna` to convert from |
| Name | Filename for the new DNA |
| Output | Folder the new DNA is written to |
| Head | Your head mesh to fit |
| Body | Your body mesh to fit |

!!! warning
    Auto-fitting is **UV based**. Your mesh must share the base MetaHuman UV layout with no overlapping or extra islands, or the fit won't be correct. See [Mesh Wrapping](../workflows/mesh-wrapping.md) for how to produce a compatible mesh.

## Advanced

Expand **Advanced** for finer control:

- **Reference Meshes** — Base-DNA meshes (eyes, teeth, etc.) to fit during conversion. Each can use a **Wrap** or **Relative** fit method.
- **Constrain Head to Body** — align overlapping head/body bones so the two components meet cleanly (avoids a seam).
- **Zero Shape Deltas** — reset shape key deltas during conversion.
- **Validate UVs** / **Tolerance** — check UV consistency within a tolerance before fitting.

**Normalize to A-Pose** selects the regions whose posture is corrected during conversion:

| Mode | Behavior |
| ---- | -------- |
| Off | Retain the input posture. |
| Limbs Only (default) | Normalize clavicles, arms, hands, fingers, legs, feet, and toes. |
| Spine Only | Normalize the spine, neck, and head alignment. Limbs follow their attachments while retaining local articulation. |
| All | Normalize the limbs and spine together. |

## LOD Calibration

The Converter calibrates lower LODs from your LOD0 mesh automatically when the DNA conversion happens. This keeps all levels in sync with that initial `Convert to DNA` operation.

<video autoplay loop muted playsinline class="rounded-image" style="width:100%" poster="../../images/converter/lod_calibration.png">
  <source src="../../images/converter/lod_calibration.webm" type="video/webm">
</video>

## Convert to DNA

Runs the conversion and writes the new `.dna` files in a similar fashion to the **MetaHuman Creator** DCC export process.
