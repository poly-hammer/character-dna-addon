# RBF Editor

!!! info "Pro Feature"
    The RBF Editor is part of the **Pro** edition and only appears when Pro is installed.

Modify, add, and delete corrective bone poses on the selected **Head** or **Body** component. These correctives interpolate between one another using **Radial Basis Functions (RBF)**. As the driver bones move, the corresponding corrective outputs blend according to the solver's falloff function.

The **Component** selector is shared with the Raw and Shape Key editors and is locked during editing. Head DNA can contain RBF solvers too. The Driven list also displays existing shape and animated-map relationships; shape geometry is edited in the Shape Key Editor.

<video autoplay loop muted playsinline class="rounded-image" style="width:100%" poster="../../images/rbf-editor/rbf_editor.png">
  <source src="../../images/rbf-editor/rbf_editor.webm" type="video/webm">
</video>

## Solvers & Poses

- The **Solvers** list holds the RBF solvers in the selected component's DNA. A solver can use multiple drivers and joint groups.
- The **Poses** list holds the `driven` corrective bone poses for the active solver. You can have as many or as few poses as needed to make your solver interpolate to all the corrective positions.

Click **Edit** to modify a solver. While editing, you can **Add**, **Remove**, and **Mirror** solvers (mirroring maps the driving bones left ↔ right).

## Settings

Expand **Settings** while editing to choose the **Function Type**, the RBF kernel (e.g. Gaussian, Linear), with a live curve preview showing how the interpolation falls off.

## Committing

**Commit** writes your changes into the selected component's DNA; **Revert** discards them.

!!! note
    While in RBF editing mode, you must click **Apply Pose Transform Edits** before switching to another pose. This is how you associate the changes with a pose. Please note these are still ephemeral until committed to the DNA.
