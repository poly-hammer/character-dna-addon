# Shape Key Editor

!!! info "Pro Feature"
    The Shape Key Editor is part of the **Pro** edition and only appears when Pro is installed.

Edit the blend shapes (shape keys) a DNA drives. With this editor, you will sculpt directly onto the activated shape key while locking all others.

!!! note
    A basic shape key like `jaw_open` sits on top of a single `jawOpen` Raw Control value and its posed bones. Other shape keys like `MCornerDepress_NSwrinkle_Jopen_R` are only visible as correctives that activate in the face when other raw controls or shape keys are active as well.
    Thus, entering editing mode activates the dependents backward from the desired shape key you want to edit.

<video autoplay loop muted playsinline class="rounded-image" style="width:100%" poster="../../images/shape-key-editor/shape_key_editor.png">
  <source src="../../images/shape-key-editor/shape_key_editor.webm" type="video/webm">
</video>

## Importing Shape Keys

Choose **Head** or **Body** with the shared **Component** selector. Import and editing use that component's DNA. The selector is locked during an import or edit session. A component with no blend-shape targets has no shape keys to import.

If you just imported your MetaHuman, then you need to run the **Import Shape Keys** operator to actually load them from the DNA into the Blender scene. Enable **Generate Neutral** to auto-generate neutral shapes during import if you want to work with a clean slate.

## Choosing a Shape Key

The list shows every shape key channel. Filter by:

- **Mesh** — scope shape keys to a specific LOD0 mesh.
- **Name** — search a shape key by name.
- **Non-Zero** — hide-zero-value shape keys. (This can be useful to find which shape keys are activated.)
- **Value Sort** — sort by value so you can see which shape keys have the highest values.
- **Side** — a side filter (Left / Right / Center). Helpful if you only want to work on one side of the face.
- **Deltas** — the ghost button is a filter to hide shape keys that do not actually contain vertex changes (a sculpted offset different from the basis). This so you can quickly spot which shape keys hold edits. This is handy when sculpting a custom character where most shape keys have been zeroed out from the beginning.
- **Freeze** — the snowflake button locks the current list order and filtering in place so rows do not reshuffle while you scrub shape key values.

## Editing with Dependencies

Select a channel and click **Edit**. Because expressions combine through [PSDs](../terminology.md#how-an-expression-is-evaluated), a shape key often depends on other shapes and raw controls. The editor shows a **dependency list** so you can see exactly what contributes to the shape you're editing:

- The dependencies are **locked** (read-only context).
- The edited key is **unlocked**.
- Toggle each dependency's visibility to diff what you're sculpting vs what is below it.

You can make changes in **Sculpt** or **Edit** mode, using Mirror / Flip / Reset as needed, then **Commit** to write the change into the DNA or **Revert** to discard it.
