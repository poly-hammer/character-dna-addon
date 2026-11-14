# Custom topology metadata

For a custom character, select **Output → Method → Overwrite**, then enter a
**Topology Type**, for example `3D Scan Store`. Head and body have independent Topology Type fields.
Import restores those fields from the DNA. An empty field inherits the source
DNA's type. Exporting custom DNA can be used for building custom templates in the Converter.

## Preparing a neckline

On each custom `head_lodN_mesh` and `body_lodN_mesh`, create the vertex group
`SEAM_neckline` and assign the complete neck boundary.

Merge requires both boundary loops, equal vertex counts, and an unambiguous
one-to-one correspondence within 1 mm in rest space.

## Descriptor keys for other tools

| Key | Value |
| --- | --- |
| `character_dna.topology.schema` | `1` |
| `character_dna.topology.head.type` | Head type, for example `3D Scan Store` |
| `character_dna.topology.body.type` | Body type |
| `character_dna.mesh_data.head` | Head payload manifest |
| `character_dna.mesh_data.body` | Body payload manifest |
| `character_dna.mesh_data.<component>.000000` | First payload chunk; subsequent chunks use consecutive six-digit suffixes |

Type values are ordinary strings, readable with the SDK's `getMetaDataValue`.

## Mesh payload schema 1

```json
{
  "schema": 1,
  "component": "head",
  "space": "component",
  "meshes": [
    {
      "mesh_index": 0,
      "name": "head_lod0_mesh",
      "lods": [0],
      "topology": {
        "vertices": 12576,
        "faces": 12520,
        "corners": 50080,
        "sha256": "<topology fingerprint>"
      },
      "colors": [],
      "neckline": ["<ordered DNA position indices>"]
    }
  ]
}
```
