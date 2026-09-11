## Major Changes

* Re-designed RigLogic integration for x3.7 speed up to total rig instance evaluation

## Minor Changes

* Updated Legacy migration operator to convert existing rig instances to newly re-designed RigLogic integration
* Updated Append and Link functionality to append data within a rig instance collection, while maintaining RigLogic connection
* The Link operation now has a `Editable Rig` option to make rig controls native to the scene, while linking everything else

## Patch Changes

* Fixed face board import to properly exclude bones not effecting DNA GUI controls
* Fixed a character's collections being moved out of the view layer when its asset collection was deleted
* Fixed the face rig silently stopping evaluation after a re-initialize that Blender would not let complete
* Fixed baking the face board when its action held channels other than pose bone transforms
* Fixed Blender freezing or crashing part way through rendering an animation with rig logic evaluation turned on
* Fixed the character's face lagging a frame behind the face board when rendering an animation
* Fixed DNA files never being released and from memory
* Fixed Face Board origin on second MetaHuman import
* Fixed the eyes not aiming at the eyes aim control when the head bone is rotated and only the head DNA has been imported [#309](https://github.com/poly-hammer/character-dna-addon/issues/309)
* Fixed the head RBFs not evaluating in real time when the head bone is rotated and only the head DNA has been imported [#359](https://github.com/poly-hammer/character-dna-addon/issues/359)

## Tests Passing On

* Blender `4.5`, `5.2` (installed from blender.org)
* Unreal `5.6`, `5.7`, `5.8`
