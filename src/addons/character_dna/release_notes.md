## Major Changes

* Added initial implementation of the Swing and Twist Editor. Now the Swing and twist values in DNA can be customized.
* Convertor now supports merging and splitting DNA components at the Blender scene level, as well as, the DNA file level.

## Minor Changes

* Added Shape Key support to RBF Editor. Assign Shape Key values to a RBF Pose individually, or batch per poses in all solvers.

## Patch Changes

* Fixed baking bug [#67](https://github.com/orgs/poly-hammer/discussions/67)
* Reduced repeated DNA export allocation and native driver setup work.
* Fixed stale mesh lookups during shape migration and RBF pose duplication after collection changes.

## Tests Passing On

* Blender `4.5`, `5.2` (installed from blender.org)
* Unreal `5.6`, `5.7`, `5.8`
