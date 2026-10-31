# Test suite

Keep behavior checks at the smallest boundary that can detect the regression.
The complete supported CI matrix still runs every existing numerical case.

## Run tests

```powershell
# Python domain tests; this command fails if anything imports Blender.
python -m pytest tests/unit --require-no-blender -o addopts='' -q

# Registered Blender, native DNA and complete character integration.
python -m pytest tests/blender

# Complete suite with phase timings, including fixture construction.
python -m pytest --test-run-report=reports/test-run.json

# Diagnose order dependence or compare cached fixtures against real imports.
python -m pytest tests/blender/test_rig_fixtures.py --test-order-seed=17
python -m pytest tests/blender/test_swing_twist_editor.py --fresh-rigs
```

Use the project's configured Python environment for Blender tests. The standalone
domain job needs only pytest and coverage, which the measurement-tool tests use.
`RUNNING_CI=1` selects the same corpus as CI; do not compare timings with different
values. PyTorch solver tests keep their own worker process to avoid loading Torch
into Blender's process.

## Choose a fixture

| Contract | Preferred boundary |
| --- | --- |
| Names, classification and domain rules | Production Python functions in `unit/`, without `bpy` substitutes |
| RNA selection, property callbacks and validation | `minimal_character` or small real Blender objects |
| Reader/writer and RigLogic numerical behavior | Owned real SDK handles, without a scene when the scene is irrelevant |
| Editor commit/revert, deformation and native bindings | A `rig_factory.lease` with the required `RigProfile` |
| DNA import, portability and process lifecycle | The actual importer, load operation or subprocess under test |

For example, a body editor test can request its component without constructing
an unused head or face board:

```python
@pytest.fixture
def character(rig_factory, tmp_path):
    profile = RigProfile(components=("body",))
    with rig_factory.lease(profile, tmp_path) as instance:
        yield instance
```

Profiles explicitly describe LODs, geometry, materials, face boards, shape keys,
animation and document mode. Default imports contain LOD0 head and body geometry.
An importer regression must use `fresh=True` or call the importer directly.
Do not turn off native evaluation in a test whose assertion depends on it.

For transaction and lifecycle checks, `compact_dna` retains Ada's complete native
joint/control/solver definitions and every LOD, with one triangle per mesh and
one existing target per mapped mesh. The fixture supplies its matching seam map.
The native behavior canary compares every joint, shape and animated-map output
against the original DNA at every LOD. Leases still use the real importer,
RigLogic, bindings, file writes, reload and rollback operations.

`RigProfile(pose_bones=...)` can further limit scene bones to observed outputs,
all native input bones, the head attachment and their ancestors. It leaves the
DNA and evaluated native output arrays complete. Use this only for ownership or
transaction checks; full skeleton and deformation comparisons retain the original
scene. Geometry goldens, full skinning, UV reconstruction, all 42 long-name
targets and the calf-back-90 preview regression use the original full-size DNA.

## State ownership

The factory caches pristine files within one pytest process. Every lease owns
its writable DNA files, output directory and reopened Blender scene. Combined
documents receive private authoritative files and disposable SDK partitions.
All RNA objects and SDK handles must be reacquired after a scene reload.
Lease teardown releases native handles before removing its own materialized SDK
partitions. Pytest removes successful per-test directories and retains failed ones.

Never cache a live rig instance, mesh, key block or mutable SDK handle at session
scope. Module-scoped scenes are reserved for existing read-only numerical
matrices; a test that replaces the scene cannot borrow those fixtures. Cache
immutable arrays or exported reference files instead. Release native state before
its manager and reader, and always pair editor entry with commit/revert or the
owner's failure cleanup.

The fixture canaries exercise pose, key, visibility, auto-key, file-path and
failed-editor isolation. Extend those checks when adding a new profile option.

Native graph tests share only an immutable, process-local archive. Each module
restores its own containers and verifies that its reference graphs were not
mutated. Tests that edit a graph must copy it first. The archive is built through
the real SDK and composition path, and that construction time remains part of
the complete run.

## Mocking and measurement

Mock a collaborator only when that collaborator is outside the behavior being
asserted. Prefer autospecced boundaries and verify arguments and effects. Keep a
real integration anchor for the behavior moved out of a parameter matrix, and
record the boundary in `scripts/test_suite_refactor/contracts.json`. Do not mock
the calculation being checked, replace numerical goldens, loosen tolerances or
drop parameter cases to meet a runtime budget.

The [benchmark guide](../scripts/test_suite_refactor/README.md) covers complete
matched runs, cold caches, exact case/coverage comparison and mutation checks.
Parallel local Blender runs, incomplete baselines and focused microbenchmarks
cannot establish the full-suite runtime target.
