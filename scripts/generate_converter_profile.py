"""Run with Blender --background --python this_file -- [options]."""

import argparse
import sys

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Transfer MetaHuman fitting landmarks to aligned template DNA")
    resources = Path(__file__).resolve().parents[1] / "src/addons/character_dna/editors/shared/resources"
    parser.add_argument("--reference", type=Path, default=resources / "dna/default")
    parser.add_argument("--target", type=Path, default=resources / "dna/3d_scan_store/full_body")
    parser.add_argument("--profile", default="3d_scan_store/full_body")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pose", action="store_true", help="Also transfer the reviewed anatomical pose landmarks")
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    sys.path.insert(0, str(resources.parents[3]))
    from character_dna.editors.shared.mapping_transfer import generate

    destination = arguments.output or resources / "topologies" / arguments.profile
    if not arguments.output and not destination.resolve().is_relative_to((resources / "topologies").resolve()):
        raise ValueError("Profile ID must stay within the topology resource directory")
    generate(
        {kind: arguments.reference / f"{kind}.dna" for kind in ("head", "body")},
        {kind: arguments.target / f"{kind}.dna" for kind in ("head", "body")},
        destination,
        arguments.profile,
    )
    if arguments.pose:
        from character_dna.editors.shared.pose_transfer import generate as generate_pose

        generate_pose(arguments.reference / "body.dna", arguments.target / "body.dna", destination, arguments.profile)


if __name__ == "__main__":
    main()
