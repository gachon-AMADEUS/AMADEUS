from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_INPUT = Path("input/test.stl")
OUTPUT_DIR = Path("output")


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    if not input_path.exists():
        print(f"Error: STL file not found: {input_path}")
        print("Place an STL file in the input folder, for example: input/test.stl")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{input_path.stem}_repaired.stl"

    try:
        from mesh_pipeline import inspect_stl, repair_with_pymeshlab

        before = inspect_stl(input_path)
        repair_info = repair_with_pymeshlab(input_path, output_path)
        after = inspect_stl(output_path)
    except Exception as exc:
        print(f"Error: PyMeshLab repair failed for: {input_path}")
        print(f"Reason: {exc}")
        return 1

    print("PyMeshLab repair complete")
    print(f"Input file:  {input_path}")
    print(f"Output file: {output_path}")
    print("\nBefore:")
    print(json.dumps(before, indent=2, ensure_ascii=False))
    print("\nAfter:")
    print(json.dumps(after, indent=2, ensure_ascii=False))
    print("\nPyMeshLab filter details:")
    print(json.dumps(repair_info, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
