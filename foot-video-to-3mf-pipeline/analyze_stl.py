from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_INPUT = Path("input/test.stl")


def main() -> int:
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT

    if not input_path.exists():
        print(f"Error: STL file not found: {input_path}")
        print("Place an STL file in the input folder, for example: input/test.stl")
        return 1

    try:
        from mesh_pipeline import inspect_stl

        report = inspect_stl(input_path)
    except Exception as exc:
        print(f"Error: failed to analyze STL: {input_path}")
        print(f"Reason: {exc}")
        return 1

    print(f"STL analysis: {input_path}")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
