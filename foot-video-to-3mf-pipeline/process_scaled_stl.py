from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STL_INPUT = Path("input/test.stl")
DEFAULT_PLY_INPUT = Path("input/foot_for_scale_2.ply")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("reports")

DEFAULT_SLICE_AFTER_PROCESSING = True
DEFAULT_SLICER_ENGINE = "orca-legacy"
DEFAULT_SUPPORT_TYPE = "tree(auto)"
DEFAULT_SUPPORT_THRESHOLD_ANGLE = 30
DEFAULT_SUPPORT_ON_BUILD_PLATE_ONLY = "0"


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _require_file(path: str | Path, label: str) -> Path:
    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def _get(report: dict[str, Any] | None, key: str, default: Any = "n/a") -> Any:
    if not isinstance(report, dict) or "error" in report:
        return default
    return report.get(key, default)


def _format_report_section(title: str, report: dict[str, Any] | None) -> list[str]:
    lines = [title]
    if not isinstance(report, dict):
        lines.append("  n/a")
        return lines
    if "error" in report:
        lines.append(f"  error: {report['error']}")
        return lines

    for key in (
        "triangle_count",
        "vertex_count",
        "watertight",
        "winding_consistent",
        "boundary_edges",
        "non_manifold_edges",
        "volume_available",
        "volume",
    ):
        lines.append(f"  {key}: {report.get(key, 'n/a')}")
    return lines


def scale_stl(input_stl: str | Path, output_stl: str | Path, scale_factor: float) -> dict[str, Any]:
    """Scale a processed STL by the PLY-derived mm scale factor."""
    from mesh_pipeline import load_mesh

    input_file = _require_file(input_stl, "Processed STL")
    output_file = _as_path(output_stl)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(input_file)
    before_bounds = mesh.bounds.tolist()
    mesh.apply_scale(float(scale_factor))
    after_bounds = mesh.bounds.tolist()
    mesh.export(output_file)

    return {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "scale_factor": float(scale_factor),
        "before_bounds": before_bounds,
        "after_bounds": after_bounds,
    }


def _format_txt_report(result: dict[str, Any]) -> str:
    scale = result.get("scale", {})
    stl_processing = result.get("stl_processing", {})
    slicing = result.get("slicing", {})

    lines = [
        "Scaled STL Processing Report",
        "============================",
        "",
        f"Input STL:          {result.get('input_stl')}",
        f"Input PLY:          {result.get('input_ply')}",
        f"Processed STL:      {stl_processing.get('final_file', 'n/a')}",
        f"Scaled STL:         {result.get('scaled_stl')}",
        f"Sliced 3MF:         {slicing.get('output_file', 'n/a')}",
        "",
        "Scale",
        f"  scale_factor:       {scale.get('scale_factor', 'n/a')}",
        f"  pixel_distance:     {scale.get('pixel_distance', 'n/a')}",
        f"  checker_square_mm:  {scale.get('square_real_size_mm', 'n/a')}",
        f"  projected_image:    {scale.get('projected_image', 'n/a')}",
        f"  debug_lines_image:  {scale.get('debug_lines_image', 'n/a')}",
        "",
    ]

    lines.extend(_format_report_section("Before STL Processing", stl_processing.get("before")))
    lines.append("")
    lines.extend(_format_report_section("After STL Processing", stl_processing.get("final")))
    lines.append("")
    lines.extend(_format_report_section("After Scaling", result.get("scaled_report")))
    lines.append("")

    floating = result.get("scaled_floating_regions")
    if isinstance(floating, dict):
        lines.append("Scaled Floating Regions")
        if "error" in floating:
            lines.append(f"  error: {floating['error']}")
        else:
            after_floating = floating.get("after", {})
            lines.extend(
                [
                    f"  action: {floating.get('action', 'n/a')}",
                    f"  removed_component_count: {floating.get('removed_component_count', 'n/a')}",
                    f"  floating_components_left: {after_floating.get('floating_component_count', 'n/a')}",
                    f"  unsupported_downward_faces: {after_floating.get('unsupported_downward_face_count', 'n/a')}",
                    f"  unsupported_downward_area_ratio: {after_floating.get('unsupported_downward_area_ratio', 'n/a')}",
                    f"  support_recommended: {floating.get('support_recommended', False)}",
                ]
            )
        lines.append("")

    lines.extend(
        [
            "Slicing",
            f"  enabled: {slicing.get('enabled', False)}",
            f"  status: {slicing.get('status', 'n/a')}",
            f"  engine: {slicing.get('engine', 'n/a')}",
            f"  support: {slicing.get('enable_support', False)}",
            f"  contains_gcode: {slicing.get('contains_gcode', False)}",
            f"  output_file: {slicing.get('output_file', 'n/a')}",
        ]
    )
    if slicing.get("error"):
        lines.append(f"  error: {slicing['error']}")

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run STL repair/simplify, compute PLY scale, scale STL, and slice 3MF."
    )
    parser.add_argument("input_stl", nargs="?", default=str(DEFAULT_STL_INPUT))
    parser.add_argument("--scale-ply", default=str(DEFAULT_PLY_INPUT))
    parser.add_argument("--checker-square-mm", type=float, default=30.0)
    parser.add_argument("--scale-resolution", type=int, default=1000)
    parser.add_argument("--simplify-threshold", type=int, default=None)
    parser.add_argument("--target-triangles", type=int, default=None)
    parser.add_argument("--floating-bed-tolerance", type=float, default=None)
    parser.add_argument("--floating-remove-ratio", type=float, default=None)
    parser.add_argument("--no-slice", action="store_true")
    parser.add_argument("--sliced-output", default=None)
    parser.add_argument(
        "--slicer-engine",
        choices=["orca-legacy", "bambu", "orca"],
        default=DEFAULT_SLICER_ENGINE,
    )
    parser.add_argument("--slicer-bin", default=None)
    parser.add_argument("--force-support", action="store_true")
    parser.add_argument("--no-auto-support", action="store_true")
    parser.add_argument("--support-type", default=DEFAULT_SUPPORT_TYPE)
    parser.add_argument("--support-threshold-angle", type=int, default=DEFAULT_SUPPORT_THRESHOLD_ANGLE)
    parser.add_argument("--support-on-build-plate-only", default=DEFAULT_SUPPORT_ON_BUILD_PLATE_ONLY)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_stl = Path(args.input_stl)
    input_ply = Path(args.scale_ply)

    try:
        _require_file(input_stl, "Input STL")
        _require_file(input_ply, "Scale PLY")
    except Exception as exc:
        print(f"Error: {exc}")
        print("Example: python process_scaled_stl.py input/test.stl --scale-ply input/foot_for_scale_2.ply")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from mesh_pipeline import inspect_stl, process_stl, resolve_floating_regions
        from scale_from_ply import compute_scale_factor_from_ply

        process_kwargs: dict[str, Any] = {}
        if args.simplify_threshold is not None:
            process_kwargs["simplify_threshold"] = args.simplify_threshold
        if args.target_triangles is not None:
            process_kwargs["target_triangles"] = args.target_triangles
        if args.floating_bed_tolerance is not None:
            process_kwargs["floating_bed_tolerance"] = args.floating_bed_tolerance
        if args.floating_remove_ratio is not None:
            process_kwargs["floating_remove_triangle_ratio"] = args.floating_remove_ratio

        print("--- [1] STL analyze / repair / simplify ---")
        stl_result = process_stl(input_stl, OUTPUT_DIR, **process_kwargs)

        print("--- [2] PLY checkerboard scale factor ---")
        scale_report = compute_scale_factor_from_ply(
            input_ply,
            output_dir=OUTPUT_DIR / "scale_debug",
            resolution=args.scale_resolution,
            square_real_size_mm=args.checker_square_mm,
        )

        print("--- [3] Apply scale factor to processed STL ---")
        scaled_raw = OUTPUT_DIR / f"{input_stl.stem}_scaled_mm_raw.stl"
        scale_stl_result = scale_stl(
            stl_result["final_file"],
            scaled_raw,
            float(scale_report["scale_factor"]),
        )

        scaled_floating_file = OUTPUT_DIR / f"{input_stl.stem}_scaled_mm.stl"
        scaled_floating = resolve_floating_regions(
            scaled_raw,
            scaled_floating_file,
            bed_tolerance=0.05,
        )
        scaled_report = inspect_stl(scaled_floating_file)
    except Exception as exc:
        print("Error: scaled STL pipeline failed")
        print(f"Reason: {exc}")
        return 1

    run_slicing = DEFAULT_SLICE_AFTER_PROCESSING and not args.no_slice
    support_recommended = bool(
        scaled_floating.get("support_recommended", False)
        or stl_result.get("support_recommended", False)
    )
    enable_support = bool(args.force_support or (support_recommended and not args.no_auto_support))
    sliced_output = (
        Path(args.sliced_output)
        if args.sliced_output
        else OUTPUT_DIR / f"{input_stl.stem}_scaled_sliced.3mf"
    )

    result: dict[str, Any] = {
        "input_stl": str(input_stl),
        "input_ply": str(input_ply),
        "stl_processing": stl_result,
        "scale": scale_report,
        "scale_stl": scale_stl_result,
        "scaled_stl": str(scaled_floating_file),
        "scaled_report": scaled_report,
        "scaled_floating_regions": scaled_floating,
        "printable_enough": bool(
            scaled_report.get("triangle_count", 0) > 0
            and scaled_report.get("boundary_edges", -1) == 0
            and scaled_report.get("non_manifold_edges", -1) == 0
            and scaled_report.get("watertight", False)
        ),
        "slicing": {
            "enabled": run_slicing,
            "status": "not_run" if not run_slicing else "pending",
            "engine": args.slicer_engine if run_slicing else None,
            "support_recommended": support_recommended,
            "enable_support": enable_support,
            "output_file": str(sliced_output) if run_slicing else None,
        },
    }

    slicing_failed = False
    if run_slicing:
        try:
            print("--- [4] Slice scaled STL to 3MF ---")
            if args.slicer_engine == "orca-legacy":
                from slicer import slice_with_orca_legacy_cli_safe

                slice_result = slice_with_orca_legacy_cli_safe(
                    scaled_floating_file,
                    sliced_output,
                    slicer_bin=args.slicer_bin,
                    enable_support=enable_support,
                    support_type=args.support_type,
                    support_threshold_angle=args.support_threshold_angle,
                    support_on_build_plate_only=args.support_on_build_plate_only,
                )
            elif args.slicer_engine == "orca":
                from slicer import slice_with_orca_slicer

                slice_result = slice_with_orca_slicer(
                    scaled_floating_file,
                    sliced_output,
                    slicer_bin=args.slicer_bin,
                    enable_support=enable_support,
                    support_type=args.support_type,
                    support_threshold_angle=args.support_threshold_angle,
                )
            else:
                from slicer import slice_with_bambu_studio

                slice_result = slice_with_bambu_studio(
                    scaled_floating_file,
                    sliced_output,
                    bambu_studio_bin=args.slicer_bin,
                    enable_support=enable_support,
                    support_type=args.support_type,
                    support_threshold_angle=args.support_threshold_angle,
                )

            result["slicing"].update(
                {
                    "status": "success",
                    "success": True,
                    "engine": slice_result.get("slicer", args.slicer_engine),
                    "output_file": slice_result.get("output_file", str(sliced_output)),
                    "machine_json": slice_result.get("machine_json"),
                    "process_json": slice_result.get("process_json"),
                    "filament_json": slice_result.get("filament_json"),
                    "profile_root": slice_result.get("profile_root"),
                    "safe_profile_dir": slice_result.get("safe_profile_dir"),
                    "contains_gcode": slice_result.get("contains_gcode", False),
                    "gcode_files": slice_result.get("gcode_files", []),
                    "command": slice_result.get("command"),
                    "output_exists": slice_result.get("output_exists", False),
                    "stdout_tail": (slice_result.get("stdout") or "")[-4000:],
                    "stderr_tail": (slice_result.get("stderr") or "")[-4000:],
                }
            )
        except Exception as exc:
            slicing_failed = True
            result["slicing"].update({"status": "failed", "success": False, "error": str(exc)})

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{input_stl.stem}_scaled_{timestamp}_report.json"
    txt_report_path = REPORTS_DIR / f"{input_stl.stem}_scaled_{timestamp}_report.txt"
    try:
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        txt_report_path.write_text(_format_txt_report(result), encoding="utf-8")
    except Exception as exc:
        print(f"Error: failed to write report files: {report_path}, {txt_report_path}")
        print(f"Reason: {exc}")
        return 1

    print("Scaled STL pipeline complete")
    print(f"- Input STL:                   {input_stl}")
    print(f"- Scale PLY:                   {input_ply}")
    print(f"- Pixel distance:              {scale_report['pixel_distance']:.2f} px")
    print(f"- Scale factor:                {scale_report['scale_factor']:.8f}")
    print(f"- Processed STL:               {stl_result['final_file']}")
    print(f"- Scaled STL:                  {scaled_floating_file}")
    print(f"- Final triangle count:        {_get(scaled_report, 'triangle_count')}")
    print(f"- Final boundary edges:        {_get(scaled_report, 'boundary_edges')}")
    print(f"- Final non-manifold edges:    {_get(scaled_report, 'non_manifold_edges')}")
    print(f"- Final watertight:            {_get(scaled_report, 'watertight')}")
    print(f"- Support recommended:         {support_recommended}")
    print(f"- JSON report file:            {report_path}")
    print(f"- TXT report file:             {txt_report_path}")
    if result["slicing"]["enabled"]:
        print(f"- Slicing engine:              {result['slicing'].get('engine')}")
        print(f"- Slicing status:              {result['slicing'].get('status')}")
        print(f"- Slicing support enabled:     {result['slicing'].get('enable_support')}")
        print(f"- Sliced 3MF file:             {result['slicing'].get('output_file')}")
        print(f"- Sliced G-code inside 3MF:    {result['slicing'].get('contains_gcode', False)}")
        if slicing_failed:
            print("- Slicing error:               see report TXT/JSON for details")
            return 1
    else:
        print("- Slicing status:              skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
