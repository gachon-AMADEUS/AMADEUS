from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("input/test.stl")
OUTPUT_DIR = Path("output")
REPORTS_DIR = Path("reports")

# process_stl.py의 기본 동작입니다.
# True이면 python process_stl.py input/test.stl 실행만으로 repair/simplify 후 slicing까지 시도합니다.
# STL 처리까지만 하고 싶으면 실행할 때 --no-slice 옵션을 붙이세요.
DEFAULT_SLICE_AFTER_PROCESSING = True

# 기본 slicer는 Orca legacy CLI입니다.
# BambuStudio 02.06.00.51 CLI와 최신 Orca CLI는 이 Mac에서 특정 BBL 프로필을
# 읽을 때 크래시가 재현되어, 실제 slicing 성공이 확인된 구버전 Orca CLI를
# 기본값으로 사용합니다. 필요하면 실행 시 --slicer-engine bambu 로 바꿀 수 있습니다.
DEFAULT_SLICER_ENGINE = "orca-legacy"

# Floating/overhang 위험이 감지되면 support를 자동으로 켭니다.
# support_type: "tree(auto)" 또는 "normal(auto)" 같은 값을 시도할 수 있습니다.
# support_threshold_angle: 각도 값입니다. 낮출수록 support가 더 많이 생깁니다.
DEFAULT_SUPPORT_TYPE = "tree(auto)"
DEFAULT_SUPPORT_THRESHOLD_ANGLE = 30
DEFAULT_SUPPORT_ON_BUILD_PLATE_ONLY = "0"


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

    lines.extend(
        [
            f"  triangle_count:      {_get(report, 'triangle_count')}",
            f"  vertex_count:        {_get(report, 'vertex_count')}",
            f"  watertight:          {_get(report, 'watertight')}",
            f"  winding_consistent:  {_get(report, 'winding_consistent')}",
            f"  boundary_edges:      {_get(report, 'boundary_edges')}",
            f"  non_manifold_edges:  {_get(report, 'non_manifold_edges')}",
            f"  volume_available:    {_get(report, 'volume_available')}",
            f"  volume:              {_get(report, 'volume')}",
        ]
    )
    return lines


def _format_txt_report(result: dict[str, Any]) -> str:
    lines = [
        "STL Processing Report",
        "=====================",
        "",
        f"Input file:        {result['input_file']}",
        f"Final file:        {result['final_file']}",
        f"Repair used:       {result['repair_used']}",
        f"Simplified:        {result['simplified']}",
        f"Printable enough:  {result['printable_enough']}",
        "",
    ]

    sections = [
        ("Before", result.get("before")),
        ("After PyMeshLab", result.get("after_pymeshlab")),
        ("After MeshFix", result.get("after_meshfix")),
        ("Final", result.get("final")),
    ]
    for title, report in sections:
        lines.extend(_format_report_section(title, report))
        lines.append("")

    floating = result.get("floating_regions")
    if isinstance(floating, dict):
        lines.append("Floating Regions")
        if "error" in floating:
            lines.append(f"  error:        {floating['error']}")
        else:
            before_floating = floating.get("before", {})
            after_floating = floating.get("after", {})
            lines.extend(
                [
                    f"  action:                         {floating.get('action', 'n/a')}",
                    f"  removed_component_count:        {floating.get('removed_component_count', 'n/a')}",
                    f"  before floating components:     {before_floating.get('floating_component_count', 'n/a')}",
                    f"  after floating components:      {after_floating.get('floating_component_count', 'n/a')}",
                    f"  unsupported downward faces:     {after_floating.get('unsupported_downward_face_count', 'n/a')}",
                    f"  unsupported downward area ratio:{after_floating.get('unsupported_downward_area_ratio', 'n/a')}",
                    f"  support_recommended:            {floating.get('support_recommended', False)}",
                ]
            )
        lines.append("")

    slicing = result.get("slicing")
    if isinstance(slicing, dict):
        lines.extend(
            [
                "Slicing",
                f"  enabled:      {slicing.get('enabled', False)}",
                f"  status:       {slicing.get('status', 'n/a')}",
                f"  engine:       {slicing.get('engine', 'n/a')}",
                f"  support_recommended: {slicing.get('support_recommended', False)}",
                f"  support:      {slicing.get('enable_support', False)}",
                f"  output_file:  {slicing.get('output_file', 'n/a')}",
            ]
        )
        if slicing.get("support_note"):
            lines.append(f"  support_note: {slicing['support_note']}")
        if slicing.get("error"):
            lines.append(f"  error:        {slicing['error']}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STL repair/simplify pipeline.")
    parser.add_argument("input_stl", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument(
        "--simplify-threshold",
        type=int,
        default=None,
        help="Simplify only when triangle count is above this value.",
    )
    parser.add_argument(
        "--target-triangles",
        type=int,
        default=None,
        help="Target triangle count after simplify.",
    )
    parser.add_argument(
        "--slice",
        action="store_true",
        help="Deprecated compatibility option. Slicing is enabled by default.",
    )
    parser.add_argument(
        "--no-slice",
        action="store_true",
        help="Skip CLI slicing and only output the repaired/simplified STL.",
    )
    parser.add_argument(
        "--sliced-output",
        default=None,
        help="Output path for sliced .3mf. Defaults to output/<name>_sliced.3mf.",
    )
    parser.add_argument(
        "--slicer-engine",
        choices=["orca-legacy", "bambu", "orca"],
        default=DEFAULT_SLICER_ENGINE,
        help="Slicer CLI engine. Default is orca-legacy because it avoids the BambuStudio CLI crash.",
    )
    parser.add_argument(
        "--slicer-bin",
        default=None,
        help="Optional explicit slicer executable path.",
    )
    parser.add_argument(
        "--force-support",
        action="store_true",
        help="Enable support generation even if floating-region inspection does not request it.",
    )
    parser.add_argument(
        "--no-auto-support",
        action="store_true",
        help="Do not auto-enable supports when floating/overhang risk is detected.",
    )
    parser.add_argument(
        "--support-type",
        default=DEFAULT_SUPPORT_TYPE,
        help='Support style for orca-legacy generated profile, for example "tree(auto)" or "normal(auto)".',
    )
    parser.add_argument(
        "--support-threshold-angle",
        type=int,
        default=DEFAULT_SUPPORT_THRESHOLD_ANGLE,
        help="Support threshold angle in degrees. Lower values create more support.",
    )
    parser.add_argument(
        "--support-on-build-plate-only",
        default=DEFAULT_SUPPORT_ON_BUILD_PLATE_ONLY,
        help='Use "1" for build-plate-only support or "0" for support from model as needed.',
    )
    parser.add_argument(
        "--floating-bed-tolerance",
        type=float,
        default=None,
        help="Z tolerance for detecting components floating above the bed.",
    )
    parser.add_argument(
        "--floating-remove-ratio",
        type=float,
        default=None,
        help="Remove floating fragments only when they are under this triangle ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input_stl)

    if not input_path.exists():
        print(f"Error: STL file not found: {input_path}")
        print("Place an STL file in the input folder, for example: input/test.stl")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from mesh_pipeline import process_stl

        process_kwargs: dict[str, Any] = {}
        if args.simplify_threshold is not None:
            process_kwargs["simplify_threshold"] = args.simplify_threshold
        if args.target_triangles is not None:
            process_kwargs["target_triangles"] = args.target_triangles
        if args.floating_bed_tolerance is not None:
            process_kwargs["floating_bed_tolerance"] = args.floating_bed_tolerance
        if args.floating_remove_ratio is not None:
            process_kwargs["floating_remove_triangle_ratio"] = args.floating_remove_ratio

        result = process_stl(input_path, OUTPUT_DIR, **process_kwargs)
    except Exception as exc:
        print(f"Error: full STL pipeline failed for: {input_path}")
        print(f"Reason: {exc}")
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"{input_path.stem}_{timestamp}_report.json"
    txt_report_path = REPORTS_DIR / f"{input_path.stem}_{timestamp}_report.txt"

    run_slicing = (DEFAULT_SLICE_AFTER_PROCESSING or args.slice) and not args.no_slice
    slicing_failed = False
    support_recommended = bool(result.get("support_recommended", False))
    enable_support = bool(args.force_support or (support_recommended and not args.no_auto_support))
    support_note = (
        "Floating/overhang risk was detected, so the generated CLI-safe slicing "
        "profile enables support automatically."
        if support_recommended
        else None
    )
    result["slicing"] = {
        "enabled": run_slicing,
        "status": "not_run" if not run_slicing else "pending",
        "engine": args.slicer_engine if run_slicing else None,
        "support_recommended": support_recommended,
        "enable_support": enable_support,
        "support_note": support_note,
        "output_file": None,
    }

    if run_slicing:
        sliced_output = (
            Path(args.sliced_output)
            if args.sliced_output
            else OUTPUT_DIR / f"{input_path.stem}_sliced.3mf"
        )
        result["slicing"]["output_file"] = str(sliced_output)
        try:
            if args.slicer_engine == "orca-legacy":
                from slicer import slice_with_orca_legacy_cli_safe

                slice_result = slice_with_orca_legacy_cli_safe(
                    result["final_file"],
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
                    result["final_file"],
                    sliced_output,
                    slicer_bin=args.slicer_bin,
                    enable_support=enable_support,
                    support_type=args.support_type,
                    support_threshold_angle=args.support_threshold_angle,
                )
            else:
                from slicer import slice_with_bambu_studio

                slice_result = slice_with_bambu_studio(
                    result["final_file"],
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
                    "enable_support": slice_result.get("enable_support", False),
                    "engine": slice_result.get("slicer", args.slicer_engine),
                    "output_file": slice_result["output_file"],
                    "machine_json": slice_result.get("machine_json"),
                    "process_json": slice_result.get("process_json"),
                    "filament_json": slice_result.get("filament_json"),
                    "profile_root": slice_result.get("profile_root"),
                    "safe_profile_dir": slice_result.get("safe_profile_dir"),
                    "contains_gcode": slice_result.get("contains_gcode", False),
                    "gcode_files": slice_result.get("gcode_files", []),
                    "command": slice_result.get("command"),
                    "output_exists": slice_result.get("output_exists"),
                    "stdout_tail": (slice_result.get("stdout") or "")[-4000:],
                    "stderr_tail": (slice_result.get("stderr") or "")[-4000:],
                }
            )
        except Exception as exc:
            slicing_failed = True
            result["slicing"].update(
                {
                    "status": "failed",
                    "success": False,
                    "error": str(exc),
                }
            )

    try:
        report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        txt_report_path.write_text(_format_txt_report(result), encoding="utf-8")
    except Exception as exc:
        print(f"Error: failed to write report files: {report_path}, {txt_report_path}")
        print(f"Reason: {exc}")
        return 1

    before = result["before"]
    final = result["final"]

    print("STL processing complete")
    print(f"- Before triangle count:        {_get(before, 'triangle_count')}")
    print(f"- Before boundary edges:        {_get(before, 'boundary_edges')}")
    print(f"- Before non-manifold edges:    {_get(before, 'non_manifold_edges')}")
    print(f"- Repair used:                  {result['repair_used']}")
    print(f"- Simplify applied:             {result['simplified']}")
    floating = result.get("floating_regions")
    if isinstance(floating, dict) and "error" not in floating:
        after_floating = floating.get("after", {})
        print(f"- Floating cleanup action:      {floating.get('action', 'n/a')}")
        print(f"- Floating components left:     {after_floating.get('floating_component_count', 'n/a')}")
        print(f"- Support recommended:          {floating.get('support_recommended', False)}")
    elif isinstance(floating, dict):
        print(f"- Floating cleanup warning:     {floating.get('error')}")
    print(f"- Final triangle count:         {_get(final, 'triangle_count')}")
    print(f"- Final boundary edges:         {_get(final, 'boundary_edges')}")
    print(f"- Final non-manifold edges:     {_get(final, 'non_manifold_edges')}")
    print(f"- Final watertight:             {_get(final, 'watertight')}")
    print(f"- Printable enough:             {result['printable_enough']}")
    print(f"- Final file:                   {result['final_file']}")
    print(f"- JSON report file:             {report_path}")
    print(f"- TXT report file:              {txt_report_path}")

    if isinstance(result.get("after_pymeshlab"), dict) and "error" in result["after_pymeshlab"]:
        print(f"- PyMeshLab warning:            {result['after_pymeshlab']['error']}")
    if isinstance(result.get("after_meshfix"), dict) and "error" in result["after_meshfix"]:
        print(f"- MeshFix warning:              {result['after_meshfix']['error']}")

    slicing = result["slicing"]
    if slicing["enabled"]:
        print(f"- Slicing engine:               {slicing.get('engine', 'n/a')}")
        print(f"- Slicing status:               {slicing['status']}")
        print(f"- Slicing support enabled:      {slicing.get('enable_support', False)}")
        print(f"- Sliced 3MF file:              {slicing['output_file']}")
        print(f"- Sliced G-code inside 3MF:     {slicing.get('contains_gcode', False)}")
        if slicing.get("support_note"):
            print(f"- Slicing support note:         {slicing['support_note']}")
        if slicing_failed:
            print("- Slicing error:                see report TXT/JSON for full details")
            return 1
    else:
        print("- Slicing status:               skipped")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
