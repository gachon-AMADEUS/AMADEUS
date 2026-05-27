from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_PROFILE_ROOT = Path.home() / "Library/Application Support/BambuStudio/system/BBL"
DEFAULT_OUTPUT_DIR = Path("output/cli_safe_profiles")

LIST_KEYS_TO_KEEP = {
    "bed_exclude_area",
    "bed_shape",
    "compatible_printers",
    "compatible_prints",
    "printable_area",
    "upward_compatible_machine",
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4, ensure_ascii=False)
        handle.write("\n")


def _resolve_inherits(
    profile_root: Path,
    profile_type: str,
    data: dict[str, Any],
    seen: set[str] | None = None,
) -> dict[str, Any]:
    """Inline Orca/Bambu profile inheritance so copied safe profiles are standalone."""
    inherited = data.get("inherits")
    if not inherited:
        return dict(data)

    inherited_names = inherited if isinstance(inherited, list) else [inherited]
    merged: dict[str, Any] = {}
    seen = seen or set()

    for inherited_name in inherited_names:
        parent_name = str(inherited_name)
        if parent_name in seen:
            continue
        seen.add(parent_name)
        parent_path = profile_root / profile_type / f"{parent_name}.json"
        if not parent_path.exists():
            continue
        parent = _resolve_inherits(
            profile_root,
            profile_type,
            _read_json(parent_path),
            seen=seen,
        )
        merged.update(parent)

    merged.update(data)
    merged.pop("inherits", None)
    return merged


def _first_only(value: Any) -> Any:
    if isinstance(value, list) and value:
        return [value[0]]
    return value


def _flatten_keys(data: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    patched = dict(data)
    for key in keys:
        if key in patched:
            patched[key] = _first_only(patched[key])
    return patched


def _flatten_most_lists(data: dict[str, Any]) -> dict[str, Any]:
    patched = dict(data)
    for key, value in list(patched.items()):
        if key in LIST_KEYS_TO_KEEP:
            continue
        patched[key] = _first_only(value)
    return patched


def create_cli_safe_profiles(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    profile_root: Path = DEFAULT_PROFILE_ROOT,
    machine_name: str = "Bambu Lab X1 Carbon 0.4 nozzle",
    process_name: str = "0.20mm Standard @BBL X1C",
    filament_name: str = "Bambu PLA Basic @BBL X1C",
    flatten_all_lists: bool = False,
    enable_support: bool = False,
    support_type: str = "tree(auto)",
    support_threshold_angle: str = "30",
    support_on_build_plate_only: str = "0",
) -> dict[str, str]:
    machine_path = profile_root / "machine" / f"{machine_name}.json"
    process_path = profile_root / "process" / f"{process_name}.json"
    filament_path = profile_root / "filament" / f"{filament_name}.json"

    machine = _resolve_inherits(profile_root, "machine", _read_json(machine_path))
    process = _resolve_inherits(profile_root, "process", _read_json(process_path))
    filament = _resolve_inherits(profile_root, "filament", _read_json(filament_path))

    machine = _flatten_keys(
        machine,
        [
            "deretraction_speed",
            "hotend_cooling_rate",
            "hotend_heating_rate",
            "long_retractions_when_cut",
            "machine_max_acceleration_e",
            "machine_max_acceleration_extruding",
            "machine_max_acceleration_retracting",
            "machine_max_acceleration_travel",
            "machine_max_acceleration_x",
            "machine_max_acceleration_y",
            "machine_max_acceleration_z",
            "machine_max_jerk_e",
            "machine_max_jerk_x",
            "machine_max_jerk_y",
            "machine_max_jerk_z",
            "machine_max_speed_e",
            "machine_max_speed_x",
            "machine_max_speed_y",
            "machine_max_speed_z",
            "nozzle_type",
            "nozzle_volume",
            "nozzle_flush_dataset",
            "printer_extruder_id",
            "printer_extruder_variant",
            "retract_before_wipe",
            "retract_length_toolchange",
            "retract_lift_above",
            "retract_lift_below",
            "retraction_length",
            "retraction_minimum_travel",
            "retraction_speed",
            "retract_restart_extra",
            "retract_restart_extra_toolchange",
            "retract_when_changing_layer",
            "retraction_distances_when_cut",
            "wipe_distance",
            "wipe",
            "z_hop",
            "z_hop_types",
        ],
    )
    process = _flatten_keys(
        process,
        [
            "bridge_speed",
            "default_acceleration",
            "enable_overhang_speed",
            "enable_height_slowdown",
            "gap_infill_speed",
            "initial_layer_speed",
            "inner_wall_speed",
            "internal_solid_infill_speed",
            "initial_layer_travel_acceleration",
            "initial_layer_acceleration",
            "initial_layer_infill_speed",
            "inner_wall_acceleration",
            "outer_wall_speed",
            "overhang_totally_speed",
            "outer_wall_acceleration",
            "overhang_1_4_speed",
            "overhang_2_4_speed",
            "overhang_3_4_speed",
            "overhang_4_4_speed",
            "print_extruder_id",
            "print_extruder_variant",
            "sparse_infill_speed",
            "support_interface_speed",
            "support_speed",
            "slowdown_start_height",
            "slowdown_start_speed",
            "slowdown_start_acc",
            "slowdown_end_height",
            "slowdown_end_speed",
            "slowdown_end_acc",
            "small_perimeter_speed",
            "small_perimeter_threshold",
            "sparse_infill_acceleration",
            "top_surface_speed",
            "top_solid_infill_flow_ratio",
            "travel_acceleration",
            "travel_short_distance_acceleration",
            "travel_speed",
            "top_surface_acceleration",
            "travel_speed_z",
            "vertical_shell_speed",
        ],
    )
    filament = _flatten_keys(
        filament,
        [
            "filament_adaptive_volumetric_speed",
            "filament_deretraction_speed",
            "filament_extruder_variant",
            "filament_flow_ratio",
            "filament_flush_temp",
            "filament_flush_volumetric_speed",
            "filament_long_retractions_when_cut",
            "filament_max_volumetric_speed",
            "filament_pre_cooling_temperature",
            "filament_ramming_volumetric_speed",
            "filament_retract_before_wipe",
            "filament_retract_restart_extra",
            "filament_retract_when_changing_layer",
            "filament_retraction_distances_when_cut",
            "filament_retraction_length",
            "filament_retraction_minimum_travel",
            "filament_retraction_speed",
            "filament_wipe",
            "filament_wipe_distance",
            "filament_z_hop",
            "filament_z_hop_types",
            "filament_ramming_travel_time",
            "long_retractions_when_ec",
            "nozzle_temperature",
            "nozzle_temperature_initial_layer",
            "retraction_distances_when_ec",
            "slow_down_min_speed",
            "filament_start_gcode",
            "volumetric_speed_coefficients",
        ],
    )

    if flatten_all_lists:
        machine = _flatten_most_lists(machine)
        process = _flatten_most_lists(process)
        filament = _flatten_most_lists(filament)

    # Older Orca/Bambu CLI builds validate relative extruder mode strictly.
    # BBL start G-code uses M83, so reset E at each layer for CLI slicing.
    for key in ("layer_gcode", "layer_change_gcode"):
        layer_gcode = str(machine.get(key) or "").strip()
        if "G92 E0" not in layer_gcode:
            machine[key] = "G92 E0" if not layer_gcode else f"{layer_gcode}\nG92 E0"

    if enable_support:
        # These are the slicer-side support parameters that are easiest to tune.
        # support_type examples seen in Bambu/Orca profiles: "normal(auto)",
        # "tree(auto)", "normal(manual)", "tree(manual)".
        # support_threshold_angle is degrees: lower values create more support.
        process["enable_support"] = "1"
        process["support_type"] = support_type
        process["support_threshold_angle"] = str(support_threshold_angle)
        process["support_on_build_plate_only"] = str(support_on_build_plate_only)

    machine_out = output_dir / "machine" / f"{machine_name}.json"
    process_out = output_dir / "process" / f"{process_name}.json"
    filament_out = output_dir / "filament" / f"{filament_name}.json"

    _write_json(machine_out, machine)
    _write_json(process_out, process)
    _write_json(filament_out, filament)

    return {
        "machine_json": str(machine_out),
        "process_json": str(process_out),
        "filament_json": str(filament_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create local single-extruder Bambu CLI-safe profiles.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--profile-root", default=str(DEFAULT_PROFILE_ROOT))
    parser.add_argument("--flatten-all-lists", action="store_true")
    parser.add_argument("--enable-support", action="store_true")
    parser.add_argument("--support-type", default="tree(auto)")
    parser.add_argument("--support-threshold-angle", default="30")
    parser.add_argument("--support-on-build-plate-only", default="0")
    args = parser.parse_args()

    result = create_cli_safe_profiles(
        output_dir=Path(args.output_dir),
        profile_root=Path(args.profile_root),
        flatten_all_lists=args.flatten_all_lists,
        enable_support=args.enable_support,
        support_type=args.support_type,
        support_threshold_angle=args.support_threshold_angle,
        support_on_build_plate_only=args.support_on_build_plate_only,
    )
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
