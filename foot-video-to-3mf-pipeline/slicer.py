from __future__ import annotations

import argparse
import subprocess
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("output/test_processed.stl")
DEFAULT_OUTPUT = Path("output/test_sliced.3mf")

# Default slicing profile names.
# Change these three values when you want a different printer, layer profile,
# or filament. The code searches Bambu Studio's local profile folders for
# files named like "<profile name>.json".
DEFAULT_MACHINE_NAME = "Bambu Lab X1 Carbon 0.4 nozzle"
DEFAULT_PROCESS_NAME = "0.20mm Standard @BBL X1C"
DEFAULT_FILAMENT_NAME = "Bambu PLA Basic @BBL X1C"

# Bambu Studio keeps system/user profile JSON files under these folders.
# Add another folder here if you export or store custom profile JSONs elsewhere.
BAMBU_PROFILE_ROOTS = [
    Path.home() / "Library/Application Support/BambuStudio/system/BBL",
    Path.home() / "Library/Application Support/BambuStudio/user/default",
]

# Default slicer executable locations on this Mac.
# If Bambu Studio or OrcaSlicer is installed somewhere else, either add that
# path here or pass --slicer-bin from VS Code/terminal.
BAMBU_BIN_CANDIDATES = [
    Path("/Applications/BambuStudio.app/Contents/MacOS/BambuStudio"),
    Path.home() / "Desktop/BambuStudio.app/Contents/MacOS/BambuStudio",
]

ORCA_BIN_CANDIDATES = [
    # This older Orca CLI has been verified to slice successfully on this Mac.
    # It avoids the BambuStudio 02.06 / recent Orca crash in
    # update_values_to_printer_extruders_for_multiple_filaments.
    Path.home() / "Applications/OrcaSlicer-CLI-1.9.5.app/Contents/MacOS/OrcaSlicer",
    Path("/Volumes/OrcaSlicer 1/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"),
    Path("/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"),
    Path("/Volumes/OrcaSlicer/OrcaSlicer.app/Contents/MacOS/OrcaSlicer"),
    Path.home() / "Desktop/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
]


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser()


def _require_file(path: str | Path, label: str) -> Path:
    file_path = _as_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"{label} not found: {file_path}")
    if not file_path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {file_path}")
    return file_path


def _ensure_output_file(path: str | Path) -> Path:
    file_path = _as_path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists() and path.is_file():
            return path
    return None


def find_bambu_studio_bin() -> Path:
    found = _first_existing(BAMBU_BIN_CANDIDATES)
    if found:
        return found
    raise FileNotFoundError(
        "Bambu Studio executable was not found. Install Bambu Studio or pass --slicer-bin."
    )


def find_orca_slicer_bin() -> Path:
    found = _first_existing(ORCA_BIN_CANDIDATES)
    if found:
        return found
    raise FileNotFoundError(
        "OrcaSlicer executable was not found. Install OrcaSlicer or pass --slicer-bin."
    )


def _find_profile(profile_type: str, profile_name: str) -> Path:
    wanted = f"{profile_name}.json"
    matches: list[Path] = []

    for root in BAMBU_PROFILE_ROOTS:
        profile_dir = root / profile_type
        if not profile_dir.exists():
            continue
        exact = profile_dir / wanted
        if exact.exists():
            return exact
        matches.extend(sorted(profile_dir.glob(f"*{profile_name}*.json")))

    if matches:
        return matches[0]

    searched = ", ".join(str(root / profile_type) for root in BAMBU_PROFILE_ROOTS)
    raise FileNotFoundError(
        f"Could not find {profile_type} profile '{profile_name}'. Searched: {searched}"
    )


def find_bambu_profiles(
    machine_name: str = DEFAULT_MACHINE_NAME,
    process_name: str = DEFAULT_PROCESS_NAME,
    filament_name: str = DEFAULT_FILAMENT_NAME,
) -> dict[str, Path]:
    return {
        "machine_json": _find_profile("machine", machine_name),
        "process_json": _find_profile("process", process_name),
        "filament_json": _find_profile("filament", filament_name),
    }


def _resolve_profiles(
    machine_json: str | Path | None,
    process_json: str | Path | None,
    filament_json: str | Path | None,
    machine_name: str,
    process_name: str,
    filament_name: str,
) -> dict[str, Path]:
    if machine_json and process_json and filament_json:
        return {
            "machine_json": _as_path(machine_json),
            "process_json": _as_path(process_json),
            "filament_json": _as_path(filament_json),
        }

    profiles = find_bambu_profiles(machine_name, process_name, filament_name)
    if machine_json:
        profiles["machine_json"] = _as_path(machine_json)
    if process_json:
        profiles["process_json"] = _as_path(process_json)
    if filament_json:
        profiles["filament_json"] = _as_path(filament_json)
    return profiles


def _run_command(command: list[str], timeout_seconds: int) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(
            f"Slicer command timed out after {timeout_seconds} seconds."
        ) from exc

    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "success": completed.returncode == 0,
    }


def _profile_root_from_slicer_bin(slicer_bin: str | Path) -> Path:
    """Return the bundled BBL profile root inside a Bambu/Orca app bundle."""
    bin_path = _as_path(slicer_bin)
    return bin_path.parents[1] / "Resources/profiles/BBL"


def verify_sliced_3mf(path: str | Path) -> dict[str, Any]:
    output_file = _require_file(path, "Sliced 3MF")
    try:
        with zipfile.ZipFile(output_file) as archive:
            names = archive.namelist()
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Output 3MF is not a readable ZIP archive: {output_file}") from exc

    gcode_files = [
        name
        for name in names
        if name.startswith("Metadata/") and name.endswith(".gcode")
    ]
    return {
        "contains_gcode": bool(gcode_files),
        "gcode_files": gcode_files,
        "entry_count": len(names),
    }


def build_bambu_studio_command(
    input_stl: str | Path,
    output_3mf: str | Path,
    bambu_studio_bin: str | Path,
    machine_json: str | Path,
    process_json: str | Path,
    filament_json: str | Path,
    plate_index: int = 0,
    # True: let Bambu Studio try to choose a good orientation.
    # False: keep the STL orientation as-is.
    orient: bool = True,
    # True: place the object on the build plate automatically.
    # False: preserve the loaded placement.
    arrange: bool = True,
    # True: lift/adjust the object so it is on the print bed.
    ensure_on_bed: bool = True,
    # True: use the first loaded filament when the STL has no filament mapping.
    load_default_filament: bool = True,
    # Explicit object-to-filament mapping. For one STL, "1" tells Bambu Studio
    # to use filament slot 1 and avoids an empty loaded_filament_ids list.
    filament_ids: list[int] | None = None,
    # 0=fatal, 1=error, 2=warning, 3=info, 4=debug, 5=trace.
    # Use 2 for normal runs; use 3-5 only when debugging slicer failures.
    debug_level: int = 2,
    extra_args: list[str] | None = None,
) -> list[str]:
    input_file = _require_file(input_stl, "Input STL")
    output_file = _ensure_output_file(output_3mf)
    slicer_bin = _require_file(bambu_studio_bin, "Bambu Studio executable")
    machine_file = _require_file(machine_json, "Machine profile JSON")
    process_file = _require_file(process_json, "Process profile JSON")
    filament_file = _require_file(filament_json, "Filament profile JSON")
    settings_files = [str(machine_file.resolve()), str(process_file.resolve())]

    command = [
        str(slicer_bin),
        "--load-settings",
        ";".join(settings_files),
        "--load-filaments",
        str(filament_file.resolve()),
        "--slice",
        str(plate_index),
        "--debug",
        str(debug_level),
        "--export-3mf",
        str(output_file.resolve()),
    ]

    if orient:
        command.extend(["--orient", "1"])
    if arrange:
        command.extend(["--arrange", "1"])
    if ensure_on_bed:
        command.append("--ensure-on-bed")
    if load_default_filament:
        # Bambu Studio's CLI parses "--load-defaultfila 1" as if "1" were
        # another input model on some versions, so keep the value attached.
        command.append("--load-defaultfila=1")
    if filament_ids:
        command.extend(["--load-filament-ids", ",".join(str(int(item)) for item in filament_ids)])
    if extra_args:
        command.extend(extra_args)

    command.append(str(input_file.resolve()))
    return command


def slice_with_bambu_studio(
    input_stl: str | Path,
    output_3mf: str | Path,
    bambu_studio_bin: str | Path | None = None,
    machine_json: str | Path | None = None,
    process_json: str | Path | None = None,
    filament_json: str | Path | None = None,
    machine_name: str = DEFAULT_MACHINE_NAME,
    process_name: str = DEFAULT_PROCESS_NAME,
    filament_name: str = DEFAULT_FILAMENT_NAME,
    plate_index: int = 0,
    orient: bool = True,
    arrange: bool = True,
    ensure_on_bed: bool = True,
    load_default_filament: bool = True,
    filament_ids: list[int] | None = None,
    enable_support: bool = False,
    support_type: str = "tree(auto)",
    support_threshold_angle: int = 30,
    debug_level: int = 2,
    timeout_seconds: int = 900,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Slice an STL with Bambu Studio CLI and export a sliced 3MF.

    If profile JSON paths are omitted, this function searches Bambu Studio's installed
    profile folders using the given profile names.
    """
    slicer_bin = _as_path(bambu_studio_bin) if bambu_studio_bin else find_bambu_studio_bin()

    profiles = _resolve_profiles(
        machine_json=machine_json,
        process_json=process_json,
        filament_json=filament_json,
        machine_name=machine_name,
        process_name=process_name,
        filament_name=filament_name,
    )
    active_process_json = profiles["process_json"]
    support_cli_warning = None
    if enable_support:
        support_cli_warning = (
            "Bambu Studio CLI 02.06.00.51 does not accept an external support "
            "override profile reliably. Enable supports in the GUI or use a saved "
            "Bambu process profile that already has supports enabled."
        )

    command = build_bambu_studio_command(
        input_stl=input_stl,
        output_3mf=output_3mf,
        bambu_studio_bin=slicer_bin,
        machine_json=profiles["machine_json"],
        process_json=active_process_json,
        filament_json=profiles["filament_json"],
        plate_index=plate_index,
        orient=orient,
        arrange=arrange,
        ensure_on_bed=ensure_on_bed,
        load_default_filament=load_default_filament,
        filament_ids=filament_ids if filament_ids is not None else [1],
        debug_level=debug_level,
        extra_args=extra_args,
    )

    result: dict[str, Any] = {
        "implemented": True,
        "slicer": "bambu_studio",
        "input_file": str(_as_path(input_stl)),
        "output_file": str(_as_path(output_3mf)),
        "machine_json": str(profiles["machine_json"]),
        "process_json": str(active_process_json),
        "base_process_json": str(profiles["process_json"]),
        "filament_json": str(profiles["filament_json"]),
        "support_process_json": None,
        "enable_support": enable_support,
        "support_cli_warning": support_cli_warning,
        "command": command,
        "dry_run": dry_run,
    }

    if dry_run:
        result.update({"success": True, "message": "Dry run only. Command was not executed."})
        return result

    run_result = _run_command(command, timeout_seconds)
    result.update(run_result)
    if not result["success"]:
        raise RuntimeError(
            "Bambu Studio slicing failed.\n"
            f"Return code: {run_result['returncode']}\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{run_result['stdout']}\n"
            f"STDERR:\n{run_result['stderr']}"
        )

    output_file = _as_path(output_3mf)
    result["output_exists"] = output_file.exists()
    if not output_file.exists():
        raise RuntimeError(f"Slicing finished but output file was not created: {output_file}")
    result.update(verify_sliced_3mf(output_file))
    if not result["contains_gcode"]:
        raise RuntimeError(f"3MF was created but does not contain sliced G-code: {output_file}")
    return result


def build_orca_slicer_command(
    input_stl: str | Path,
    output_3mf: str | Path,
    slicer_bin: str | Path,
    machine_json: str | Path,
    process_json: str | Path,
    filament_json: str | Path,
    plate_index: int = 0,
    orient: bool = True,
    arrange: bool = True,
    ensure_on_bed: bool = True,
    load_default_filament: bool = True,
    filament_ids: list[int] | None = None,
    debug_level: int = 2,
    extra_args: list[str] | None = None,
) -> list[str]:
    return build_bambu_studio_command(
        input_stl=input_stl,
        output_3mf=output_3mf,
        bambu_studio_bin=slicer_bin,
        machine_json=machine_json,
        process_json=process_json,
        filament_json=filament_json,
        plate_index=plate_index,
        orient=orient,
        arrange=arrange,
        ensure_on_bed=ensure_on_bed,
        load_default_filament=load_default_filament,
        filament_ids=filament_ids,
        debug_level=debug_level,
        extra_args=extra_args,
    )


def slice_with_orca_slicer(
    input_stl: str | Path,
    output_3mf: str | Path,
    slicer_bin: str | Path | None = None,
    machine_json: str | Path | None = None,
    process_json: str | Path | None = None,
    filament_json: str | Path | None = None,
    machine_name: str = DEFAULT_MACHINE_NAME,
    process_name: str = DEFAULT_PROCESS_NAME,
    filament_name: str = DEFAULT_FILAMENT_NAME,
    plate_index: int = 0,
    orient: bool = True,
    arrange: bool = True,
    ensure_on_bed: bool = True,
    load_default_filament: bool = True,
    filament_ids: list[int] | None = None,
    enable_support: bool = False,
    support_type: str = "tree(auto)",
    support_threshold_angle: int = 30,
    debug_level: int = 2,
    timeout_seconds: int = 900,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Slice an STL with OrcaSlicer CLI and export a sliced 3MF.

    OrcaSlicer shares much of this CLI shape with Bambu Studio. Profile JSON paths
    can be passed explicitly, or Bambu-style profile names can be searched locally.
    """
    orca_bin = _as_path(slicer_bin) if slicer_bin else find_orca_slicer_bin()
    profiles = _resolve_profiles(
        machine_json=machine_json,
        process_json=process_json,
        filament_json=filament_json,
        machine_name=machine_name,
        process_name=process_name,
        filament_name=filament_name,
    )
    active_process_json = profiles["process_json"]
    support_cli_warning = None
    if enable_support:
        support_cli_warning = (
            "Support override was requested, but this wrapper does not inject "
            "external process profiles for Orca/Bambu CLI yet. Use a saved "
            "support-enabled process profile instead."
        )

    command = build_orca_slicer_command(
        input_stl=input_stl,
        output_3mf=output_3mf,
        slicer_bin=orca_bin,
        machine_json=profiles["machine_json"],
        process_json=active_process_json,
        filament_json=profiles["filament_json"],
        plate_index=plate_index,
        orient=orient,
        arrange=arrange,
        ensure_on_bed=ensure_on_bed,
        load_default_filament=load_default_filament,
        filament_ids=filament_ids if filament_ids is not None else [1],
        debug_level=debug_level,
        extra_args=extra_args,
    )

    result: dict[str, Any] = {
        "implemented": True,
        "slicer": "orca_slicer",
        "input_file": str(_as_path(input_stl)),
        "output_file": str(_as_path(output_3mf)),
        "machine_json": str(profiles["machine_json"]),
        "process_json": str(active_process_json),
        "base_process_json": str(profiles["process_json"]),
        "filament_json": str(profiles["filament_json"]),
        "support_process_json": None,
        "enable_support": enable_support,
        "support_cli_warning": support_cli_warning,
        "command": command,
        "dry_run": dry_run,
    }
    if dry_run:
        result.update({"success": True, "message": "Dry run only. Command was not executed."})
        return result

    run_result = _run_command(command, timeout_seconds)
    result.update(run_result)
    if not result["success"]:
        raise RuntimeError(
            "OrcaSlicer slicing failed.\n"
            f"Return code: {run_result['returncode']}\n"
            f"Command: {' '.join(command)}\n"
            f"STDOUT:\n{run_result['stdout']}\n"
            f"STDERR:\n{run_result['stderr']}"
        )

    output_file = _as_path(output_3mf)
    result["output_exists"] = output_file.exists()
    if not output_file.exists():
        raise RuntimeError(f"Slicing finished but output file was not created: {output_file}")
    result.update(verify_sliced_3mf(output_file))
    if not result["contains_gcode"]:
        raise RuntimeError(f"3MF was created but does not contain sliced G-code: {output_file}")
    return result


def slice_with_orca_legacy_cli_safe(
    input_stl: str | Path,
    output_3mf: str | Path,
    slicer_bin: str | Path | None = None,
    machine_name: str = DEFAULT_MACHINE_NAME,
    process_name: str = DEFAULT_PROCESS_NAME,
    filament_name: str = DEFAULT_FILAMENT_NAME,
    safe_profile_dir: str | Path = Path("output/cli_safe_profiles_orca_legacy"),
    profile_root: str | Path | None = None,
    enable_support: bool = False,
    support_type: str = "tree(auto)",
    support_threshold_angle: int = 30,
    support_on_build_plate_only: str = "0",
    orient: bool = True,
    arrange: bool = True,
    ensure_on_bed: bool = True,
    load_default_filament: bool = True,
    filament_ids: list[int] | None = None,
    debug_level: int = 2,
    timeout_seconds: int = 900,
    dry_run: bool = False,
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Slice with the verified legacy Orca CLI and generated CLI-safe profiles."""
    orca_bin = _as_path(slicer_bin) if slicer_bin else find_orca_slicer_bin()
    bundled_profile_root = (
        _as_path(profile_root)
        if profile_root
        else _profile_root_from_slicer_bin(orca_bin)
    )

    from make_cli_safe_profiles import create_cli_safe_profiles

    profiles = create_cli_safe_profiles(
        output_dir=_as_path(safe_profile_dir),
        profile_root=bundled_profile_root,
        machine_name=machine_name,
        process_name=process_name,
        filament_name=filament_name,
        flatten_all_lists=True,
        enable_support=enable_support,
        support_type=support_type,
        support_threshold_angle=str(support_threshold_angle),
        support_on_build_plate_only=support_on_build_plate_only,
    )

    result = slice_with_orca_slicer(
        input_stl=input_stl,
        output_3mf=output_3mf,
        slicer_bin=orca_bin,
        machine_json=profiles["machine_json"],
        process_json=profiles["process_json"],
        filament_json=profiles["filament_json"],
        orient=orient,
        arrange=arrange,
        ensure_on_bed=ensure_on_bed,
        load_default_filament=load_default_filament,
        filament_ids=filament_ids if filament_ids is not None else [1],
        enable_support=enable_support,
        support_type=support_type,
        support_threshold_angle=support_threshold_angle,
        debug_level=debug_level,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
        extra_args=extra_args,
    )
    result["slicer"] = "orca_legacy_cli_safe"
    result["profile_root"] = str(bundled_profile_root)
    result["safe_profile_dir"] = str(_as_path(safe_profile_dir))
    result["support_process_json"] = profiles["process_json"] if enable_support else None
    result["support_cli_warning"] = None
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Slice a repaired STL with Bambu Studio or OrcaSlicer CLI."
    )
    parser.add_argument("input_stl", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output sliced .3mf path.")
    parser.add_argument(
        "--engine",
        choices=["orca-legacy", "bambu", "orca"],
        default="orca-legacy",
        help="orca-legacy is the default because it avoids the BambuStudio CLI crash seen on this Mac.",
    )
    parser.add_argument("--slicer-bin", default=None, help="Path to Bambu Studio/OrcaSlicer CLI executable.")
    parser.add_argument("--machine-json", default=None, help="Machine profile JSON path.")
    parser.add_argument("--process-json", default=None, help="Process profile JSON path.")
    parser.add_argument("--filament-json", default=None, help="Filament profile JSON path.")
    parser.add_argument("--profile-root", default=None, help="Profile root for orca-legacy, for example .../Resources/profiles/BBL.")
    parser.add_argument("--safe-profile-dir", default="output/cli_safe_profiles_orca_legacy")
    parser.add_argument("--machine-name", default=DEFAULT_MACHINE_NAME)
    parser.add_argument("--process-name", default=DEFAULT_PROCESS_NAME)
    parser.add_argument("--filament-name", default=DEFAULT_FILAMENT_NAME)
    parser.add_argument("--plate-index", type=int, default=0)
    parser.add_argument("--no-orient", action="store_true")
    parser.add_argument("--no-arrange", action="store_true")
    parser.add_argument("--no-ensure-on-bed", action="store_true")
    parser.add_argument("--no-load-default-filament", action="store_true")
    parser.add_argument("--filament-ids", default="1", help="Comma-separated filament ids for loaded objects.")
    parser.add_argument(
        "--enable-support",
        action="store_true",
        help="Enable support in the generated orca-legacy CLI-safe process profile.",
    )
    parser.add_argument("--support-type", default="tree(auto)", help='Support style, for example "tree(auto)" or "normal(auto)".')
    parser.add_argument("--support-threshold-angle", type=int, default=30, help="Support threshold angle in degrees.")
    parser.add_argument("--debug-level", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Pass one raw extra argument to the slicer CLI. Repeat for multiple args.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without running slicer.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.engine == "orca-legacy":
            result = slice_with_orca_legacy_cli_safe(
                input_stl=args.input_stl,
                output_3mf=args.output,
                slicer_bin=args.slicer_bin,
                machine_name=args.machine_name,
                process_name=args.process_name,
                filament_name=args.filament_name,
                safe_profile_dir=args.safe_profile_dir,
                profile_root=args.profile_root,
                enable_support=args.enable_support,
                support_type=args.support_type,
                support_threshold_angle=args.support_threshold_angle,
                orient=not args.no_orient,
                arrange=not args.no_arrange,
                ensure_on_bed=not args.no_ensure_on_bed,
                load_default_filament=not args.no_load_default_filament,
                filament_ids=[int(item.strip()) for item in args.filament_ids.split(",") if item.strip()],
                debug_level=args.debug_level,
                extra_args=args.extra_arg,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
        elif args.engine == "bambu":
            result = slice_with_bambu_studio(
                input_stl=args.input_stl,
                output_3mf=args.output,
                bambu_studio_bin=args.slicer_bin,
                machine_json=args.machine_json,
                process_json=args.process_json,
                filament_json=args.filament_json,
                machine_name=args.machine_name,
                process_name=args.process_name,
                filament_name=args.filament_name,
                plate_index=args.plate_index,
                orient=not args.no_orient,
                arrange=not args.no_arrange,
                ensure_on_bed=not args.no_ensure_on_bed,
                load_default_filament=not args.no_load_default_filament,
                filament_ids=[int(item.strip()) for item in args.filament_ids.split(",") if item.strip()],
                enable_support=args.enable_support,
                support_type=args.support_type,
                support_threshold_angle=args.support_threshold_angle,
                debug_level=args.debug_level,
                extra_args=args.extra_arg,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
        else:
            result = slice_with_orca_slicer(
                input_stl=args.input_stl,
                output_3mf=args.output,
                slicer_bin=args.slicer_bin,
                machine_json=args.machine_json,
                process_json=args.process_json,
                filament_json=args.filament_json,
                machine_name=args.machine_name,
                process_name=args.process_name,
                filament_name=args.filament_name,
                plate_index=args.plate_index,
                orient=not args.no_orient,
                arrange=not args.no_arrange,
                ensure_on_bed=not args.no_ensure_on_bed,
                load_default_filament=not args.no_load_default_filament,
                filament_ids=[int(item.strip()) for item in args.filament_ids.split(",") if item.strip()],
                enable_support=args.enable_support,
                support_type=args.support_type,
                support_threshold_angle=args.support_threshold_angle,
                debug_level=args.debug_level,
                extra_args=args.extra_arg,
                timeout_seconds=args.timeout_seconds,
                dry_run=args.dry_run,
            )
    except Exception as exc:
        print("Slicing failed")
        print(f"Reason: {exc}")
        return 1

    print("Slicing complete" if not args.dry_run else "Slicing dry run")
    print(f"- Engine:        {result['slicer']}")
    print(f"- Input STL:     {result['input_file']}")
    print(f"- Output 3MF:    {result['output_file']}")
    print(f"- Machine JSON:  {result['machine_json']}")
    print(f"- Process JSON:  {result['process_json']}")
    print(f"- Filament JSON: {result['filament_json']}")
    print(f"- Support:       {result.get('enable_support', False)}")
    if result.get("support_cli_warning"):
        print(f"- Support note:  {result['support_cli_warning']}")
    print("- Command:")
    print("  " + " ".join(result["command"]))
    if not args.dry_run:
        print(f"- Output exists: {result.get('output_exists', False)}")
        print(f"- Contains G-code: {result.get('contains_gcode', False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
