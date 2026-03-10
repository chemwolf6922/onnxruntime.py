from argparse import ArgumentParser
from pathlib import Path
from glob import glob
import shutil
import subprocess
import tempfile
from get_version import get_version, get_lib_version, get_dependency_string
import platform
import sys

def copy_file_with_replacements(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    src_content = src.read_text()
    for old, new in replacements.items():
        src_content = src_content.replace(old, new)
    dst.write_text(src_content)

def get_wheel_platform_tag() -> str:
    """Return the platform component of the wheel tag for the current host."""
    system = platform.system()
    machine = platform.machine()
    if system == "Windows":
        arch_map = {"AMD64": "amd64", "ARM64": "arm64"}
        return f"win_{arch_map[machine]}"
    elif system == "Linux":
        # e.g. linux_x86_64, linux_aarch64
        return f"linux_{machine}"
    elif system == "Darwin":
        # Must match CMAKE_OSX_DEPLOYMENT_TARGET in CMakeLists.txt
        return f"macosx_14_0_{machine}"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")

def find_native_extension(search_dirs: list[Path], module_name: str) -> Path:
    """Find the compiled native extension (.pyd on Windows, .so on Linux/macOS)."""
    extensions = (".pyd", ".so", ".dylib")
    for d in search_dirs:
        for ext in extensions:
            matches = glob(str(d / f"{module_name}.*{ext}"))
            if matches:
                return Path(matches[0])
    raise FileNotFoundError(
        f"Could not find native extension '{module_name}' in: {[str(d) for d in search_dirs]}"
    )

def get_shared_libs(lib_dir: Path) -> list[Path]:
    """Return the list of ONNX Runtime shared libraries to bundle."""
    system = platform.system()
    if system == "Windows":
        names = ["onnxruntime.dll", "onnxruntime_providers_shared.dll"]
    elif system == "Linux":
        names = ["libonnxruntime.so", "libonnxruntime_providers_shared.so"]
    elif system == "Darwin":
        names = ["libonnxruntime.dylib"]
    else:
        raise RuntimeError(f"Unsupported platform: {system}")
    libs = []
    for name in names:
        p = lib_dir / name
        if p.exists():
            libs.append(p)
        else:
            print(f"Warning: expected shared library not found, skipping: {p}")
    if not libs:
        raise FileNotFoundError(f"No ONNX Runtime shared libraries found in {lib_dir}")
    return libs

def fix_rpath(native_ext: Path) -> None:
    """Set the RPATH/LC_RPATH on the native extension so it finds shared libs
    in the same directory at runtime. This is done after copying the binary
    into the wheel staging directory so the build-time RPATH (which points
    to the onnxruntime source tree) is not baked into the distributed wheel."""
    system = platform.system()
    if system == "Linux":
        subprocess.run(
            ["patchelf", "--set-rpath", "$ORIGIN", str(native_ext)],
            check=True,
        )
    elif system == "Darwin":
        # Remove any existing rpaths, then add @loader_path
        result = subprocess.run(
            ["otool", "-l", str(native_ext)],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith("path "):
                old_rpath = line.split()[1]
                subprocess.run(
                    ["install_name_tool", "-delete_rpath", old_rpath, str(native_ext)],
                    check=True,
                )
        subprocess.run(
            ["install_name_tool", "-add_rpath", "@loader_path", str(native_ext)],
            check=True,
        )

def repair_wheel(whl_path: Path) -> None:
    """Repair a wheel in-place using platform-specific tools.
    
    On Linux, runs auditwheel to retag linux_* -> manylinux_*.
    On macOS, runs delocate-wheel to fix dylib rpaths.
    On Windows, no repair is needed.
    Raises RuntimeError if the repair tool is missing or repair fails.
    """
    system = platform.system()
    if system == "Windows":
        return

    if system == "Linux":
        # Exclude all onnxruntime shared libs — they ship in the ortpy_lib wheel.
        # Must also match versioned names (e.g. libonnxruntime.so.1.24.3) since
        # auditwheel resolves libraries by SONAME. Include symlinks because the
        # SONAME (e.g. libonnxruntime.so.1) is typically a symlink.
        ort_lib_dir = Path(__file__).parent.parent / "onnxruntime" / "lib"
        exclude_args = []
        for lib in list(ort_lib_dir.glob("*.so")) + list(ort_lib_dir.glob("*.so.*")):
            exclude_args += ["--exclude", lib.name]
        tool_cmd = [sys.executable, "-m", "auditwheel", "repair"] + exclude_args
        tool_name = "auditwheel"
    elif system == "Darwin":
        # Exclude all onnxruntime shared libs — they ship in the ortpy_lib wheel
        ort_lib_dir = Path(__file__).parent.parent / "onnxruntime" / "lib"
        exclude_args = []
        for lib in ort_lib_dir.glob("*.dylib"):
            exclude_args += ["--exclude", lib.name]
        tool_cmd = [sys.executable, "-m", "delocate.cmd.delocate_wheel", "-v"] + exclude_args
        tool_name = "delocate"
    else:
        return

    # Check if the tool is available
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", tool_name],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            f"{tool_name} is not installed. Install it with: pip install {tool_name}"
        )

    output_dir = whl_path.parent
    with tempfile.TemporaryDirectory() as tmp_dir:
        result = subprocess.run(
            tool_cmd + [str(whl_path), "-w", tmp_dir],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{tool_name} repair failed for {whl_path.name}:\n{result.stderr}"
            )
        # Replace the original wheel with the repaired one
        repaired = list(Path(tmp_dir).glob("*.whl"))
        if repaired:
            whl_path.unlink()
            for r in repaired:
                shutil.move(str(r), output_dir / r.name)
            print(f"Repaired: {whl_path.name} -> {', '.join(r.name for r in repaired)}")

def pack_and_repair(wheel_build_dir: Path, output_dir: Path, native_ext_name: str | None = None) -> None:
    """Pack a wheel from the build directory and repair it.
    If native_ext_name is provided, fix its RPATH after repair."""
    existing = set(output_dir.glob("*.whl"))
    subprocess.run(
        [sys.executable, "-m", "wheel", "pack", str(wheel_build_dir), "--dest-dir", str(output_dir)],
        check=True,
    )
    new_wheels = set(output_dir.glob("*.whl")) - existing
    for whl in new_wheels:
        repair_wheel(whl)
    # Fix RPATH after repair so the repair tool can still resolve dependencies
    # using the build-time rpath, and we overwrite it afterward.
    # Re-scan because repair may rename wheels (e.g. linux -> manylinux).
    if native_ext_name:
        repaired_wheels = set(output_dir.glob("*.whl")) - existing
        for whl in repaired_wheels:
            _fix_rpath_in_wheel(whl, native_ext_name)

def _fix_rpath_in_wheel(whl_path: Path, native_ext_name: str) -> None:
    """Unpack a wheel, fix the RPATH on the native extension, and repack."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        result = subprocess.run(
            [sys.executable, "-m", "wheel", "unpack", "-d", str(tmp), str(whl_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"wheel unpack failed: {result.stderr}")
            raise RuntimeError(f"wheel unpack failed for {whl_path.name}:\n{result.stderr}")
        # wheel unpack creates a subdirectory named <name>-<version>
        unpacked_dirs = list(tmp.iterdir())
        if len(unpacked_dirs) != 1:
            raise RuntimeError(f"Expected one unpacked directory, got {unpacked_dirs}")
        unpacked = unpacked_dirs[0]
        # Find and fix the native extension
        matches = list(unpacked.rglob(f"{native_ext_name}*"))
        for m in matches:
            if m.suffix in (".so", ".dylib") or ".so." in m.name or m.name.endswith(".pyd"):
                fix_rpath(m)
        # Repack
        whl_path.unlink()
        subprocess.run(
            [sys.executable, "-m", "wheel", "pack", str(unpacked), "--dest-dir", str(whl_path.parent)],
            check=True,
        )

parser = ArgumentParser(description="Pack the ortpy and ortpy-lib wheels.")
parser.add_argument(
    "--build-type", "-b", 
    choices=["Release", "Debug"],
    default="Release",
    help="Select the build configuration to pack")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).parent.parent
WHEEL_BUILD_DIR = PROJECT_DIR / "build-wheel"
WHEEL_OUTPUT_DIR = PROJECT_DIR / "dist"
WHEEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# parse the python version from python library or use the current interpreter's version

python_version = f"{sys.version_info.major}{sys.version_info.minor}"
platform_tag = get_wheel_platform_tag()
wheel_tag = f"cp{python_version}-cp{python_version}-{platform_tag}"

# Pack the ortpy wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "ortpy"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(PROJECT_DIR / "src" / "ortpy", wheel_build_source_dir, dirs_exist_ok=True)
build_dir = PROJECT_DIR / "build"
binary_dir = build_dir / args.build_type
# On multi-config generators (MSVC) the binary is under build/{config}/,
# on single-config generators (Ninja/Make) it's directly under build/.
ortpy_native_path = find_native_extension([binary_dir, build_dir], "_ortpy")
shutil.copy(ortpy_native_path, wheel_build_source_dir)

ortpy_pyi_path = build_dir / "_ortpy.pyi"
if not ortpy_pyi_path.exists():
    raise FileNotFoundError("The type stub file is missing")
shutil.copy(ortpy_pyi_path, wheel_build_source_dir)
license_path = PROJECT_DIR / "LICENSE"
shutil.copy(license_path, wheel_build_source_dir / "LICENSE")
wheel_build_dist_info_dir = WHEEL_BUILD_DIR / f"ortpy-{get_version()}.dist-info"
wheel_build_dist_info_dir.mkdir(parents=True, exist_ok=True)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "ortpy.dist-info.in" / "METADATA.in",
    wheel_build_dist_info_dir / "METADATA",
    {
        "ORTPY_VERSION": get_version(),
        "ORTPY_LIB_REQUIREMENT": get_dependency_string()
    }
)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "ortpy.dist-info.in" / "WHEEL.in",
    wheel_build_dist_info_dir / "WHEEL",
    {
        "ORTPY_WHEEL_TAG": wheel_tag
    }
)
pack_and_repair(WHEEL_BUILD_DIR, WHEEL_OUTPUT_DIR, native_ext_name="_ortpy")

# Pack the ortpy-lib wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "ortpy"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
onnxruntime_lib_path = PROJECT_DIR / "onnxruntime" / "lib"
for lib_path in get_shared_libs(onnxruntime_lib_path):
    # On Linux, lib names like libfoo.so are symlinks to the SONAME (libfoo.so.1).
    # Follow one symlink level to get the SONAME as the filename, so the dynamic
    # linker can find it at runtime. shutil.copy resolves the full chain to get
    # the actual file content. On macOS/Windows this is typically a no-op.
    if lib_path.is_symlink():
        soname = lib_path.parent / Path(lib_path.readlink())
        shutil.copy(soname.resolve(), wheel_build_source_dir / soname.name)
    else:
        shutil.copy(lib_path, wheel_build_source_dir)
onnxruntime_license_path = PROJECT_DIR / "onnxruntime" / "LICENSE"
shutil.copy(onnxruntime_license_path, wheel_build_source_dir / "ONNXRUNTIME_LICENSE")
wheel_build_dist_info_dir = WHEEL_BUILD_DIR / f"ortpy_lib-{get_lib_version()}.dist-info"
wheel_build_dist_info_dir.mkdir(parents=True, exist_ok=True)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "ortpy_lib.dist-info.in" / "METADATA.in",
    wheel_build_dist_info_dir / "METADATA",
    {
        "ORTPY_LIB_VERSION": get_lib_version(),
    }
)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "ortpy_lib.dist-info.in" / "WHEEL.in",
    wheel_build_dist_info_dir / "WHEEL",
    {
        "ORTPY_WHEEL_TAG": wheel_tag
    }
)
shutil.copy(PROJECT_DIR / "src" / "ortpy_lib.dist-info.in" / "top_level.txt", wheel_build_dist_info_dir)
pack_and_repair(WHEEL_BUILD_DIR, WHEEL_OUTPUT_DIR)
