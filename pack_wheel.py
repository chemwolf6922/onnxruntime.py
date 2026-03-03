from argparse import ArgumentParser
from pathlib import Path
from glob import glob
import shutil
import subprocess
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
        # Target macOS 11.0 as the minimum (required for ARM Macs)
        return f"macosx_11_0_{machine}"
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

parser = ArgumentParser(description="Pack the ortpy and ortpy-lib wheels.")
parser.add_argument(
    "--build-type", "-b", 
    choices=["Release", "Debug"],
    default="Release",
    help="Select the build configuration to pack")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).parent
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
subprocess.run(
    [sys.executable, "-m", "wheel", "pack", str(WHEEL_BUILD_DIR), "--dest-dir", str(WHEEL_OUTPUT_DIR)],
    check=True,
)

# Pack the ortpy-lib wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "ortpy"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
onnxruntime_lib_path = PROJECT_DIR / "onnxruntime" / "lib"
for lib_path in get_shared_libs(onnxruntime_lib_path):
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
subprocess.run(
    [sys.executable, "-m", "wheel", "pack", str(WHEEL_BUILD_DIR), "--dest-dir", str(WHEEL_OUTPUT_DIR)],
    check=True,
)
