from argparse import ArgumentParser
from pathlib import Path
from glob import glob
import shutil
from wheel.cli.pack import pack
from get_version import get_version, get_lib_version, get_dependency_string

def copy_file_with_replacements(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    src_content = src.read_text()
    for old, new in replacements.items():
        src_content = src_content.replace(old, new)
    dst.write_text(src_content)

parser = ArgumentParser(description="Pack the pyort and pyort-lib wheels.")
parser.add_argument(
    "--build-type", "-b", 
    choices=["Release", "Debug"],
    default="Release",
    help="Select the .pyd flavor to pack")
parser.add_argument(
    "--python-lib", "-p",
    type=str,
    default=None,
    help="Path to the Python library the pyd linked against (optional)")
parser.add_argument(
    "--arch", "-a",
    choices=["x64", "AMD64", "ARM64"],
    default="x64",
    help="Select the architecture for the wheel (default: x64)")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).parent
WHEEL_BUILD_DIR = PROJECT_DIR / "build-wheel"
WHEEL_OUTPUT_DIR = PROJECT_DIR / "dist"
WHEEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# parse the python version from python library or use the current interpreter's version

python_version = None
if args.python_lib:
    python_version = Path(args.python_lib).name.replace("python", "").replace(".lib", "")
else:
    import sys
    python_version = sys.version.split()[0].replace(".", "")
target_arch = {
    'AMD64': 'amd64',
    'x64': 'amd64',
    'ARM64': 'arm64'
}[args.arch]

# Pack the pyort wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "pyort"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(PROJECT_DIR / "src" / "pyort", wheel_build_source_dir, dirs_exist_ok=True)
binary_dir = PROJECT_DIR / "build" / args.build_type
pyort_pyd_path = Path(glob(str(binary_dir / "_pyort.pyd"))[0])
shutil.copy(pyort_pyd_path, wheel_build_source_dir / f"_pyort.cp{python_version}-win_{target_arch}.pyd")
wheel_build_dist_info_dir = WHEEL_BUILD_DIR / f"pyort-{get_version()}.dist-info"
wheel_build_dist_info_dir.mkdir(parents=True, exist_ok=True)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "pyort.dist-info.in" / "METADATA.in",
    wheel_build_dist_info_dir / "METADATA",
    {
        "PYORT_VERSION": get_version(),
        "PYORT_LIB_REQUIREMENT": get_dependency_string()
    }
)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "pyort.dist-info.in" / "WHEEL.in",
    wheel_build_dist_info_dir / "WHEEL",
    {
        "PYORT_WHEEL_TAG": f"cp{python_version}-cp{python_version}-win_{target_arch}"
    }
)
pack(str(WHEEL_BUILD_DIR), str(WHEEL_OUTPUT_DIR), None)

# Pack the pyort-lib wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "pyort_lib"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
onnxruntime_lib_path = {
    'amd64': PROJECT_DIR / "onnxruntime" / "lib",
    'arm64': PROJECT_DIR / "onnxruntime-arm64" / "lib"
}[target_arch]
shutil.copy(onnxruntime_lib_path / "onnxruntime.dll", wheel_build_source_dir)
shutil.copy(onnxruntime_lib_path / "onnxruntime_providers_shared.dll", wheel_build_source_dir)
wheel_build_dist_info_dir = WHEEL_BUILD_DIR / f"pyort_lib-{get_lib_version()}.dist-info"
wheel_build_dist_info_dir.mkdir(parents=True, exist_ok=True)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "pyort_lib.dist-info.in" / "METADATA.in",
    wheel_build_dist_info_dir / "METADATA",
    {
        "PYORT_LIB_VERSION": get_lib_version(),
    }
)
copy_file_with_replacements(
    PROJECT_DIR / "src" / "pyort_lib.dist-info.in" / "WHEEL.in",
    wheel_build_dist_info_dir / "WHEEL",
    {
        "PYORT_WHEEL_TAG": f"cp3-none-win_{target_arch}"
    }
)
shutil.copy(PROJECT_DIR / "src" / "pyort_lib.dist-info.in" / "top_level.txt", wheel_build_dist_info_dir)
pack(str(WHEEL_BUILD_DIR), str(WHEEL_OUTPUT_DIR), None)
