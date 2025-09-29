from argparse import ArgumentParser
from pathlib import Path
from glob import glob
import shutil
from wheel.cli.pack import pack
from get_version import get_version, get_lib_version, get_dependency_string
import platform
import sys

def copy_file_with_replacements(src: Path, dst: Path, replacements: dict[str, str]) -> None:
    src_content = src.read_text()
    for old, new in replacements.items():
        src_content = src_content.replace(old, new)
    dst.write_text(src_content)

parser = ArgumentParser(description="Pack the ortpy and ortpy-lib wheels.")
parser.add_argument(
    "--build-type", "-b", 
    choices=["Release", "Debug"],
    default="Release",
    help="Select the .pyd flavor to pack")
args = parser.parse_args()

PROJECT_DIR = Path(__file__).parent
WHEEL_BUILD_DIR = PROJECT_DIR / "build-wheel"
WHEEL_OUTPUT_DIR = PROJECT_DIR / "dist"
WHEEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# parse the python version from python library or use the current interpreter's version

python_version = f"{sys.version_info.major}{sys.version_info.minor}"
target_arch = {
    'AMD64': 'amd64',
    'ARM64': 'arm64'
}[platform.machine()]

# Pack the ortpy wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "ortpy"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(PROJECT_DIR / "src" / "ortpy", wheel_build_source_dir, dirs_exist_ok=True)
build_dir = PROJECT_DIR / "build"
binary_dir = build_dir / args.build_type
ortpy_pyd_path = Path(glob(str(binary_dir / "_ortpy.*.pyd"))[0])
shutil.copy(ortpy_pyd_path, wheel_build_source_dir)
ortpy_pyi_path = build_dir / "_ortpy.pyi"
if not ortpy_pyi_path.exists():
    raise FileNotFoundError("The type stub file is missing")
shutil.copy(ortpy_pyi_path, wheel_build_source_dir)
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
        "ORTPY_WHEEL_TAG": f"cp{python_version}-cp{python_version}-win_{target_arch}"
    }
)
pack(str(WHEEL_BUILD_DIR), str(WHEEL_OUTPUT_DIR), None)

# Pack the ortpy-lib wheel

shutil.rmtree(WHEEL_BUILD_DIR, ignore_errors=True)
WHEEL_BUILD_DIR.mkdir(parents=True, exist_ok=True)
wheel_build_source_dir = WHEEL_BUILD_DIR / "ortpy"
wheel_build_source_dir.mkdir(parents=True, exist_ok=True)
onnxruntime_lib_path = PROJECT_DIR / "onnxruntime" / "lib"
shutil.copy(onnxruntime_lib_path / "onnxruntime.dll", wheel_build_source_dir)
shutil.copy(onnxruntime_lib_path / "onnxruntime_providers_shared.dll", wheel_build_source_dir)
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
        "ORTPY_WHEEL_TAG": f"cp{python_version}-cp{python_version}-win_{target_arch}"
    }
)
shutil.copy(PROJECT_DIR / "src" / "ortpy_lib.dist-info.in" / "top_level.txt", wheel_build_dist_info_dir)
pack(str(WHEEL_BUILD_DIR), str(WHEEL_OUTPUT_DIR), None)
