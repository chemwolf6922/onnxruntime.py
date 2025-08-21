from pathlib import Path
import subprocess

_PROJECT_PATH = Path(__file__).parent.resolve()

def _get_tag() -> str | None:
    try:
        result = subprocess.run(
            ["git", "describe", "--exact-match", "--tags", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def _get_commit_id() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def get_version() -> str:
    tag = _get_tag()
    if tag is not None:
        tag_parts = tag.split(".")
        if len(tag_parts) >= 2:
            tag_major_minor = f"{tag_parts[0]}.{tag_parts[1]}"
            lib_version = get_lib_version()
            lib_parts = lib_version.split(".")
            lib_major_minor = f"{lib_parts[0]}.{lib_parts[1]}"
            if tag_major_minor != lib_major_minor:
                raise ValueError(f"Tag '{tag}' does not match library version '{lib_version}'")
            return tag

    commit_id = _get_commit_id()
    if commit_id is None:
        raise RuntimeError("Unable to determine commit ID. Ensure you are in a git repository.")
    lib_version = get_lib_version()
    version_parts = lib_version.split(".")
    major_minor = f"{version_parts[0]}.{version_parts[1]}"
    commit_id = _get_commit_id()
    return f"{major_minor}.dev0+{commit_id}"

def get_lib_version() -> str:
    with open(_PROJECT_PATH / "onnxruntime" / "VERSION_NUMBER", "r") as f:
        return f.read().strip()

def get_dependency_string() -> str:
    lib_version = get_lib_version()
    version_parts = lib_version.split(".")
    major_minor = f"{version_parts[0]}.{version_parts[1]}"
    return f"pyort_lib~={major_minor}.0"

if __name__ == "__main__":
    print(get_version())
