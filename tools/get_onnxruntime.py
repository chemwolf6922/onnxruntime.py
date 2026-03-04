import argparse
import json
import os
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.request import Request, urlopen
import git
import platform
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# All platform/architecture combos that must be present in a release
# when --require-all-platforms is used.
ALL_SUPPORTED_PLATFORMS = (
    "win-x64",
    "win-arm64",
    "linux-x64",
    "linux-aarch64",
    "osx-arm64",
)

@dataclass(frozen=True)
class GitHubAsset:
    name: str
    browser_download_url: str
    
    @staticmethod
    def from_dict(data: dict) -> 'GitHubAsset':
        if not isinstance(data, dict):
            raise ValueError("Invalid data for GitHubAsset")
        name = data['name']
        if not isinstance(name, str):
            raise ValueError("Invalid name in GitHubAsset data")
        browser_download_url = data['browser_download_url']
        if not isinstance(browser_download_url, str):
            raise ValueError("Invalid browser_download_url in GitHubAsset data")
        return GitHubAsset(name=name, browser_download_url=browser_download_url)

@dataclass(frozen=True)
class GitHubRelease:
    tag_name: str
    assets: list[GitHubAsset]

    @staticmethod
    def from_dict(data: dict) -> 'GitHubRelease':
        if not isinstance(data, dict):
            raise ValueError("Invalid data for Release")
        tag_name = data['tag_name']
        if not isinstance(tag_name, str):
            raise ValueError("Invalid tag_name in Release data")
        assets = data['assets']
        if not isinstance(assets, list):
            raise ValueError("Invalid assets in Release data")
        return GitHubRelease(tag_name=tag_name, assets=[GitHubAsset.from_dict(asset) for asset in assets])


def try_get_git_tag(repo: git.Repo) -> Optional[str]:
    head = repo.head.commit
    matching_tags = [tag for tag in repo.tags if tag.commit == head]
    if not matching_tags:
        return None
    if len(matching_tags) > 1:
        raise RuntimeError(f"Multiple tags found for HEAD: {', '.join(tag.name for tag in matching_tags)}")
    return matching_tags[0].name

def git_tag_to_version_hint(tag: str) -> str:
    version = re.sub(r'(a\d+|b\d+|rc\d+)$', '', tag)
    segments = version.split(".")
    if len(segments) < 2:
        raise RuntimeError(f"Git tag '{tag}' is not a valid version.")
    return ".".join(segments[:2])

def validate_version_hint(version_hint: str) -> None:
    segments = version_hint.split(".")
    if len(segments) < 2:
        raise RuntimeError(f"Version hint '{version_hint}' is not valid. Provide a value like '1.22'.")
    if not all(segment.isdigit() for segment in segments):
        raise RuntimeError(f"Version hint '{version_hint}' is not valid. Provide a value like '1.22'.")

def fetch_releases(max_releases: int) -> List[GitHubRelease]:
    url = f"https://api.github.com/repos/microsoft/onnxruntime/releases?per_page={max_releases}"
    request = Request(url)
    github_token = os.environ.get("GITHUB_TOKEN")
    if github_token:
        request.add_header("Authorization", f"token {github_token}")
    with urlopen(request) as response:
        payload = response.read()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from GitHub API.")
    return [GitHubRelease.from_dict(item) for item in data]

def find_release(
    releases: Iterable[GitHubRelease],
    version_hint: Optional[str],
    require_all_platforms: bool = False,
    platform_override: Optional[str] = None,
) -> GitHubRelease:
    host_arch = platform_override or get_host_architecture()
    required_archs = ALL_SUPPORTED_PLATFORMS if require_all_platforms else (host_arch,)

    if version_hint:
        print(f"Looking for releases matching '{version_hint}.*'.")
    if require_all_platforms:
        print(f"Requiring assets for all supported platforms: {', '.join(required_archs)}")

    for release in releases:
        clean_version = release.tag_name.lstrip("vV")
        if version_hint and not clean_version.startswith(version_hint):
            continue

        missing = [
            arch for arch in required_archs
            if not any(asset.name.startswith(f"onnxruntime-{arch}-") for asset in release.assets)
        ]
        if missing:
            print(f"Skipping release {release.tag_name}: missing assets for {', '.join(missing)}.")
            continue
        return release
    raise RuntimeError("No valid package found.")

def _extract_archive(archive_path: Path, destination: Path) -> None:
    """Extract a .zip or .tgz/.tar.gz archive into *destination*."""
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination)
    elif name.endswith(".tgz") or name.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(destination, filter="data")
    else:
        raise RuntimeError(f"Unsupported archive format: {archive_path.name}")

def download_and_extract_asset(asset: GitHubAsset, destination: Path, skip_top_layer: bool = False) -> None:
    request = Request(asset.browser_download_url)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        archive_path = tmp_dir / asset.name
        with urlopen(request) as response, archive_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        if not skip_top_layer:
            _extract_archive(archive_path, destination)
        else:
            extract_dir = tmp_dir / "extracted"
            _extract_archive(archive_path, extract_dir)
            top_level_items = list(extract_dir.iterdir())
            if len(top_level_items) != 1 or not top_level_items[0].is_dir():
                raise RuntimeError("Unexpected archive structure.")
            shutil.rmtree(destination, ignore_errors=True)
            shutil.move(str(top_level_items[0]), destination)
            
def get_host_architecture() -> str:
    system_map = {
        'Windows': 'win',
        'Linux': 'linux',
        'Darwin': 'osx',
    }
    system = system_map.get(platform.system())
    if system is None:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")

    # ONNX Runtime uses platform-specific architecture names in release assets:
    #   win-x64, win-arm64, linux-x64, linux-aarch64, osx-arm64, osx-x86_64
    machine_map = {
        ('win', 'AMD64'):       'x64',
        ('win', 'ARM64'):       'arm64',
        ('linux', 'x86_64'):    'x64',
        ('linux', 'aarch64'):   'aarch64',
        ('osx', 'x86_64'):     'x86_64',
        ('osx', 'arm64'):      'arm64',
    }
    machine = machine_map.get((system, platform.machine()))
    if machine is None:
        raise RuntimeError(
            f"Unsupported architecture: {platform.machine()} on {platform.system()}"
        )

    return f"{system}-{machine}"

def find_host_asset(release: GitHubRelease, platform_override: Optional[str] = None) -> GitHubAsset:
    arch = platform_override or get_host_architecture()
    for asset in release.assets:
        if asset.name.startswith(f"onnxruntime-{arch}-"):
            return asset
    raise RuntimeError(f"No package found for architecture '{arch}' in release '{release.tag_name}'.")


parser = argparse.ArgumentParser(
    description="Download the ONNX Runtime distribution for the current platform."
)
parser.add_argument(
    "--version",
    "-v",
    dest="version",
    help="Major.minor version hint (e.g. 1.22). The latest release matching this prefix will be used.",
)
parser.add_argument(
    "--out-dir",
    dest="out_dir",
    default=str(PROJECT_ROOT / "onnxruntime"),
    help=f"Root directory to extract the x64 package into (default: {str(PROJECT_ROOT / "onnxruntime")})",
)
parser.add_argument(
    "--max-releases",
    dest="max_releases",
    type=int,
    default=50,
    help="Maximum number of recent releases to consider (default: 50).",
)
parser.add_argument(
    "--require-all-platforms",
    dest="require_all_platforms",
    action="store_true",
    default=False,
    help="Only accept releases that have assets for all supported platforms.",
)
parser.add_argument(
    "--platform",
    dest="platform",
    default=None,
    choices=ALL_SUPPORTED_PLATFORMS,
    help="Force downloading a specific platform/arch flavor instead of auto-detecting.",
)
args = parser.parse_args()
out_dir = Path(args.out_dir).resolve()

version_hint = args.version
if not version_hint:
    tag = try_get_git_tag(git.Repo(PROJECT_ROOT))
    if tag:
        version_hint = git_tag_to_version_hint(tag)
if version_hint:
    validate_version_hint(version_hint)

releases = fetch_releases(args.max_releases)
release = find_release(releases, version_hint, require_all_platforms=args.require_all_platforms, platform_override=args.platform)
asset = find_host_asset(release, platform_override=args.platform)

shutil.rmtree(out_dir, ignore_errors=True)
out_dir.mkdir(parents=True, exist_ok=True)

download_and_extract_asset(asset, out_dir, skip_top_layer=True)

print(f"Downloaded and extracted {asset.name} to {out_dir}")
