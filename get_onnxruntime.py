import argparse
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from urllib.request import Request, urlopen
import git
import platform

PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_ARCHITECTURES = ("win-x64", "win-arm64")

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
    # The git tag should strictly follow the v + PyPI version format.
    version = tag.lstrip("vV")
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
    with urlopen(request) as response:
        payload = response.read()
    data = json.loads(payload)
    if not isinstance(data, list):
        raise RuntimeError("Unexpected response from GitHub API.")
    return [GitHubRelease.from_dict(item) for item in data]

def find_release(releases: Iterable[GitHubRelease], version_hint: Optional[str]) -> GitHubRelease:
    if version_hint:
        print(f"Looking for releases matching '{version_hint}.*'.")

    for release in releases:
        clean_version = release.tag_name.lstrip("vV")
        if version_hint and not clean_version.startswith(version_hint):
            continue

        all_required_found = all(
            any(asset.name.startswith(f"onnxruntime-{arch}-") for asset in release.assets)
            for arch in REQUIRED_ARCHITECTURES
        )
        if not all_required_found:
            print(f"Skipping release {release.tag_name} due to missing required architectures.")
            continue
        return release
    raise RuntimeError("No valid package found.")

def download_and_extract_asset(asset: GitHubAsset, destination: Path, skip_top_layer: bool = False) -> None:
    request = Request(asset.browser_download_url)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        zip_file_path = tmp_dir / asset.name
        with urlopen(request) as response, zip_file_path.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        with zipfile.ZipFile(zip_file_path) as zip_file:
            if not skip_top_layer:
                zip_file.extractall(destination)
            else:
                extract_dir = tmp_dir / "extracted"
                zip_file.extractall(extract_dir)
                top_level_items = list(extract_dir.iterdir())
                if len(top_level_items) != 1 or not top_level_items[0].is_dir():
                    raise RuntimeError("Unexpected archive structure.")
                shutil.rmtree(destination, ignore_errors=True)
                shutil.move(str(top_level_items[0]), destination)
            
def get_host_architecture() -> str:
    system = {
        'Windows': 'win'
    }[platform.system()]
    machine = {
        'AMD64': 'x64',
        'x86_64': 'x64',
        'ARM64': 'arm64',
        'aarch64': 'arm64'
    }[platform.machine()]
    return f"{system}-{machine}"

def find_host_asset(release: GitHubRelease) -> GitHubAsset:
    host_arch = get_host_architecture()
    for asset in release.assets:
        if asset.name.startswith(f"onnxruntime-{host_arch}-"):
            return asset
    raise RuntimeError(f"No package found for host architecture '{host_arch}' in release '{release.tag_name}'.")


parser = argparse.ArgumentParser(
    description="Download ONNX Runtime Windows distributions (x64 and arm64)."
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
release = find_release(releases, version_hint)
asset = find_host_asset(release)

shutil.rmtree(out_dir, ignore_errors=True)
out_dir.mkdir(parents=True, exist_ok=True)

download_and_extract_asset(asset, out_dir, skip_top_layer=True)

print(f"Downloaded and extracted {asset.name} to {out_dir}")
