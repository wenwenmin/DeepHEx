"""Download the public standard Visium dataset used by the DeepHEx tutorial."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
import shutil
import tarfile
from urllib.request import Request, urlopen


DATASETS = {
    "human_cervical_cancer_1": {
        "page": "https://www.10xgenomics.com/datasets/human-cervical-cancer-1-standard",
        "matrix_url": (
            "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/"
            "Visium_FFPE_Human_Cervical_Cancer/"
            "Visium_FFPE_Human_Cervical_Cancer_filtered_feature_bc_matrix.h5"
        ),
        "spatial_url": (
            "https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/"
            "Visium_FFPE_Human_Cervical_Cancer/"
            "Visium_FFPE_Human_Cervical_Cancer_spatial.tar.gz"
        ),
        "spatial_archive_name": "Visium_FFPE_Human_Cervical_Cancer_spatial.tar.gz",
    },
    "mouse_brain_sagittal_anterior": {
        "page": (
            "https://www.10xgenomics.com/datasets/"
            "mouse-brain-serial-section-1-sagittal-anterior-1-standard-1-0-0"
        ),
        "matrix_url": (
            "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/"
            "V1_Mouse_Brain_Sagittal_Anterior/"
            "V1_Mouse_Brain_Sagittal_Anterior_filtered_feature_bc_matrix.h5"
        ),
        "spatial_url": (
            "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/"
            "V1_Mouse_Brain_Sagittal_Anterior/"
            "V1_Mouse_Brain_Sagittal_Anterior_spatial.tar.gz"
        ),
        "spatial_archive_name": "V1_Mouse_Brain_Sagittal_Anterior_spatial.tar.gz",
    },
}
DEFAULT_DATASET = "human_cervical_cancer_1"


def _display_path(path: Path) -> str:
    """Redact the user's home directory in tutorial output."""

    return str(path).replace(str(Path.home()), "~")


def _download(url: str, destination: Path) -> None:
    """Download to a temporary file, then atomically move it into place."""

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    print(f"Downloading {url}\n  -> {_display_path(destination)}")
    try:
        request = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "Chrome/124.0 Safari/537.36"
                ),
                "Accept": "application/octet-stream,*/*",
            },
        )
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _extract_spatial_archive(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in spatial archive: {member.name}")
        extract_kwargs = {"members": members}
        if "filter" in inspect.signature(archive.extractall).parameters:
            extract_kwargs["filter"] = "data"
        archive.extractall(destination, **extract_kwargs)


def download_public_example(
    output_dir: str | Path,
    dataset: str = DEFAULT_DATASET,
) -> Path:
    """Download and prepare a standard Space Ranger ``outs`` directory.

    The function is idempotent: existing non-empty matrix and extracted spatial
    files are reused. It returns the path expected by
    :func:`data.processing_Visium.discover_visium_inputs`.
    """

    if dataset not in DATASETS:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset {dataset!r}; choose one of: {choices}")
    preset = DATASETS[dataset]
    root = Path(output_dir).expanduser().resolve()
    outs = root / "outs"
    spatial = outs / "spatial"
    matrix_path = outs / "filtered_feature_bc_matrix.h5"
    spatial_archive = root / preset["spatial_archive_name"]
    required_spatial = [
        spatial / "scalefactors_json.json",
        spatial / "tissue_hires_image.png",
    ]
    position_candidates = [
        spatial / "tissue_positions.csv",
        spatial / "tissue_positions_list.csv",
    ]

    outs.mkdir(parents=True, exist_ok=True)
    if not matrix_path.is_file() or matrix_path.stat().st_size == 0:
        _download(preset["matrix_url"], matrix_path)
    else:
        print(f"Reusing matrix: {_display_path(matrix_path)}")

    if not all(path.is_file() for path in required_spatial) or not any(
        path.is_file() for path in position_candidates
    ):
        if not spatial_archive.is_file() or spatial_archive.stat().st_size == 0:
            _download(preset["spatial_url"], spatial_archive)
        else:
            print(f"Reusing spatial archive: {_display_path(spatial_archive)}")
        _extract_spatial_archive(spatial_archive, outs)
    else:
        print(f"Reusing extracted spatial files: {_display_path(spatial)}")

    missing = [path for path in required_spatial if not path.is_file()]
    if not any(path.is_file() for path in position_candidates):
        missing.append(spatial / "tissue_positions.csv or tissue_positions_list.csv")
    if missing:
        raise FileNotFoundError(
            "The public archive did not contain the expected files: "
            + ", ".join(_display_path(path) for path in missing)
        )
    print(f"Dataset: {dataset}")
    print(f"Dataset page: {preset['page']}")
    print(f"Space Ranger outs: {_display_path(outs)}")
    return outs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a public standard Visium tutorial dataset."
    )
    parser.add_argument("--dataset", choices=sorted(DATASETS), default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        default="example_data/visium_mouse_brain_sagittal_anterior",
        help="Directory in which to create the Space Ranger 'outs' folder",
    )
    args = parser.parse_args()
    download_public_example(args.output_dir)


if __name__ == "__main__":
    main()
