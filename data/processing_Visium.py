"""Prepare standard 10x Genomics Visium data for the DeepHEx pipeline.

This module intentionally keeps the model-facing data format identical to the
existing Visium HD and Xenium preprocessing paths.  Space Ranger inputs are
validated, the selected histology image is rescaled, and the following files
are written to the DeepHEx working directory:

``he.jpg``, ``locs.csv``, ``cnts.csv``, ``gene-names.txt``, ``mask.png``,
``radius.txt``, ``pixel-size-raw.txt``, ``pixel-size.txt`` and
``visium_preprocessing.json``.

The implementation supports Space Ranger versions that write either
``spatial/tissue_positions.csv`` (with a header) or the legacy
``spatial/tissue_positions_list.csv`` (without a header).
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from scipy.spatial import cKDTree
import scanpy as sc

from utils.reproducibility import portable_path


Image.MAX_IMAGE_PIXELS = None

_POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


class VisiumInputError(ValueError):
    """Raised when a standard Visium input cannot be prepared safely."""


@dataclass(frozen=True)
class VisiumInputs:
    """Resolved files in a Space Ranger ``outs`` directory."""

    outs_dir: Path
    matrix_path: Path
    positions_path: Path
    scalefactors_path: Path
    image_path: Path
    image_level: str


def _first_existing(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    choices = ", ".join(str(path) for path in candidates)
    raise VisiumInputError(f"Could not find {description}. Checked: {choices}")


def discover_visium_inputs(
    outs_dir: str | os.PathLike[str],
    *,
    matrix_path: str | os.PathLike[str] | None = None,
    positions_path: str | os.PathLike[str] | None = None,
    scalefactors_path: str | os.PathLike[str] | None = None,
    image_path: str | os.PathLike[str] | None = None,
    image_level: str = "auto",
) -> VisiumInputs:
    """Resolve the files needed for standard Visium preprocessing.

    ``outs_dir`` may point to a Space Ranger ``outs`` directory or to its
    ``spatial`` subdirectory.  Any file can be overridden explicitly, which is
    useful for custom full-resolution H&E images or renamed downloads.
    """

    outs = Path(outs_dir).expanduser().resolve()
    if outs.name == "spatial":
        outs = outs.parent
    if not outs.is_dir():
        raise VisiumInputError(f"Space Ranger outs directory does not exist: {outs}")

    spatial = outs / "spatial"
    if matrix_path is None:
        matrix = _first_existing(
            [
                outs / "filtered_feature_bc_matrix.h5",
                outs / "filtered_feature_bc_matrix",
                outs / "raw_feature_bc_matrix.h5",
                outs / "raw_feature_bc_matrix",
            ],
            "a Space Ranger feature-barcode matrix",
        )
    else:
        matrix = Path(matrix_path).expanduser().resolve()
        if not matrix.exists():
            raise VisiumInputError(f"Feature-barcode matrix does not exist: {matrix}")

    positions = (
        Path(positions_path).expanduser().resolve()
        if positions_path is not None
        else _first_existing(
            [
                spatial / "tissue_positions.csv",
                spatial / "tissue_positions_list.csv",
            ],
            "tissue positions",
        )
    )
    scalefactors = (
        Path(scalefactors_path).expanduser().resolve()
        if scalefactors_path is not None
        else _first_existing(
            [spatial / "scalefactors_json.json"], "scalefactors_json.json"
        )
    )

    two_utc = outs.parent.parent if outs.name == "outs" else outs.parent
    if image_level not in {"auto", "fullres", "hires", "lowres"}:
        raise VisiumInputError(
            "image_level must be one of: auto, fullres, hires, lowres"
        )

    hires = spatial / "tissue_hires_image.png"
    lowres = spatial / "tissue_lowres_image.png"
    fullres_candidates = [
        spatial / "tissue_fullres_image.png",
        spatial / "tissue_fullres_image.tif",
        spatial / "tissue_fullres_image.tiff",
        two_utc / "tissue_image.tif",
        two_utc / "tissue_image.tiff",
    ]

    if image_path is not None:
        image = Path(image_path).expanduser().resolve()
        if not image.exists():
            raise VisiumInputError(f"Histology image does not exist: {image}")
        if image_level == "auto":
            # A user-supplied image is normally in the Space Ranger full-resolution
            # coordinate system.  Callers can still pass image_level explicitly.
            image_level = "fullres"
    elif image_level == "fullres":
        image = _first_existing(fullres_candidates, "a full-resolution histology image")
    elif image_level == "lowres":
        image = _first_existing([lowres], "tissue_lowres_image.png")
    elif image_level == "hires":
        image = _first_existing([hires], "tissue_hires_image.png")
    else:
        image = _first_existing([hires], "tissue_hires_image.png")
        image_level = "hires"

    for path, label in [
        (positions, "tissue positions"),
        (scalefactors, "scalefactors"),
    ]:
        if not path.exists():
            raise VisiumInputError(f"{label} file does not exist: {path}")

    return VisiumInputs(
        outs_dir=outs,
        matrix_path=matrix,
        positions_path=positions,
        scalefactors_path=scalefactors,
        image_path=image,
        image_level=image_level,
    )


def _read_feature_matrix(matrix_path: Path) -> sc.AnnData:
    if matrix_path.is_dir():
        adata = sc.read_10x_mtx(
            matrix_path,
            var_names="gene_symbols",
            make_unique=True,
            cache=False,
        )
    else:
        adata = sc.read_10x_h5(matrix_path, gex_only=True)
        adata.var_names_make_unique()
    if adata.n_obs == 0 or adata.n_vars == 0:
        raise VisiumInputError(f"Feature-barcode matrix is empty: {matrix_path}")
    return adata


def _read_tissue_positions(positions_path: Path) -> pd.DataFrame:
    with positions_path.open("r", encoding="utf-8") as handle:
        first_line = handle.readline().strip().lower()

    has_header = "barcode" in first_line
    if has_header:
        positions = pd.read_csv(positions_path)
    else:
        positions = pd.read_csv(
            positions_path, header=None, names=_POSITION_COLUMNS
        )

    missing = [column for column in _POSITION_COLUMNS if column not in positions]
    if missing:
        raise VisiumInputError(
            f"tissue positions are missing required columns {missing}: {positions_path}"
        )
    if positions["barcode"].duplicated().any():
        raise VisiumInputError("tissue positions contain duplicated barcodes")

    positions = positions.copy()
    positions["in_tissue"] = pd.to_numeric(positions["in_tissue"], errors="coerce")
    positions = positions.loc[positions["in_tissue"].eq(1)].copy()
    if positions.empty:
        raise VisiumInputError("No spots with in_tissue == 1 were found")

    for column in [
        "pxl_row_in_fullres",
        "pxl_col_in_fullres",
        "array_row",
        "array_col",
    ]:
        positions[column] = pd.to_numeric(positions[column], errors="raise")
    coordinate_columns = ["pxl_row_in_fullres", "pxl_col_in_fullres"]
    if positions[coordinate_columns].isna().any().any():
        raise VisiumInputError("Pixel coordinates contain missing values")
    return positions.set_index("barcode", drop=True)


def _select_highly_variable_genes(
    adata: sc.AnnData,
    *,
    n_top_genes: int,
    min_cells: int,
    hvg_flavor: str,
) -> tuple[sc.AnnData, pd.DataFrame]:
    if n_top_genes < 1:
        raise VisiumInputError("num_top_genes must be at least 1")
    if min_cells < 0:
        raise VisiumInputError("min_cells must be non-negative")

    sc.pp.filter_genes(adata, min_cells=min_cells)
    if adata.n_vars < 2:
        raise VisiumInputError(
            "Fewer than two genes remain after filtering; relax min_cells"
        )

    n_top = min(int(n_top_genes), int(adata.n_vars))
    if n_top == adata.n_vars:
        hvg_table = pd.DataFrame(
            {
                "highly_variable": np.ones(adata.n_vars, dtype=bool),
                "highly_variable_rank": np.arange(adata.n_vars, dtype=float),
                "means": np.asarray(adata.X.mean(axis=0)).ravel(),
                "variances": np.asarray(adata.X.multiply(adata.X).mean(axis=0)).ravel()
                - np.asarray(adata.X.mean(axis=0)).ravel() ** 2,
            },
            index=adata.var_names.copy(),
        )
        hvg_table.index.name = "gene"
    else:
        hvg_table = sc.pp.highly_variable_genes(
            adata,
            flavor=hvg_flavor,
            n_top_genes=n_top,
            subset=False,
            inplace=False,
        )
    hvg_table = hvg_table.copy()
    hvg_table.index.name = "gene"
    selected = hvg_table.index[hvg_table["highly_variable"].to_numpy()].tolist()
    if len(selected) < 2:
        raise VisiumInputError("HVG selection returned fewer than two genes")
    return adata[:, selected].copy(), hvg_table


def _read_histology(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        image.load()
        return image.convert("RGB")


def _resize_and_pad_image(
    image: Image.Image,
    scale: float,
    output_path: Path,
    tile_size: int = 224,
    fill_value: int = 255,
) -> tuple[np.ndarray, dict[str, int]]:
    if scale <= 0:
        raise VisiumInputError("Image scale must be positive")
    new_width = max(1, int(round(image.width * scale)))
    new_height = max(1, int(round(image.height * scale)))
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    padded_width = int(np.ceil(new_width / tile_size) * tile_size)
    padded_height = int(np.ceil(new_height / tile_size) * tile_size)
    canvas = Image.new("RGB", (padded_width, padded_height), (fill_value,) * 3)
    canvas.paste(resized, (0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95, subsampling=0)
    return np.asarray(canvas), {
        "original_width": image.width,
        "original_height": image.height,
        "resized_width": new_width,
        "resized_height": new_height,
        "padded_width": padded_width,
        "padded_height": padded_height,
    }


def _build_spot_mask(
    coordinates: np.ndarray,
    image_shape: tuple[int, int, int],
    *,
    output_um_per_pixel: float,
    spot_diameter_um: float,
    superpixel_size: int,
    dilation_iterations: int,
    closing_radius: int | None,
) -> tuple[np.ndarray, float]:
    height, width = image_shape[:2]
    if height % superpixel_size or width % superpixel_size:
        raise VisiumInputError(
            "Padded image dimensions must be divisible by superpixel_size"
        )
    mask = np.zeros(
        (height // superpixel_size, width // superpixel_size), dtype=bool
    )
    radius_px = spot_diameter_um / output_um_per_pixel / 2
    radius_cells = radius_px / superpixel_size

    centers = np.empty((len(coordinates), 2), dtype=np.float64)
    for spot_index, (x, y) in enumerate(coordinates):
        center_x = int(round(x)) // superpixel_size
        center_y = int(round(y)) // superpixel_size
        centers[spot_index] = (center_y, center_x)
        row_min = max(0, int(np.floor(center_y - radius_cells)))
        row_max = min(mask.shape[0] - 1, int(np.ceil(center_y + radius_cells)))
        col_min = max(0, int(np.floor(center_x - radius_cells)))
        col_max = min(mask.shape[1] - 1, int(np.ceil(center_x + radius_cells)))
        if row_min > row_max or col_min > col_max:
            continue
        yy, xx = np.ogrid[row_min : row_max + 1, col_min : col_max + 1]
        disk = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius_cells**2
        mask[row_min : row_max + 1, col_min : col_max + 1] |= disk

    if not mask.any():
        raise VisiumInputError("The generated tissue mask is empty")
    if dilation_iterations > 0:
        mask = binary_dilation(mask, iterations=dilation_iterations)

    # Standard Visium has a 100-um centre-to-centre lattice.  The individual
    # spot footprints alone leave a checkerboard of unpredicted pixels at the
    # histology scale.  Closing the footprint mask at half the measured lattice
    # spacing restores the contiguous tissue domain expected by the shared
    # DeepHEx inference step, without treating unmapped background as tissue.
    if closing_radius is None:
        distances, _ = cKDTree(centers).query(centers, k=2)
        lattice_spacing = float(np.median(distances[:, 1]))
        closing_radius = max(1, int(np.ceil(lattice_spacing / 2.0)))
    if closing_radius < 0:
        raise VisiumInputError("mask_closing_radius must be non-negative")
    if closing_radius > 0:
        axis = np.arange(-closing_radius, closing_radius + 1)
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        disk = yy**2 + xx**2 <= closing_radius**2
        mask = binary_closing(mask, structure=disk)
        mask = binary_fill_holes(mask)
    return mask, radius_px, int(closing_radius)


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def prepare_standard_visium(
    outs_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    matrix_path: str | os.PathLike[str] | None = None,
    positions_path: str | os.PathLike[str] | None = None,
    scalefactors_path: str | os.PathLike[str] | None = None,
    image_path: str | os.PathLike[str] | None = None,
    image_level: str = "auto",
    output_um_per_pixel: float = 0.5,
    spot_diameter_um: float = 55.0,
    source_um_per_pixel: float | None = None,
    superpixel_size: int = 16,
    num_top_genes: int = 1000,
    min_cells: int = 10,
    hvg_flavor: str = "seurat_v3",
    mask_dilation: int = 1,
    mask_closing_radius: int | None = None,
) -> dict:
    """Convert a standard Visium Space Ranger run into DeepHEx inputs.

    Returns a JSON-serialisable metadata dictionary.  The function writes all
    files consumed by the shared UNI, interpolation and BCAF stages.
    """

    inputs = discover_visium_inputs(
        outs_dir,
        matrix_path=matrix_path,
        positions_path=positions_path,
        scalefactors_path=scalefactors_path,
        image_path=image_path,
        image_level=image_level,
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    if output_um_per_pixel <= 0 or spot_diameter_um <= 0:
        raise VisiumInputError("Pixel size and spot diameter must be positive")
    if superpixel_size < 1:
        raise VisiumInputError("superpixel_size must be positive")

    with inputs.scalefactors_path.open("r", encoding="utf-8") as handle:
        scalefactors = json.load(handle)

    level_to_scale_key = {
        "fullres": None,
        "hires": "tissue_hires_scalef",
        "lowres": "tissue_lowres_scalef",
    }
    level_scale_key = level_to_scale_key[inputs.image_level]
    image_level_factor = (
        1.0 if level_scale_key is None else float(scalefactors[level_scale_key])
    )
    if image_level_factor <= 0:
        raise VisiumInputError("The selected image scale factor must be positive")

    if source_um_per_pixel is None:
        spot_diameter_fullres = scalefactors.get("spot_diameter_fullres")
        if spot_diameter_fullres is None or float(spot_diameter_fullres) <= 0:
            raise VisiumInputError(
                "scalefactors_json.json lacks spot_diameter_fullres; provide "
                "source_um_per_pixel explicitly"
            )
        spot_diameter_selected_image = float(spot_diameter_fullres) * image_level_factor
        source_um_per_pixel = spot_diameter_um / spot_diameter_selected_image
    source_um_per_pixel = float(source_um_per_pixel)
    if source_um_per_pixel <= 0:
        raise VisiumInputError("source_um_per_pixel must be positive")

    adata = _read_feature_matrix(inputs.matrix_path)
    n_genes_input = int(adata.n_vars)
    positions = _read_tissue_positions(inputs.positions_path)
    common_barcodes = [barcode for barcode in positions.index if barcode in adata.obs_names]
    if len(common_barcodes) < 2:
        raise VisiumInputError(
            "Fewer than two tissue-position barcodes occur in the feature matrix"
        )
    if len(common_barcodes) != adata.n_obs:
        warnings.warn(
            "Only barcodes present in both the feature matrix and tissue positions "
            f"are retained ({len(common_barcodes)}/{adata.n_obs}).",
            stacklevel=2,
        )
    positions = positions.loc[common_barcodes].copy()
    adata = adata[common_barcodes].copy()

    adata, hvg_table = _select_highly_variable_genes(
        adata,
        n_top_genes=num_top_genes,
        min_cells=min_cells,
        hvg_flavor=hvg_flavor,
    )

    image = _read_histology(inputs.image_path)
    output_spatial_scale = source_um_per_pixel / output_um_per_pixel
    image_array, image_dimensions = _resize_and_pad_image(
        image, output_spatial_scale, destination / "he.jpg"
    )

    coordinates_in_fullres = positions[
        ["pxl_col_in_fullres", "pxl_row_in_fullres"]
    ].to_numpy(dtype=np.float64)
    coordinates_in_selected_image = coordinates_in_fullres * image_level_factor
    coordinates = coordinates_in_selected_image * output_spatial_scale
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    outside = (
        (x_coordinates < 0)
        | (y_coordinates < 0)
        | (x_coordinates >= image_dimensions["resized_width"])
        | (y_coordinates >= image_dimensions["resized_height"])
    )
    if outside.any():
        raise VisiumInputError(
            f"{int(outside.sum())} tissue spots fall outside the selected histology "
            "image. Check image_level and scalefactors_json.json, or supply the "
            "matching full-resolution image explicitly."
        )

    locs = pd.DataFrame(
        {"x": x_coordinates, "y": y_coordinates},
        index=positions.index,
    )
    locs.index.name = "barcode"
    locs.to_csv(destination / "locs.csv", float_format="%.6f")

    expression = adata.X.toarray() if sparse.issparse(adata.X) else np.asarray(adata.X)
    counts = pd.DataFrame(
        np.asarray(expression, dtype=np.float32),
        index=positions.index,
        columns=adata.var_names.astype(str),
    )
    counts.to_csv(destination / "cnts.csv", float_format="%.8g")
    pd.Series(adata.var_names.astype(str), name="gene").to_csv(
        destination / "gene-names.txt", index=False, header=False
    )
    hvg_table.to_csv(destination / "highly-variable-genes.csv")

    mask, radius_px, effective_closing_radius = _build_spot_mask(
        coordinates,
        image_array.shape,
        output_um_per_pixel=output_um_per_pixel,
        spot_diameter_um=spot_diameter_um,
        superpixel_size=superpixel_size,
        dilation_iterations=mask_dilation,
        closing_radius=mask_closing_radius,
    )
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
        destination / "mask.png"
    )
    full_mask = Image.fromarray((mask.astype(np.uint8) * 255), mode="L").resize(
        (image_dimensions["padded_width"], image_dimensions["padded_height"]),
        Image.Resampling.NEAREST,
    )
    full_mask.save(destination / "mask_fullres.png")

    (destination / "radius.txt").write_text(f"{radius_px:.8g}\n", encoding="utf-8")
    (destination / "pixel-size-raw.txt").write_text(
        f"{source_um_per_pixel:.8g}\n", encoding="utf-8"
    )
    (destination / "pixel-size.txt").write_text(
        f"{output_um_per_pixel:.8g}\n", encoding="utf-8"
    )

    metadata = {
        "input_type": "standard_visium",
        "space_ranger_outs": str(inputs.outs_dir),
        "inputs": {
            "matrix": str(inputs.matrix_path),
            "tissue_positions": str(inputs.positions_path),
            "scalefactors": str(inputs.scalefactors_path),
            "histology_image": str(inputs.image_path),
            "image_level": inputs.image_level,
        },
        "coordinate_convention": {
            "locs_csv": "x = pxl_col_in_fullres, y = pxl_row_in_fullres; origin at top-left",
            "array_layout": "array_row is vertical, array_col is horizontal",
            "output_units": "scaled histology-image pixels",
        },
        "preprocessing": {
            "n_spots": int(adata.n_obs),
            "n_genes_input": n_genes_input,
            "n_genes_after_min_cells": int(hvg_table.shape[0]),
            "n_genes_selected": int(adata.n_vars),
            "min_cells": int(min_cells),
            "hvg_flavor": hvg_flavor,
            "num_top_genes_requested": int(num_top_genes),
            "source_um_per_pixel": source_um_per_pixel,
            "output_um_per_pixel": float(output_um_per_pixel),
            "image_rescale_factor": float(output_spatial_scale),
            "spot_diameter_um": float(spot_diameter_um),
            "spot_radius_output_pixels": float(radius_px),
            "superpixel_size_output_pixels": int(superpixel_size),
            "mask_grid_shape": [int(mask.shape[0]), int(mask.shape[1])],
            "mask_dilation_iterations": int(mask_dilation),
            "mask_closing_radius_superpixels": effective_closing_radius,
            "mask_domain": "contiguous tissue region inferred from spot lattice",
            "expression_matrix_shape": [int(counts.shape[0]), int(counts.shape[1])],
        },
        "image": image_dimensions,
        "software_versions": {
            package: _package_version(package)
            for package in ["numpy", "pandas", "scanpy", "Pillow", "scipy"]
        },
    }
    (destination / "visium_preprocessing.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(
        "Prepared standard Visium data: "
        f"{adata.n_obs} spots x {adata.n_vars} genes -> {portable_path(destination)}"
    )
    return metadata


def run_visium_preprocessing(config: dict) -> dict:
    """Run standard Visium preparation from a DeepHEx YAML configuration."""

    params = config["preprocessing"]["visium"]
    root = Path(config["global"]["directory"]).expanduser().resolve()
    outs = Path(params["outs_dir"]).expanduser()
    if not outs.is_absolute():
        outs = root / outs

    def optional_path(key: str) -> Path | None:
        value = params.get(key)
        if value in {None, "", "null"}:
            return None
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path

    return prepare_standard_visium(
        outs_dir=outs,
        output_dir=root,
        matrix_path=optional_path("matrix_path"),
        positions_path=optional_path("positions_path"),
        scalefactors_path=optional_path("scalefactors_path"),
        image_path=optional_path("image_path"),
        image_level=params.get("image_level", "auto"),
        output_um_per_pixel=float(params.get("output_um_per_pixel", 0.5)),
        spot_diameter_um=float(params.get("spot_diameter_um", 55.0)),
        source_um_per_pixel=params.get("source_um_per_pixel"),
        superpixel_size=int(config["global"]["superpixel_size"]),
        num_top_genes=int(params.get("num_top_genes", 1000)),
        min_cells=int(params.get("min_cells", 10)),
        hvg_flavor=params.get("hvg_flavor", "seurat_v3"),
        mask_dilation=int(params.get("mask_dilation", 1)),
        mask_closing_radius=params.get("mask_closing_radius"),
    )
