"""Export DeepHEx predictions into portable expression and coordinate files."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import pickle

import anndata as ad
import numpy as np
import pandas as pd
from PIL import Image

from .reproducibility import (
    environment_metadata,
    peak_cuda_memory_mb,
    peak_host_memory_mb,
    portable_path,
    redact_config,
    write_json,
)
from .utils import load_mask, read_lines, read_string


def export_deephex_results(
    prefix: str | Path,
    config: dict,
    *,
    stage_metadata: dict | None = None,
) -> dict:
    """Assemble per-gene pickles into an h5ad matrix and coordinate tables.

    The output is defined on the tissue-mask superpixels at the histology scale
    selected in preprocessing.  Each output row is one ``superpixel_size`` by
    ``superpixel_size`` pixel region; ``obs['x_um']`` and ``obs['y_um']`` give
    the corresponding physical position.
    """

    root = Path(prefix).expanduser().resolve()
    export_cfg = config.get("export", {})
    if not bool(export_cfg.get("enabled", True)):
        return {"enabled": False}

    gene_names = read_lines(root / "gene-names.txt")
    if not gene_names:
        raise ValueError(f"No gene names found in {root / 'gene-names.txt'}")
    missing = [
        gene
        for gene in gene_names
        if not (root / "cnts-super" / f"{gene}.pickle").is_file()
    ]
    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"Missing {len(missing)} per-gene prediction files, for example: {examples}"
        )

    mask = load_mask(root / "mask.png")
    tissue_cells = np.argwhere(mask)
    if tissue_cells.size == 0:
        raise ValueError("mask.png contains no tissue superpixels")

    expression = np.empty((len(tissue_cells), len(gene_names)), dtype=np.float32)
    reference_shape: tuple[int, int] | None = None
    for gene_index, gene in enumerate(gene_names):
        path = root / "cnts-super" / f"{gene}.pickle"
        with path.open("rb") as handle:
            gene_map = np.asarray(pickle.load(handle), dtype=np.float32)
        if gene_map.ndim != 2:
            raise ValueError(f"Expected a 2-D prediction array for {gene}: {path}")
        if reference_shape is None:
            reference_shape = gene_map.shape
            if reference_shape != mask.shape:
                raise ValueError(
                    f"Prediction shape {reference_shape} does not match mask shape "
                    f"{mask.shape}. Re-run with a consistent superpixel size."
                )
        elif gene_map.shape != reference_shape:
            raise ValueError(f"Inconsistent prediction shape for gene {gene}")
        expression[:, gene_index] = gene_map[tissue_cells[:, 0], tissue_cells[:, 1]]

    superpixel_size = int(config["global"]["superpixel_size"])
    output_um_per_pixel = float(read_string(root / "pixel-size.txt"))
    rows = tissue_cells[:, 0].astype(int)
    cols = tissue_cells[:, 1].astype(int)
    x_pixels = (cols + 0.5) * superpixel_size
    y_pixels = (rows + 0.5) * superpixel_size
    output_ids = pd.Index(
        [f"superpixel_{row}_{col}" for row, col in zip(rows, cols, strict=True)],
        name="spatial_id",
    )
    obs = pd.DataFrame(
        {
            "superpixel_row": rows,
            "superpixel_col": cols,
            "x": x_pixels,
            "y": y_pixels,
            "x_um": x_pixels * output_um_per_pixel,
            "y_um": y_pixels * output_um_per_pixel,
        },
        index=output_ids,
    )
    var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
    adata = ad.AnnData(X=expression, obs=obs, var=var)
    adata.uns["deephex"] = {
        "input_type": config.get("global", {}).get("input_type", "unknown"),
        "output_um_per_pixel": output_um_per_pixel,
        "superpixel_size": superpixel_size,
        "coordinate_convention": (
            "x/y are centres of superpixels in the rescaled histology image; "
            "x_um/y_um are physical coordinates"
        ),
    }

    h5ad_path = root / str(export_cfg.get("h5ad_name", "deephex_expression.h5ad"))
    adata.write_h5ad(h5ad_path, compression="gzip")
    coordinates_path = root / "deephex_coordinates.csv"
    obs.to_csv(coordinates_path, index=True)

    csv_path: Path | None = None
    if bool(export_cfg.get("csv", False)):
        csv_path = root / str(export_cfg.get("csv_name", "deephex_expression.csv"))
        pd.DataFrame(expression, index=output_ids, columns=gene_names).to_csv(
            csv_path, float_format="%.8g"
        )

    preprocessing_metadata: dict = {}
    preprocessing_json = root / "visium_preprocessing.json"
    if preprocessing_json.exists():
        preprocessing_metadata = json.loads(preprocessing_json.read_text(encoding="utf-8"))

    output_metadata = {
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "outputs": {
            "expression_h5ad": str(h5ad_path),
            "coordinates_csv": str(coordinates_path),
            "expression_csv": str(csv_path) if csv_path is not None else None,
        },
        "dimensions": {
            "n_superpixels": int(adata.n_obs),
            "n_genes": int(adata.n_vars),
            "expression_matrix": [int(adata.n_obs), int(adata.n_vars)],
            "histology_mask_grid": [int(mask.shape[0]), int(mask.shape[1])],
            "superpixel_size_pixels": superpixel_size,
            "output_um_per_pixel": output_um_per_pixel,
        },
        "config": redact_config(config),
        "preprocessing": preprocessing_metadata.get("preprocessing"),
        "environment": environment_metadata(),
        "resources": {
            "peak_host_memory_mb": peak_host_memory_mb(),
            "peak_cuda_memory_mb": peak_cuda_memory_mb(),
        },
        "stages": stage_metadata or {},
    }
    metadata_path = root / str(export_cfg.get("metadata_name", "run_metadata.json"))
    write_json(metadata_path, output_metadata)
    output_metadata["outputs"]["run_metadata_json"] = str(metadata_path)
    print(
        f"Exported DeepHEx results: {adata.n_obs} superpixels x {adata.n_vars} genes "
        f"to {portable_path(h5ad_path)}"
    )
    return output_metadata
