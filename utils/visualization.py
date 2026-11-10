"""Reusable publication-style visualizations for DeepHEx results.

The functions in this module deliberately contain no sample-specific absolute
paths.  Notebooks can therefore be shared without exposing local storage roots.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse
from scipy.stats import rankdata
from skimage.metrics import structural_similarity


PALETTE = {
    "ours": "#0F4D92",
    "ours_light": "#DCE8F4",
    "neutral": "#4D4D4D",
    "neutral_light": "#D9D9D9",
    "teal": "#42949E",
    "red": "#B64342",
}


def apply_publication_style(font_size: float = 8.0) -> None:
    """Apply consistent, editable-text figure defaults."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def save_publication_figure(fig: plt.Figure, stem: str | Path, dpi: int = 600) -> dict[str, str]:
    """Save editable vector outputs, a high-resolution TIFF and a PNG preview."""

    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": str(stem.with_suffix(".svg")),
        "pdf": str(stem.with_suffix(".pdf")),
        "tiff": str(stem.with_suffix(".tiff")),
        "png": str(stem.with_suffix(".png")),
    }
    fig.savefig(paths["svg"], bbox_inches="tight")
    fig.savefig(paths["pdf"], bbox_inches="tight")
    fig.savefig(paths["tiff"], dpi=dpi, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(paths["png"], dpi=220, bbox_inches="tight")
    return paths


def select_available_genes(
    available: list[str] | pd.Index,
    requested: list[str] | tuple[str, ...],
    fallback: list[str] | tuple[str, ...],
) -> tuple[list[str], list[str]]:
    """Select requested genes and fill missing slots from a review-backed fallback."""

    available_set = set(map(str, available))
    selected = [gene for gene in requested if gene in available_set]
    missing = [gene for gene in requested if gene not in available_set]
    for gene in fallback:
        if len(selected) >= len(requested):
            break
        if gene in available_set and gene not in selected:
            selected.append(gene)
    if not selected:
        raise ValueError("None of the requested or fallback genes are available")
    return selected, missing


def _read_mask(path: str | Path) -> np.ndarray:
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask > 0


def _normalise(values: np.ndarray, upper_quantile: float = 99.5) -> tuple[np.ndarray, mcolors.Normalize]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Expression vector contains no finite values")
    low = float(np.min(finite))
    high = float(np.quantile(finite, upper_quantile / 100.0))
    if not np.isfinite(high) or high <= low:
        high = low + 1.0
    norm = mcolors.Normalize(vmin=low, vmax=high, clip=True)
    return values, norm


def _style_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_standard_visium_expression(
    work_dir: str | Path,
    *,
    requested_genes: tuple[str, ...] = ("TP53", "RB1", "CDK4"),
    fallback_genes: tuple[str, ...] = ("TOP2A", "EPCAM", "KRT5", "ISG15"),
    figure_dir: str | Path | None = None,
) -> tuple[plt.Figure, list[str], dict[str, str]]:
    """Plot raw Visium spots beside native DeepHEx superpixel maps (Fig. 6c)."""

    root = Path(work_dir).expanduser()
    figure_dir = Path(figure_dir or root / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    apply_publication_style()

    spots = pd.read_csv(root / "cnts.csv", index_col=0)
    locs = pd.read_csv(root / "locs.csv", index_col=0)
    result = ad.read_h5ad(root / "deephex_expression.h5ad", backed="r")
    genes, missing = select_available_genes(result.var_names, requested_genes, fallback_genes)
    if missing:
        print("Genes not retained as HVGs:", ", ".join(missing))
    print("Visualized genes:", ", ".join(genes))

    mask = _read_mask(root / "mask.png")
    superpixel_size = int(result.uns.get("deephex", {}).get("superpixel_size", 16))
    ncols = len(genes)
    fig, axes = plt.subplots(2, ncols, figsize=(2.55 * ncols, 4.8), squeeze=False)
    # Match Fig. 6c: viridis encodes low-to-high expression for both rows.
    # Zero or missing spots remain white rather than receiving a false low value.
    raw_cmap = matplotlib.colormaps["viridis"].copy()
    raw_cmap.set_bad("white")
    deephex_cmap = matplotlib.colormaps["viridis"].copy()
    deephex_cmap.set_bad("white")

    for column, gene in enumerate(genes):
        spot_values = spots.loc[locs.index, gene].to_numpy(dtype=float)
        finite = np.isfinite(spot_values)
        positive = finite & (spot_values > 0)
        display_values = np.full_like(spot_values, np.nan)
        if positive.any():
            # Expression units differ substantially between genes.  Rank
            # normalisation gives all three raw panels the same viridis density.
            ranks = rankdata(spot_values[positive], method="average")
            if ranks.size > 1:
                ranks = (ranks - 1) / (ranks.size - 1)
            else:
                ranks = np.ones_like(ranks)
            display_values[positive] = ranks
        spot_norm = mcolors.Normalize(vmin=0.0, vmax=1.0, clip=True)
        ax = axes[0, column]
        scatter = ax.scatter(
            locs["x"],
            locs["y"],
            c=display_values,
            cmap=raw_cmap,
            norm=spot_norm,
            s=2.8,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )
        ax.invert_yaxis()
        _style_spatial_axis(ax)
        ax.set_title(gene, fontstyle="italic", pad=5)

        grid = np.full(mask.shape, np.nan, dtype=float)
        rows = result.obs["superpixel_row"].to_numpy(dtype=int)
        cols = result.obs["superpixel_col"].to_numpy(dtype=int)
        values = np.asarray(result[:, gene].X).ravel().astype(float)
        finite_values = values[np.isfinite(values)]
        high = float(np.quantile(finite_values, 0.99)) if finite_values.size else 1.0
        if not np.isfinite(high) or high <= 0:
            high = 1.0
        high_norm = mcolors.PowerNorm(gamma=0.55, vmin=0.0, vmax=high, clip=True)
        grid[rows, cols] = values
        ax = axes[1, column]
        image = ax.imshow(
            np.ma.masked_invalid(grid),
            cmap=deephex_cmap,
            norm=high_norm,
            interpolation="nearest",
            extent=(
                0,
                mask.shape[1] * superpixel_size,
                mask.shape[0] * superpixel_size,
                0,
            ),
            rasterized=True,
        )
        _style_spatial_axis(ax)

        tissue_rows, tissue_cols = np.nonzero(mask)
        margin = 2 * superpixel_size
        x_min = max(0, tissue_cols.min() * superpixel_size - margin)
        x_max = min(mask.shape[1] * superpixel_size, (tissue_cols.max() + 1) * superpixel_size + margin)
        y_min = max(0, tissue_rows.min() * superpixel_size - margin)
        y_max = min(mask.shape[0] * superpixel_size, (tissue_rows.max() + 1) * superpixel_size + margin)
        axes[0, column].set_xlim(x_min, x_max)
        axes[0, column].set_ylim(y_max, y_min)
        axes[1, column].set_xlim(x_min, x_max)
        axes[1, column].set_ylim(y_max, y_min)

    axes[0, 0].set_ylabel("Raw", fontsize=10)
    axes[1, 0].set_ylabel("DeepHEx", fontsize=10)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.04, wspace=0.07, hspace=0.06)
    paths = save_publication_figure(fig, figure_dir / "standard_visium_expression_maps")
    result.file.close()
    return fig, genes, paths


def _grid_adjacency(mask: np.ndarray) -> tuple[sparse.csr_matrix, int]:
    """Build a symmetric four-neighbour graph for tissue superpixels."""

    rows, cols = np.nonzero(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[rows, cols] = np.arange(rows.size)
    padded = np.pad(mask, ((0, 1), (0, 1)), mode="constant", constant_values=False)
    left, right = [], []
    for dr, dc in ((0, 1), (1, 0)):
        valid = padded[rows + dr, cols + dc]
        if np.any(valid):
            left.append(index[rows[valid], cols[valid]])
            right.append(index[rows[valid] + dr, cols[valid] + dc])
    src = np.concatenate(left + right)
    dst = np.concatenate(right + left)
    weights = np.ones(src.size, dtype=np.float64)
    matrix = sparse.coo_matrix((weights, (src, dst)), shape=(rows.size, rows.size)).tocsr()
    return matrix, int(rows.size)


def compute_spatial_autocorrelation(
    h5ad_path: str | Path,
    mask_path: str | Path,
    *,
    chunk_size: int = 64,
) -> pd.DataFrame:
    """Compute gene-wise Moran's I and Geary's C on the superpixel lattice."""

    result = ad.read_h5ad(h5ad_path, backed="r")
    mask = _read_mask(mask_path)
    graph, n_spots = _grid_adjacency(mask)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    edge_rows, edge_cols = graph.nonzero()
    total_weight = float(graph.sum())
    spatial_ids = pd.MultiIndex.from_arrays(
        [result.obs["superpixel_row"].astype(int), result.obs["superpixel_col"].astype(int)]
    )
    if spatial_ids.has_duplicates:
        raise ValueError("Duplicate superpixel coordinates in the result")

    moran, geary, nonzero = [], [], []
    for start in range(0, result.n_vars, chunk_size):
        stop = min(start + chunk_size, result.n_vars)
        block = result.X[:, start:stop]
        if sparse.issparse(block):
            block = block.toarray()
        block = np.asarray(block, dtype=np.float64)
        if block.ndim != 2:
            raise ValueError(f"Expected a 2-D expression block, got shape {block.shape}")
        if block.shape[0] != n_spots and block.shape[1] == n_spots:
            block = block.T
        if block.shape[0] != n_spots:
            raise ValueError(f"Unexpected expression block shape {block.shape}")
        centre = block - block.mean(axis=0, keepdims=True)
        variance_sum = np.square(centre).sum(axis=0)
        neighbour_sum = graph @ block
        numerator = np.sum(
            centre * (neighbour_sum - degree[:, None] * block.mean(axis=0, keepdims=True)),
            axis=0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            moran.extend(n_spots / total_weight * numerator / variance_sum)
            squared_difference = np.square(block[edge_rows] - block[edge_cols]).sum(axis=0)
            geary.extend(
                (n_spots - 1) / (2.0 * total_weight) * squared_difference / variance_sum
            )
            nonzero.extend(np.mean(block > 0, axis=0))

    frame = pd.DataFrame(
        {
            "gene": np.asarray(result.var_names),
            "moran_i": moran,
            "geary_c": geary,
            "nonzero_fraction": nonzero,
        }
    )
    result.file.close()
    return frame


def plot_metric_boxplots(
    metrics: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    titles: tuple[str, ...] | None = None,
    direction: tuple[str, ...] | None = None,
    output_stem: str | Path,
) -> tuple[plt.Figure, dict[str, str]]:
    """Plot one compact boxplot per metric, without benchmark/baseline boxes."""

    apply_publication_style()
    titles = titles or columns
    rows = len(metrics)
    fig, axes = plt.subplots(1, len(columns), figsize=(2.35 * len(columns), 2.8), squeeze=False)
    rng = np.random.default_rng(42)
    for ax, column, title, order in zip(axes[0], columns, titles, direction or ("",) * len(columns)):
        values = metrics[column].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No finite values for metric {column}")
        sample = rng.choice(values, size=min(300, values.size), replace=False)
        ax.scatter(
            rng.normal(1.0, 0.035, sample.size),
            sample,
            s=4,
            alpha=0.18,
            color=PALETTE["neutral"],
            linewidths=0,
            rasterized=True,
            zorder=1,
        )
        box = ax.boxplot(
            values,
            positions=[1],
            widths=0.42,
            patch_artist=True,
            whis=1.5,
            showfliers=False,
            zorder=3,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(PALETTE["ours_light"])
            patch.set_edgecolor(PALETTE["ours"])
            patch.set_linewidth(1.0)
        for key in ("whiskers", "caps", "medians"):
            for artist in box[key]:
                artist.set_color(PALETTE["ours"] if key == "medians" else PALETTE["neutral"])
                artist.set_linewidth(1.0)
        ax.set_xlim(0.65, 1.35)
        ax.set_xticks([1])
        ax.set_xticklabels(["DeepHEx"])
        ax.set_ylabel(order)
        ax.set_title(f"{title}\n$n_{{genes}}$ = {values.size}", fontsize=9)
        lower, upper = float(values.min()), float(values.max())
        margin = max((upper - lower) * 0.08, 1e-6)
        ax.set_ylim(lower - margin, upper + margin)
        ax.tick_params(axis="x", length=0)
    fig.tight_layout(pad=0.8, w_pad=1.4)
    paths = save_publication_figure(fig, output_stem)
    return fig, paths


def load_visium_hd_ground_truth(path: str | Path) -> np.ndarray:
    """Load the dense per-superpixel Visium HD ground-truth tensor."""

    with Path(path).open("rb") as handle:
        return np.asarray(pickle.load(handle), dtype=np.float32)


def _joint_minmax_01(
    first: np.ndarray,
    second: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalize paired maps on a shared scale while preserving amplitude."""

    first_values = np.asarray(first, dtype=float)[valid]
    second_values = np.asarray(second, dtype=float)[valid]
    if first_values.size == 0:
        raise ValueError("The tissue mask contains no values to normalize")
    low = min(float(np.min(first_values)), float(np.min(second_values)))
    high = max(float(np.max(first_values)), float(np.max(second_values)))
    scale = high - low
    if not np.isfinite(scale) or scale <= 0:
        return np.zeros_like(first, dtype=float), np.zeros_like(second, dtype=float)
    return (
        (np.asarray(first, dtype=float) - low) / scale,
        (np.asarray(second, dtype=float) - low) / scale,
    )


def compute_visium_hd_metrics(
    run_dir: str | Path,
    ground_truth: str | Path | np.ndarray,
    *,
    max_genes: int | None = None,
) -> pd.DataFrame:
    """Compute gene-wise metrics after joint min-max normalization to ``[0, 1]``."""

    root = Path(run_dir).expanduser()
    genes = [line.strip() for line in (root / "gene-names.txt").read_text().splitlines() if line.strip()]
    if max_genes is not None:
        genes = genes[:max_genes]
    mask = _read_mask(root / "mask.png")
    if not isinstance(ground_truth, np.ndarray):
        ground_truth = load_visium_hd_ground_truth(ground_truth)
    if ground_truth.ndim != 3:
        raise ValueError(f"Expected a 3-D Visium HD ground-truth tensor, got {ground_truth.shape}")
    if ground_truth.shape[2] < len(genes):
        raise ValueError(
            f"Ground truth has {ground_truth.shape[2]} genes but {len(genes)} genes were requested"
        )
    if ground_truth.shape[:2] != mask.shape:
        raise ValueError(f"Ground truth shape {ground_truth.shape[:2]} does not match mask {mask.shape}")
    valid = mask
    if not np.any(valid):
        raise ValueError("The tissue mask contains no valid superpixels")
    tissue_rows, tissue_cols = np.nonzero(valid)
    row_slice = slice(tissue_rows.min(), tissue_rows.max() + 1)
    col_slice = slice(tissue_cols.min(), tissue_cols.max() + 1)
    rows = []
    gene_indices = {gene: index for index, gene in enumerate(genes)}
    for gene in genes:
        with (root / "cnts-super" / f"{gene}.pickle").open("rb") as handle:
            prediction = np.asarray(pickle.load(handle), dtype=np.float32)
        truth = np.asarray(ground_truth[..., gene_indices[gene]], dtype=float)
        truth = np.where(np.isfinite(truth) & (truth >= 0), truth, 0.0)
        prediction = np.where(np.isfinite(prediction) & (prediction >= 0), prediction, 0.0)
        truth[~valid] = 0.0
        prediction[~valid] = 0.0
        truth_normalized, prediction_normalized = _joint_minmax_01(truth, prediction, valid)
        error = prediction_normalized[valid] - truth_normalized[valid]
        truth_crop = truth_normalized[row_slice, col_slice]
        prediction_crop = prediction_normalized[row_slice, col_slice]
        ssim = float(
            structural_similarity(
                truth_crop,
                prediction_crop,
                data_range=1.0,
            )
        )
        rows.append(
            {
                "gene": gene,
                "rmse": float(np.sqrt(np.mean(np.square(error)))),
                "mae": float(np.mean(np.abs(error))),
                "ssim": ssim,
            }
        )
    return pd.DataFrame(rows)


def plot_visium_hd_expression(
    run_dir: str | Path,
    ground_truth: np.ndarray,
    *,
    requested_genes: tuple[str, ...] = ("Olfm1", "Hpca"),
    fallback_genes: tuple[str, ...] = ("Snap25", "Slc17a7", "Pcp4", "Mbp"),
    figure_dir: str | Path | None = None,
    display_vmax: float = 0.15,
) -> tuple[plt.Figure, list[str], dict[str, str]]:
    """Compare jointly min-max normalized maps with a publication display scale."""

    root = Path(run_dir).expanduser()
    figure_dir = Path(figure_dir or root / "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    if not 0.0 < display_vmax <= 1.0:
        raise ValueError("display_vmax must be in the interval (0, 1]")
    apply_publication_style()
    genes_available = [line.strip() for line in (root / "gene-names.txt").read_text().splitlines() if line.strip()]
    genes, missing = select_available_genes(genes_available, requested_genes, fallback_genes)
    if missing:
        print("Genes absent from this HD panel:", ", ".join(missing))
    print("Visualized genes:", ", ".join(genes))
    mask = _read_mask(root / "mask.png")
    ncols = len(genes)
    fig, axes = plt.subplots(2, ncols, figsize=(2.35 * ncols, 4.8), squeeze=False)
    cmap = plt.get_cmap("viridis")
    for column, gene in enumerate(genes):
        gene_index = genes_available.index(gene)
        with (root / "cnts-super" / f"{gene}.pickle").open("rb") as handle:
            prediction = np.asarray(pickle.load(handle), dtype=float)
        truth = np.asarray(ground_truth[..., gene_index], dtype=float)
        truth = np.where(np.isfinite(truth) & (truth >= 0), truth, 0.0)
        prediction = np.where(np.isfinite(prediction) & (prediction >= 0), prediction, 0.0)
        truth[~mask] = 0.0
        prediction[~mask] = 0.0
        truth, prediction = _joint_minmax_01(truth, prediction, mask)
        # Expression values remain min-max normalized to [0, 1]. The lower display
        # ceiling prevents rare extremes from making most tissue pixels look black.
        norm = mcolors.Normalize(vmin=0.0, vmax=display_vmax, clip=True)
        for row, values in (
            (0, truth),
            (1, prediction),
        ):
            ax = axes[row, column]
            image = ax.imshow(
                values,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                rasterized=True,
            )
            _style_spatial_axis(ax)
            if row == 0:
                ax.set_title(gene, fontstyle="italic", pad=5)
            cb = fig.colorbar(image, ax=ax, fraction=0.038, pad=0.01)
            cb.set_ticks([0.0, display_vmax])
            cb.set_ticklabels(["Low", "High"])
            cb.ax.tick_params(length=0, width=0.6)
    axes[0, 0].set_ylabel("Visium HD ground truth\n(8 µm)", fontsize=9)
    axes[1, 0].set_ylabel("DeepHEx\n(super-resolution)", fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.04, wspace=0.09, hspace=0.04)
    paths = save_publication_figure(fig, figure_dir / "visium_hd_expression_maps")
    return fig, genes, paths
