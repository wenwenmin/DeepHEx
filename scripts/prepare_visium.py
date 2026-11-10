"""Command-line wrapper for standard 10x Genomics Visium preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.processing_Visium import prepare_standard_visium


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a standard Visium Space Ranger outs directory into the "
            "canonical inputs used by DeepHEx."
        )
    )
    parser.add_argument("--outs", required=True, help="Space Ranger outs directory")
    parser.add_argument("--output-dir", required=True, help="DeepHEx working directory")
    parser.add_argument("--matrix", default=None, help="Override feature-barcode matrix")
    parser.add_argument("--positions", default=None, help="Override tissue positions CSV")
    parser.add_argument(
        "--scalefactors", default=None, help="Override scalefactors_json.json"
    )
    parser.add_argument("--image", default=None, help="Override histology image")
    parser.add_argument(
        "--image-level",
        choices=["auto", "fullres", "hires", "lowres"],
        default="auto",
        help="Coordinate system of the selected histology image (default: auto)",
    )
    parser.add_argument(
        "--output-um-per-pixel",
        type=float,
        default=0.5,
        help="Target histology pixel size in micrometres (default: 0.5)",
    )
    parser.add_argument(
        "--spot-diameter-um",
        type=float,
        default=55.0,
        help="Physical Visium spot diameter (default: 55)",
    )
    parser.add_argument(
        "--source-um-per-pixel",
        type=float,
        default=None,
        help="Override the source image pixel size derived from scalefactors",
    )
    parser.add_argument("--superpixel-size", type=int, default=16)
    parser.add_argument("--num-top-genes", type=int, default=1000)
    parser.add_argument("--min-cells", type=int, default=10)
    parser.add_argument(
        "--hvg-flavor",
        choices=["seurat_v3", "seurat", "cell_ranger"],
        default="seurat_v3",
    )
    parser.add_argument("--mask-dilation", type=int, default=1)
    parser.add_argument(
        "--mask-closing-radius",
        type=int,
        default=None,
        help=(
            "Radius in superpixels for closing spot footprints into a contiguous "
            "tissue mask; default infers half the spot-lattice spacing"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = prepare_standard_visium(
        outs_dir=args.outs,
        output_dir=args.output_dir,
        matrix_path=args.matrix,
        positions_path=args.positions,
        scalefactors_path=args.scalefactors,
        image_path=args.image,
        image_level=args.image_level,
        output_um_per_pixel=args.output_um_per_pixel,
        spot_diameter_um=args.spot_diameter_um,
        source_um_per_pixel=args.source_um_per_pixel,
        superpixel_size=args.superpixel_size,
        num_top_genes=args.num_top_genes,
        min_cells=args.min_cells,
        hvg_flavor=args.hvg_flavor,
        mask_dilation=args.mask_dilation,
        mask_closing_radius=args.mask_closing_radius,
    )
    print(json.dumps(metadata["preprocessing"], indent=2))


if __name__ == "__main__":
    main()
