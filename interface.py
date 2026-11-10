from utils import load_config
from utils import set_global_seed
from data import (
    run_visium_hd_preprocessing,
    run_visium_preprocessing,
    run_xenium_preprocessing,
)
from model import run_get_hr
from model import run_feature_extraction
from pipeline import run_main
import argparse

def _infer_input_type(config):
    configured = config.get('global', {}).get('input_type')
    if configured:
        return configured
    preprocessing = config.get('preprocessing', {})
    if 'visium' in preprocessing:
        return 'visium'
    if 'xenium' in preprocessing:
        return 'xenium'
    if 'visium_hd' in preprocessing:
        return 'visium_hd'
    raise ValueError('Could not infer input_type from the configuration file')


def main(config_path, input_type=None):
    config = load_config(config_path)
    input_type = input_type or _infer_input_type(config)
    set_global_seed(int(config.get('global', {}).get('seed', 42)))

    preprocessors = {
        'visium': run_visium_preprocessing,
        'standard_visium': run_visium_preprocessing,
        'visium_hd': run_visium_hd_preprocessing,
        'xenium': run_xenium_preprocessing,
    }
    if input_type not in preprocessors:
        raise ValueError(
            f"Unsupported input_type {input_type!r}; choose visium, visium_hd or xenium"
        )
    preprocessors[input_type](config)

    run_get_hr(config)

    run_feature_extraction(config)

    run_main(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main script bridging data preprocessing and model training.")

    parser.add_argument('--config',
                        type=str,
                        default='configs/config_demo.yaml',
                        help='Path to the YAML configuration file')
    parser.add_argument(
        '--input-type',
        choices=['visium', 'standard_visium', 'visium_hd', 'xenium'],
        default=None,
        help='Input assay. Inferred from preprocessing.* when omitted.',
    )

    args = parser.parse_args()

    main(args.config, args.input_type)
