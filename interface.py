from utils import load_config
from data import run_visium_hd_preprocessing
from model import run_get_hr
from model import run_feature_extraction
from pipeline import run_main
import argparse

def main(config_path):
    config = load_config(config_path)

    run_visium_hd_preprocessing(config)

    run_get_hr(config)

    run_feature_extraction(config)

    run_main(config)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main script bridging data preprocessing and model training.")

    parser.add_argument('--config',
                        type=str,
                        default='configs/config_demo.yaml',
                        help='Path to the YAML configuration file')

    args = parser.parse_args()

    main(args.config)
