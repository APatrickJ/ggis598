import argparse
import logging
from pathlib import Path

import pandas as pd

from lfmc.common.const import LABELS_PATH


def analyze_csv(input_csv_path: Path):
    logging.info("Reading the CSV file")
    df = pd.read_csv(input_csv_path)

    logging.info("Analyzing the CSV file")
    logging.info("Number of rows: %d", len(df))
    logging.info("Number of sites: %d", df["site_name"].nunique())
    logging.info("Number of species: %d", df["species_collected"].nunique())
    logging.info("Number of functional types: %d", df["species_functional_type"].nunique())
    logging.info("Median LFMC: %f", df["lfmc_value"].median())
    logging.info("Min LFMC: %f", df["lfmc_value"].min())
    logging.info("Max LFMC: %f", df["lfmc_value"].max())
    logging.info(
        "Percentiles:\n%s",
        df["lfmc_value"].quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]),
    )
    logging.info("Protocols and counts:\n%s", df["protocol"].value_counts())


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Creates the LFMC CSV file")
    parser.add_argument(
        "--input_csv_path",
        type=Path,
        default=LABELS_PATH,
    )
    args = parser.parse_args()
    analyze_csv(args.input_csv_path)


if __name__ == "__main__":
    main()
