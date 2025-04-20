import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from lfmc.model.dataset import LFMCDataset

logger = logging.getLogger(__name__)


def load_normalizer(config_dir: Path) -> Normalizer:
    normalization_dicts = Dataset.load_normalization_values(config_dir / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=normalization_dicts)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Create H5pys from TIFs")
    parser.add_argument(
        "--config_dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--data_folder",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--h5py_folder",
        type=Path,
        required=True,
    )
    args = parser.parse_args()

    lfmc_dataset = LFMCDataset(
        normalizer=load_normalizer(args.config_dir),
        data_folder=args.data_folder,
        h5py_folder=args.h5py_folder,
        h5pys_only=False,
    )
    for i in tqdm(range(len(lfmc_dataset))):
        result = lfmc_dataset[i]
        if result is None:
            logger.error(f"None at {i}: {lfmc_dataset.tifs[i]}")


if __name__ == "__main__":
    main()
