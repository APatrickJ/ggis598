import argparse
import json
import logging
from pathlib import Path

from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from galileo.galileo import Encoder
from galileo.utils import device
from lfmc.model.eval import evaluate_all

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
    parser = argparse.ArgumentParser("Evaluate the LFMC model")
    parser.add_argument(
        "--pretrained_model_folder",
        type=Path,
        default=Path("lib/galileo/data/models/nano"),
        required=True,
    )
    parser.add_argument(
        "--output_folder",
        type=Path,
        default=Path("data/models"),
        required=True,
    )
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
    parser.add_argument(
        "--h5pys_only",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output_hw",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    logger.info("Device: %s", device)

    results = evaluate_all(
        normalizer=load_normalizer(args.config_dir),
        pretrained_model_folder=Encoder.load_from_folder(args.pretrained_model_folder),
        data_folder=args.data_folder,
        h5py_folder=args.h5py_folder,
        output_folder=args.output_folder,
        h5pys_only=args.h5pys_only,
        patch_size=args.patch_size,
        output_hw=args.output_hw,
    )

    with open(args.output_folder / "results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
