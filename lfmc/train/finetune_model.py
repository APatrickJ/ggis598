import argparse
import logging
from pathlib import Path

from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from galileo.galileo import Encoder
from lfmc.model.eval import LFMCEval


def load_normalizer(config_dir: Path) -> Normalizer:
    normalization_dicts = Dataset.load_normalization_values(config_dir / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=normalization_dicts)


def finetune_model(lfmc_eval: LFMCEval, pretrained_model_folder: Path, output_folder: Path):
    encoder = Encoder.load_from_folder(pretrained_model_folder)
    lfmc_eval.finetune(pretrained_model=encoder, output_folder=output_folder)


def main():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser("Finetune the LFMC model")
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
    args = parser.parse_args()

    lfmc_eval = LFMCEval(
        normalizer=load_normalizer(args.config_dir),
        data_folder=args.data_folder,
        h5py_folder=args.h5py_folder,
        h5pys_only=args.h5pys_only,
    )
    finetune_model(
        lfmc_eval=lfmc_eval,
        pretrained_model_folder=args.pretrained_model_folder,
        output_folder=args.output_folder,
    )


if __name__ == "__main__":
    main()
