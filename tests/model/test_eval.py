from pathlib import Path

import lfmc
from galileo.data.config import NORMALIZATION_DICT_FILENAME
from galileo.data.dataset import Dataset, Normalizer
from galileo.galileo import Encoder
from lfmc.model.eval import FinetuningConfig, LFMCEval

ROOT_DIR = Path(lfmc.__file__).parent.parent


def load_normalizer(config_dir: Path) -> Normalizer:
    normalization_dicts = Dataset.load_normalization_values(config_dir / NORMALIZATION_DICT_FILENAME)
    return Normalizer(std=True, normalizing_dicts=normalization_dicts)


def test_finetune(tmp_path: Path):
    config_dir = ROOT_DIR / "data" / "config"
    current_dir = Path(__file__).parent
    data_folder = Path(current_dir / "data")
    h5py_folder = tmp_path / "h5pys"
    h5py_folder.mkdir(parents=True, exist_ok=True)
    output_folder = tmp_path / "finetuned"
    output_folder.mkdir(parents=True, exist_ok=True)
    lfmc_eval = LFMCEval(
        normalizer=load_normalizer(config_dir),
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
    )
    encoder = Encoder.load_from_folder(Path(ROOT_DIR / "data" / "models" / "nano"))
    finetuning_config = FinetuningConfig(
        max_epochs=1,
        weight_decay=0.001,
        learning_rate=0.001,
        batch_size=16,
        patience=5,
    )
    result = lfmc_eval.finetune(
        pretrained_model=encoder,
        output_folder=output_folder,
        finetuning_config=finetuning_config,
    )
    assert result is not None
    assert result.encoder is not None
    assert result.head is not None
