from pathlib import Path

from galileo.data.dataset import Normalizer
from galileo.galileo import Encoder
from lfmc.model.eval import FinetuningConfig, LFMCEval


def test_finetune(tmp_path: Path, normalizer: Normalizer, encoder: Encoder, data_folder: Path, h5py_folder: Path):
    output_folder = tmp_path / "finetuned"
    output_folder.mkdir(parents=True, exist_ok=True)
    lfmc_eval = LFMCEval(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
    )
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
