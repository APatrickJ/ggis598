from pathlib import Path

from galileo.data.dataset import Normalizer
from galileo.galileo import Encoder
from lfmc.model.eval import FinetuningConfig, LFMCEval


def test_finetune_and_evaluate(
    tmp_path: Path, normalizer: Normalizer, encoder: Encoder, data_folder: Path, h5py_folder: Path
):
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
    finetuned_model = lfmc_eval.finetune(
        pretrained_model=encoder,
        output_folder=output_folder,
        finetuning_config=finetuning_config,
    )
    assert finetuned_model is not None
    assert finetuned_model.encoder is not None
    assert finetuned_model.head is not None

    metrics = lfmc_eval.evaluate(
        name="test",
        finetuned_model=finetuned_model,
        filter=None,
    )
    assert metrics is not None
    assert "test" in metrics
    assert "r2_score" in metrics["test"]
    assert "mae" in metrics["test"]
    assert "rmse" in metrics["test"]
    assert isinstance(metrics["test"]["r2_score"], float)
    assert isinstance(metrics["test"]["mae"], float)
    assert isinstance(metrics["test"]["rmse"], float)
