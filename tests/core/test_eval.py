from pathlib import Path

from galileo.data.dataset import Normalizer
from galileo.galileo import Encoder
from lfmc.core.const import MeteorologicalSeason, WorldCoverClass
from lfmc.core.eval import FinetuningConfig, LFMCEval, evaluate_all


def test_finetune_and_test(
    tmp_path: Path,
    normalizer: Normalizer,
    encoder: Encoder,
    data_folder: Path,
    h5py_folder: Path,
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

    labels, preds = lfmc_eval.test(
        name="test",
        finetuned_model=finetuned_model,
        filter=None,
    )
    assert labels is not None
    assert preds is not None
    assert labels.shape == preds.shape
    assert 0 < labels.shape[0] <= len(list(data_folder.glob("*.tif")))
    assert 0 < preds.shape[0] <= len(list(data_folder.glob("*.tif")))


def test_evaluate_all(
    tmp_path: Path,
    normalizer: Normalizer,
    encoder: Encoder,
    data_folder: Path,
    h5py_folder: Path,
):
    output_folder = tmp_path / "results"
    output_folder.mkdir(parents=True, exist_ok=True)
    results = evaluate_all(
        normalizer=normalizer,
        pretrained_model=encoder,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        output_folder=output_folder,
    )
    assert results is not None
    assert isinstance(results, dict)
    filter_names = [
        "all",
        MeteorologicalSeason.WINTER,
        MeteorologicalSeason.SPRING,
        MeteorologicalSeason.SUMMER,
        MeteorologicalSeason.AUTUMN,
        WorldCoverClass.TREE_COVER,
        WorldCoverClass.SHRUBLAND,
        WorldCoverClass.GRASSLAND,
        "elevation_500_1000",
        "elevation_1000_1500",
        "elevation_1500_2000",
        "elevation_2000_2500",
        "high_fire_danger",
        "low_fire_danger",
    ]

    for filter_name in filter_names:
        assert filter_name in results
        assert "r2_score" in results[filter_name]
        assert "mae" in results[filter_name]
        assert "rmse" in results[filter_name]
        assert isinstance(results[filter_name]["r2_score"], float)
        assert isinstance(results[filter_name]["mae"], float)
        assert isinstance(results[filter_name]["rmse"], float)
