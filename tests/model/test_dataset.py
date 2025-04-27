from pathlib import Path
from typing import Sequence

import pytest

from galileo.data.dataset import Normalizer
from lfmc.common.const import MeteorologicalSeason, WorldCoverClass
from lfmc.common.filter import Filter
from lfmc.model.dataset import LFMCDataset
from lfmc.model.splits import num_splits


def test_dataset_all_samples(data_folder: Path, h5py_folder: Path, normalizer: Normalizer):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
    )
    assert len(dataset) == 10


def test_dataset_split(data_folder: Path, h5py_folder: Path, normalizer: Normalizer):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        split_id=0,
    )
    train_dataset, validation_dataset = dataset.split()
    assert len(train_dataset) == 7
    assert len(validation_dataset) == 3


def test_dataset_splits_are_different(data_folder: Path, h5py_folder: Path, normalizer: Normalizer):
    def assert_sets_unique(sets: Sequence[set[float]]):
        assert len(sets) == len({frozenset(s) for s in sets})

    training_samples_by_split_id: dict[int, set[float]] = {}
    validation_samples_by_split_id: dict[int, set[float]] = {}
    for split_id in range(num_splits()):
        dataset = LFMCDataset(
            normalizer=normalizer,
            data_folder=data_folder,
            h5py_folder=h5py_folder,
            h5pys_only=False,
            split_id=split_id,
        )
        train_dataset, validation_dataset = dataset.split()
        for i in range(len(train_dataset)):
            _, lfmc_value = train_dataset[i]
            training_samples_by_split_id.setdefault(split_id, set()).add(lfmc_value)
        for i in range(len(validation_dataset)):
            _, lfmc_value = validation_dataset[i]
            validation_samples_by_split_id.setdefault(split_id, set()).add(lfmc_value)

    assert_sets_unique(list(training_samples_by_split_id.values()))
    assert_sets_unique(list(validation_samples_by_split_id.values()))


@pytest.mark.parametrize(
    "season, expected_len",
    [
        (None, 10),
        (MeteorologicalSeason.WINTER, 1),
        (MeteorologicalSeason.SPRING, 1),
        (MeteorologicalSeason.SUMMER, 5),
        (MeteorologicalSeason.AUTUMN, 3),
    ],
)
def test_dataset_filter_seasons(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    season: MeteorologicalSeason,
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(seasons={season} if season is not None else None),
    )
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "landcover, expected_len",
    [
        (None, 10),
        (WorldCoverClass.TREE_COVER, 5),
        (WorldCoverClass.SHRUBLAND, 1),
        (WorldCoverClass.GRASSLAND, 4),
        (WorldCoverClass.CROPLAND, 0),
        (WorldCoverClass.BUILT_UP, 0),
        (WorldCoverClass.BARE_VEGETATION, 0),
        (WorldCoverClass.SNOW_AND_ICE, 0),
        (WorldCoverClass.WATER, 0),
        (WorldCoverClass.HERBACEOUS_WETLAND, 0),
        (WorldCoverClass.MANGROVES, 0),
        (WorldCoverClass.MOSS_AND_LICHEN, 0),
    ],
)
def test_dataset_filter_landcover(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    landcover: WorldCoverClass,
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(landcover={landcover} if landcover is not None else None),
    )
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "elevation, expected_len",
    [
        (None, 10),
        ((0, 1000), 1),
        ((1000, 2000), 7),
        ((2000, 3000), 2),
        ((3000, 4000), 0),
    ],
)
def test_dataset_filter_elevation(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    elevation: tuple[int, int],
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(elevation=elevation),
    )
    assert len(dataset) == expected_len


@pytest.mark.parametrize(
    "high_fire_danger, expected_len",
    [
        (None, 10),
        (True, 5),
        (False, 5),
    ],
)
def test_dataset_filter_high_fire_danger(
    data_folder: Path,
    h5py_folder: Path,
    normalizer: Normalizer,
    high_fire_danger: bool,
    expected_len: int,
):
    dataset = LFMCDataset(
        normalizer=normalizer,
        data_folder=data_folder,
        h5py_folder=h5py_folder,
        h5pys_only=False,
        filter=Filter(high_fire_danger=high_fire_danger),
    )
    assert len(dataset) == expected_len
