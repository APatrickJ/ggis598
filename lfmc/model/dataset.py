from pathlib import Path

from galileo.data.dataset import Dataset, Normalizer


class LFMCDataset(Dataset):
    def __init__(
        self,
        normalizer: Normalizer,
        data_folder: Path,
        h5py_folder: Path,
        download: bool = False,
        h5pys_only: bool = False,
        output_hw: int = 32,
        output_timesteps: int = 12,
    ):
        super().__init__(
            data_folder,
            download,
            h5py_folder,
            h5pys_only,
            output_hw,
            output_timesteps,
        )
        self.normalizer = normalizer

    def __len__(self):
        return len(self.data)
