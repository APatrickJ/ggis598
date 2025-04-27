from copy import deepcopy
from logging import getLogger
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm

from galileo.data.dataset import Normalizer
from galileo.galileo import Encoder
from galileo.utils import device
from lfmc.common.const import MAX_LFMC_VALUE, MeteorologicalSeason, WorldCoverClass
from lfmc.common.filter import Filter
from lfmc.model.dataset import LFMCDataset
from lfmc.model.finetuning import DEFAULT_FINETUNING_CONFIG, FinetuningConfig, FineTuningModel
from lfmc.model.hyperparameters import DEFAULT_HYPERPARAMETERS, HyperParameters
from lfmc.model.mode import Mode
from lfmc.model.splits import num_splits

logger = getLogger(__name__)

ResultsDict = dict[str, dict[str, float]]


class LFMCEval:
    def __init__(
        self,
        normalizer: Normalizer,
        data_folder: Path,
        h5py_folder: Path,
        h5pys_only: bool = False,
        output_hw: int = 32,
        output_timesteps: int = 12,
        patch_size: int = 16,
        split_id: int = 0,
    ):
        self.normalizer = normalizer
        self.data_folder = data_folder
        self.h5py_folder = h5py_folder
        self.h5pys_only = h5pys_only
        self.output_hw = output_hw
        self.output_timesteps = output_timesteps
        self.patch_size = patch_size
        self.split_id = split_id

    @classmethod
    def _new_finetuning_model(cls, model: Encoder) -> FineTuningModel:
        num_classes = 1
        head = nn.Linear(model.embedding_size, num_classes)
        finetuning_model = FineTuningModel(model, head).to(device)
        finetuning_model.train()
        return finetuning_model

    def finetune(
        self,
        pretrained_model: Encoder,
        output_folder: Path,
        hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
        finetuning_config: FinetuningConfig = DEFAULT_FINETUNING_CONFIG,
    ) -> FineTuningModel:
        loss_fn = nn.MSELoss()

        dataset = LFMCDataset(
            normalizer=self.normalizer,
            data_folder=self.data_folder,
            h5py_folder=self.h5py_folder,
            h5pys_only=self.h5pys_only,
            output_hw=self.output_hw,
            output_timesteps=self.output_timesteps,
            split_id=self.split_id,
        )
        train_dataset, validation_dataset = dataset.split()

        train_loader = DataLoader(
            train_dataset,
            batch_size=finetuning_config.batch_size,
            shuffle=True,
            num_workers=hyperparams.num_workers,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=finetuning_config.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        finetuning_model = self._new_finetuning_model(pretrained_model)

        optimizer = torch.optim.AdamW(
            finetuning_model.parameters(),
            lr=finetuning_config.learning_rate,
            weight_decay=finetuning_config.weight_decay,
        )

        train_losses = []
        validation_losses = []
        best_loss = None
        best_model_dict = None
        epochs_since_improvement = 0

        for epoch in (pbar := tqdm(range(finetuning_config.max_epochs), desc="Finetuning")):
            finetuning_model.train()
            epoch_train_loss = 0.0

            for masked_output, label in tqdm(train_loader, desc="Training", leave=False):
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [x.to(device) for x in masked_output]
                optimizer.zero_grad()
                predictions = finetuning_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )[:, 0]
                loss = loss_fn(predictions, label.float().to(device))
                epoch_train_loss += loss.item()
                loss.backward()
                optimizer.step()

            train_losses.append(epoch_train_loss / len(train_loader))

            finetuning_model.eval()
            all_predictions = []
            all_labels = []
            for masked_output, label in tqdm(validation_loader, desc="Validation", leave=False):
                s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [x.to(device) for x in masked_output]
                with torch.no_grad():
                    predictions = finetuning_model(
                        s_t_x,
                        sp_x,
                        t_x,
                        st_x,
                        s_t_m,
                        sp_m,
                        t_m,
                        st_m,
                        months,
                        patch_size=self.patch_size,
                    )[:, 0]
                    all_predictions.append(predictions)
                    all_labels.append(label)

            validation_losses.append(
                torch.mean(loss_fn(torch.cat(all_predictions), torch.cat(all_labels).float().to(device)))
            )
            pbar.set_description(f"Train loss: {train_losses[-1]:.4f}, Validation loss: {validation_losses[-1]:.4f}")
            if best_loss is None or validation_losses[-1] < best_loss:
                best_loss = validation_losses[-1]
                best_model_dict = deepcopy(finetuning_model.state_dict())
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1
                if epochs_since_improvement >= finetuning_config.patience:
                    logger.info(f"Early stopping at epoch {epoch} with validation loss {validation_losses[-1]:.4f}")
                    break

        if best_model_dict is None:
            raise ValueError("No best model found")

        finetuning_model.load_state_dict(best_model_dict)
        torch.save(finetuning_model.state_dict(), output_folder / "lfmc_model.pth")
        finetuning_model.eval()
        return finetuning_model

    def test(
        self,
        name: str,
        finetuned_model: FineTuningModel,
        filter: Filter | None = None,
        hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
    ) -> tuple[np.ndarray, np.ndarray]:
        test_dataset = LFMCDataset(
            normalizer=self.normalizer,
            data_folder=self.data_folder,
            h5py_folder=self.h5py_folder,
            h5pys_only=self.h5pys_only,
            output_hw=self.output_hw,
            output_timesteps=self.output_timesteps,
            mode=Mode.VALIDATION,
            split_id=self.split_id,
            filter=filter,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=hyperparams.batch_size,
            shuffle=False,
            num_workers=hyperparams.num_workers,
        )

        labels_list = []
        preds_list = []
        for masked_output, label in tqdm(test_loader, desc=f"Evaluating {name}", leave=False):
            s_t_x, sp_x, t_x, st_x, s_t_m, sp_m, t_m, st_m, months = [x.to(device) for x in masked_output]
            finetuned_model.eval()
            with torch.no_grad():
                predictions = finetuned_model(
                    s_t_x,
                    sp_x,
                    t_x,
                    st_x,
                    s_t_m,
                    sp_m,
                    t_m,
                    st_m,
                    months,
                    patch_size=self.patch_size,
                )[:, 0]
                labels_list.append(label.cpu().numpy())
                preds_list.append(predictions.cpu().numpy())

        all_labels = np.concatenate(labels_list) if len(labels_list) > 0 else np.array([])
        all_preds = np.concatenate(preds_list) if len(preds_list) > 0 else np.array([])
        return all_labels, all_preds

    def compute_metrics(self, name: str, preds: np.ndarray, labels: np.ndarray) -> ResultsDict:
        if preds.size == 0 or labels.size == 0:
            return {}
        adjusted_preds = preds * MAX_LFMC_VALUE
        adjusted_labels = labels * MAX_LFMC_VALUE
        return {
            name: {
                "r2_score": r2_score(adjusted_labels, adjusted_preds),
                "mae": mean_absolute_error(adjusted_labels, adjusted_preds),
                "rmse": root_mean_squared_error(adjusted_labels, adjusted_preds),
            }
        }


def evaluate_all(
    normalizer: Normalizer,
    pretrained_model: Encoder,
    data_folder: Path,
    h5py_folder: Path,
    output_folder: Path,
    h5pys_only: bool = False,
    output_hw: int = 32,
    output_timesteps: int = 12,
    patch_size: int = 16,
    hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
    finetuning_config: FinetuningConfig = DEFAULT_FINETUNING_CONFIG,
) -> ResultsDict:
    filters = {
        "all": None,
        MeteorologicalSeason.WINTER: Filter(seasons={MeteorologicalSeason.WINTER}),
        MeteorologicalSeason.SPRING: Filter(seasons={MeteorologicalSeason.SPRING}),
        MeteorologicalSeason.SUMMER: Filter(seasons={MeteorologicalSeason.SUMMER}),
        MeteorologicalSeason.AUTUMN: Filter(seasons={MeteorologicalSeason.AUTUMN}),
        WorldCoverClass.TREE_COVER: Filter(landcover={WorldCoverClass.TREE_COVER}),
        WorldCoverClass.GRASSLAND: Filter(landcover={WorldCoverClass.GRASSLAND}),
        WorldCoverClass.SHRUBLAND: Filter(landcover={WorldCoverClass.SHRUBLAND}),
        WorldCoverClass.BUILT_UP: Filter(landcover={WorldCoverClass.BUILT_UP}),
        WorldCoverClass.BARE_VEGETATION: Filter(landcover={WorldCoverClass.BARE_VEGETATION}),
        "elevation_0_500": Filter(elevation=(0, 500)),
        "elevation_500_1000": Filter(elevation=(500, 1000)),
        "elevation_1000_1500": Filter(elevation=(1000, 1500)),
        "elevation_1500_2000": Filter(elevation=(1500, 2000)),
        "elevation_2000_2500": Filter(elevation=(2000, 2500)),
        "elevation_2500_3000": Filter(elevation=(2500, 3000)),
        "elevation_3000_3500": Filter(elevation=(3000, 3500)),
        "high_fire_danger": Filter(high_fire_danger=True),
        "low_fire_danger": Filter(high_fire_danger=False),
    }

    all_labels_by_name: dict[str, list[np.ndarray]] = {}
    all_preds_by_name: dict[str, list[np.ndarray]] = {}
    for split_id in tqdm(range(num_splits()), desc="Processing splits"):
        lfmc_eval = LFMCEval(
            normalizer=normalizer,
            data_folder=data_folder,
            h5py_folder=h5py_folder,
            h5pys_only=h5pys_only,
            output_hw=output_hw,
            output_timesteps=output_timesteps,
            patch_size=patch_size,
            split_id=split_id,
        )

        split_output_folder = output_folder / f"split_{split_id}"
        split_output_folder.mkdir(parents=True, exist_ok=True)
        finetuned_model = lfmc_eval.finetune(pretrained_model, split_output_folder, hyperparams, finetuning_config)

        for filter_name, filter in filters.items():
            labels, preds = lfmc_eval.test(filter_name, finetuned_model, filter=filter)
            all_labels_by_name.setdefault(filter_name, []).append(labels)
            all_preds_by_name.setdefault(filter_name, []).append(preds)

    all_results = {}
    for filter_name in filters.keys():
        all_labels = np.concatenate(all_labels_by_name[filter_name])
        all_preds = np.concatenate(all_preds_by_name[filter_name])
        results = lfmc_eval.compute_metrics(filter_name, all_preds, all_labels)
        all_results.update(results)
    return all_results
