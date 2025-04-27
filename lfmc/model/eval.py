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
from lfmc.common.const import MAX_LFMC_VALUE
from lfmc.common.filter import Filter
from lfmc.model.dataset import LFMCDataset
from lfmc.model.finetuning import DEFAULT_FINETUNING_CONFIG, FinetuningConfig, FineTuningModel
from lfmc.model.hyperparameters import DEFAULT_HYPERPARAMETERS, HyperParameters
from lfmc.model.mode import Mode

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

    def evaluate(
        self,
        name: str,
        finetuned_model: FineTuningModel,
        filter: Filter | None = None,
        hyperparams: HyperParameters = DEFAULT_HYPERPARAMETERS,
    ) -> ResultsDict:
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

        preds_list = []
        labels_list = []
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
                preds_list.append(predictions.cpu().numpy())
                labels_list.append(label.cpu().numpy())

        all_preds = np.concatenate(preds_list)
        all_labels = np.concatenate(labels_list)
        return self._compute_metrics(name, all_preds, all_labels)

    def _compute_metrics(self, name: str, preds: np.ndarray, labels: np.ndarray) -> ResultsDict:
        adjusted_preds = preds * MAX_LFMC_VALUE
        adjusted_labels = labels * MAX_LFMC_VALUE
        return {
            name: {
                "r2_score": r2_score(adjusted_labels, adjusted_preds),
                "mae": mean_absolute_error(adjusted_labels, adjusted_preds),
                "rmse": root_mean_squared_error(adjusted_labels, adjusted_preds),
            }
        }
