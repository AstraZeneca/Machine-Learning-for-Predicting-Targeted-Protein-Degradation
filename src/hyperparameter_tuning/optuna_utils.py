from __future__ import annotations
from typing import Any, Dict, Set, Optional, Union, Type, Callable

import optuna
import copy
import warnings
import logging
import importlib
from data.protac_dataloader import PROTACDataModule
from models.wrapper_model import WrapperModel, ProtacModel # noqa: F401

from hyperparameter_tuning.cli import TuneLightningCLI
from packaging import version
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback

from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
    Namespace,
    register_unresolvable_import_paths,
    set_config_read_mode,
    CLI,
    lazy_instance,
)


def suggest_optuna_param(trial: optuna.Trial,
                          name: str,
                          config: Dict[str, Any]) -> Any:
    """Given a config, suggest a value for the hyperparameter.

    Args:
        trial (optuna.Trial): The current Optuna trial
        name (str): The name of the hyperparameter
        config (Dict[str, Any]): The configuration of the hyperparameter

    Returns:
        Any: Suggested hyperparameter value
    """
    return getattr(trial, config['function'])(name, **config['kwargs'])


def objective(trial: optuna.Trial, cli: Optional[TuneLightningCLI] = None, config: Optional[Dict[str, Any]] = None) -> float:
    """Optuna objective function. Creates a new TuneLightningCLI object with the suggested hyperparameters and then fits the newly instantiated model.

    Args:
        trial (optuna.Trial): Optuna trial
        cli (Optional[TuneLightningCLI], optional): Custom LightningCLI object. Defaults to None.
        config (Optional[Dict[str, Any]], optional): Extra additional configurations as parsed arguments to the custom CLI. Defaults to None.

    Returns:
        float: Value of the objective function
    """
    if cli is None and config is None:
        raise ValueError('Either cli or config must be provided')
    elif cli is None and config is not None:
        cli_trial = TuneLightningCLI(ProtacModel,
                                     PROTACDataModule,
                                     seed_everything_default=42,
                                     parser_kwargs={'parser_mode': 'omegaconf'},
                                     args=config,
                                     run=False)
    else:
        cli_trial = copy.deepcopy(cli)
    if config is not None:
        print('WARNING. Both cli and config are provided. CLI "optuna" config will be updated.')
        cli_trial.config['optuna'].update(config)
    # Change CLI config with Optuna suggested values
    for hparam, cfg in cli_trial.config['optuna']['hparams'].items():
        cli_trial.config[hparam] = suggest_optuna_param(trial, hparam, cfg)
    # Change logging name if TensorBoardLogger is used
    if 'trainer.logger.TensorBoardLogger' in cli_trial.config:
        name = cli_trial.config['trainer.logger.TensorBoardLogger.init_args.name']
        name = f'{name}_{trial.number}'
        cli_trial.config['trainer.logger.TensorBoardLogger.init_args.name'] = name
    # Store current configuration in the trial attributes
    trial.set_user_attr('config', cli_trial.config)
    # TODO: Change Trainer config `overfit_batches` if not there already, then
    # fit the model and check its performance. If train accuracy not close to
    # 100%, then return a bad score. Else, remove the `overfit_batches` config
    # and fit the model.
    # 
    # Turn on `overfit_batches` (with 5% of the data) and fit the model
    cli_trial.config['trainer.overfit_batches'] = 1
    # cli_trial.config['trainer.limit_val_batches'] = 1.0
    cli_trial.instantiate_classes()
    cli_trial.trainer.fit(cli_trial.model, cli_trial.datamodule)
    # Check if train accuracy is close to 100%
    if cli_trial.trainer.callback_metrics['train_acc'] < 0.95:
        logging.warning(f'WARNING. Train accuracy is {cli_trial.trainer.callback_metrics["train_acc"]}. Returning bad score.')
        return 0.0
    # Turn off `overfit_batches` and fit the model
    cli_trial.config['trainer.overfit_batches'] = 0.0
    # NOTE: Instantiating again the classes would hopefully overwrite the model
    # and the dataloader, without wasting memory...
    # TODO: Add Optuna callback for pruning to Trainer callbacks
    # Instantiate classes and fit model
    cli_trial.instantiate_classes()
    cli_trial.trainer.fit(cli_trial.model, cli_trial.datamodule)
    # Obtain the metric value
    return cli_trial.trainer.callback_metrics[cli_trial.config.optuna['metric']]


# Define key names of `Trial.system_attrs`.
_PRUNED_KEY = "ddp_pl:pruned"
_EPOCH_KEY = "ddp_pl:epoch"

class CustomPyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.

    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.

    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.

    .. note::
        For the distributed data parallel training, the version of PyTorchLightning needs to be
        higher than or equal to v1.5.0. In addition, :class:`~optuna.study.Study` should be
        instantiated with RDB storage.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor
        self.is_ddp_backend = False

    def on_init_start(self, trainer: Trainer) -> None:
        self.is_ddp_backend = (
            trainer._accelerator_connector.distributed_backend is not None  # type: ignore
        )
        if self.is_ddp_backend:
            if version.parse(pl.__version__) < version.parse("1.5.0"):  # type: ignore
                raise ValueError("PyTorch Lightning>=1.5.0 is required in DDP.")
            if not (
                isinstance(self._trial.study._storage, _CachedStorage)
                and isinstance(self._trial.study._storage._backend, RDBStorage)
            ):
                raise ValueError(
                    "optuna.integration.PyTorchLightningPruningCallback"
                    " supports only optuna.storages.RDBStorage in DDP."
                )

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:

        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        should_stop = False
        if trainer.is_global_zero:
            self._trial.report(current_score.item(), step=epoch)
            should_stop = self._trial.should_prune()
        # TODO: The following line breaks the current version of Pytorch
        # Lightning. But I suspect it's necessary in a distributed training
        # environment... so it shouldn't matter for us...
        # should_stop = trainer.training_type_plugin.broadcast(should_stop)
        trainer.should_stop = should_stop
        if not should_stop:
            return

        if not self.is_ddp_backend:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)
        else:
            # Stop every DDP process if global rank 0 process decides to stop.
            trainer.should_stop = True
            if trainer.is_global_zero:
                self._trial.storage.set_trial_system_attr(self._trial._trial_id, _PRUNED_KEY, True)
                self._trial.storage.set_trial_system_attr(self._trial._trial_id, _EPOCH_KEY, epoch)

    def on_fit_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.is_ddp_backend:
            return

        # Because on_validation_end is executed in spawned processes,
        # _trial.report is necessary to update the memory in main process, not to update the RDB.
        _trial_id = self._trial._trial_id
        _study = self._trial.study
        _trial = _study._storage._backend.get_trial(_trial_id)  # type: ignore
        _trial_system_attrs = _study._storage.get_trial_system_attrs(_trial_id)
        is_pruned = _trial_system_attrs.get(_PRUNED_KEY)
        epoch = _trial_system_attrs.get(_EPOCH_KEY)
        intermediate_values = _trial.intermediate_values
        for step, value in intermediate_values.items():
            self._trial.report(value, step=step)

        if is_pruned:
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)