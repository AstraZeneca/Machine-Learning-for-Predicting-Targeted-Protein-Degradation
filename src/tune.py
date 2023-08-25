from __future__ import annotations
from typing import Any, Dict, Set, Optional, Union, Type, List, Tuple, Callable

from data.protac_dataloader import PROTACDataModule
from models.wrapper_model import WrapperModel, ProtacModel # noqa: F401

from hyperparameter_tuning.cli import TuneLightningCLI
from hyperparameter_tuning.optuna_utils import objective

from pytorch_lightning.cli import ArgsType

def cli_main():
    """Main function for hyperparameter tuning."""
    cli = TuneLightningCLI(ProtacModel,
                           PROTACDataModule,
                           seed_everything_default=42,
                           parser_kwargs={'parser_mode': 'omegaconf'},
                           run=False)
    # Create study and launch optimization
    study = cli.create_study()
    optimize_args = cli.config.optuna.get('optimize', {})
    study.optimize(lambda trial: objective(trial, cli), **optimize_args)
    # Save best config to file
    best_config = study.best_trial.user_attrs['config']
    print(f'Best config: {best_config}')
    filename = best_config.get('best-config-filename', 'best_config.yaml')
    cli.parser.save(best_config, filename, format='yaml', overwrite=True)


if __name__ == '__main__':
    cli_main()