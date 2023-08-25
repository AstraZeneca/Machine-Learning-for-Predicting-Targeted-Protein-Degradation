from __future__ import annotations
from typing import Any, Dict, Set, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI, ArgsType, LightningArgumentParser

from lightning.pytorch.utilities.types import (
    _EVALUATE_OUTPUT,
    _PREDICT_OUTPUT,
    EVAL_DATALOADERS,
    LRSchedulerConfig,
    TRAIN_DATALOADERS,
)

from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
)

# simple demo classes for your convenience
from data.protac_dataloader import PROTACDataModule
from models.wrapper_model import WrapperModel, ProtacModel # noqa: F401
from models.smiles_encoder.mlp.rdkit_fp_model import RDKitFingerprintEncoder # noqa: F401


def cli_main():
    cli = LightningCLI(
        ProtacModel,
        PROTACDataModule,
        # subclass_mode_model=True,
        seed_everything_default=42,
        parser_kwargs={'parser_mode': 'omegaconf'},
        save_config_kwargs={'overwrite': True},
    )


if __name__ == '__main__':
    cli_main()