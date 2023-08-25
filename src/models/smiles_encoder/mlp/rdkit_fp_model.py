from __future__ import annotations
from typing import Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from torchvision.ops import MLP

import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.utils.smiles import from_smiles

from torchmetrics import (Accuracy,
                          AUROC,
                          ROC,
                          Precision,
                          Recall,
                          F1Score,
                          MeanAbsoluteError,
                          MeanSquaredError)
from torchmetrics.functional import (mean_absolute_error,
                                     mean_squared_error,
                                     mean_squared_log_error,
                                     pearson_corrcoef,
                                     r2_score)
from torchmetrics.functional.classification import (binary_accuracy,
                                                    binary_auroc,
                                                    binary_precision,
                                                    binary_recall,
                                                    binary_f1_score)
from torchmetrics import MetricCollection

MACCS_BITWIDTH = 167


class RDKitFingerprintEncoder(pl.LightningModule):

    def __init__(self,
                 fp_type: Literal['morgan_fp', 'maccs_fp', 'path_fp'] = 'morgan_fp',
                 fp_bits: int = 1024,
                 hidden_channels: List[int] = [128, 128],
                 norm_layer: Type[nn.Module] = nn.BatchNorm1d,
                 dropout: float = 0.5):
        """SMILES encoder using RDKit fingerprints.

        Args:
            fp_type (Literal['morgan_fp', 'maccs_fp', 'path_fp'], optional): Type of fingerprint to use. Defaults to 'morgan_fp'.
            fp_bits (int, optional): Number of bits in the fingerprint. Defaults to 1024.
            hidden_channels (List[int], optional): Number of hidden channels in the MLP. Defaults to [128, 128].
            norm_layer (Type[nn.Module], optional): Normalization layer to use. Defaults to nn.BatchNorm1d.
            dropout (float, optional): Dropout to use. Defaults to 0.5.
        """
        super().__init__()
        # Set our init args as class attributes
        self.__dict__.update(locals()) # Add arguments as attributes
        self.save_hyperparameters()
        self.fp_bits = MACCS_BITWIDTH if fp_type == 'maccs_fp' else fp_bits
        # Define PyTorch model
        self.fp_encoder = MLP(in_channels=self.fp_bits,
                              hidden_channels=hidden_channels,
                              norm_layer=norm_layer,
                              inplace=False,
                              dropout=dropout)
        self.maccs_encoder = MLP(in_channels=MACCS_BITWIDTH,
                                 hidden_channels=hidden_channels,
                                 norm_layer=norm_layer,
                                 inplace=False,
                                 dropout=dropout)

    def forward(self, x_in):
        # return self.fp_encoder(x_in[self.fp_type])
        morgan_emb = self.fp_encoder(x_in[self.fp_type])
        maccs_emb = self.maccs_encoder(x_in['maccs_fp'])
        return morgan_emb + maccs_emb

    def get_embedding_size(self):
        return self.hidden_channels[-1]


class FingerprintSubModel(pl.LightningModule):

    def __init__(self,
                 fp_type: Literal['morgan_fp', 'maccs_fp', 'path_fp'] = 'morgan_fp',
                 fp_bits: int = 1024,
                 hidden_channels: List[int] = [128, 128],
                 norm_layer: object = nn.BatchNorm1d,
                 dropout: float = 0.5):
        super().__init__()
        # Set our init args as class attributes
        self.__dict__.update(locals()) # Add arguments as attributes
        self.save_hyperparameters()
        self.fp_bits = MACCS_BITWIDTH if fp_type == 'maccs_fp' else fp_bits
        # Define PyTorch model
        self.fp_encoder = MLP(in_channels=self.fp_bits,
                              hidden_channels=hidden_channels,
                              norm_layer=norm_layer,
                              inplace=False,
                              dropout=dropout)

    def forward(self, x_in):
        return self.fp_encoder(x_in[self.fp_type])

    def get_smiles_embedding_size(self):
        return self.hidden_channels[-1]