from __future__ import annotations
from typing import Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type

from sklearn.preprocessing import OrdinalEncoder

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np
import torch
from torch import nn

import joblib
import os


class CellTypeEncoder(pl.LightningModule):

    def __init__(self,
                 normalize_output: bool = False,
                 cell_type_encoder_filepath: str | None = 'cell_type_encoder.joblib',
                 run_ordinal_encoder: bool = False,
                 cell_type_train_data: np.ndarray | None = None,
                 use_linear_layer: bool = False,
                 embedding_size: int = 1,
                 ):
        """Encode cell type as an embedding vector.

        Args:
            normalize_output (bool, optional): Whether to normalize the output vector. Defaults to False.
            cell_type_encoder_filepath (str | None, optional): Path to get the ordinal encoder from. Defaults to 'cell_type_encoder.joblib'.
            run_ordinal_encoder (bool, optional): Whether to run the ordinal encoder. If False, the encoder will handle the raw string class. Defaults to False.
            cell_type_train_data (np.ndarray | None, optional): Training data to fit the ordinal encoder. Defaults to None.
            use_linear_layer (bool, optional): Whether to use a linear layer to encode the cell type. If False, the embedding size will be overwritten to 1. Defaults to False.
        Raises:
            ValueError: cell_type_train_data must be passed if cell_type_encoder_filepath does not exist'
        """
        super().__init__()
        # Set our init args as class attributes
        self.__dict__.update(locals()) # Add arguments as attributes
        self.save_hyperparameters(ignore='cell_type_train_data')
        # Load the pre-trained ordinal encoder if it exists, otherwise train it
        if run_ordinal_encoder:
            if os.path.exists(cell_type_encoder_filepath):
                self.cell_type_encoder = joblib.load(cell_type_encoder_filepath)
            else:
                self.cell_type_encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                                unknown_value=-1,
                                                encoded_missing_value=-1)
                if cell_type_train_data is not None:
                    self.fit_ordinal_encoder(cell_type_train_data)
                else:
                    raise ValueError('cell_type_train_data must be passed if cell_type_encoder_filepath does not exist')
        if use_linear_layer:
            self.lin_layer = nn.Linear(1, self.embedding_size)
        else:
            self.embedding_size = 1

    def forward(self, x_in):
        if self.run_ordinal_encoder:
            cell_emb = self.cell_type_encoder.transform(x_in['cell_type'].numpy())
            if self.normalize_output:
                cell_emb /= len(self.cell_type_encoder.categories_)
            cell_emb = torch.tensor(cell_emb, dtype=torch.float32)
        else:
            cell_emb = x_in['cell_type']
        if self.use_linear_layer:
            return self.lin_layer(cell_emb)
        else:
            return cell_emb

    def get_embedding_size(self):
        return self.embedding_size

    def fit_ordinal_encoder(self, cell_type_train_data: np.ndarray):
        self.cell_type_encoder.fit(cell_type_train_data.reshape(-1, 1))
        joblib.dump(self.cell_type_encoder, self.cell_type_encoder_filepath)