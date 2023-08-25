from __future__ import annotations
from typing import Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type

from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_extraction.text import CountVectorizer

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import numpy as np
import torch
from torch import nn

import joblib
import os


class POISequenceEncoder(pl.LightningModule):

    def __init__(self,
                 ngram_min_range: int = 2,
                 ngram_max_range: int = 2,
                 normalize_output: bool = False,
                 poi_seq_encoder_filepath: str | None = 'poi_seq_encoder.joblib',
                 run_count_vectorizer: bool = False,
                 poi_seq_train_data: np.ndarray | None = None,
                 input_size: int = 403,
                 use_linear_layer: bool = False,
                 embedding_size: int = 32,
                 ):
        """Encode POI sequence as an embedding vector.
        
        Args:
            ngram_min_range (int, optional): Minimum ngram range for the count vectorizer. Defaults to 2.
            ngram_max_range (int, optional): Maximum ngram range for the count vectorizer. Defaults to 2.
            normalize_output (bool, optional): Whether to normalize the output vector. Defaults to False.
            poi_seq_encoder_filepath (str | None, optional): Path to get the count vectorizer from. Defaults to 'poi_seq_encoder.joblib'.
            run_count_vectorizer (bool, optional): Whether to run the count vectorizer. If False, the encoder will handle the raw string class. Defaults to False.
            poi_seq_train_data (np.ndarray | None, optional): Training data to fit the count vectorizer. Defaults to None.
            input_size (int, optional): Input size of the linear layer. Defaults to 403.
            use_linear_layer (bool, optional): Whether to use a linear layer. Defaults to False.
            embedding_size (int, optional): Embedding size of the linear layer. Defaults to 32.
        """
        super().__init__()
        # Set our init args as class attributes
        self.__dict__.update(locals()) # Add arguments as attributes
        self.save_hyperparameters(ignore='poi_seq_train_data')
        # Load the pre-trained ordinal encoder if it exists, otherwise train it
        if os.path.exists(poi_seq_encoder_filepath):
            self.poi_seq_encoder = joblib.load(poi_seq_encoder_filepath)
            input_size = self.poi_seq_encoder.get_feature_names_out().shape[-1]
        elif run_count_vectorizer:
            ngram_range = (ngram_min_range, ngram_max_range)
            self.poi_seq_encoder = CountVectorizer(analyzer='char',
                                                ngram_range=ngram_range)
            if poi_seq_train_data is not None:
                self.fit_count_vectorizer(poi_seq_train_data)
            else:
                raise ValueError('poi_seq_train_data must be passed if poi_seq_encoder_filepath does not exist')
            input_size = self.poi_seq_encoder.get_feature_names_out().shape[-1]
        if use_linear_layer:
            self.lin_layer = nn.Linear(input_size, self.embedding_size)
        else:
            self.embedding_size = input_size

    def forward(self, x_in):
        if self.run_count_vectorizer:
            poi_emb = self.poi_seq_encoder.transform(x_in['poi_seq'].tolist())
            if self.normalize_output:
                poi_emb /= len(self.poi_seq_encoder.categories_)
            poi_emb = torch.tensor(poi_emb, dtype=torch.float32)
        else:
            poi_emb = x_in['poi_seq']
        if self.use_linear_layer:
            return self.lin_layer(poi_emb)
        else:
            return poi_emb

    def get_embedding_size(self):
        return self.embedding_size

    def fit_count_vectorizer(self, poi_seq_train_data: List[str]):
        self.poi_seq_encoder.fit(poi_seq_train_data)
        joblib.dump(self.poi_seq_encoder, self.poi_seq_encoder_filepath)