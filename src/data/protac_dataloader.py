from __future__ import annotations
from typing import Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type

from data.protac_dataset import ProtacDataset

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from torch_geometric.data import Data, Batch
from torch.utils.data._utils.collate import collate
from torch.utils.data._utils.collate import default_collate_fn_map

from torchvision import transforms

import pandas as pd


def graph_collate(batch, *, collate_fn_map=None):
    # Handle graph data separately: graph representation and computation can be
    # greatly optimized due to their sparse nature. In fact, multiple graphs in
    # a batch can be seen as a 'big' graph of unconnected sub-graphs. Hence,
    # their adjecency matrices can be combined together to form a single one.
    return Batch.from_data_list(batch)


def custom_collate(batch):
    collate_map = default_collate_fn_map.copy()
    collate_map.update({Data: graph_collate})
    return collate(batch, collate_fn_map=collate_map)


class PROTACDataModule(pl.LightningDataModule):

    def __init__(self,
                 train_df_path: str = './data/train/train_bin_upsampled.csv',
                 val_df_path: str = './data/val/val_bin.csv',
                 test_df_path: str = './data/test/test_bin.csv',
                 predict_df_path: str = './data/test/test_bin.csv',
                 protac_dataset_args: Mapping[str, Any] = {},
                 batch_size: int = 32):
        """Wrapper DataModule for PROTAC datasets.
        
        Args:
            train_df_path (str, optional): Path to train dataset CSV. Defaults to './data/train/train_bin.csv'.
            val_df_path (str, optional): Path to validation dataset CSV. Defaults to './data/val/val_bin.csv'.
            test_df_path (str, optional): Path to test dataset CSV. Defaults to './data/test/test_bin.csv'.
            predict_df_path (str, optional): Path to prediction dataset CSV. Defaults to './data/test/test_bin.csv'.
            protac_dataset_args (Mapping[str, Any], optional): Arguments to pass to ProtacDataset. Defaults to {}.
            batch_size (int, optional): Batch size. Defaults to 32.
        """
        super().__init__()
        self.__dict__.update(locals()) # Add arguments as attributes
        self.save_hyperparameters()

    def prepare_data(self):
        # Download and clean PROTAC-DB and PROTAC-Pedia
        # TODO: Wrap the notebook code for data cleaning into a function
        pass

    def setup(self, stage: str = Literal['fit', 'validate', 'test', 'predict']):
        cols_to_keep = [
            'Smiles',
            'Smiles_nostereo',
            'DC50',
            'pDC50',
            'Dmax',
            'poi_gene_id',
            'poi_seq',
            'cell_type',
            'e3_ligase',
            'active',
        ]
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage == 'validate':
            train_df = pd.read_csv(self.train_df_path).reset_index(drop=True)
            train_df = train_df[cols_to_keep]
            self.train_ds = ProtacDataset(train_df, **self.protac_dataset_args)
            val_df = pd.read_csv(self.val_df_path).reset_index(drop=True)
            val_df = val_df[cols_to_keep]
            self.val_ds = ProtacDataset(val_df, **self.protac_dataset_args)
        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage == 'predict':
            test_df = pd.read_csv(self.test_df_path).reset_index(drop=True)
            test_df = test_df[cols_to_keep]
            self.test_ds = ProtacDataset(test_df, **self.protac_dataset_args)
            predict_df = pd.read_csv(self.predict_df_path).reset_index(drop=True)
            predict_df = predict_df[cols_to_keep]
            self.predict_ds = ProtacDataset(predict_df, **self.protac_dataset_args)

    def train_dataset(self):
        return self.train_ds

    def val_dataset(self):
        return self.val_ds

    def test_dataset(self):
        return self.test_ds

    def predict_dataset(self):
        return self.predict_ds
    
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset(), batch_size=self.batch_size,
                          shuffle=True, collate_fn=custom_collate,
                          drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset(), batch_size=self.batch_size,
                          shuffle=False, collate_fn=custom_collate)

    def test_dataloader(self):
        return DataLoader(self.test_dataset(), batch_size=self.batch_size,
                          shuffle=False, collate_fn=custom_collate)
    
    def test_dataloader(self):
        return DataLoader(self.predict_dataset(), batch_size=self.batch_size,
                          shuffle=False, collate_fn=custom_collate)