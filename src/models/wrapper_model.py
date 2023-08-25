from __future__ import annotations
from typing import Optional, Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type, Dict

from data.protac_dataset import ProtacDataset
from data.protac_dataloader import custom_collate

import pytorch_lightning as pl
from pytorch_lightning import LightningModule

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

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
from torchvision.ops import MLP

from torchmetrics import MetricCollection


class ProtacModel(pl.LightningModule):

    def __init__(self,
                 smiles_encoder: nn.Module | nn.Sequential,
                 poi_seq_encoder: Optional[nn.Module | nn.Sequential] = None,
                 e3_ligase_encoder: Optional[nn.Module | nn.Sequential] = None,
                 cell_type_encoder: Optional[nn.Module | nn.Sequential] = None,
                 head: Optional[nn.Module | nn.Sequential] = None,
                 join_branches: Literal['cat', 'sum'] = 'cat',):
        """Wrapper class to make prediction on PROTAC data.

        Args:
            smiles_encoder (nn.Module): SMILES encoder class or instance.
            poi_seq_encoder (Optional[nn.Module], optional): POI sequence encoder class or instance. Defaults to None.
            e3_ligase_encoder (Optional[nn.Module], optional): E3 ligase encoder class or instance. Defaults to None.
            cell_type_encoder (Optional[nn.Module], optional): Cell type encoder class or instance. Defaults to None.
            head (Optional[nn.Module], optional): Head class or instance. Defaults to None.
            join_branches (Literal['cat', 'sum'], optional): How to join the branches embeddings. Defaults to 'cat'.
        """
        super().__init__()
        # Set our init args as class attributes
        # self.__dict__.update(locals()) # Add arguments as attributes
        # # Save the arguments passed to init
        # ignore_args = []
        # self.save_hyperparameters()
        self.join_branches = join_branches
        # Define SMILES encoder and head input size
        self.smiles_encoder = smiles_encoder
        head_input_size = self.smiles_encoder.get_embedding_size()
        # Define or load POI sequence, cell type and E3 ligase encoders
        # POI sequence encoder
        self.poi_seq_encoder = None
        if poi_seq_encoder is not None:
            self.poi_seq_encoder = poi_seq_encoder
            head_input_size += self.poi_seq_encoder.get_embedding_size()
        # E3 ligase encoder
        self.e3_ligase_encoder = None
        if e3_ligase_encoder is not None:
            self.e3_ligase_encoder = e3_ligase_encoder
            head_input_size += self.e3_ligase_encoder.get_embedding_size()
        # Cell type encoder
        self.cell_type_encoder = None
        if cell_type_encoder is not None:
            self.cell_type_encoder = cell_type_encoder
            head_input_size += self.cell_type_encoder.get_embedding_size()
        
        # self.extra_feat_enc = MLP(in_channels=head_input_size - self.smiles_encoder.get_embedding_size(),
        #                           hidden_channels=[1],
        #                           norm_layer=nn.BatchNorm1d,
        #                           inplace=False,
        #                           dropout=0.3)
        
        
        # Define head module
        if head is None:
            head_args = {
                'hidden_channels': [1],
                'norm_layer': nn.BatchNorm1d,
                'inplace': False,
                'dropout': 0.3,
            }
            self.head = MLP(in_channels=head_input_size, **head_args)
        else:
            self.head = head
        # Define loss function
        self.bin_loss = nn.BCEWithLogitsLoss()
        # Metrics, a separate metrics collection is defined for each stage
        # NOTE: According to the PyTorch Lightning docs, "similar" metrics,
        # i.e., requiring the same computation, should be optimized w/in a
        # metrics collection.
        stages = ['train_metrics', 'val_metrics', 'test_metrics']
        self.metrics = nn.ModuleDict({s: MetricCollection({
            'acc': Accuracy(task='binary'),
            'roc_auc': AUROC(task='binary'),
            'precision': Precision(task='binary'),
            'recall': Recall(task='binary'),
            'f1_score': F1Score(task='binary'),
            'opt_score': Accuracy(task='binary') + F1Score(task='binary'),
            'hp_metric': Accuracy(task='binary'),
        }, prefix=s.replace('metrics', '')) for s in stages})

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters())
    #     return optimizer

    def forward(self, x_in):
        mol_emb = self.smiles_encoder(x_in)
        if self.poi_seq_encoder is not None:
            poi_seq_emb = self.poi_seq_encoder(x_in)
            if self.join_branches == 'cat':
                mol_emb = torch.cat((mol_emb, poi_seq_emb), dim=-1)
            elif self.join_branches == 'sum':
                mol_emb = mol_emb + poi_seq_emb
        if self.e3_ligase_encoder is not None:
            e3_ligase_emb = self.e3_ligase_encoder(x_in)
            if self.join_branches == 'cat':
                mol_emb = torch.cat((mol_emb, e3_ligase_emb), dim=-1)
            elif self.join_branches == 'sum':
                mol_emb = mol_emb + e3_ligase_emb
        if self.cell_type_encoder is not None:
            cell_type_emb = self.cell_type_encoder(x_in)
            if self.join_branches == 'cat':
                mol_emb = torch.cat((mol_emb, cell_type_emb), dim=-1)
            elif self.join_branches == 'sum':
                mol_emb = mol_emb + cell_type_emb
        return self.head(mol_emb)
    
    def step(self, batch, stage='train'):
        y = batch['labels']
        preds = self.forward(batch)
        loss = self.bin_loss(preds, y)
        self.metrics[f'{stage}_metrics'].update(preds, y)
        self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True)
        self.log_dict(self.metrics[f'{stage}_metrics'], on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, stage='test')


class WrapperModel(pl.LightningModule):

    def __init__(self,
                 smiles_encoder: Type[nn.Module] | nn.Module,
                 smiles_encoder_args: Optional[Dict] = None,
                 task: Literal['predict_active_inactive', 'predict_pDC50_and_Dmax'] = 'predict_active_inactive',
                 freeze_smiles_encoder: bool = False,
                 cell_type_encoder: Optional[Type[nn.Module] | nn.Module] = None,
                 cell_type_encoder_args: Optional[Dict] = None,
                 e3_ligase_encoder: Optional[Type[nn.Module] | nn.Module] = None,
                 e3_ligase_encoder_args: Optional[Dict] = None,
                 poi_seq_encoder: Optional[Type[nn.Module] | nn.Module] = None,
                 poi_seq_encoder_args: Optional[Dict] = None,
                 head_class: Optional[Type[nn.Module] | nn.Module] = None,
                 head: Optional[nn.Module] = None,
                 head_args: Optional[Type[nn.Module] | nn.Module] = None,
                 train_dataset: ProtacDataset = None,
                 val_dataset: ProtacDataset = None,
                 test_dataset: ProtacDataset = None,
                 batch_size: int = 32,
                 learning_rate: float = 1e-3,
                 regr_loss: Type[nn.Module] | nn.Module | Callable = nn.HuberLoss,
                 regr_loss_args: Optional[Dict] = None):
        """Wrapper class to make prediction on PROTAC data.

        Args:
            smiles_encoder (Type[nn.Module] | nn.Module): SMILES encoder class or instance.
            smiles_encoder_args (Optional[Dict], optional): Arguments to pass to the SMILES encoder class. Defaults to None.
            task (Literal['predict_active_inactive', 'predict_pDC50_and_Dmax'], optional): Task to perform. Defaults to 'predict_active_inactive'.
            freeze_smiles_encoder (bool, optional): Whether to freeze the SMILES encoder. Defaults to False.
            cell_type_encoder (Optional[Type[nn.Module] | nn.Module], optional): Cell type encoder class or instance. Defaults to None.
            cell_type_encoder_args (Optional[Dict], optional): Arguments to pass to the cell type encoder class. Defaults to None.
            e3_ligase_encoder (Optional[Type[nn.Module] | nn.Module], optional): E3 ligase encoder class or instance. Defaults to None.
            e3_ligase_encoder_args (Optional[Dict], optional): Arguments to pass to the E3 ligase encoder class. Defaults to None.
            poi_seq_encoder (Optional[Type[nn.Module] | nn.Module], optional): POI sequence encoder class or instance. Defaults to None.
            poi_seq_encoder_args (Optional[Dict], optional): Arguments to pass to the POI sequence encoder class. Defaults to None.
            head_class (Optional[Type[nn.Module] | nn.Module], optional): Head class or instance. Defaults to None.
            head_args (Optional[Dict[List, bool, int, float, Type[nn.Module]]], optional): Arguments to pass to the head class. Defaults to None.
            train_dataset (ProtacDataset, optional): Training dataset. Defaults to None.
            val_dataset (ProtacDataset, optional): Validation dataset. Defaults to None.
            test_dataset (ProtacDataset, optional): Test dataset. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 32.
            learning_rate (float, optional): Learning rate. Defaults to 1e-3.
            regr_loss (Type[nn.Module] | nn.Module | Callable, optional): Regression loss function. Defaults to nn.HuberLoss.
            regr_loss_args (Optional[Dict], optional): Arguments to pass to the regression loss function. Defaults to None.
        """
        super().__init__()
        # Set our init args as class attributes
        self.__dict__.update(locals()) # Add arguments as attributes
        # Save the arguments passed to init
        ignore_args_as_hyperparams = [
            'train_dataset',
            'test_dataset',
            'val_dataset',
        ]
        self.save_hyperparameters(ignore=ignore_args_as_hyperparams) 
        # Define or load SMILES encoder sub-model
        smiles_encoder_args = smiles_encoder_args or {}
        self.smiles_encoder = smiles_encoder(**smiles_encoder_args)
        if freeze_smiles_encoder:
            self.smiles_encoder.freeze()
        head_input_size = self.smiles_encoder.get_embedding_size()
        # Define or load POI sequence, cell type and E3 ligase encoders
        # POI sequence encoder
        if poi_seq_encoder_args is not None:
            if poi_seq_encoder is None:
                raise ValueError('`poi_seq_encoder` must be provided if `poi_seq_encoder_args` is not None.')
            self.poi_seq_encoder = poi_seq_encoder(**poi_seq_encoder_args)
        elif poi_seq_encoder is not None:
            self.poi_seq_encoder = poi_seq_encoder
        if self.poi_seq_encoder is not None:
            head_input_size += self.poi_seq_encoder.get_embedding_size()
        # E3 ligase encoder
        if e3_ligase_encoder_args is not None:
            if e3_ligase_encoder is None:
                raise ValueError('`e3_ligase_encoder` must be provided if `e3_ligase_encoder_args` is not None.')
            self.e3_ligase_encoder = e3_ligase_encoder(**e3_ligase_encoder_args)
        elif e3_ligase_encoder is not None:
            self.e3_ligase_encoder = e3_ligase_encoder
        if self.e3_ligase_encoder is not None:
            head_input_size += self.e3_ligase_encoder.get_embedding_size()
        # Cell type encoder
        if cell_type_encoder_args is not None:
            if cell_type_encoder is None:
                raise ValueError('`cell_type_encoder` must be provided if `cell_type_encoder_args` is not None.')
            self.cell_type_encoder = cell_type_encoder(**cell_type_encoder_args)
        elif cell_type_encoder is not None:
            self.cell_type_encoder = cell_type_encoder
        if self.cell_type_encoder is not None:
            head_input_size += self.cell_type_encoder.get_embedding_size()
        # Define head module
        if head_args is None:
            num_outputs = 2 if task == 'predict_pDC50_and_Dmax' else 1
            head_args = {
                'hidden_channels': [num_outputs],
                'norm_layer': nn.BatchNorm1d,
                'inplace': False,
                'dropout': 0.3,
            }
        if head_class is None:
            self.head = MLP(in_channels=head_input_size, **head_args)
        elif head_args is not None:
            self.head = head_class(**head_args)
        else:
            self.head = head_class
        # Define PROTAC model
        self.model = ProtacModel(smiles_encoder=self.smiles_encoder,
                                 poi_seq_encoder=self.poi_seq_encoder,
                                 e3_ligase_encoder=self.e3_ligase_encoder,
                                 cell_type_encoder=self.cell_type_encoder,
                                 head=self.head)
        # Define losses
        if task == 'predict_pDC50_and_Dmax':
            if regr_loss_args is None:
                regr_loss_args = {'reduction': 'mean'}
            self.regr_loss = regr_loss(**regr_loss_args)
        else:
            self.bin_loss = nn.BCEWithLogitsLoss()
        # Metrics, a separate metrics collection is defined for each stage
        # NOTE: According to the PyTorch Lightning docs, "similar" metrics,
        # i.e., requiring the same computation, should be optimized w/in a
        # metrics collection.
        stages = ['train_metrics', 'val_metrics', 'test_metrics']
        self.metrics = nn.ModuleDict({s: MetricCollection({
            'acc': Accuracy(task='binary'),
            'roc_auc': AUROC(task='binary'),
            'precision': Precision(task='binary'),
            'recall': Recall(task='binary'),
            'f1_score': F1Score(task='binary'),
            'opt_score': Accuracy(task='binary') + F1Score(task='binary'),
            'hp_metric': Accuracy(task='binary'),
        }, prefix=s.replace('metrics', '')) for s in stages})
        # Misc settings
        self.missing_dataset_error = \
            '''Class variable `{0}` is None. If the model was loaded from a checkpoint, the dataset must be set manually:
            
            model = {1}.load_from_checkpoint('checkpoint.ckpt')
            model.{0} = my_{0}
            '''

    def forward(self, x_in):
        return self.model(x_in)

    def step(self, batch, stage='train'):
        y = batch['labels']
        preds = self.forward(batch)
        if self.task == 'predict_active_inactive':
            loss = self.bin_loss(preds, y)
            self.metrics[f'{stage}_metrics'].update(preds, y)
            self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True)
            self.log_dict(self.metrics[f'{stage}_metrics'], on_epoch=True)
        else:
            loss = self.regr_loss(preds, y)
            self.log(f'{stage}_loss', loss, on_epoch=True, prog_bar=True)
            if stage == 'val':
                self.log('hp_metric', loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, stage='val')

    def test_step(self, batch, batch_idx):
        return self.step(batch, stage='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def load_smiles_encoder(self, checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.smiles_encoder.load_state_dict(ckpt, strict=False)

    # def prepare_data(self):
    #     train_ds = os.path.join(data_dir, 'protac', f'train_dataset_fp{self.fp_bits}.pt')
    #     test_ds = os.path.join(data_dir, 'protac', f'test_dataset_fp{self.fp_bits}.pt')
    #     self.train_dataset = torch.load(train_ds)
    #     self.train_dataset = torch.load(train_ds)
    #     self.test_dataset = torch.load(test_ds)

    def train_dataloader(self):
        if self.train_dataset is None:
            format = 'train_dataset', self.__class__.__name__
            raise ValueError(self.missing_dataset_error.format(*format))
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, collate_fn=custom_collate,
                          drop_last=True)

    def val_dataloader(self):
        if self.val_dataset is None:
            format = 'val_dataset', self.__class__.__name__
            raise ValueError(self.missing_dataset_error.format(*format))
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, collate_fn=custom_collate)

    def test_dataloader(self):
        if self.test_dataset is None:
            format = 'test_dataset', self.__class__.__name__
            raise ValueError(self.missing_dataset_error.format(*format))
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, collate_fn=custom_collate)
