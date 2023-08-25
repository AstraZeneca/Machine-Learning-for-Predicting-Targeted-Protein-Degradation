from __future__ import annotations
from typing import Mapping, Literal, Callable, List, ClassVar, Any, Tuple, Type

import torch
import pytorch_lightning as pl
import torch_geometric.nn as geom_nn
from torch_geometric.nn.models import GIN, GAT, GCN, AttentiveFP

class GnnSubModel(pl.LightningModule):

    def __init__(self,
                 num_node_features: int = 9,
                 node_edge_dim: int = 3,
                 model_type: Literal['gin', 'gat', 'gcn', 'attentivefp'] = 'gin',
                 hidden_channels: int = 32,
                 num_layers: int = 8,
                 out_channels: int = 8,
                 dropout: float = 0.1,
                 act: Literal['relu', 'elu'] = 'relu',
                 jk: Literal['max', 'last', 'cat', 'lstm'] = 'max',
                 norm: Literal['batch', 'layer'] = 'batch',
                 num_timesteps: int = 16):
        """Initialize a GNN submodel for encoding SMILES strings into a fixed-length vector representation.
        
        Args:
            num_node_features (int, optional): Number of node features. Defaults to 9. See `from_smiles` [implementation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html#from_smiles).
            node_edge_dim (int, optional): Number of edge features. Defaults to 3. See `from_smiles` [implementation](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/smiles.html#from_smiles).
            model_type (Literal['gin', 'gat', 'gcn', 'attentivefp'], optional): Type of GNN to use. Defaults to 'gin'.
            hidden_channels (int, optional): Number of hidden channels. Defaults to 32.
            num_layers (int, optional): Number of GNN layers. Defaults to 8.
            out_channels (int, optional): Number of output channels. Defaults to 8.
            dropout (float, optional): Dropout probability. Defaults to 0.1.
            act (Literal['relu', 'elu'], optional): Activation function. Defaults to 'relu'.
            jk (Literal['max', 'last', 'cat', 'lstm'], optional): JK aggregation type. Defaults to 'max'.
            norm (Literal['batch', 'layer'], optional): Normalization type. Defaults to 'batch'.
            num_timesteps (int, optional): Number of timesteps for AttentiveFP. Defaults to 16.
        """
        super().__init__()
        # Set our init args as class attributes
        self.__dict__.update(locals()) # Add arguments as attributes
        self.save_hyperparameters()
        self.smiles_embedding_size = out_channels
        if model_type == 'gin':
            self.smiles_embedding_size = hidden_channels
            self.gnn = GIN(in_channels=num_node_features,
                           hidden_channels=hidden_channels,
                           num_layers=num_layers,
                           dropout=dropout,
                           act=act,
                           norm=norm,
                           jk=jk)
        elif model_type == 'gat':
            self.gnn = GAT(in_channels=num_node_features,
                           hidden_channels=hidden_channels,
                           num_layers=num_layers,
                           out_channels=out_channels,
                           dropout=dropout,
                           act=act,
                           norm=norm,
                           jk=jk)
        elif model_type == 'gcn':
            self.gnn = GCN(in_channels=num_node_features,
                           hidden_channels=hidden_channels,
                           num_layers=num_layers,
                           out_channels=out_channels,
                           dropout=dropout,
                           act=act,
                           norm=norm,
                           jk=jk)
        elif model_type == 'attentivefp':
            self.gnn = AttentiveFP(in_channels=num_node_features,
                                   hidden_channels=hidden_channels,
                                   out_channels=out_channels,
                                   edge_dim=node_edge_dim,
                                   num_layers=num_layers,
                                   num_timesteps=num_timesteps,
                                   dropout=dropout)
        else:
            raise ValueError(f'Unknown model type: {model_type}. Available: gin, gat, gcn, attentivefp')
        
        
    def forward(self, batch):
        if self.model_type == 'gin':
            x = self.gnn(batch['smiles_graph'].x,
                         batch['smiles_graph'].edge_index)
            smiles_emb = geom_nn.global_add_pool(x, batch['smiles_graph'].batch)
        elif self.model_type == 'gat':
            x = self.gnn(x=batch['smiles_graph'].x.to(torch.float),
                         edge_index=batch['smiles_graph'].edge_index,
                         edge_attr=batch['smiles_graph'].edge_attr)
            smiles_emb = geom_nn.global_add_pool(x, batch['smiles_graph'].batch)
        elif self.model_type == 'gcn':
            x = self.gnn(x=batch['smiles_graph'].x.to(torch.float),
                         edge_index=batch['smiles_graph'].edge_index,
                         edge_attr=batch['smiles_graph'].edge_attr)
            smiles_emb = geom_nn.global_add_pool(x, batch['smiles_graph'].batch)
        elif self.model_type == 'attentivefp':
            smiles_emb = self.gnn(batch['smiles_graph'].x.to(torch.float),
                                  batch['smiles_graph'].edge_index,
                                  batch['smiles_graph'].edge_attr,
                                  batch['smiles_graph'].batch)
        return smiles_emb
    
    def get_embedding_size(self):
        return self.smiles_embedding_size