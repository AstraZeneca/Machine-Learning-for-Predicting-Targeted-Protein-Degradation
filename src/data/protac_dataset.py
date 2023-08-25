from __future__ import annotations
from typing import (Dict, Optional, Literal, Callable, Any, Tuple, Type)

from utils.fingerprints import get_fingerprint

import numpy as np

import torch
from torch.utils.data import Dataset

import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.utils.smiles import from_smiles

from torchvision.transforms import Compose

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

import pandas as pd
import joblib

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    RobertaTokenizerFast,
    RobertaForMaskedLM,    
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ProtacDataset(Dataset):

    def __init__(self,
                 dataframe,
                 task: Literal['predict_active_inactive', 'predict_pDC50_and_Dmax'] = 'predict_active_inactive',
                 scale_concentration: bool = False,
                 include_smiles_as_str: bool = False,
                 include_smiles_as_graphs: bool = False,
                 smiles_tokenizer: Optional[str | Callable] = None,
                 smiles_tokenizer_type: Type = AutoTokenizer,
                 smiles_tokenizer_args: Dict = {},
                 ngram_range: Tuple[int, int] = (2, 2),      
                 precompute_smiles_as_graphs: bool = False,
                 precompute_fingerprints: bool = False,
                 use_for_ssl: bool = False,
                 use_morgan_fp: bool = False,
                 morgan_bits: int = 1024,
                 morgan_atomic_radius: int = 2,
                 use_maccs_fp: bool = False,
                 use_path_fp: bool = False,
                 path_bits: int = 1024,
                 fp_min_path: int = 1,
                 fp_max_path: int = 16,
                 include_poi_seq: bool = True,
                 include_poi_gene: bool = False,
                 include_e3_ligase: bool = True,
                 include_cell_type: bool = True,
                 tokenize_poi_seq: bool = False,
                 poi_tokenizer: Optional[Callable | str] = None,
                 poi_seq_enc: Optional[Callable | str] = None,
                 preencode_poi_seq: bool = False,
                 poi_gene_enc: Optional[Callable | str] = None,
                 e3_ligase_enc: Optional[Callable | str] = None,
                 cell_type_enc: Optional[Callable | str] = None,
                 use_default_poi_seq_enc: bool = False,
                 use_default_e3_ligase_enc: bool = False,
                 use_default_cell_type_enc: bool = False,
                 normalize_poi_seq_enc: bool = False,
                 normalize_e3_ligase_enc: bool = False,
                 normalize_cell_type_enc: bool = False,
                 normalize_poi_gene_enc: bool = False,
                 transform: Optional[Callable] = None):
        """Pytorch Dataset for PROTAC data. Each element will consist of a dictionary of different processed features.
        When processed by a DataLoader, the dictionary structure will remain, but each value will be converted to a batch of tensors.

        Args:
            dataframe (pd.DataFrame): Dataframe containing the PROTAC data
            task (Literal['predict_active_inactive', 'predict_pDC50_and_Dmax'], optional): Task to perform. Defaults to 'predict_active_inactive'.
            scale_concentration (bool, optional): Whether to scale the concentration values. Defaults to False.
            
        """
        self.__dict__.update(locals()) # Add arguments as attributes
        self.hparams = {k: v for k, v in locals().items() if k != 'dataframe' and k != 'self'} # Store hyperparameters
        self.maccs_bits = 167 # Hardcoded, see RDKit documentation
        self.dataset_len = len(self.dataframe)
        # Handle SMILES information
        self.smiles = self.dataframe['Smiles_nostereo']
        # if include_selfies:
        #     self.selfies = [sf.encoder(s) for s in self.smiles]
        if precompute_fingerprints:
            if self.use_morgan_fp:
                self.morgan_fp = np.array([get_fingerprint(s, n_bits=self.morgan_bits, fp_type='morgan', atomic_radius=morgan_atomic_radius).astype(np.float32) for s in self.smiles])
            if self.use_maccs_fp:
                self.maccs_fp = np.array([get_fingerprint(s, fp_type='maccs').astype(np.float32) for s in self.smiles])
            if self.use_path_fp:
                self.path_fp = np.array([get_fingerprint(s, n_bits=self.path_bits, fp_type='path', min_path=self.fp_min_path, max_path=self.fp_max_path).astype(np.float32) for s in self.smiles])
        if include_smiles_as_graphs or precompute_smiles_as_graphs:
            # NOTE: self.graph_smiles is a list of PytorchGeometric Data objects
            self.graph_smiles = [from_smiles(s) for s in self.smiles]
        if smiles_tokenizer is not None:
            if isinstance(smiles_tokenizer, str):
                self.smiles_tokenizer = smiles_tokenizer_type.from_pretrained(smiles_tokenizer, **smiles_tokenizer_args)
            else:
                self.smiles_tokenizer = smiles_tokenizer
            # NOTE: Do NOT return tensors when doing SSL, i.e., MLM, as reported
            # in this conversation: https://discuss.huggingface.co/t/extra-dimension-with-datacollatorfor-languagemodeling-into-bertformaskedlm/6400/6
            if use_for_ssl:
                self.smiles_tokenized = [
                    self.smiles_tokenizer(s, padding='max_length', truncation=True) for s in self.smiles
                ]
                assert len(self.smiles_tokenized) == len(self.smiles), (
                    f'ERROR. Len tokenized {len(self.smiles_tokenized)} /= len SMILES {len(self.smiles)}'
                )
            else:
                self.smiles_tokenized = [
                    self.smiles_tokenizer(s, padding='max_length', truncation=True, return_tensors='pt') for s in self.smiles
                ]
        # Handle the POI sequence
        if include_poi_seq:
            self.poi_seq = self.dataframe['poi_seq'].to_list()
            if poi_seq_enc is not None:
                if isinstance(poi_seq_enc, str):
                    self.poi_seq_enc = joblib.load(poi_seq_enc)
                else:
                    self.poi_seq_enc = poi_seq_enc
            else:
                self.poi_seq_enc = CountVectorizer(analyzer='char',
                                                        ngram_range=ngram_range)
                self.poi_seq_enc.fit(self.poi_seq)
            if preencode_poi_seq:
                self.poi_seq = self.poi_seq_enc.transform(self.poi_seq)
                self.poi_seq = self.poi_seq.toarray().astype(np.float32)
        # Tokenize the POI sequence (for example for BERT-based models)
        if tokenize_poi_seq:
            if poi_tokenizer is None:
                self.poi_seq = self.dataframe['poi_seq']
            else:
                self.poi_seq = [self.poi_tokenizer(seq, padding='max_length', truncation=True, return_tensors='pt') for seq in self.dataframe['poi_seq']]
        # Handle the POI gene
        if include_poi_gene:
            self.gene = self.dataframe['poi_gene_id'].to_numpy().reshape(-1, 1)
            if poi_gene_enc is not None:
                self.gene = poi_gene_enc.transform(self.gene)
                self.gene = self.gene.astype(np.float32)
                if normalize_poi_gene_enc:
                    self.gene /= len(poi_gene_enc.categories_)
            else:
                self.poi_gene_enc = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    encoded_missing_value=-1
                )
                tmp = self.poi_gene_enc.fit_transform(self.gene)
                self.gene = tmp.astype(np.float32).flatten()
        # Handle the E3 ligase
        if include_e3_ligase:
            self.e3_ligase = self.dataframe['e3_ligase'].to_numpy().reshape(-1, 1)
            if e3_ligase_enc is not None:
                if isinstance(e3_ligase_enc, str):
                    self.e3_ligase_enc = joblib.load(e3_ligase_enc)
                else:
                    self.e3_ligase_enc = e3_ligase_enc
                self.e3_ligase = self.e3_ligase_enc.transform(self.e3_ligase)
                self.e3_ligase = self.e3_ligase.astype(np.float32)
                if normalize_e3_ligase_enc:
                    self.e3_ligase /= len(self.e3_ligase_enc.categories_)
            elif use_default_e3_ligase_enc:
                self.e3_ligase_enc = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    encoded_missing_value=-1
                )
                tmp = self.e3_ligase_enc.fit_transform(self.e3_ligase)
                self.e3_ligase = tmp.astype(np.float32)
        # Handle cell type
        if include_cell_type:
            self.cell_type = self.dataframe['cell_type'].to_numpy().reshape(-1, 1)
            if cell_type_enc is not None:
                if isinstance(cell_type_enc, str):
                    self.cell_type_enc = joblib.load(cell_type_enc)
                else:
                    self.cell_type_enc = cell_type_enc
                self.cell_type = self.cell_type_enc.transform(self.cell_type)
                self.cell_type = self.cell_type.astype(np.float32)
                if normalize_cell_type_enc:
                    self.cell_type /= len(self.cell_type_enc.categories_)
            elif use_default_cell_type_enc:
                self.cell_type_enc = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    encoded_missing_value=-1
                )
                tmp = self.cell_type_enc.fit_transform(self.cell_type)
                self.cell_type = tmp.astype(np.float32).flatten()
        # Handle PROTAC activity information
        if not use_for_ssl:
            if task == 'predict_active_inactive':
                num_nan = len(self.dataframe[self.dataframe['active'].isna()])
                if num_nan > 0:
                    print('-' * 80)
                    print(f'Number of NaNs in active column: {num_nan}')
                    print('-' * 80)
                    raise ValueError('NaNs found in active column')
                    # self.dataframe = dataframe.dropna(subset=['active'])
                self.dataframe['active'] = self.dataframe['active'].replace({True: 1, False: 0})
                # Get the concentration and degradation values
                self.active = self.dataframe['active'].to_numpy().astype(np.float32).reshape(-1, 1)
            else:
                # TODO: Scaling the concentrations and degradations???
                if scale_concentration:
                    self.pDC50 = (self.dataframe['pDC50'] * 0.1).astype(np.float32)
                else:
                    self.pDC50 = self.dataframe['pDC50'].astype(np.float32)
                self.Dmax = (self.dataframe['Dmax']).astype(np.float32)

    @staticmethod
    def load(pt_file):
        # TODO: Work in progress
        return torch.load(pt_file)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles = self.smiles.iloc[idx]
        if self.use_for_ssl:
            elem = {}
            if self.smiles_tokenizer:
                smiles_tokenized = self.smiles_tokenized[idx]
                elem['input_ids'] = smiles_tokenized['input_ids']
                elem['attention_mask'] = smiles_tokenized['attention_mask']
                elem['labels'] = smiles_tokenized['input_ids'].copy()
            else:
                elem['smiles'] = smiles
            return elem
        elem = {}
        if self.include_poi_seq:
            elem['poi_seq'] = self.poi_seq[idx]
            if self.poi_seq_enc is not None and not self.preencode_poi_seq:
                poi_seq = self.poi_seq_enc.transform([self.poi_seq[idx]])
                poi_seq = poi_seq.toarray().flatten().astype(np.float32)
                elem['poi_seq'] = poi_seq
            elif self.tokenize_poi_seq is not None:
                poi_seq = self.tokenize_poi_seq(self.poi_seq[idx])
                elem['poi_seq'] = poi_seq
        if self.include_poi_gene:
            elem['poi_gene_id'] = self.gene[idx]
        if self.include_e3_ligase:
            elem['e3_ligase'] = self.e3_ligase[idx]
        if self.include_cell_type:
            elem['cell_type'] = self.cell_type[idx]
        if self.task == 'predict_active_inactive':
            elem['labels'] = self.active[idx]
        elif self.task == 'predict_pDC50_and_Dmax':
            Dmax = self.Dmax.iloc[idx]
            pDC50 = self.pDC50.iloc[idx]
            elem['labels'] = np.array([Dmax, pDC50])
        else:
            raise ValueError(f'Task "{self.task}" not recognized. Available: "predict_active_inactive" \| "predict_pDC50_and_Dmax"')
        if self.include_smiles_as_graphs or self.precompute_smiles_as_graphs:
            if self.precompute_smiles_as_graphs:
                elem['smiles_graph'] = self.graph_smiles[idx]
            else:
                elem['smiles_graph'] = from_smiles(smiles)
        if self.smiles_tokenizer:
            elem['smiles_tokenized'] = self.smiles_tokenized[idx]
        if self.include_smiles_as_str:
            elem['smiles'] = smiles
        if self.use_morgan_fp:
            if self.precompute_fingerprints:
                fp = self.morgan_fp[idx].copy()
            else:
                fp = get_fingerprint(smiles, n_bits=self.morgan_bits).astype(np.float32)
            elem['morgan_fp'] = fp
        if self.use_maccs_fp:
            if self.precompute_fingerprints:
                fp = self.maccs_fp[idx].copy()
            else:
                fp = get_fingerprint(smiles, fp_type='maccs').astype(np.float32)
            elem['maccs_fp'] = fp
        if self.use_path_fp:
            if self.precompute_fingerprints:
                fp = self.path_fp[idx].copy()
            else:
                fp = get_fingerprint(smiles, n_bits=self.path_bits,
                                     fp_type='path',
                                     min_path=self.fp_min_path,
                                     max_path=self.fp_max_path).astype(np.float32)
            elem['path_fp'] = fp
        if self.transform is not None:
            elem = self.transform(elem)
        return elem

    def get_fingerprint(self, fp_type: Literal['morgan_fp', 'maccs_fp', 'path_fp'] = 'morgan_fp'):
        # TODO: Add the proper checks if fingerprints are used
        if self.precompute_fingerprints:
            if fp_type == 'morgan_fp':
                return self.morgan_fp
            elif fp_type == 'maccs_fp':
                return self.maccs_fp
            elif fp_type == 'path_fp':
                return self.path_fp
            else:
                raise ValueError(f'Fingerprint type "{fp_type}" not recognized. Available: "morgan_fp" \| "maccs_fp" \| "path_fp"')
        else:
            smiles = self.smiles
            if fp_type == 'morgan_fp':
                return np.array([get_fingerprint(s, n_bits=self.morgan_bits).astype(np.float32) for s in smiles])
            elif fp_type == 'maccs_fp':
                return np.array([get_fingerprint(s, fp_type='maccs_fp').astype(np.float32) for s in smiles])
            elif fp_type == 'path_fp':
                return np.array([get_fingerprint(s, n_bits=self.path_bits, fp_type='path_fp', min_path=self.fp_min_path, max_path=self.fp_max_path).astype(np.float32) for s in smiles])
            else:
                raise ValueError(f'Fingerprint type "{fp_type}" not recognized. Available: "morgan_fp" \| "maccs_fp" \| "path_fp"')
    
    def get_poi_seq_emb_size(self):
        if self.include_poi_seq:
            return len(self.poi_seq_enc.get_feature_names_out())
        else:
            return 0
    
    def __str__(self) -> str:
        return f'ProtacDataset for {self.task} task with {len(self)} samples.'