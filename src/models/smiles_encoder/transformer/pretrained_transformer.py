import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
import pytorch_lightning as pl

def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output['last_hidden_state']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask


class TransformerSubModel(pl.LightningModule):

    def __init__(self, checkpoint_path: str = 'seyonec/ChemBERTa-zinc-base-v1'):
        super().__init__()
        # Save the arguments passed to init
        self.save_hyperparameters()
        self.__dict__.update(locals()) # Add arguments as attributes
        # ChemBERT for SMILES
        self.config = AutoConfig.from_pretrained(checkpoint_path,
                                                 output_hidden_states=True,
                                                 num_labels=1)
        self.chembert = AutoModelForSequenceClassification.from_pretrained(
            checkpoint_path,
            config=self.config
        ).roberta

    def forward(self, x_in):
        # Run ChemBert over the tokenized SMILES
        input_ids = x_in['smiles_tokenized']['input_ids'].squeeze(dim=1)
        attention_mask = x_in['smiles_tokenized']['attention_mask'].squeeze(dim=1)
        smiles_embedding = self.chembert(input_ids, attention_mask)
        # NOTE: Due to multi-head attention, the output of the Transformer is a
        # sequence of hidden states, one for each input token. The following
        # takes the mean of all token embeddings to get a single embedding.
        smiles_embedding = mean_pooling(smiles_embedding, attention_mask)
        return smiles_embedding
    
    def get_embedding_size(self):
        return self.config.to_dict()['hidden_size']