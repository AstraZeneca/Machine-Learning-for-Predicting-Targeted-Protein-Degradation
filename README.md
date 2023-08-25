![Maturity level-0](https://img.shields.io/badge/Maturity%20Level-ML--0-red)

# ML for Predicting Targeted Protein Degradation

This repository contains the code developed within the master thesis project: _"Machine Learning for Predicting Targetd Protain Degradation"_. A brief overview of the project and the thesis report can be read at this [repository](https://github.com/ribesstefano/ml-for-protacs).

## Code Overview

The `ProtacModel` class, defined in the `src/models/wrapper_model.py` file, is a subclass of the PyTorch Lightning `LightningModule` class. It is a wrapper class that makes predictions on PROTAC data.
The `ProtacModel` class takes in several encoders and a head module, which are used to encode the input data and make predictions. The forward method defines the forward pass of the model, which takes in the input data and passes it through the encoders and head module to produce the output.

The model roughly follows the following architecture:

![image](https://github.com/ribesstefano/ml-for-protacs/assets/17163014/5adedfd7-9e5e-419b-bc8f-5334c9a41c4f)

The `ProtacModel` class is used in the `src/main.py` for training and testing the model through a `LightningCLI` module. The `LightningCLI` module is a command-line interface that allows to train and test the model using a YAML configuration file. The configuration file contains the hyperparameters and dataset arguments that are used to define and automatically instantiate the model and the dataset before performing training and testing.

The `src/models/wrapper_model.py` file contains instead the `WrapperModel` class, which is a wrapper around the PyTorch Lightning `ProtacModel` class. The purpose of this class is to provide a simple and flexible way to train and test various models using the same interface.
The `WrapperModel` class can be used as a stand-alone class to train and test the model without the `LightningCLI` module. This is useful when one wants to train and test the model using a Jupyter Notebook or a Python script.

### Encoders

In the `src/models` folder different encoders are defined. The encoders are used to encode the diverse input data about PROTACs before passing it to the head module. The encoders are all defined as subclasses of the `nn.Module` class and are used in the `ProtacModel` class to encode the input data.

### PROTACDataset and PROTACDataLoader

The `PROTACDataset` class represents a dataset used for training and testing the PROTAC models. It takes in a Pandas DataFrame containing the PROTAC data and various arguments to preprocess the data.

The `PROTACDataModule` class is defined in the `src/data/protac_dataloader.py` file and is a subclass of the PyTorch Lightning `LightningDataModule` class. It provides a convenient and customizable way to load and preprocess the PROTAC data and create PyTorch dataloaders for training, validation, and testing.

## Data Curation

The data curation process is detailed and carried on in the notebooks `notebooks/protac_db_data_curation.ipynb` and `notebooks/protac_pedia_data_curation.ipynb`. They are based on the [PROTAC-DB](http://cadd.zju.edu.cn/protacdb/about) and [PROTAC-Pedia](https://protacpedia.weizmann.ac.il/ptcb/main) datasets. The directory `data` already contains curated versions of the aforementioned datasets. In order to perform data curation, one shall download the raw datasets from the respective sources and run the respective notebooks.

## Quick Start

1. Install the required dependencies by running `conda env create -f environment.yml` in your terminal.

2. To train an MLP model, run the following command:

```bash
python main.py fit \
    --trainer="{'max_epochs': 10, 'accelerator': 'gpu', 'precision': 16}" \
    --model.smiles_encoder=src.models.smiles_encoder.mlp.rdkit_fp_model.RDKitFingerprintEncoder \
    --data.protac_dataset_args="{'use_morgan_fp': True, 'morgan_bits': 1024, 'precompute_fingerprints': True, 'poi_vectorizer': 'models/poi_encoder.joblib', 'e3_ligase_enc': 'models/e3_ligase_encoder.joblib', 'cell_type_enc': 'models/cell_type_encoder.joblib'}"
```
This will train an MLP model with the specified hyperparameters and dataset arguments.

3. To test the MLP model, run the following command:
    
```bash
python main.py test --ckpt_path=.\lightning_logs\version_7\checkpoints\epoch=9-step=930.ckpt -c .\lightning_logs\version_7\config.yaml
```
This will test the MLP model using the specified checkpoint path and configuration file.

4. To train a GNN model, for instance, run the following command instead:

```bash
python main.py fit --trainer="{'max_epochs': 10, 'accelerator': 'gpu', 'precision': 16}" \
    --model.smiles_encoder=src.models.smiles_encoder.gnn.torch_geom_architectures.GnnSubModel \
    --model.use_smiles_only=False \
    --data.protac_dataset_args="{'include_smiles_as_graphs': True, 'precompute_smiles_as_graphs': True, 'poi_vectorizer': 'models/poi_encoder.joblib', 'e3_ligase_enc': 'models/e3_ligase_encoder.joblib', 'cell_type_enc': 'models/cell_type_encoder.joblib'}" \
    --model.cell_type_encoder=src.models.cell_type_encoder.cell_type_ordinal_encoder.CellTypeEncoder \
    --model.e3_ligase_encoder=src.models.e3_encoder.e3_ordinal_encoder.E3LigaseEncoder \
    --model.poi_seq_encoder=src.models.poi_encoder.poi_count_vectorizer.POISequenceEncoder \
    --model.poi_seq_encoder_args=" {'poi_seq_encoder_filepath': 'models/poi_encoder.joblib'}"
```

This will train a GNN model with the specified hyperparameters and dataset arguments.

That's it! You can modify the hyperparameters and dataset arguments as needed to train and test different models. Additionally, there are some TODOs in the code that you can work on to improve the functionality of the code.

## Hyperparameters Search with Optuna

The `src/tune.py` file provides a convenient way to perform hyperparameter tuning using the Optuna library. It defines several functions that create PyTorch Lightning objects with the specified hyperparameters, and an objective function that is optimized by Optuna. By running it with the appropriate configuration files, one can perform hyperparameter tuning on the `ProtacModel` and obtain the best configuration as a YAML file.

The `src/config_optuna.yml` file contains an example configuration for the Optuna hyperparameter tuning. It defines the search space for the hyperparameters and the number of trials to run. The `src/config_default.yml` file contains the default configuration for the model and dataset arguments for the `ProtacModel` and `ProtacDataset`. You can modify the configuration files as needed to perform hyperparameter tuning on your models.

The `objective` function in `src/hyperparameter_tuning/optuna_utils.py` is the objective function that is optimized by Optuna during hyperparameter tuning. It returns the validation loss of the model trained with the suggested hyperparameters. Note that the CLI script will only change the arguments to be passed to `nn.Module` classes in the model before instantiating them. For more advanced hyperparameters configurations, one shall write a custom objective function. Please refer to the file `notebooks/machine_learning.ipynb` for some tailored examples.

Here's a brief overview of what the `objective` function does:

* Ovefits a trial model with the suggested hyperparameters on a minibatch.
* If the training accuracy on the minibatch is less than 0.95, returns a score of 0.0.
* Otherwise, it fits a trial model with the suggested hyperparameters.
* Finally, returns the validation loss of the model.

Example of usage:

```bash
python .\src\tune.py --config .\config_default.yml --config .\config_optuna.yml
```