# Tips and Notes

## TODOs

* The directory tree might be simplified, as it is intended to be extended in the future.

## CLI

* For any sub-module to be used in the CLI, it must have a docstring!
* To disable all logging when validating and testing, append the following to the command: `--trainer.logger false --trainer.enable_checkpointing false`
* The official Optuna pruning callback for pytorch ligthning has some issues. If one desires to add it, use instead the following as `class_path`: `hyperparameter_tuning.optuna_utils.CustomPyTorchLightningPruningCallback`

> Parsers make a best effort to determine the correct names and types that the parser should accept. However, there can be cases not yet supported or cases for which it would be impossible to support. To somewhat overcome these limitations, there is a special key `dict_kwargs` that can be used to provide arguments that will not be validated during parsing, but will be used for class instantiation.
> Multiple config files can be provided, and they will be parsed sequentially.
> `$ python main.py fit --trainer trainer.yaml --model model.yaml --data data.yaml [...]`