from __future__ import annotations
from typing import Any, Dict, Set, Optional, Union, Type, Callable
from os import PathLike

from data.protac_dataloader import PROTACDataModule
from models.wrapper_model import WrapperModel, ProtacModel # noqa: F401


from optuna.study import Study, create_study
from optuna.pruners import BasePruner

from pytorch_lightning.cli import (
    LightningCLI,
    LightningArgumentParser,
    ArgsType,
    SaveConfigCallback
)

from pytorch_lightning import LightningDataModule, LightningModule, Trainer

import importlib
from jsonargparse import (
    ActionConfigFile,
    ArgumentParser,
    class_from_function,
    Namespace,
    register_unresolvable_import_paths,
    set_config_read_mode,
    CLI,
    lazy_instance,
)


def getclass(class_string):
    """Given a string, return the class type it represents."""
    module_name, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def create_instance(class_string, args_dict, instanciate_nested: bool = False, lazy_init: bool = False):
    if isinstance(args_dict, Namespace) or isinstance(args_dict, dict) or isinstance(args_dict, list):
        tmp = args_dict
        if isinstance(args_dict, Namespace):
            tmp = args_dict.as_dict().items()
        if isinstance(args_dict, dict):
            tmp = args_dict.items()
        for arg_key, arg_value in tmp:
            # print('-' * 80)
            # print(f'arg_key: {arg_key}, arg_value: {arg_value}')
            # print('-' * 80)
            if isinstance(arg_value, dict):
                if 'class_path' in arg_value.keys():
                    # TODO: Do not create nested instances... current limitation
                    if instanciate_nested:
                        if 'init_args' in arg_value.keys():
                            args_dict[arg_key] = create_instance(arg_value['class_path'], arg_value['init_args'])
                        else:
                            args_dict[arg_key] = getclass(arg_value['class_path'])
                    else:
                        args_dict[arg_key] = getclass(arg_value['class_path'])
    class_ = getclass(class_string)
    if lazy_init:
        return lazy_instance(class_, **args_dict)
    else:
        return class_(**args_dict)


class TuneLightningCLI(LightningCLI):
    
    def __init__(
        self,
        model_class: Optional[Union[Type[LightningModule], Callable[..., LightningModule]]] = None,
        datamodule_class: Optional[Union[Type[LightningDataModule], Callable[..., LightningDataModule]]] = None,
        save_config_callback: Optional[Type[SaveConfigCallback]] = SaveConfigCallback,
        save_config_kwargs: Optional[Dict[str, Any]] = None,
        trainer_class: Union[Type[Trainer], Callable[..., Trainer]] = Trainer,
        trainer_defaults: Optional[Dict[str, Any]] = None,
        seed_everything_default: Union[bool, int] = True,
        parser_kwargs: Optional[Union[Dict[str, Any], Dict[str, Dict[str, Any]]]] = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
    ) -> None:
        """Receives as input pytorch-lightning classes (or callables which return pytorch-lightning classes), which
        are called / instantiated using a parsed configuration file and / or command line args.

        Parsing of configuration from environment variables can be enabled by setting ``parser_kwargs={"default_env":
        True}``. A full configuration yaml would be parsed from ``PL_CONFIG`` if set. Individual settings are so parsed
        from variables named for example ``PL_TRAINER__MAX_EPOCHS``.

        For more info, read :ref:`the CLI docs <lightning-cli>`.

        Args:
            model_class: An optional :class:`~lightning.pytorch.core.module.LightningModule` class to train on or a
                callable which returns a :class:`~lightning.pytorch.core.module.LightningModule` instance when
                called. If ``None``, you can pass a registered model with ``--model=MyModel``.
            datamodule_class: An optional :class:`~lightning.pytorch.core.datamodule.LightningDataModule` class or a
                callable which returns a :class:`~lightning.pytorch.core.datamodule.LightningDataModule` instance when
                called. If ``None``, you can pass a registered datamodule with ``--data=MyDataModule``.
            save_config_callback: A callback class to save the config.
            save_config_kwargs: Parameters that will be used to instantiate the save_config_callback.
            trainer_class: An optional subclass of the :class:`~lightning.pytorch.trainer.trainer.Trainer` class or a
                callable which returns a :class:`~lightning.pytorch.trainer.trainer.Trainer` instance when called.
            trainer_defaults: Set to override Trainer defaults or add persistent callbacks. The callbacks added through
                this argument will not be configurable from a configuration file and will always be present for
                this particular CLI. Alternatively, configurable callbacks can be added as explained in
                :ref:`the CLI docs <lightning-cli>`.
            seed_everything_default: Number for the :func:`~lightning.fabric.utilities.seed.seed_everything`
                seed value. Set to True to automatically choose a seed value.
                Setting it to False will avoid calling ``seed_everything``.
            parser_kwargs: Additional arguments to instantiate each ``LightningArgumentParser``.
            subclass_mode_model: Whether model can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            subclass_mode_data: Whether datamodule can be any `subclass
                <https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes>`_
                of the given class.
            args: Arguments to parse. If ``None`` the arguments are taken from ``sys.argv``. Command line style
                arguments can be given in a ``list``. Alternatively, structured config options can be given in a
                ``dict`` or ``jsonargparse.Namespace``.
            run: Whether subcommands should be added to run a :class:`~lightning.pytorch.trainer.trainer.Trainer`
                method. If set to ``False``, the trainer and model classes will be instantiated only.
        """
        self.save_config_callback = save_config_callback
        self.save_config_kwargs = save_config_kwargs or {}
        self.trainer_class = trainer_class
        self.trainer_defaults = trainer_defaults or {}
        self.seed_everything_default = seed_everything_default
        self.parser_kwargs = parser_kwargs or {}  # type: ignore[var-annotated]  # github.com/python/mypy/issues/6463
        self.auto_configure_optimizers = auto_configure_optimizers

        self.model_class = model_class
        # used to differentiate between the original value and the processed value
        self._model_class = model_class or LightningModule
        self.subclass_mode_model = (model_class is None) or subclass_mode_model

        self.datamodule_class = datamodule_class
        # used to differentiate between the original value and the processed value
        self._datamodule_class = datamodule_class or LightningDataModule
        self.subclass_mode_data = (datamodule_class is None) or subclass_mode_data

        main_kwargs, subparser_kwargs = self._setup_parser_kwargs(self.parser_kwargs)
        self.setup_parser(run, main_kwargs, subparser_kwargs)
        self.parse_arguments(self.parser, args)

        self.subcommand = self.config["subcommand"] if run else None

        self._set_seed()

        # self.before_instantiate_classes()
        # self.instantiate_classes()
        # if self.subcommand is not None:
        #     self._run_subcommand(self.subcommand)


    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """Implement to add extra arguments to the parser or link arguments.

        Args:
            parser: The parser object to which arguments can be added
        """
        parser.add_argument('--optuna', type=dict, help='Optuna hyperparameters', required=True)
        parser.add_argument('--best-config-filename', type=PathLike, help='Best config file path', default='best_config.yaml')
        # parser.add_subclass_arguments(BasePruner, '--optuna.study.pruner', instantiate=False)


    def create_study(self) -> Study:
        config = {}
        if 'study' in self.config.optuna.keys():
            for k, v in self.config.optuna['study'].items():
                if isinstance(v, dict):
                    if 'class_path' in v.keys():
                        config[k] = create_instance(v['class_path'], v['init_args'])
                else:
                    config[k] = v
        return create_study(**config)