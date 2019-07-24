import argparse
import dataclasses
import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from functools import reduce
from operator import getitem
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from logger.logger import setup_logging

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


JSON_FORMAT = ".json"
YML_FORMAT = ".yml"
DEFAULT_CONFIG_FILE_NAME = "config.json"
EXCLUTED_ARGS = ["device", "resume", "config"]


class ConfigParser:
    """
    Parser to load json or yaml based configuration files for pytorch
    """

    def __init__(
        self,
        args: Dict,
        options: Optional[List["CustomArgs"]] = None,
        timestamp: bool = True,
    ):
        if "device" in args:
            if args["device"] is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(args["device"])
                self.device = [int(device) for device in args["device"]]

        if args["resume"]:
            self.resume: Union[Path, None] = Path(args["resume"])
            self.cfg_fname = self.resume.parent / DEFAULT_CONFIG_FILE_NAME
        elif args["config"]:
            self.resume: Union[Path, None] = None
            self.cfg_fname = Path(args["config"])
        else:
            msg_no_cfg = (
                "Configuration file need to be specified.\n"
                "add -c config.json for example."
            )
            raise ValueError(msg_no_cfg)

        self._config = self.load(self.cfg_fname)
        self._apply_options(args, options)

        # set save_dir where trained model and log will be saved
        save_dir = Path(self._config["trainer"]["save_dir"])
        expr_name = self._config["name"]
        if self.resume:
            time_stamp = self.resume.parent.stem if timestamp else ""
            self._save_dir = save_dir / "models" / expr_name / time_stamp
            self._log_dir = save_dir / "log" / expr_name / time_stamp
            self._test_dir = save_dir / "test" / expr_name / time_stamp

        else:
            time_stamp = datetime.now().strftime(r"%m%d_%H%M%S") if timestamp else ""
            self._save_dir = save_dir / "models" / expr_name / time_stamp
            self._log_dir = save_dir / "log" / expr_name / time_stamp
            self._test_dir = save_dir / "test" / expr_name / time_stamp

            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._log_dir.mkdir(parents=True, exist_ok=True)
            self._test_dir.mkdir(parents=True, exist_ok=True)

            self.save(self._save_dir / DEFAULT_CONFIG_FILE_NAME)

        # configurations for logging module
        setup_logging(self.log_dir)
        self._log_level = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def __str__(self):
        text = "PARAMETERS\n" + "-" * 20 + "\n"
        for (key, value) in self.__dict__.items():
            text += f"{key}:\t{value}" + "\n"
        return text

    def __getitem__(self, name: str):
        return self._config[name]

    def initialize(self, name: str, module: str, *args: List, **kwargs: Dict):
        """
        finds a function handle with the name given as 'type' in config.
        and returns instance initialized with corresponding given keyword args nad 'args'
        """
        module_cfg = self[name]
        return getattr(module, module_cfg["type"])(
            *args, **kwargs, **module_cfg["args"]
        )

    @classmethod
    def parse_args(
        cls,
        args: "argparse.ArgumentParser",
        options: Optional[List["CustomArgs"]] = None,
        timestamp: bool = True,
    ) -> "ConfigParser":
        """
        parse arguments from dict or argumentparser

        :param args: parameters to save.
        """
        assert isinstance(args, argparse.ArgumentParser)
        if options is not None:
            for opt in options:
                if opt.choices is not None:
                    args.add_argument(
                        *opt.flags, default=None, type=opt.type, choices=opt.choices
                    )
                else:
                    args.add_argument(*opt.flags, default=None, type=opt.type)
        args = args.parse_args()

        arguments = {}
        for (key, value) in vars(args).items():
            arguments[key] = value

        return cls(arguments, options, timestamp)

    def get_param(self, key: str, default: Optional[Any] = None) -> Any:
        """
        get value from key. this is wrapper for getting parameters
        """
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def add_param(self, key: str, value: Any, update: bool = False) -> None:
        """
        add argument if key is not exists.
        it updates value when update=True
        """
        if hasattr(self, key):
            if update and getattr(self, key) != value:
                print(f"Update parameter {key}: {getattr(self, key)} -> {value}")
                setattr(self, key, value)
        else:
            setattr(self, key, value)

    def update_param(self, key: str, value: Any):
        """
        update argument if key is exists
        """
        self.add_param(key, value, update=True)

    def del_param(self, key: str) -> None:
        """
        delete argument if key is exists.
        """
        if hasattr(self, key):
            print(f"Delate parameter {key}")
            delattr(self, key)

    def get_config(self, target: List[str], default: Optional[Any] = None) -> Any:
        """
        get value from key. this is wrapper for __getitem__
        """
        return self.__get_by_path(self._config, target, default)

    def add_config(self, target: List[str], value: Any, update: bool = False) -> None:
        """
        add config parameters if key is not exists.
        it updates value when update=True
        """
        old_value = self.get_config(target)
        if old_value is not None:
            if update and old_value != value:
                print(
                    f"Update config parameter {'-> '.join(target)}: {old_value} -> {value}"
                )
                self.__get_by_path(self._config, target[:-1])[target[-1]] = value
        else:
            self.__get_by_path(self._config, target[:-1])[target[-1]] = value

    def update_config(self, target: List[str], value: Any) -> None:
        """
        update config parameters if key is exists
        """
        self.add_config(target, value, update=True)

    def del_config(self, target: List[str]) -> None:
        """
        delete config parameters if key is exists.
        """
        if self.get_config(target) is not None:
            print(f"Delate parameter {'-'.join(target)[-1]}")
            delattr(self.__get_by_path(self._config, target[:-1]), target[-1])

    def _apply_options(
        self, args: Dict, options: Optional[List["Customargs"]] = None
    ) -> None:
        """
        apply Customargs to self._config
        loaded config file is OVERWRITTEN by args.
        """
        if options:
            for opt in options:
                value = args.get(self.__get_opt_name(opt.flags))
                if value is not None:
                    self.update_config(opt.target, value)

    @staticmethod
    def __get_opt_name(flags: List[str]):
        """
        get option name from CustomArgs's flag
        """
        for flag in flags:
            if flag.startswith("--"):
                return flag.replace("--", "")
        return flags[0].replace("--", "")

    def __set_by_path(
        self, tree: Union[Dict, "OrderedDict"], target: List[str], value: Any
    ) -> None:
        """
        update value on the target position
        """
        self.__get_by_path(self._config, target[:-1])[target[-1]] = value

    @staticmethod
    def __get_by_path(
        tree: Union[Dict, "OrderedDict"], target: List[str], default: bool = None
    ):
        """
        get config parameter based on target

        :param tree: dict or OrderedDict to fetch target value
        :param target: list of target.
        :param default: default value when specified target does not exists.
        """
        try:
            value = reduce(getitem, target, tree)
        except KeyError:
            value = default

        return value

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> Dict:
        """
        parse config file of YML or JSON format.

        :param config_path: path to config file.
        :return dict of configurations.
        """
        if isinstance(config_path, str):
            config_path = Path(config_path)
        assert config_path.exists()

        with config_path.open() as handle:
            if config_path.suffix == JSON_FORMAT:
                config = json.load(handle, object_hook=OrderedDict)

            elif config_path.suffix == YML_FORMAT:
                yaml.add_constructor(  # For loading key and values with OrderedDict style.
                    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                    lambda loader, node: OrderedDict(loader.construct_pairs(node)),
                )
                config = yaml.load(handle, Loader=Loader)

            else:
                raise Exception("config_loader: config format is unknown.")

        return config

    def save(self, save_path: Union[None, str, Path] = None) -> None:
        """
        save config parameters to save_path.

        :param save_path: path to save config parameters.
        """
        if save_path is None:
            save_path = self._save_dir / DEFAULT_CONFIG_FILE_NAME
        if isinstance(save_path, str):
            save_path = Path(save_path)
        invalid_format_msg = "config_loader: config format is unknown."
        assert save_path.suffix in [JSON_FORMAT, YML_FORMAT], invalid_format_msg

        if not save_path.parent.exists():
            save_path.parent.mkdir(parents=True)

        with save_path.open("w") as handle:
            if save_path.suffix == JSON_FORMAT:
                json.dump(self._config, handle, indent=4, sort_keys=False)
            elif save_path.suffix == YML_FORMAT:
                handle.write(yaml.dump(self._config, default_flow_style=False))

    def get_logger(self, name: str, verbosity: int = 2):
        """
        get logger

        :param name:
        :param verbosity:
        :return: logger
        """
        msg_verbosity = f"verbosity option {verbosity} is invalid"
        assert verbosity in self._log_level, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self._log_level[verbosity])
        return logger

    @property
    def config(self):
        "get registerd configurations"
        return self._config

    @property
    def save_dir(self):
        "get the path to save directory."
        return self._save_dir

    @property
    def log_dir(self):
        "get the path to logging directory."
        return self._log_dir

    @property
    def test_dir(self):
        "get the path to test directory."
        return self._test_dir


@dataclasses.dataclass
class CustomArgs:
    flags: List[str]
    type: str
    target: List[str]
    choices: Optional[List[str]] = None

    def __repr__(self) -> str:
        return (
            f"<CustomArgs flags={self.flags}, type={self.type}, target={self.target}>"
        )

    def __str__(self) -> str:
        text = "CUSTOMARGS\n" + "-" * 20 + "\n"
        for (key, value) in self.__dict__.items():
            text += f"{key}:\t{value}" + "\n"
        return text
