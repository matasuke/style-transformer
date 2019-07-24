import argparse
from pathlib import Path

import torch

from data import load_dataset
# from models import Discriminator, StyleTransformer
import models as module_arch
import models.optim as module_optim
from train import auto_eval, train
from utils.evaluator import Evaluator

from config_parser import ConfigParser


def main(config: ConfigParser):
    device = torch.device("cuda:0" if config["n_gpu"] > 0 else "cpu")

    train_iters, test_iters, vocab = load_dataset(config, device)
    print("Vocab size:", len(vocab))

    print("prepare Evaluator...")
    evaluator_path = config["evaluator"]["save_path"]
    if evaluator_path is None or not Path(evaluator_path).exists():
        evaluator = Evaluator.create(**config["evaluator"]["args"])
        config["evaluator"]["save_path"] = (config.save_dir / "evaluator").as_posix()
        evaluator.save(config["evaluator"]["save_path"])
    else:
        evaluator = Evaluator.load(evaluator_path)
    config.save()

    model_F = config.initialize("arch_generator", module_arch, vocab=vocab).to(device)
    model_D = config.initialize("arch_discriminator", module_arch, vocab=vocab).to(device)

    opt_F = config.initialize("generator_optimizer", module_optim)
    opt_D = config.initialize("discriminator_optimizer", module_optim)
    opt_F.set_parameters(model_F.parameters())
    opt_D.set_parameters(model_D.parameters())

    train(
        config=config,
        vocab=vocab,
        model_F=model_F,
        model_D=model_D,
        opt_F=opt_F,
        opt_D=opt_D,
        train_iters=train_iters,
        test_iters=test_iters,
        evaluator=evaluator
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style transformer")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        nargs="+",
        help="indices of GPUs to enable (default: all)",
    )
    config = ConfigParser.parse_args(parser)

    main(config)
