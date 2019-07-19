import time
from pathlib import Path

import torch
import torchtext

from config_parser import ConfigParser
from torchtext import data


class DatasetIterator(object):
    def __init__(self, pos_iter, neg_iter):
        self.pos_iter = pos_iter
        self.neg_iter = neg_iter

    def __iter__(self):
        for batch_pos, batch_neg in zip(iter(self.pos_iter), iter(self.neg_iter)):
            if batch_pos.text.size(0) == batch_neg.text.size(0):
                yield batch_pos.text, batch_neg.text


def load_dataset(config: ConfigParser, device):

    data_config = config["data"]
    root = Path(data_config["data_root"])
    TEXT = data.Field(batch_first=True, eos_token="<eos>")

    def dataset_fn(name):
        return data.TabularDataset(
            path=(root / name).as_posix(), format="tsv", fields=[("text", TEXT)]
        )

    train_pos_set, train_neg_set = map(
        dataset_fn, [data_config["train_pos"], data_config["train_neg"]]
    )
    test_pos_set, test_neg_set = map(
        dataset_fn, [data_config["test_pos"], data_config["test_neg"]]
    )

    TEXT.build_vocab(train_pos_set, train_neg_set, min_freq=data_config["min_freq"])

    if data_config["load_pretrained_embed"]:
        start = time.time()

        vectors = torchtext.vocab.GloVe(
            "6B", dim=data_config["embed_size"], cache=data_config["pretrained_embed_path"]
        )
        TEXT.vocab.set_vectors(vectors.stoi, vectors.vectors, vectors.dim)
        print("vectors", TEXT.vocab.vectors.size())

        print("load embedding took {:.2f} s.".format(time.time() - start))

    vocab = TEXT.vocab

    def dataiter_fn(dataset, train):
        return data.BucketIterator(
            dataset=dataset,
            batch_size=data_config["batch_size"],
            shuffle=train,
            repeat=train,
            sort_key=lambda x: len(x.text),
            sort_within_batch=False,
            device=device,
        )

    train_pos_iter, train_neg_iter = map(
        lambda x: dataiter_fn(x, True), [train_pos_set, train_neg_set]
    )
    test_pos_iter, test_neg_iter = map(
        lambda x: dataiter_fn(x, False), [test_pos_set, test_neg_set]
    )

    train_iters = DatasetIterator(train_pos_iter, train_neg_iter)
    test_iters = DatasetIterator(test_pos_iter, test_neg_iter)

    return train_iters, test_iters, vocab
