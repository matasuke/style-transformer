import hashlib
import math
import pickle
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import fasttext
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score

import kenlm
from utils import convert_to_pathlib, ensure_dir, save_text_list

DEFAULT_LM_CONFIG = {"n_order": 5}
DEFAULT_CLASSIFIER_CONFIG = {
    "epoch": 25,
    "lr": 1.0,
    "n_order": 2,
    "verbose": 2,
    "min_count": 1,
    "label_prefix": "__style__",
    "style_labels": ["positive", "negative"],
}


class Evaluator:
    """
    Evaluator for style control, context preservation, and fluency.
    """

    __all__ = ["Ealuator"]

    __slot__ = [
        "lm_path",
        "classifier_path",
        "reference_paths",
        "language_model",
        "classifier",
        "_reference_dict",
        "_classifier_config",
        "_lm_config",
        "label_prefix",
        "style_labels",
    ]

    MODEL_PREFIX = "evaluator"
    CLASSIFIER_FILE_NAME = MODEL_PREFIX + ".classifier.model"
    LANGUAGE_MODEL_FILE_NAME = MODEL_PREFIX + ".lm.model"
    CONFIG_FILE_NAME = MODEL_PREFIX + ".config.pkl"

    # save text file used when training is saved to temporal directory
    TMO_DIR_PREFIX = Path(".tmp/evaluator/")
    TMP_CASSIFIER_FILE_NAME = "classification_text.txt"
    TMP_LM_FILE_NAME = "language_model.txt"

    def __init__(
        self,
        lm_path: Union[str, Path],
        classifier_path: Union[str, Path],
        reference_paths: Union[None, List[str], List[Path]] = None,
        lm_config: Dict[str, Any] = DEFAULT_LM_CONFIG,
        classifier_config: Dict[str, Any] = DEFAULT_CLASSIFIER_CONFIG,
    ) -> None:
        """
        Calculate Classification accucacy, Bleu score, perplexity,
        which are for measuring controlness of style transfer,
        how much contexual information is preserved, and how much fluent
        the transfered sentences are for each.

        :param lm_path: path to pre-trained language model's
            binary file, which is trained by kenLM.
        :param classifier_path: path to pre-trained language model's
            binary file, which is trained by fastText.
        :parma reference_paths: paths to reference text file coresponding to each styles,
        which is used for calculating bleu score.
        :param lm_config: language model configuration dictionary.
        :param classifier_config: classifier configuration dictionalry.

        """
        self.lm_path = lm_path
        self.classifier_path = classifier_path
        self.reference_paths = reference_paths
        assert reference_paths is None or \
            len(reference_paths) == len(classifier_config["style_labels"])

        # check file existance and convert them to pathlib format
        language_model_path = convert_to_pathlib(lm_path)
        classification_model_path = convert_to_pathlib(classifier_path)

        # load all of them above with required format.
        self.language_model = kenlm.Model(language_model_path.as_posix())
        self.classifier = fasttext.load_model(classification_model_path.as_posix())

        # concat label_prefix with style_label
        classifier_labels = self.classifier.get_labels()
        self.label_prefix = classifier_config["label_prefix"]
        self.style_labels = [
            self.label_prefix + style_label
            for style_label in classifier_config["style_labels"]
        ]

        # set reference sentences
        if reference_paths:
            assert len(reference_paths) == len(self.style_labels)
            self._reference_dict = {}

            for path, style_label in zip(reference_paths, self.style_labels):
                path = convert_to_pathlib(path)
                self._reference_dict[style_label] = []
                with path.open() as f:
                    for sentence in f.readlines():
                        self._reference_dict[style_label].append(sentence.strip())
        else:
            self._reference_dict = None

        # set configuration
        self._classifier_config = classifier_config
        self._lm_config = lm_config

        # ensure all style label exists in classifier_labels
        for style_label in self.style_labels:
            assert style_label in classifier_labels

    def __str__(self) -> str:
        _text = "Classifier configuration\n"
        _text += "-" * 30 + "\n"
        for key, value in self._classifier_config.items():
            _text += f"{key}: {value}\n"

        _text += "\nlanguage model configuration\n"
        _text += "-" * 30 + "\n"
        for key, value in self._lm_config.items():
            _text += f"{key}: {value}\n"

        return _text

    @property
    def classifier_config(self) -> None:
        return self._classifier_config

    @property
    def lm_config(self) -> None:
        return self._lm_config

    @property
    def reference(self) -> Dict[str, List[str]]:
        return self._reference_dict

    def get_style_accuracy(
        self, transfered_sentences: List[str], style_label: str
    ) -> float:
        """
        calculate style transfer accuracy based on given original style label.
        accucacy incleases when classification results don't match with original style.
        style of all given sentences must be same

        :param transfered_sentences: list of sentences which is transfered.
        :param original_style_label: original style label before transferring style
        """
        if not style_label.startswith(self.label_prefix):
            style_label = f"{self.label_prefix}{style_label}"

        msg = f"original_style_labels has to be in {self.style_labels}"
        assert style_label in self.style_labels, msg

        num_correct = 0
        for sentence in transfered_sentences:
            sentence = " ".join(word_tokenize(sentence.lower().strip()))
            if not len(sentence):
                continue
            predicted_label = self.classifier.predict(sentence)[0][0]

            if predicted_label != style_label:
                num_correct += 1

        return num_correct / len(transfered_sentences)

    def get_bleu_score(
        self, original_sentence: str, transfered_sentence: str, n_gram: int = 4
    ) -> float:
        """
        Calculate bleu score using nltk.

        :param original_sentence: original sentences.
        :parma transfered_sentence: transfered sentences.
        :parma n_gram: n-gram
        """
        original_sentences = [word_tokenize(original_sentence.lower().strip())]
        transfered_sentence = word_tokenize(transfered_sentence.lower().strip())
        weights = [1.0 / n_gram] * n_gram

        return (
            bleu_score.sentence_bleu(original_sentences, transfered_sentence, weights)
            * 100
        )

    def get_self_bleu_score(
        self, original_sentences: List[str], transfered_sentences: List[str]
    ) -> float:
        """
        Calculate bleu score between two pair of sentences.

        :param original_sentences: list of original sentences.
        :param transfered_sentences: list of transfered sentences.
        """
        msg = "length of original_sentences and transfered_sentences does not match."
        assert len(original_sentences) == len(transfered_sentences), msg
        sum_bleu_score = 0
        for origin, transfered in zip(original_sentences, transfered_sentences):
            sum_bleu_score += self.get_bleu_score(origin, transfered)

        return sum_bleu_score / len(original_sentences)

    def get_reference_bleu_score(
        self, transfered_sentences: List[str], style_label: str
    ) -> float:
        """
        Calcucate bleu score between transfered sentences
        from target to source and original source sentences.

        :param transfered_sentences: list of transfered sentences from target to source
        :param style_label: style label
        """
        if not style_label.startswith(self.label_prefix):
            style_label = f"{self.label_prefix}{style_label}"
        assert style_label in self.style_labels

        if not len(self._reference_dict) or not len(self._reference_dict[style_label]):
            raise Exception("reference files are not initialized.")

        return self.get_self_bleu_score(
            transfered_sentences, self._reference_dict[style_label]
        )

    def get_perplexity(self, transfered_sentences: List[str]) -> float:
        """
        Calculate perplexity using kenLM

        :param: transfered_sentences: list of transfered sentences
        """
        num_words = 0
        sum_ppl = 0
        for idx, sentence in enumerate(transfered_sentences):
            tokens = word_tokenize(sentence.lower().strip())
            num_words += len(tokens)
            sum_ppl += self.language_model.score(" ".join(tokens))

        return math.pow(10, -sum_ppl / num_words)

    @classmethod
    def create(
        cls,
        text_paths: Union[List[str], List[Path]],
        style_list: List[str],
        reference_paths: Union[None, List[str], List[Path]] = None,
        lm_config: Dict[str, Any] = DEFAULT_LM_CONFIG,
        classifier_config: Dict[str, Any] = DEFAULT_CLASSIFIER_CONFIG,
        sample_size: Optional[int] = None,
    ) -> "Evaluator":
        """
        create evaluator with training language model and classifier

        :param text_paths: list of text paths, which is used for traininig style-transformer,
            and inner list only contains specific stylistic sentences
        :param style_list: list of style, which is corresponding to sentence_list
        :param reference_paths: paths to reference file corresponding to each sentence_list
        :param lm_config: configuration for language model.
        :param classifier_config: configuration for classifier.
        :param sample_size: sampling specified number of sentences from each style

        NOTE
        ----
        input sentences has to be pre-tokenized for training language model.
        """
        # load text
        print("Loading text...")
        num_sentences = 0
        sentence_list = []
        for path in text_paths:
            path = convert_to_pathlib(path)
            assert path.exists()

            with path.open() as f:
                sentences = [sentence.strip() for sentence in f.readlines()]
            sentence_list.append(sentences)
            num_sentences += len(sentences)
        print(f"text size: {num_sentences}")

        # sample sentences
        if sample_size is not None and sample_size > 0:
            min_sentence_size = min(
                [len(sub_sentences) for sub_sentences in sentence_list]
            )
            style_sample_size = min(sample_size, min_sentence_size)

            if sample_size != style_sample_size:
                print(f"change sampling size from {sample_size} to {style_sample_size}")
            print("Sampling sentences...")

            for style_idx, style_sentences in enumerate(sentence_list):
                sentence_list[style_idx] = style_sentences[:style_sample_size]

        # create temporal directory
        hash_seed = str(sentence_list) + str(style_list)
        setting_hash = hashlib.md5(hash_seed.encode("utf-8")).hexdigest()
        tmp_dir = cls.TMO_DIR_PREFIX / setting_hash
        lm_model_path = tmp_dir / cls.LANGUAGE_MODEL_FILE_NAME
        classifier_model_path = tmp_dir / cls.CLASSIFIER_FILE_NAME
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)

        if not classifier_model_path.exists() or not lm_model_path.exists():
            cls._create_model(
                sentence_list, style_list, tmp_dir, lm_config, classifier_config
            )

        classifier_config["style_labels"] = style_list

        return cls(
            lm_model_path,
            classifier_model_path,
            reference_paths,
            lm_config,
            classifier_config,
        )

    @classmethod
    def _create_model(
        cls,
        sentence_list: List[List[str]],
        style_list: List[str],
        tmp_dir: Union[str, Path],
        lm_config: Dict[str, Any] = DEFAULT_LM_CONFIG,
        classifier_config: Dict[str, Any] = DEFAULT_CLASSIFIER_CONFIG,
    ) -> None:
        """
        create language model by kenlm and classifier by fasttext

        :param sentence_list: nested list of sentence, which is splitted based on style
        :param style_list: list of style corresponding to sentence_list
        :param tmp_dir: directory to save model and config and so on.
        :param lm_config: configuration dictionary of language model.
        :param classifier_config: configuration dictionary of classifier.

        NOTE
        ----

        - about kenlm
        language model by kenLM can be trained like below this.
        More resources can be found here https://github.com/kpu/kenlm
        >>> bin/lmplz -o 5 < train_data.txt > output.bin

        - about fasttext
        classification model by fasttext can be trained like below this.
        More resources can be found here https://github.com/facebookresearch/fasttext
        >>> import fasttext
        >>> model = fasttext.train_supervised(
                        'input.txt',
                        epoch=25,
                        lr=1.0,
                        wordNgrams=2,
                        verbose=2,
                        minCount=1,
                        label='__style__'
                    )
        model.save_model("output.bin")
        """
        tmp_dir = ensure_dir(tmp_dir)
        tmp_lm_file_path = tmp_dir / cls.TMP_LM_FILE_NAME
        tmp_classifier_file_path = tmp_dir / cls.TMP_CASSIFIER_FILE_NAME
        lm_model_path = tmp_dir / cls.LANGUAGE_MODEL_FILE_NAME
        classifier_model_path = tmp_dir / cls.CLASSIFIER_FILE_NAME

        # prepare tmporaly file
        language_model_sentences = []
        classifier_sentences = []
        label_prefix = classifier_config["label_prefix"]

        # load source sentences
        for style_sentence_list, style_label in zip(sentence_list, style_list):
            for sentence in style_sentence_list:
                sentence = sentence.lower().strip()

                # preprocess for training classifier
                classifier_sentence = f"{label_prefix}{style_label} {sentence}"
                classifier_sentences.append(classifier_sentence)

                # for language model training
                language_model_sentences.append(sentence)

        # save each text for training classifier and language model
        save_text_list(language_model_sentences, tmp_lm_file_path)
        save_text_list(classifier_sentences, tmp_classifier_file_path)

        # train kenlm's language model as subprocesses
        print("Training language model...")
        with tmp_lm_file_path.open() as fi, lm_model_path.open("wb") as fo:
            lm_model = subprocess.Popen(
                ["lmplz", "-o", str(lm_config["n_order"])], stdin=fi, stdout=fo
            )
            lm_model.wait()

        # train fasttext classifier
        print("Training classifier...")
        model = fasttext.train_supervised(
            tmp_classifier_file_path.as_posix(),
            epoch=classifier_config["epoch"],
            lr=classifier_config["lr"],
            wordNgrams=classifier_config["n_order"],
            verbose=2,
            minCount=classifier_config["min_count"],
            label=classifier_config["label_prefix"],
        )
        model.save_model(classifier_model_path.as_posix())

        # delete temporal files
        tmp_lm_file_path.unlink()
        tmp_classifier_file_path.unlink()

    @classmethod
    def load(cls, dir_path: Union[str, Path]) -> "Evaluator":
        """
        Load saved evaluator, which is saved as pickle format

        :param dir_path: path to saved model dir.
        :param return: Evalutor
        """
        dir_path = convert_to_pathlib(dir_path)

        lm_path = dir_path / cls.LANGUAGE_MODEL_FILE_NAME
        classifier_path = dir_path / cls.CLASSIFIER_FILE_NAME
        config_path = dir_path / cls.CONFIG_FILE_NAME

        if not classifier_path.exists():
            raise FileNotFoundError(
                f"Classifier path found: {classifier_path.as_posix()}"
            )
        if not lm_path.exists():
            raise FileNotFoundError(f"language model not found: {lm_path.as_posix()}")
        if not config_path.exists():
            raise FileNotFoundError(f"config file not found: {config_path.as_posix()}")

        with config_path.open("rb") as f:
            (reference_paths, lm_config, classifier_config) = pickle.loads(f.read())

        return cls(
            lm_path, classifier_path, reference_paths, lm_config, classifier_config
        )

    def save(self, save_dir: Union[str, Path]) -> None:
        """
        save configurations

        :param save_dir: directory to save configurations.
        """
        save_dir = ensure_dir(save_dir)
        lm_path = save_dir / self.LANGUAGE_MODEL_FILE_NAME
        classifier_path = save_dir / self.CLASSIFIER_FILE_NAME
        config_path = save_dir / self.CONFIG_FILE_NAME

        if not save_dir.exists():
            save_dir.mkdir(parents=True)

        shutil.copyfile(self.lm_path, lm_path)
        shutil.copyfile(self.classifier_path, classifier_path)
        with config_path.open("wb") as f:
            pickle.dump(
                (self.reference_paths, self.lm_config, self.classifier_config), f
            )


if __name__ == "__main__":
    parser = ArgumentParser("create evaluator for style transfer")
    parser.add_argument(
        "--text_paths",
        type=str,
        nargs="+",
        required=True,
        help="list of paths to source text",
    )
    parser.add_argument(
        "--style_labels",
        type=str,
        nargs="+",
        required=True,
        help="list of style corresponding to sentence_paths",
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="path to save models"
    )
    parser.add_argument(
        "--reference_paths",
        type=str,
        nargs="+",
        default=None,
        help="path to reference source, which is used for calculating bleu score with it",
    )
    parser.add_argument(
        "--lm_n_order",
        type=int,
        default=5,
        help="the number of order for language model training",
    )
    parser.add_argument(
        "--classifier_epoch", type=int, default=25, help="language model epoch"
    )
    parser.add_argument(
        "--classifier_lr", type=float, default=1.0, help="language model learning rate"
    )
    parser.add_argument(
        "--classifier_n_order",
        type=int,
        default=2,
        help="the nuber of order for classifier",
    )
    parser.add_argument(
        "--classifier_min_count", type=int, default=1, help="the number of word count"
    )
    parser.add_argument(
        "--label_prefix",
        type=str,
        default="__style__",
        help="label prefix used for classifier training",
    )
    parser.add_argument(
        "--verbose", type=int, default=2, help="show logging for training models"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="sampling specified number of sentences from top to bottom in the files",
    )
    args = parser.parse_args()

    lm_config = {"n_order": args.lm_n_order}
    classifier_config = {
        "epoch": args.classifier_epoch,
        "lr": args.classifier_lr,
        "n_order": args.classifier_n_order,
        "verbose": args.verbose,
        "min_count": args.classifier_min_count,
        "label_prefix": args.label_prefix,
    }

    print("Start model training...")
    evaluator = Evaluator.create(
        sentence_list=args.text_paths,
        style_list=args.style_labels,
        reference_paths=args.reference_paths,
        lm_config=lm_config,
        classifier_config=classifier_config,
        sample_size=args.sample_size,
    )

    print(f"Saving models to {args.save_dir}")
    evaluator.save(args.save_dir)

    print("Done")
