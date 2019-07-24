from pathlib import Path

import pytest

from utils.evaluator import Evaluator

# load sentences for testing
YELP_TEST_POSITIVE_PATH = Path("test/tmp/yelp.test.pos")
YELP_TEST_NEGATIVE_PATH = Path("test/tmp/yelp.test.neg")
YELP_POSITIVE_REFERENCE_PATH = Path("test/tmp/yelp.ref.pos")
YELP_NEGATIVE_REFERENCE_PATH = Path("test/tmp/yelp.ref.neg")

STYLE_INDICATORS = ["positive", "negative"]

with YELP_TEST_POSITIVE_PATH.open() as f:
    YELP_TEST_POSITIVE_SENTENCES = [sen.lower().strip() for sen in f.readlines()]
with YELP_TEST_NEGATIVE_PATH.open() as f:
    YELP_TEST_NEGATIVE_SENTENCES = [sen.lower().strip() for sen in f.readlines()]

LM_CONFIG = {"n_order": 5}
CLASSIFIER_CONFIG = {
    "epoch": 5,
    "lr": 1.0,
    "n_order": 2,
    "verbose": 2,
    "min_count": 1,
    "label_prefix": "__style__",
}


class TestEvaluator:
    """
    Evaluator test class
    """

    @pytest.fixture()
    def test_create(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        for style_label in STYLE_INDICATORS:
            assert (
                f"{CLASSIFIER_CONFIG['label_prefix']}{style_label}"
                in self.evaluator.style_labels
            )

    def test_get_style_accuracy(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        pos_acc = evaluator.get_style_accuracy(
            YELP_TEST_POSITIVE_SENTENCES, STYLE_INDICATORS[1]
        )
        neg_acc = evaluator.get_style_accuracy(
            YELP_TEST_NEGATIVE_SENTENCES, STYLE_INDICATORS[0]
        )

        assert pos_acc > 0.8
        assert neg_acc > 0.8

    def test_get_bleu_score(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        bleu_score = evaluator.get_bleu_score(
            YELP_TEST_POSITIVE_SENTENCES[0], YELP_TEST_POSITIVE_SENTENCES[0]
        )
        assert bleu_score == 100

    def test_self_bleu_score(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        self_bleu_score = evaluator.get_self_bleu_score(
            YELP_TEST_POSITIVE_SENTENCES, YELP_TEST_POSITIVE_SENTENCES
        )

        assert self_bleu_score == 100

    def test_get_positive_reference_bleu_score(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        ref_bleu = evaluator.get_reference_bleu_score(
            YELP_TEST_POSITIVE_SENTENCES, STYLE_INDICATORS[0]
        )
        assert ref_bleu > 20

    def test_get_negative_reference_bleu_score(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        ref_bleu = evaluator.get_reference_bleu_score(
            YELP_TEST_NEGATIVE_SENTENCES, STYLE_INDICATORS[1]
        )

        assert ref_bleu > 20

    def test_get_perplexity(self):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        ppl = evaluator.get_perplexity(YELP_TEST_POSITIVE_SENTENCES)
        assert ppl < 100

    def test_save_and_load(self, tmpdir):
        evaluator = Evaluator.create(
            text_paths=[YELP_TEST_POSITIVE_PATH, YELP_TEST_NEGATIVE_PATH],
            style_list=STYLE_INDICATORS,
            reference_paths=[
                YELP_POSITIVE_REFERENCE_PATH,
                YELP_NEGATIVE_REFERENCE_PATH,
            ],
            lm_config=LM_CONFIG,
            classifier_config=CLASSIFIER_CONFIG,
            sample_size=None,
        )

        evaluator.save(Path(tmpdir))

        assert Path(tmpdir).exists()
        assert (Path(tmpdir) / evaluator.LANGUAGE_MODEL_FILE_NAME).exists()
        assert (Path(tmpdir) / evaluator.CLASSIFIER_FILE_NAME).exists()
        assert (Path(tmpdir) / evaluator.CONFIG_FILE_NAME).exists()

        loaded_evaluator = Evaluator.load(Path(tmpdir))

        for style_label in STYLE_INDICATORS:
            assert (
                f"{CLASSIFIER_CONFIG['label_prefix']}{style_label}"
                in loaded_evaluator.style_labels
            )
