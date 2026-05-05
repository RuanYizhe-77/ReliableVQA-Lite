from src.evaluation.vqa_accuracy import vqa_soft_accuracy
from src.utils.text import normalize_answer


def test_normalize_answer_articles_punctuation_and_numbers():
    assert normalize_answer("The Two, dogs!") == "2 dogs"
    assert normalize_answer("  an APPLE. ") == "apple."


def test_vqa_soft_accuracy():
    answers = ["cat", "cat", "dog", "cat", "bird"]
    assert vqa_soft_accuracy("the cat", answers) == 1.0
    assert vqa_soft_accuracy("dog", answers) == 1 / 3

