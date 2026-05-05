from src.evaluation.selective_metrics import compute_selective_metrics


def test_selective_metrics_threshold():
    records = [
        {"confidence": 0.9, "vqa_score": 1.0},
        {"confidence": 0.4, "vqa_score": 0.0},
        {"confidence": 0.7, "vqa_score": 0.5},
    ]
    metrics = compute_selective_metrics(records, threshold=0.5)
    assert metrics.coverage == 2 / 3
    assert metrics.selective_accuracy == 0.75
    assert metrics.risk == 0.25
    assert metrics.effective_reliability_score == 0.5

