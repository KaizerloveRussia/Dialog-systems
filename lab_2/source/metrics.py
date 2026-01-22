from typing import Dict
from trectools import TrecQrel, TrecRun, TrecEval


def compute_metrics_at_5(qrels_path: str, run_path: str) -> Dict[str, float]:
    qrels = TrecQrel(qrels_path)
    run = TrecRun(run_path)

    evaluator = TrecEval(run, qrels)

    precision_at_5 = float(evaluator.get_precision(depth=5))
    recall_at_5 = float(evaluator.get_recall(depth=5))
    map_at_5 = float(evaluator.get_map(depth=5))
    mrr_at_5 = float(evaluator.get_reciprocal_rank(depth=5))

    return {
        "P@5": precision_at_5,
        "R@5": recall_at_5,
        "MAP@5": map_at_5,
        "MRR@5": mrr_at_5,
    }


if __name__ == "__main__":
    import sys

    qrels_path = sys.argv[1]
    run_path = sys.argv[2]

    metrics = compute_metrics_at_5(qrels_path, run_path)

    print("Агрегированные метрики (усреднены по запросам):")
    print(f"Precision@5: {metrics['P@5']:.4f}")
    print(f"Recall@5:    {metrics['R@5']:.4f}")
    print(f"MAP@5:       {metrics['MAP@5']:.4f}")
    print(f"MRR@5:       {metrics['MRR@5']:.4f}")