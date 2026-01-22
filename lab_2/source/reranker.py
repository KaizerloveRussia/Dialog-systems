from typing import Dict, List, Tuple
import csv
from sentence_transformers import CrossEncoder
from create_index import get_es_client
from find_relevant_docs import bm25_search


BM25_TOP_K: int = 50
RUN_ID: str = "bm25+cross-encoder"
CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_queries(path: str) -> Dict[int, str]:
    queries: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or len(row) < 2:
                continue
            try:
                qid = int(row[0])
            except ValueError:
                continue
            query = row[1]
            queries[qid] = query
    return queries


def build_ce_pairs(query: str, docs: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """
    Строим пары (query, документ) для Cross-Encoder.
    Документ представляем как конкатенацию title и text.
    """
    pairs: List[Tuple[str, str]] = []
    for d in docs:
        title = d.get("title") or ""
        text = d.get("text") or ""
        if title and text:
            doc_text = f"{title}. {text}"
        else:
            doc_text = title or text
        pairs.append((query, doc_text))
    return pairs


def rerank_with_cross_encoder(model: CrossEncoder, query: str, docs: List[Dict[str, str]], batch_size: int = 32,) -> List[Dict[str, str]]:
    if not docs:
        return []

    pairs = build_ce_pairs(query, docs)
    scores = model.predict(pairs, batch_size=batch_size)

    for d, s in zip(docs, scores):
        d["ce_score"] = float(s)

    # По убыванию нового скора
    docs_sorted = sorted(docs, key=lambda x: x["ce_score"], reverse=True)
    return docs_sorted


if __name__ == "__main__":
    queries_path = "./assets/topics.txt"
    run_path = "./assets/run_rerank.txt"

    queries = load_queries(queries_path)
    es = get_es_client()

    print(f"Загружаем Cross-Encoder'{CROSS_ENCODER_MODEL_NAME}'...")
    model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    print("Модель загружена.")

    with open(run_path, "w", encoding="utf-8") as out:
        for qid, query in queries.items():
            # top-50 документов по BM25
            bm25_docs = bm25_search(es, query, BM25_TOP_K)
            # реранк через Cross-Encoder
            reranked_docs = rerank_with_cross_encoder(model, query, bm25_docs)

            for rank, d in enumerate(reranked_docs, start=1):
                docid = d["docid"]
                score = d["ce_score"]
                line = f"{qid} Q0 {docid} {rank} {score} {RUN_ID}\n"
                out.write(line)

    print(f"Run-файл с переранжированными результатами сохранён в {run_path}")