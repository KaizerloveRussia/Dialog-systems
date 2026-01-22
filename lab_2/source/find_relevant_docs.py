from typing import List, Dict, Any
from elasticsearch import Elasticsearch
from create_index import DOC_INDEX_NAME, get_es_client


TOP_K = 50


def bm25_search(es: Elasticsearch, query: str, top_k: int = TOP_K,) -> List[Dict[str, Any]]:
    if not query:
        return []

    body = {
        "query": {
            "multi_match": {
                "query": query,
                "fields": ["title", "text"],
            }
        }
    }

    resp = es.search(index=DOC_INDEX_NAME, body=body, size=top_k)
    hits = resp.get("hits", {}).get("hits", [])

    results: List[Dict[str, Any]] = []
    for h in hits:
        src = h.get("_source", {})
        results.append(
            {
                "docid": src.get("docid"),
                "title": src.get("title", ""),
                "text": src.get("text", ""),
                "score": h.get("_score", 0.0),
            }
        )

    return results


if __name__ == '__main__':
    query = input("Введите текст запроса: ").strip()

    es = get_es_client()
    results = bm25_search(es, query, TOP_K)

    print(f"Top-{TOP_K} релевантных документов:")
    for i, r in enumerate(results, start=1):
        title = (r["title"] or "").replace("\n", " ")
        text = (r["text"] or "").replace("\n", " ")
        print(f"{i:2d}. docid={r['docid']}, score={r['score']:.4f}")
        if title:
            print(f"    title: {title[:200]}")
        if text:
            print(f"    text: {text[:500]}")