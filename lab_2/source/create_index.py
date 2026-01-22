import pickle
from typing import Iterable, Any, Dict
from elasticsearch import Elasticsearch, helpers


DOC_INDEX_NAME = "sw_corpus"
QRELS_INDEX_NAME = "sw_qrels"


def get_es_client() -> Elasticsearch:
    return Elasticsearch("http://localhost:9200")


def ensure_doc_index(es: Elasticsearch) -> None:
    body = {
        "settings": {
            "analysis": {
                "analyzer": {
                    "default": {
                        "type": "standard"
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "docid": {"type": "keyword"},
                "title": {"type": "text"},
                "text": {"type": "text"},
            }
        },
    }
    es.options(ignore_status=[400]).indices.create(index=DOC_INDEX_NAME, body=body)


def ensure_qrels_index(es: Elasticsearch) -> None:
    body = {
        "mappings": {
            "properties": {
                "query_id": {"type": "integer"},
                "doc_id": {"type": "keyword"},
                "relevance": {"type": "integer"},
            }
        }
    }
    es.options(ignore_status=[400]).indices.create(index=QRELS_INDEX_NAME, body=body)


def load_corpus() -> Any:
    with open("./assets/corpus.pkl", "rb") as f:
        corpus = pickle.load(f)
    return corpus


def iter_corpus_docs(corpus: Any) -> Iterable[Dict[str, Any]]:
    if hasattr(corpus, "keys"):
        split = corpus.get("train", None)
        if split is None:
            first_key = next(iter(corpus.keys()))
            split = corpus[first_key]
        dataset = split
    else:
        dataset = corpus

    for doc in dataset:
        yield {
            "docid": doc.get("docid"),
            "title": doc.get("title", ""),
            "text": doc.get("text", ""),
        }


def bulk_index_corpus(es: Elasticsearch, corpus: Any) -> None:
    def gen_actions():
        for d in iter_corpus_docs(corpus):
            if d["docid"] is None:
                continue
            yield {
                "_index": DOC_INDEX_NAME,
                "_id": d["docid"],
                "_source": d,
            }

    helpers.bulk(es, gen_actions())


def bulk_index_qrels(es: Elasticsearch, qrels_path: str = "./assets/qrels.tsv") -> None:
    def gen_actions():
        with open(qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 4:
                    continue
                qid_str, _, doc_id, rel_str = parts
                try:
                    qid = int(qid_str)
                    rel = int(rel_str)
                except ValueError:
                    continue

                yield {
                    "_index": QRELS_INDEX_NAME,
                    "_id": f"{qid}_{doc_id}",
                    "_source": {
                        "query_id": qid,
                        "doc_id": doc_id,
                        "relevance": rel,
                    },
                }

    helpers.bulk(es, gen_actions())


if __name__ == "__main__":
    es = get_es_client()

    ensure_doc_index(es)
    ensure_qrels_index(es)

    corpus = load_corpus()
    bulk_index_corpus(es, corpus)
    bulk_index_qrels(es)

    doc_count = es.count(index=DOC_INDEX_NAME)["count"]
    qrels_count = es.count(index=QRELS_INDEX_NAME)["count"]
    print(f"Проиндексировано {doc_count} документов в индексе '{DOC_INDEX_NAME}'")
    print(f"Проиндексировано {qrels_count} qrels в индексе '{QRELS_INDEX_NAME}'")