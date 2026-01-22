from typing import Dict
import csv

from create_index import get_es_client
from find_relevant_docs import bm25_search, TOP_K


RUN_ID = "bm25"


def load_queries(path: str) -> Dict[int, str]:
    """
    Ожидается формат:
        qid \t query
    """
    queries: Dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            try:
                qid = int(row[0])
            except ValueError:
                continue
            query = row[1]
            queries[qid] = query
    return queries


if __name__ == '__main__':
    queries_path = "./assets/topics.txt"
    run_path = "./assets/run.txt"

    queries = load_queries(queries_path)
    es = get_es_client()

    with open(run_path, "w", encoding="utf-8") as out:
        for qid, query in queries.items():
            results = bm25_search(es, query, TOP_K)
            for rank, r in enumerate(results, start=1):
                docid = r["docid"]
                score = r["score"]
                # Формат TREC-run:
                # qid Q0 docid rank score run_id
                line = f"{qid} Q0 {docid} {rank} {score} {RUN_ID}\n"
                out.write(line)

    print(f"Run-файл сохранён в {run_path}")