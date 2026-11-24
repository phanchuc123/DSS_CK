"""Generate manual review artifacts from the model currently used by the web."""

import csv
from pathlib import Path

from app import perform_recommend


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "recipe_model1"
SOURCE_REVIEW = BASE_DIR / "recipe_model" / "manual_review.csv"


def load_queries():
    return [
        ("1", "cà chua, muối, dầu ăn, trứng"),
        ("2", "thịt gà, gừng"),
        ("3", "thịt heo, trứng"),
    ]


def generate():
    queries = load_queries()
    ranking_rows = []
    review_rows = []
    methods = ("cbf", "encoder", "hybrid", "main_aware")

    for query_id, query in queries:
        for method in methods:
            result = perform_recommend(query, top_k=10, method=method, assume_pantry=True)
            for group in ("ready", "needs_shopping"):
                rank = 0
                for item in result[group]:
                    rank += 1
                    ranking_rows.append({
                        "query_id": query_id,
                        "method": method,
                        "group": group,
                        "rank": rank,
                        "dish_id": item["dish_id"],
                        "dish": item["dish"],
                        "score": item["score"],
                    })

        # The hybrid output is the web's default and is the list to review.
        result = perform_recommend(query, top_k=10, method="hybrid", assume_pantry=True)
        for group in ("ready", "needs_shopping"):
            rank = 0
            for item in result[group]:
                rank += 1
                review_rows.append({
                    "query_id": query_id,
                    "query": query,
                    "dish_id": item["dish_id"],
                    "dish": item["dish"],
                    "matched": ", ".join(item.get("matched", [])),
                    "missing_required": ", ".join(item.get("missing_required", [])),
                    "missing_main_provisional": ", ".join(item.get("missing_main_provisional", [])),
                    "relevant_0_or_1": "",
                    "can_cook_0_or_1": "",
                    "notes": "",
                })

    ranking_path = OUTPUT_DIR / "manual_review_rankings.csv"
    with ranking_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ranking_rows[0].keys())
        writer.writeheader()
        writer.writerows(ranking_rows)

    review_path = OUTPUT_DIR / "manual_review.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=review_rows[0].keys())
        writer.writeheader()
        writer.writerows(review_rows)

    print(f"Created {ranking_path} ({len(ranking_rows)} rows)")
    print(f"Created {review_path} ({len(review_rows)} rows)")


if __name__ == "__main__":
    generate()
