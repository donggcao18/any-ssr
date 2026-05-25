import json
import sys


def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["source"]: item for item in data["predictions"]}


def main():
    if len(sys.argv) != 3:
        print("Usage: python draft.py <file1.json> <file2.json>")
        sys.exit(1)

    file1, file2 = sys.argv[1], sys.argv[2]
    preds1 = load_predictions(file1)
    preds2 = load_predictions(file2)

    diffs = []
    for source, item1 in preds1.items():
        if source not in preds2:
            continue
        item2 = preds2[source]
        if item1["prediction"] != item2["prediction"]:
            diffs.append({
                "source": source,
                "ground-truth": item1["ground-truth"],
                f"prediction ({file1})": item1["prediction"],
                f"prediction ({file2})": item2["prediction"],
            })

    with open("diff.jsonl", "w", encoding="utf-8") as f:
        for entry in diffs:
            f.write(json.dumps(entry, ensure_ascii=False, indent=2) + "\n")

    print(f"{len(diffs)} differing predictions written to diff.jsonl")


if __name__ == "__main__":
    main()
