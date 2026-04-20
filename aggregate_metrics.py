import json
import os

aggregate_metrics = {
    "all": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "BD": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "SD": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "TT": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "HH": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "CY + RD": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "Cowbell": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
    "Claves": {
        "precision": 0,
        "recall": 0,
        "f_measure": 0,
    },
}

for split in [0, 1, 2]:
    results_path = f"results/ENST-tau-0.8/ENST_inference/{split}/"
    metrics_path = os.path.join(results_path, "metrics.json")
    metrics = json.load(open(metrics_path))
    for key, value in metrics.items():
        aggregate_metrics[key]["precision"] += value["precision"]
        aggregate_metrics[key]["recall"] += value["recall"]
        aggregate_metrics[key]["f_measure"] += value["f_measure"]

for key, value in aggregate_metrics.items():
    aggregate_metrics[key]["precision"] /= 3
    aggregate_metrics[key]["recall"] /= 3
    aggregate_metrics[key]["f_measure"] /= 3

print(aggregate_metrics)
with open("results/ENST-tau-0.8/ENST_inference/aggregate_metrics.json", "w") as f:
    json.dump(aggregate_metrics, f)
