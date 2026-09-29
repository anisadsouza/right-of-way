from __future__ import annotations

import argparse
import csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the final 20-scenario yield evaluation from a completed CSV sheet.")
    parser.add_argument("results", help="Completed docs/scenario_results_template.csv file")
    return parser.parse_args()


def yes(value: str) -> bool:
    return value.strip().lower() in {"yes", "y", "true", "1", "pass"}


def main() -> None:
    args = parse_args()
    with open(args.results, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    completed = [row for row in rows if row.get("detected", "").strip()]
    if not completed:
        raise SystemExit("No completed trials. Fill detected/yielded/resumed_after_pass before evaluating.")

    expected_emergency = [row for row in completed if yes(row["expected_yield"])]
    expected_normal = [row for row in completed if not yes(row["expected_yield"])]
    correct_yields = sum(yes(row["detected"]) and yes(row["yielded"]) and yes(row["resumed_after_pass"]) for row in expected_emergency)
    false_yields = sum(yes(row["false_yield"]) or yes(row["yielded"]) for row in expected_normal)
    false_negatives = sum(not (yes(row["detected"]) and yes(row["yielded"])) for row in expected_emergency)

    print(f"Completed trials: {len(completed)}/{len(rows)}")
    print(f"Correct emergency yields: {correct_yields}/{len(expected_emergency)}")
    print(f"False negatives: {false_negatives}/{len(expected_emergency)}")
    print(f"False yields: {false_yields}/{len(expected_normal)} normal trials")
    print(f"End-to-end success rate: {correct_yields / len(expected_emergency) * 100:.1f}%")


if __name__ == "__main__":
    main()
