"""Create the disjoint valuation/final-evaluation split for corrected runs."""
import json
from pathlib import Path
from pile_main_100k.common import dataset_fingerprint, load_rows, stratified_split_indices

def main():
    rows = load_rows("val")
    valuation, evaluation = stratified_split_indices(rows, 0.2, 2027)
    payload = {
        "version": "pile_100k_protocol_v2", "seed": 2027,
        "valuation_fraction": 0.2, "val_fingerprint": dataset_fingerprint(rows),
        "valuation_indices": valuation.tolist(), "evaluation_indices": evaluation.tolist(),
    }
    path = Path("pile_main_100k/protocol_v2_split.json")
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"valuation={len(valuation)} evaluation={len(evaluation)} output={path}")
if __name__ == "__main__": main()
