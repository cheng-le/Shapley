"""Create local synthetic inputs and a tiny RANDOM model for integration tests."""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

from pile_10b.test_pipeline import fixture_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    model = a.output / "model"
    model.mkdir()
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tok.save_pretrained(model)
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=32, n_embd=32,
                              n_layer=1, n_head=2)).save_pretrained(model)
    source = a.output / "source.jsonl"
    with source.open("w") as f:
        for row in fixture_rows():
            f.write(json.dumps(row) + "\n")
    c = json.loads(Path(__file__).with_name("config.json").read_text())
    c.update(model=str(model), local_jsonl=[str(source)], train_tokens=134,
             val_tokens=64, sequence_length=16, query_blocks=2, feature_dim=8,
             feature_batch=2, feature_chunk=8, graph_k=3, graph_backend="exact",
             batch_per_gpu=2, checkpoint_steps=2, eval_steps=2, pilot_steps=1,
             max_estimated_hours_per_seed=0, seeds=[0], precision="fp32")
    (a.output / "config.json").write_text(json.dumps(c, indent=2) + "\n")
    print(a.output / "config.json")


if __name__ == "__main__":
    main()
