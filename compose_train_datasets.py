#!/usr/bin/env python3
"""
Dump Shitao/bge-reranker-data to a single JSONL file, preserving ALL fields.

Requires:
  pip install -U huggingface_hub tqdm
"""
import argparse
import json
import re
import tarfile
from typing import Iterable, Any
from huggingface_hub import hf_hub_download
from tqdm import tqdm

REPO_ID = "Shitao/bge-reranker-data"
SHARDS = [
    "data_v1/data_v1_split0.tar.gz",
    "data_v1/data_v1_split1.tar.gz",
]

def _yield_loaded(obj: Any) -> Iterable[dict]:
    """Yield dict-like records from obj (dict or list of dicts)."""
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for x in obj:
            # if some entries are not dicts, still dump them as-is
            yield x if isinstance(x, dict) else {"value": x}
    else:
        # unknown type: still write something so nothing is lost
        yield {"value": obj}

def iter_json_any(text: str) -> Iterable[dict]:
    """
    Yield 1+ JSON objects from text:
      1) a single JSON dict or list
      2) JSON Lines (one JSON per line)
      3) concatenated JSON objects like '}{'
    """
    s = text.strip()
    # 1) try single JSON (could be dict or list)
    try:
        obj = json.loads(s)
        yield from _yield_loaded(obj)
        return
    except json.JSONDecodeError:
        pass

    # 2) try JSONL
    got_any = False
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            for rec in _yield_loaded(obj):
                yield rec
            got_any = True
        except json.JSONDecodeError:
            # keep trying other formats
            pass
    if got_any:
        return

    # 3) concatenated objects: split at '}{' boundaries
    parts = re.split(r'}\s*{', s)
    if len(parts) > 1:
        for i, p in enumerate(parts):
            chunk = (p + '}') if i == 0 else ('{' + p) if i == len(parts) - 1 else ('{' + p + '}')
            try:
                obj = json.loads(chunk)
                for rec in _yield_loaded(obj):
                    yield rec
            except json.JSONDecodeError:
                # skip malformed piece but continue
                continue
        return

    # 4) last resort: emit raw text (so we don't lose data)
    yield {"raw": s}

def stream_shard(local_path: str) -> Iterable[dict]:
    # Stream tar members to keep memory low
    with tarfile.open(local_path, mode="r|gz") as tf:
        for member in tf:
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            f = tf.extractfile(member)
            if not f:
                continue
            text = f.read().decode("utf-8", errors="ignore")
            yield from iter_json_any(text)

def main(out_path: str, limit: int | None):
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for shard in SHARDS:
            local = hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=shard)
            for rec in tqdm(stream_shard(local), desc=f"Converting {shard}"):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total += 1
                if limit and total >= limit:
                    print(f"Hit limit={limit}. Wrote {total} lines to {out_path}")
                    return
    print(f"Wrote {total} lines to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser("Dump Shitao/bge-reranker-data to JSONL (preserve all fields)")
    ap.add_argument("--out", default="bge_reranker_data.jsonl", help="Output JSONL path")
    ap.add_argument("--limit", type=int, help="Stop after N records (for quick tests)")
    args = ap.parse_args()
    main(args.out, args.limit)