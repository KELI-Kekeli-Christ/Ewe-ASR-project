#!/usr/bin/env python3
"""
Build an audio-hash-disjoint Ewe ASR package on the server.

Policy
------
1. SHA1 over raw MP3 bytes for every row in the current package.
2. Drop test rows with MP3 duration <= 0 (corrupt), and drop the same SHA1 everywhere.
3. Each SHA1 kept in at most ONE split; priority: test > validation > train.
4. Within a split, prefer speechUG-like schema (has locale) over WAXAL-like.
5. Write a NEW directory; never overwrite the original package.
6. Assert train∩val∩test SHA1 intersections are 0.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from mutagen.mp3 import MP3

SRC = Path(
    "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/Ewe_parquet/data_ewe"
)
OUT = Path(
    "/home/data/projets-aps/projet6/data/data_ewe/speech_ug/parquet/"
    "Ewe_parquet/data_ewe_audio_disjoint_v1"
)
REPORT = OUT / "cleaning_report"


def sha1_bytes(raw: bytes) -> str:
    return hashlib.sha1(raw).hexdigest()


def mp3_duration(raw: bytes) -> float:
    try:
        return float(MP3(io.BytesIO(raw)).info.length)
    except Exception:
        return -1.0


def schema_score(cols: set[str]) -> int:
    if "locale" in cols:
        return 2
    if "language" in cols:
        return 1
    return 0


def iter_index(files: list[Path], split: str):
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        cols = set(pf.schema_arrow.names)
        prefer = schema_score(cols)
        want = [
            c
            for c in [
                "audio",
                "transcription",
                "id",
                "speaker_id",
                "locale",
                "language",
                "gender",
                "age",
            ]
            if c in cols
        ]
        local_i = 0
        for batch in pf.iter_batches(columns=want, batch_size=64):
            d = batch.to_pydict()
            for i in range(len(d["audio"])):
                audio = d["audio"][i]
                raw = audio.get("bytes") if isinstance(audio, dict) else None
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                if not raw:
                    local_i += 1
                    continue
                dur = mp3_duration(raw) if split == "test" else None
                uid = None
                if "id" in d and d["id"][i] is not None:
                    uid = str(d["id"][i])
                elif isinstance(audio, dict) and audio.get("path"):
                    uid = str(audio["path"])
                yield {
                    "split": split,
                    "file": str(fpath),
                    "local_i": local_i,
                    "sha1": sha1_bytes(raw),
                    "prefer": prefer,
                    "dur": dur,
                    "uid": uid,
                    "nbytes": len(raw),
                }
                local_i += 1


def choose_keepers(index_rows: list[dict]) -> tuple[dict, dict]:
    by_sha: dict[str, list[dict]] = defaultdict(list)
    for r in index_rows:
        by_sha[r["sha1"]].append(r)

    split_rank = {"test": 0, "validation": 1, "train": 2}
    keep = {}
    stats = {
        "input_rows": len(index_rows),
        "unique_sha1": len(by_sha),
        "dropped_corrupt_test": 0,
        "dropped_as_leak_or_dup": 0,
        "kept_by_split": {"train": 0, "validation": 0, "test": 0},
        "reason_counts": defaultdict(int),
    }

    corrupt_sha1 = {
        r["sha1"]
        for r in index_rows
        if r["split"] == "test" and r["dur"] is not None and r["dur"] <= 0.0
    }
    stats["corrupt_test_sha1"] = len(corrupt_sha1)

    for h, rows in by_sha.items():
        if h in corrupt_sha1:
            stats["dropped_corrupt_test"] += len(rows)
            stats["reason_counts"]["corrupt_test_sha1"] += len(rows)
            continue

        best_split = min({r["split"] for r in rows}, key=lambda s: split_rank[s])
        candidates = [r for r in rows if r["split"] == best_split]
        candidates.sort(key=lambda r: (-r["prefer"], -r["nbytes"], r["file"], r["local_i"]))
        winner = candidates[0]
        keep[(winner["file"], winner["local_i"])] = winner
        stats["kept_by_split"][best_split] += 1
        dropped = len(rows) - 1
        stats["dropped_as_leak_or_dup"] += dropped
        if dropped:
            splits_present = {r["split"] for r in rows}
            if len(splits_present) > 1:
                stats["reason_counts"]["cross_split_audio_leak"] += dropped
            else:
                stats["reason_counts"]["within_split_or_cross_source_dup"] += dropped

    stats["reason_counts"] = dict(stats["reason_counts"])
    return keep, stats


def write_kept_split(
    files: list[Path],
    split: str,
    keep: dict,
    shard_size: int | None = 3000,
) -> tuple[int, set[str]]:
    kept_sha1: set[str] = set()
    n_kept = 0
    chunks: list[pd.DataFrame] = []
    chunk_rows = 0
    shard_i = 0

    def flush(force: bool = False) -> None:
        nonlocal chunks, chunk_rows, shard_i
        if not chunks:
            return
        if (
            split == "train"
            and shard_size is not None
            and not force
            and chunk_rows < shard_size
        ):
            return
        df = pd.concat(chunks, ignore_index=True)
        chunks = []
        chunk_rows = 0
        if split != "train":
            outp = OUT / (
                "ewe_validation_clean.parquet"
                if split == "validation"
                else "ewe_test_clean.parquet"
            )
            df.to_parquet(outp, index=False)
            print(f"  wrote {outp.name} n={len(df)}", flush=True)
            return
        # train shards
        for start in range(0, len(df), shard_size or len(df)):
            part = df.iloc[start : start + (shard_size or len(df))]
            outp = OUT / f"ewe-train-{shard_i:05d}.parquet"
            part.to_parquet(outp, index=False)
            print(f"  wrote {outp.name} n={len(part)}", flush=True)
            shard_i += 1

    for fpath in files:
        pf = pq.ParquetFile(fpath)
        local_i = 0
        for batch in pf.iter_batches(batch_size=128):
            df = batch.to_pandas()
            rows = []
            for i in range(len(df)):
                key = (str(fpath), local_i)
                local_i += 1
                if key not in keep:
                    continue
                row = df.iloc[i].to_dict()
                audio = row["audio"]
                raw = audio["bytes"]
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                    row["audio"] = {**audio, "bytes": raw}
                h = sha1_bytes(raw)
                if h in kept_sha1:
                    continue
                kept_sha1.add(h)
                row["audio_sha1"] = h
                rows.append(row)
                n_kept += 1
            if rows:
                cdf = pd.DataFrame(rows)
                # Harmonize dtypes across WAXAL/speechUG schemas.
                for col in ("speaker_id", "id", "locale", "language", "gender"):
                    if col in cdf.columns:
                        cdf[col] = cdf[col].map(
                            lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else str(x)
                        )
                if "age" in cdf.columns:
                    cdf["age"] = pd.to_numeric(cdf["age"], errors="coerce")
                chunks.append(cdf)
                chunk_rows += len(rows)
            if split == "train" and chunk_rows >= (shard_size or 3000) * 2:
                flush(force=False)
        print(f"  scanned {fpath.name} rows={local_i}", flush=True)

    flush(force=True)
    return n_kept, kept_sha1


def hours_of(pattern: str) -> float:
    total = 0.0
    for f in sorted(OUT.glob(pattern)):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(columns=["audio"], batch_size=64):
            for audio in batch.to_pydict()["audio"]:
                raw = audio["bytes"]
                if isinstance(raw, memoryview):
                    raw = raw.tobytes()
                d = mp3_duration(raw)
                if d > 0:
                    total += d
    return total / 3600.0


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    for p in list(OUT.glob("ewe-train-*.parquet")) + list(OUT.glob("ewe_*.parquet")):
        p.unlink()

    train_files = sorted(SRC.glob("ewe-train-*.parquet"))
    val_files = [SRC / "ewe_validation_clean.parquet"]
    test_files = [SRC / "ewe_test_clean.parquet"]

    print("=== Pass 1: index SHA1 ===", flush=True)
    index_rows: list[dict] = []
    for split, files in [
        ("train", train_files),
        ("validation", val_files),
        ("test", test_files),
    ]:
        print(f" indexing {split}...", flush=True)
        n0 = len(index_rows)
        index_rows.extend(iter_index(files, split))
        print(f"  +{len(index_rows) - n0}", flush=True)

    print("=== Choose keepers ===", flush=True)
    keep, stats = choose_keepers(index_rows)
    print(json.dumps(stats, indent=2), flush=True)

    corrupt = [
        r
        for r in index_rows
        if r["split"] == "test" and r["dur"] is not None and r["dur"] <= 0.0
    ]
    (REPORT / "excluded_corrupt_test.json").write_text(
        json.dumps(corrupt, indent=2, ensure_ascii=False)
    )

    print("=== Pass 2: rewrite ===", flush=True)
    n_train, sha_train = write_kept_split(train_files, "train", keep, shard_size=3000)
    n_val, sha_val = write_kept_split(val_files, "validation", keep, shard_size=None)
    n_test, sha_test = write_kept_split(test_files, "test", keep, shard_size=None)

    audit = {
        "train_n": n_train,
        "validation_n": n_val,
        "test_n": n_test,
        "train_cap_validation_sha1": len(sha_train & sha_val),
        "train_cap_test_sha1": len(sha_train & sha_test),
        "validation_cap_test_sha1": len(sha_val & sha_test),
        "unique_sha1_total": len(sha_train | sha_val | sha_test),
    }

    (OUT / "test_scored_ids.txt").write_text(
        "\n".join(f"ewe_test_{i:07d}" for i in range(n_test)) + "\n"
    )
    (OUT / "test_sha1.txt").write_text("\n".join(sorted(sha_test)) + "\n")
    (OUT / "train_sha1.txt").write_text("\n".join(sorted(sha_train)) + "\n")
    (OUT / "validation_sha1.txt").write_text("\n".join(sorted(sha_val)) + "\n")

    print("=== Hours ===", flush=True)
    hours_stats = {
        "train_h": round(hours_of("ewe-train-*.parquet"), 4),
        "validation_h": round(hours_of("ewe_validation_clean.parquet"), 4),
        "test_h": round(hours_of("ewe_test_clean.parquet"), 4),
    }
    hours_stats["total_h"] = round(sum(hours_stats.values()), 4)

    report = {
        "policy": {
            "sha1": "raw MP3 bytes",
            "split_priority": ["test", "validation", "train"],
            "schema_preference": "locale(speechUG) > language(WAXAL)",
            "corrupt_test": "duration<=0 dropped everywhere",
            "source_dir": str(SRC),
            "output_dir": str(OUT),
        },
        "selection_stats": stats,
        "audit_disjoint": audit,
        "hours": hours_stats,
        "train_shards": [p.name for p in sorted(OUT.glob("ewe-train-*.parquet"))],
        "elapsed_s": round(time.time() - t0, 1),
    }
    (REPORT / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    (OUT / "README_CLEANING.md").write_text(
        "\n".join(
            [
                "# Ewe ASR audio-hash-disjoint package v1",
                "",
                f"- Source: `{SRC}` (untouched)",
                f"- Output: `{OUT}`",
                "- One row per MP3 SHA1; priority test > validation > train.",
                "- Corrupt test MP3s (duration 0) removed.",
                "",
                "## Audit (must be 0)",
                f"- train ∩ validation SHA1: **{audit['train_cap_validation_sha1']}**",
                f"- train ∩ test SHA1: **{audit['train_cap_test_sha1']}**",
                f"- validation ∩ test SHA1: **{audit['validation_cap_test_sha1']}**",
                "",
                f"## Sizes: train={audit['train_n']} val={audit['validation_n']} test={audit['test_n']}",
                f"## Hours: {hours_stats}",
                "",
                "Retrain models before reporting metrics on this cleaned set.",
            ]
        )
        + "\n"
    )
    print("REPORT", json.dumps(report, indent=2), flush=True)
    assert audit["train_cap_validation_sha1"] == 0
    assert audit["train_cap_test_sha1"] == 0
    assert audit["validation_cap_test_sha1"] == 0
    print("ALL CHECKS PASSED", flush=True)


if __name__ == "__main__":
    main()
