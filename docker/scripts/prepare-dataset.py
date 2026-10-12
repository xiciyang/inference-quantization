#!/usr/bin/env python3
"""
校准/测试集准备 + 去泄漏（在量化容器内运行，产出由宿主独立复核）。

为什么放在容器里：宿主机没有 `datasets`，也不该为了量化往宿主装 Python 包
（serving 已确立"不在宿主装东西"的规则）。

去泄漏判据（用户裁决 #3 + 三）：**内容 sha256 交集为空**，不是索引不重叠。
同一段文本在数据集的 train/test 两个 split 里同时出现是很常见的，按索引判断会漏。

产出（写到 --out 目录，由调用方挂载）：
  calib.jsonl     校准样本（每行 {"h": 内容哈希, "t": 文本}）
  test.jsonl      测试样本
  lineage.json    血缘：来源 / revision / 种子 / 计数 / 两段哈希清单的摘要 + 完整清单
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time


def item_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def log(msg: str) -> None:
    """逐段进度输出 —— 无条件长静默会被当成卡死。"""
    print(f"[prepare-dataset] {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--calib", type=int, default=512)
    ap.add_argument("--test", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--revision", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    log(f"下载数据集 {args.repo} config={args.config} split={args.split}（镜像 {os.environ.get('HF_ENDPOINT')}）")
    t0 = time.time()
    try:
        from datasets import load_dataset
    except Exception as e:  # pragma: no cover
        print(json.dumps({"ok": False, "stage": "import", "error": str(e)}))
        return 3

    try:
        ds = load_dataset(args.repo, args.config, split=args.split, revision=args.revision)
    except Exception as e:
        print(json.dumps({"ok": False, "stage": "load_dataset", "error": str(e)[:600]}))
        return 4
    log(f"下载完成，{len(ds)} 条，用时 {time.time() - t0:.1f}s")

    # --- 取文本字段（不同数据集字段名不同）---
    text_key = None
    for k in ("text", "content", "raw_content", "document", "sentence"):
        if k in ds.column_names:
            text_key = k
            break
    if text_key is None:
        print(json.dumps({"ok": False, "stage": "field", "error": f"未找到文本字段，现有：{ds.column_names}"}))
        return 5
    log(f"文本字段 = {text_key}")

    # --- 归一化 + 内容哈希去重 ---
    log("归一化并计算内容哈希…")
    seen = {}
    duplicates = 0
    for i, row in enumerate(ds):
        t = row.get(text_key) or ""
        t = t.strip()
        if len(t) < args.min_chars:
            continue
        h = item_hash(t)
        if h in seen:
            duplicates += 1
            continue
        seen[h] = t
        if (i + 1) % 20000 == 0:
            log(f"  已扫描 {i + 1} 条，去重后 {len(seen)} 条")
    log(f"去重完成：唯一 {len(seen)} 条，丢弃重复 {duplicates} 条")

    if len(seen) < args.calib + args.test:
        print(json.dumps({"ok": False, "stage": "size",
                          "error": f"去重后仅 {len(seen)} 条，不足校准 {args.calib} + 测试 {args.test}"}))
        return 6

    # --- 按种子切分（全集排序后确定性打乱，保证同种子同结果）---
    log(f"按种子 {args.seed} 切分：校准 {args.calib} + 测试 {args.test}")
    hashes = sorted(seen.keys())
    rng = random.Random(args.seed)
    rng.shuffle(hashes)
    calib_h = hashes[: args.calib]
    test_h = hashes[args.calib : args.calib + args.test]

    # --- 去泄漏断言（容器内先自己判一次）---
    inter = set(calib_h) & set(test_h)
    log(f"去泄漏自检：交集 {len(inter)} 条")
    if inter:
        print(json.dumps({"ok": False, "stage": "leakage", "error": f"校准与测试有 {len(inter)} 条内容重复"}))
        return 7

    def write_jsonl(path, hs):
        with open(path, "w", encoding="utf-8") as f:
            for h in hs:
                f.write(json.dumps({"h": h, "t": seen[h]}, ensure_ascii=False) + "\n")

    write_jsonl(os.path.join(args.out, "calib.jsonl"), calib_h)
    write_jsonl(os.path.join(args.out, "test.jsonl"), test_h)

    def list_digest(hs):
        return hashlib.sha256("\n".join(hs).encode("utf-8")).hexdigest()

    lineage = {
        "formatVersion": 1,
        "source": {
            "repo": args.repo, "config": args.config, "split": args.split,
            "revision": args.revision, "mirror": os.environ.get("HF_ENDPOINT"),
            "rowsInSplit": len(ds), "textField": text_key,
        },
        "dedup": {"minChars": args.min_chars, "unique": len(seen), "dropped": duplicates},
        "split": {
            "seed": args.seed, "algorithm": "sorted-sha256-keys + Random(seed).shuffle",
            "calibration": {"count": len(calib_h), "listDigest": list_digest(calib_h), "hashes": calib_h},
            "test": {"count": len(test_h), "listDigest": list_digest(test_h), "hashes": test_h},
        },
        "leakage": {
            "criterion": "content-sha256-intersection-empty",
            "intersection": 0, "clean": True,
            "unionEqualsDeduped": len(set(calib_h) | set(test_h)) == len(seen) and len(calib_h) + len(test_h) == len(set(calib_h) | set(test_h)),
        },
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(os.path.join(args.out, "lineage.json"), "w", encoding="utf-8") as f:
        json.dump(lineage, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        "ok": True, "calibration": len(calib_h), "test": len(test_h),
        "unique": len(seen), "dropped": duplicates,
        "calibListDigest": lineage["split"]["calibration"]["listDigest"],
        "testListDigest": lineage["split"]["test"]["listDigest"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
