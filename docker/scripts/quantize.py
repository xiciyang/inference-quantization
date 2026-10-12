#!/usr/bin/env python3
"""
单档量化（在量化容器内运行）。

输入：源模型目录（只读挂载）+ 校准集 jsonl
输出：量化后的独立目录 + quant-result.json

设计要点：
  * **源目录只读**：只读挂载，且本脚本从不写回源目录。
  * **逐段进度输出**：无条件长静默会被当成卡死。
  * **方法表可扩展**：新增量化方法 = 在 RECIPES 里加一项，不改流程（题目"量化方法插件化"）。
  * **失败即结构化返回**：非零退出 + JSON 错误，不靠猜日志。
"""
import argparse
import json
import os
import sys
import time

METHODS = ("awq-w4a16", "int8-w8a8", "fp8-dynamic", "w8a16")


def log(msg: str) -> None:
    print(f"[quantize] {msg}", flush=True)


def build_recipe(method: str):
    """
    返回 (recipe, 说明)。核心流程不认识具体方法，只拿一个 recipe —— 与 JS 侧的
    backends 注册表同构：新方法在这里加一行，流程零改动。
    """
    from llmcompressor.modifiers.quantization import QuantizationModifier
    ignore = ["lm_head"]

    if method == "awq-w4a16":
        # AWQ 是**变换器**（weight smoothing，需校准数据），量化本身由 QuantizationModifier 承担。
        # 这就是 "AWQ + INT4" 的语义，且全程纯 Python（aarch64 可用）。
        from llmcompressor.modifiers.transform.awq import AWQModifier
        return [
            AWQModifier(),
            QuantizationModifier(targets="Linear", scheme="W4A16", ignore=ignore),
        ], "AWQ transform + W4A16 group-wise（默认档）"

    if method == "int8-w8a8":
        return [QuantizationModifier(targets="Linear", scheme="INT8", ignore=ignore)], "INT8 W8A8（第 2 档）"

    if method == "fp8-dynamic":
        return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ignore)], "FP8 dynamic（第 3 档）"

    if method == "w8a16":
        return [QuantizationModifier(targets="Linear", scheme="W8A16", ignore=ignore)], "W8A16 仅权重 8bit（第 4 档）"

    raise ValueError(f"未知量化方法：{method}；可用：{METHODS}")


def load_calibration(path: str, limit: int):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if len(items) >= limit:
                break
    # llmcompressor 的 oneshot 走的是 HF datasets 管线（要 .column_names），
    # 传纯 list 会报 'list' object has no attribute 'column_names' —— 实测踩过。
    rows = [{"text": it["t"]} for it in items]
    try:
        from datasets import Dataset
        return Dataset.from_list(rows)
    except Exception:
        return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--result", default=None)
    args = ap.parse_args()

    result_path = args.result or os.path.join(args.out, "..", "quant-result.json")
    t_start = time.time()

    def emit(payload, code):
        print(json.dumps(payload, ensure_ascii=False))
        try:
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return code

    # 安全阀：绝不写到源目录里或源目录本身
    src = os.path.realpath(args.model)
    dst = os.path.realpath(args.out)
    if dst == src or dst.startswith(src + os.sep):
        return emit({"ok": False, "stage": "guard",
                     "error": f"输出目录不得等于或位于源目录内部：src={src} dst={dst}"}, 10)
    if os.path.exists(dst) and os.listdir(dst):
        log(f"输出目录已存在且非空，将被覆写：{dst}")
    os.makedirs(dst, exist_ok=True)

    log(f"方法 = {args.method}，源 = {src}，输出 = {dst}")
    log(f"校准样本 = {args.samples}（来自 {args.calib}），max_seq_len = {args.max_seq_len}")

    try:
        import torch
        from llmcompressor import oneshot
    except Exception as e:
        return emit({"ok": False, "stage": "import", "error": str(e)}, 11)

    try:
        recipe, desc = build_recipe(args.method)
    except Exception as e:
        return emit({"ok": False, "stage": "recipe", "error": str(e)}, 12)
    log(f"配方：{desc}")

    try:
        calib = load_calibration(args.calib, args.samples)
    except Exception as e:
        return emit({"ok": False, "stage": "calibration", "error": str(e)}, 13)
    if len(calib) < 16:
        return emit({"ok": False, "stage": "calibration",
                     "error": f"校准样本仅 {len(calib)} 条，过少（<16），拒绝量化"}, 14)
    log(f"校准集载入 {len(calib)} 条")

    gpu_free = None
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu_free = free
            log(f"GPU 可用：free {free / 1024 ** 3:.1f} GiB / total {total / 1024 ** 3:.1f} GiB")
        else:
            log("警告：CUDA 不可用，将在 CPU 上量化（会非常慢）")
    except Exception as e:
        log(f"GPU 探测异常：{e}")

    log("开始 oneshot 量化（这是最耗时的一步）…")
    try:
        oneshot(
            model=src,
            dataset=calib,
            recipe=recipe,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=len(calib),
            output_dir=dst,
            save_compressed=True,
        )
    except TypeError as e:
        # 不同版本的 oneshot 形参名可能不同，退化为关键字最小集
        log(f"oneshot 形参不匹配（{e}），尝试最小参数集")
        try:
            oneshot(model=src, dataset=calib, recipe=recipe, output_dir=dst)
        except Exception as e2:
            return emit({"ok": False, "stage": "oneshot", "error": f"{type(e2).__name__}: {str(e2)[:500]}"}, 15)
    except Exception as e:
        return emit({"ok": False, "stage": "oneshot", "error": f"{type(e).__name__}: {str(e)[:500]}"}, 15)

    log(f"量化完成，用时 {time.time() - t_start:.1f}s。检查产出…")

    # --- 产出自检（缺什么就说清楚，不要"跑完了"就算成功）---
    checks = {}
    import glob
    cfg_path = os.path.join(dst, "config.json")
    quant_cfg = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        quant_cfg = cfg.get("quantization_config")
    checks["hasConfig"] = os.path.exists(cfg_path)
    checks["hasQuantizationConfig"] = quant_cfg is not None
    weights = glob.glob(os.path.join(dst, "*.safetensors")) + glob.glob(os.path.join(dst, "*.bin"))
    checks["weightFileCount"] = len(weights)
    checks["outputBytes"] = sum(os.path.getsize(w) for w in weights)
    checks["scheme"] = (quant_cfg or {}).get("quant_method")

    payload = {
        "ok": checks["hasQuantizationConfig"] and checks["weightFileCount"] > 0,
        "stage": "done",
        "method": args.method,
        "recipeDesc": desc,
        "src": src, "dst": dst,
        "calibrationSamples": len(calib),
        "maxSeqLen": args.max_seq_len,
        "seed": args.seed,
        "durationSec": round(time.time() - t_start, 1),
        "checks": checks,
        "quantizationConfig": quant_cfg,
        "gpuFreeBytesAtStart": gpu_free,
        "toolchain": {
            "llmcompressor": __import__("llmcompressor").__version__,
            "compressed_tensors": __import__("compressed_tensors").__version__,
            "transformers": __import__("transformers").__version__,
            "torch": torch.__version__,
        },
    }
    log(f"产出检查：{json.dumps(checks, ensure_ascii=False)}")
    return emit(payload, 0 if payload["ok"] else 16)


if __name__ == "__main__":
    sys.exit(main())
