#!/usr/bin/env python3
"""
精度评测（testing-core 契约的执行端，容器内跑）。

为什么用**直接模型打分**而不是走 HTTP API：
  精度损失是"权重变了多少"的直接后果，直接算 loss 最精确；
  走 API 会掺进采样、停止条件、序列化等噪声。
  性能评测则相反 —— 必须走真实服务端点（见 src/testing/perf.mjs）。

指标：**平均负对数似然（mean NLL）** 与 **困惑度（perplexity = exp(mean NLL)）**。
两者都按 token 加权（不是按样本平均），否则长文本会被短文本淹没。

逐条 NLL 会全部写出，使调用方可以用同一个产物做"抽样粗筛"，不必重跑。

--------------------------------------------------------------------------------
测量成立性（validity）—— 2026-09-23 真机事故后补建
--------------------------------------------------------------------------------
事故：一份**下载中断**的源模型（model.safetensors 只有前 0.747 GiB 是真数据，
其余 2.128 GiB 全为 0x00，含 model.norm.weight）跑出了 meanNll = ln(vocab_size)
= 11.9312144574、perplexity = 151935.97 —— 即"输出均匀分布"。基线与候选**逐位相同**，
"精度损失 0%"被当成"无损"签了收。这个数在数学上是对的，在语义上毫无意义。

因此本脚本在给出指标的同时，**必须同时给出"这次测量是否成立"的证据**：

  A. 权重健康度：整块为零 / 非有限值的大张量（权重没真的加载进来的直接证据）
  B. 输出健康度：logits 在词表维度上的标准差、argmax 去重后的个数
                 （均匀分布 ⇒ std=0、distinct=1）
  C. 退化签名：|meanNll - ln(vocab_size)| ≤ uniformTol  ⇒ 输出均匀分布
  D. 逐条方差：所有样本的 per-item meanNll 完全相同 ⇒ 与输入无关，不构成测量

判定与测量**刻意分离**：本脚本只报事实（ok 仍为 true，数值照给），
"要不要据此拒绝签收"由调用方的门禁决定（src/gate.mjs 的 assertMeasurementValid）。
这样"测量"这一层保持可被独立复核，而"判定"这一层可以被穷举测试。
"""
import argparse
import json
import math
import os
import sys
import time

# 输出均匀分布的判定容差（nats）。float32 上 ln(151936) 的 ULP ≈ 9.5e-7，
# 取 1e-4 留足累积误差余量，同时远小于真实模型的逐条 loss 散布（实测 1.98~3.66）。
UNIFORM_TOL = 1e-4

# 只对元素数达到此下限的张量做"整块为零"判定：
# 长度 1 的 bias 恰好为 0 在真实模型里常见；1536×8960 的权重矩阵全 0 不可能是有意为之。
ZERO_MIN_ELEMENTS = 1024
ZERO_MAX_NAMES = 32


def log(msg):
    print(f"[parity] {msg}", flush=True)


def param_health(model):
    """权重健康度：整块为零 / 含非有限值的大张量。

    这是"权重到底有没有真的加载进来"最直接的证据 —— 比 loss 数值更靠近现场。
    """
    zero_names = []
    nonfinite_names = []
    zero_count = 0
    nonfinite_count = 0
    tensor_count = 0
    param_count = 0
    total_bytes = 0
    skipped_small = 0
    t0 = time.time()
    try:
        items = list(model.state_dict().items())
    except Exception as e:  # 极端情况下 state_dict 不可用，不许因此让整次测量失败
        return {"available": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}

    import torch
    for name, t in items:
        try:
            if not torch.is_tensor(t) or t.numel() == 0:
                continue
            tensor_count += 1
            param_count += int(t.numel())
            total_bytes += int(t.numel()) * int(t.element_size())
            if t.is_floating_point() and not bool(torch.isfinite(t).all()):
                nonfinite_count += 1
                if len(nonfinite_names) < ZERO_MAX_NAMES:
                    nonfinite_names.append(name)
            if t.numel() < ZERO_MIN_ELEMENTS:
                skipped_small += 1
                continue
            if bool((t == 0).all()):
                # 计数**不设上限**，只有"点名清单"截断 —— 否则 271 个零张量会被报成 32 个，
                # 严重程度被低估。清单截断的事实由 zeroTensorNamesTruncated 显式标出。
                zero_count += 1
                if len(zero_names) < ZERO_MAX_NAMES:
                    zero_names.append(name)
        except Exception:
            continue

    return {
        "available": True,
        "tensorCount": tensor_count,
        "paramCount": param_count,
        "paramBytes": total_bytes,
        "paramGiB": round(total_bytes / 2 ** 30, 4),
        "zeroTensorCount": zero_count,
        "zeroTensorNames": zero_names,
        "zeroTensorNamesTruncated": zero_count > len(zero_names),
        "nonFiniteTensorCount": nonfinite_count,
        "nonFiniteTensorNames": nonfinite_names,
        "zeroMinElements": ZERO_MIN_ELEMENTS,
        "smallTensorsSkipped": skipped_small,
        "scanSec": round(time.time() - t0, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", required=True, help="test.jsonl（每行 {h,t}）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--label", default="model")
    ap.add_argument("--skip-param-health", action="store_true",
                    help="跳过权重健康度扫描（仅用于超大模型的省时场景；会削弱测量成立性证据）")
    args = ap.parse_args()

    t0 = time.time()

    def emit(payload, code):
        try:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(json.dumps(payload, ensure_ascii=False)[:2000])
        return code

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        return emit({"ok": False, "stage": "import", "error": str(e)}, 10)

    items = []
    with open(args.test, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line)["t"])
    if args.limit and args.limit > 0:
        items = items[: args.limit]
    if not items:
        return emit({"ok": False, "stage": "data", "error": "测试集为空"}, 11)
    log(f"{args.label}: 测试样本 {len(items)} 条，模型 {args.model}")

    try:
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="cuda" if torch.cuda.is_available() else "cpu",
            trust_remote_code=False,
        )
        model.eval()
    except Exception as e:
        return emit({"ok": False, "stage": "load", "error": f"{type(e).__name__}: {str(e)[:600]}"}, 12)
    log(f"{args.label}: 模型已加载（{time.time() - t0:.1f}s）")

    vocab_size = None
    try:
        vocab_size = int(getattr(model.config, "vocab_size", 0)) or None
    except Exception:
        vocab_size = None

    # ---- A. 权重健康度 ----
    if args.skip_param_health:
        health = {"available": False, "skipped": True,
                  "reason": "--skip-param-health：本档位未采集权重健康度证据"}
    else:
        health = param_health(model)
        log(f"{args.label}: 权重健康度 张量 {health.get('tensorCount')} / 整块为零 "
            f"{health.get('zeroTensorCount')} / 非有限 {health.get('nonFiniteTensorCount')} "
            f"（{health.get('scanSec')}s）")

    per_item = []
    total_nll = 0.0
    total_tokens = 0
    logit_probe = None
    with torch.no_grad():
        for idx, text in enumerate(items):
            try:
                ids = tok(text, return_tensors="pt", truncation=True, max_length=args.max_len).input_ids
                if ids.shape[1] < 8:
                    per_item.append({"i": idx, "tokens": 0, "nll": None, "skipped": "too-short"})
                    continue
                ids = ids.to(model.device)
                out = model(ids, labels=ids)
                n = ids.shape[1] - 1
                # HF 的 loss 已经按 token 平均（ignore_index 之外）
                nll_sum = float(out.loss) * n
                per_item.append({"i": idx, "tokens": n, "nll": nll_sum, "meanNll": float(out.loss)})
                total_nll += nll_sum
                total_tokens += n

                # ---- B. 输出健康度：只在第一条上探一次，够用且不拖慢 ----
                if logit_probe is None:
                    try:
                        lg = out.logits[0].float()                     # [T, V]
                        std_per_pos = lg.std(dim=-1)                   # [T]
                        argmax = lg.argmax(dim=-1)                     # [T]
                        logit_probe = {
                            "sampledAt": idx,
                            "positions": int(lg.shape[0]),
                            "vocabSize": int(lg.shape[1]),
                            "logitsStdMean": float(std_per_pos.mean()),
                            "logitsStdMin": float(std_per_pos.min()),
                            "logitsStdMax": float(std_per_pos.max()),
                            "distinctArgmax": int(torch.unique(argmax).numel()),
                            "logitsAbsMax": float(lg.abs().max()),
                        }
                    except Exception as e:
                        logit_probe = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            except Exception as e:
                per_item.append({"i": idx, "tokens": 0, "nll": None, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            if (idx + 1) % 25 == 0:
                log(f"{args.label}: {idx + 1}/{len(items)}，累计 token {total_tokens}")

    if total_tokens == 0:
        return emit({"ok": False, "stage": "score", "error": "没有任何样本成功评分"}, 13)

    mean_nll = total_nll / total_tokens

    # ---- C/D. 退化签名 ----
    uniform_loss = math.log(vocab_size) if vocab_size else None
    uniform_gap = abs(mean_nll - uniform_loss) if uniform_loss is not None else None

    scored_means = [p["meanNll"] for p in per_item if p.get("tokens")]
    item_std = None
    if len(scored_means) > 1:
        m = sum(scored_means) / len(scored_means)
        item_std = math.sqrt(sum((x - m) ** 2 for x in scored_means) / (len(scored_means) - 1))

    reasons = []
    if health.get("available") and health.get("zeroTensorCount", 0) > 0:
        reasons.append({
            "code": "zero-weight-tensors",
            "detail": f"{health['zeroTensorCount']} 个大张量整块为零（如 {', '.join(health['zeroTensorNames'][:3])}）—— 权重没有真的加载进来",
        })
    if health.get("available") and health.get("nonFiniteTensorCount", 0) > 0:
        reasons.append({
            "code": "non-finite-weights",
            "detail": f"{health['nonFiniteTensorCount']} 个张量含 NaN/Inf",
        })
    if uniform_gap is not None and uniform_gap <= UNIFORM_TOL:
        reasons.append({
            "code": "uniform-output",
            "detail": f"meanNll={mean_nll:.10f} 与 ln(vocab_size)={uniform_loss:.10f} 相差 {uniform_gap:.2e} ≤ {UNIFORM_TOL} "
                      f"—— 模型输出近似均匀分布，\"精度损失\"在此没有区分力",
        })
    if logit_probe and not logit_probe.get("error"):
        if logit_probe["logitsStdMean"] == 0.0:
            reasons.append({"code": "constant-logits", "detail": "logits 在词表维度上的标准差为 0（逐位相同）"})
        if logit_probe["distinctArgmax"] <= 1:
            reasons.append({"code": "constant-argmax", "detail": f"整条序列的 argmax 只有 {logit_probe['distinctArgmax']} 个不同取值"})
    if item_std is not None and item_std == 0.0:
        reasons.append({"code": "no-item-variance", "detail": "所有样本的逐条 meanNll 完全相同 —— 与输入无关，不构成一次测量"})

    validity = {
        "valid": len(reasons) == 0,
        "reasons": reasons,
        "uniformTol": UNIFORM_TOL,
        "checks": {
            "weightsAvailable": bool(health.get("available")),
            "paramGiB": health.get("paramGiB"),
            "zeroTensorCount": health.get("zeroTensorCount"),
            "nonFiniteTensorCount": health.get("nonFiniteTensorCount"),
            "vocabSize": vocab_size,
            "uniformLoss": uniform_loss,
            "uniformGap": uniform_gap,
            "itemMeanNllStd": item_std,
            "logitProbe": logit_probe,
        },
    }

    if not validity["valid"]:
        log(f"{args.label}: ⚠ 测量不成立 —— {[r['code'] for r in reasons]}")
    else:
        log(f"{args.label}: 测量成立性检查通过")

    payload = {
        "ok": True,
        "label": args.label,
        "model": args.model,
        "samples": len(items),
        "scoredSamples": sum(1 for p in per_item if p.get("tokens")),
        "totalTokens": total_tokens,
        "meanNll": mean_nll,
        "perplexity": math.exp(min(mean_nll, 50)),
        "durationSec": round(time.time() - t0, 1),
        "limit": args.limit,
        "maxLen": args.max_len,
        "vocabSize": vocab_size,
        "weightHealth": health,
        "validity": validity,
        "perItem": per_item,
    }
    log(f"{args.label}: meanNll={mean_nll:.4f} ppl={payload['perplexity']:.3f} tokens={total_tokens}")
    return emit(payload, 0)


if __name__ == "__main__":
    sys.exit(main())
