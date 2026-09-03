#!/usr/bin/env python3
"""Small, dependency-free concurrency benchmark for OpenAI-compatible chat APIs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ALLOWED_CONCURRENCY = (5, 10, 80)


@dataclass
class RequestResult:
    request_id: int
    ok: bool
    latency_s: float
    ttft_s: float | None
    tpot_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "压测 OpenAI 兼容的 /v1/chat/completions；并发只允许 5、10、80。"
        )
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://127.0.0.1:9108"),
        help="服务根地址，默认读取 API_BASE_URL。",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SERVED_MODEL_NAME", "DeepSeek-V4-Flash"),
        help="served model name，默认读取 SERVED_MODEL_NAME。",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", os.getenv("EVAL_API_KEY", "EMPTY")),
        help="API key；只用于请求头，不写入结果文件。",
    )
    parser.add_argument(
        "--concurrency",
        nargs="+",
        type=int,
        choices=ALLOWED_CONCURRENCY,
        default=list(ALLOWED_CONCURRENCY),
        help="依次测试的并发档位，默认：5 10 80。",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="每个档位总请求数；0 表示 max(20, concurrency)。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.getenv("BENCH_OUTPUT_LEN", "128")),
        help="每个请求最大输出 token 数。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="单请求超时秒数；并发 80 在单发服务上会长时间排队。",
    )
    parser.add_argument(
        "--prompt",
        default="请用中文解释大型语言模型中混合专家网络的路由机制，并给出三个要点。",
        help="基础测试提示词。",
    )
    parser.add_argument("--prompt-file", help="从 UTF-8 文本文件读取提示词。")
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="采样温度，默认 0。"
    )
    parser.add_argument(
        "--allow-eos",
        action="store_true",
        help="允许模型提前遇到 EOS；默认 ignore_eos=true 以固定输出长度。",
    )
    parser.add_argument(
        "--same-prompt",
        action="store_true",
        help="所有请求完全相同；默认追加请求编号，避免整条请求完全重复。",
    )
    parser.add_argument(
        "--skip-warmup", action="store_true", help="跳过每个档位前的 1 次预热。"
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getenv("RESULT_ROOT", "results"), "concurrency"),
        help="JSON 结果目录。",
    )
    args = parser.parse_args()
    if args.requests < 0:
        parser.error("--requests 不能小于 0")
    if args.max_tokens <= 0:
        parser.error("--max-tokens 必须大于 0")
    return args


def endpoint_from_base(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(percent / 100.0 * len(ordered)))
    return ordered[rank - 1]


def send_request(
    *,
    endpoint: str,
    model: str,
    api_key: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    ignore_eos: bool,
    timeout: float,
    request_id: int,
) -> RequestResult:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": ignore_eos,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    usage: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                chunk = json.loads(data)
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    emitted = delta.get("content") or delta.get("reasoning_content")
                    if emitted and first_token_at is None:
                        first_token_at = time.perf_counter()
    except urllib.error.HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        elapsed = time.perf_counter() - started
        return RequestResult(
            request_id, False, elapsed, None, None, None, None, None,
            f"HTTP {exc.code}: {detail}",
        )
    except Exception as exc:  # noqa: BLE001 - benchmark must record each failure
        elapsed = time.perf_counter() - started
        return RequestResult(
            request_id, False, elapsed, None, None, None, None, None,
            f"{type(exc).__name__}: {exc}",
        )

    finished = time.perf_counter()
    latency = finished - started
    ttft = first_token_at - started if first_token_at is not None else None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    tpot_ms: float | None = None
    if (
        ttft is not None
        and isinstance(completion_tokens, int)
        and completion_tokens > 1
    ):
        tpot_ms = max(0.0, latency - ttft) * 1000.0 / (completion_tokens - 1)

    return RequestResult(
        request_id=request_id,
        ok=True,
        latency_s=latency,
        ttft_s=ttft,
        tpot_ms=tpot_ms,
        prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
        completion_tokens=(
            completion_tokens if isinstance(completion_tokens, int) else None
        ),
        total_tokens=total_tokens if isinstance(total_tokens, int) else None,
        error=None,
    )


def metric_block(values: list[float], scale: float = 1.0) -> dict[str, float | None]:
    scaled = [value * scale for value in values]
    return {
        "mean": statistics.fmean(scaled) if scaled else None,
        "p50": percentile(scaled, 50),
        "p95": percentile(scaled, 95),
        "p99": percentile(scaled, 99),
    }


def summarize(
    concurrency: int,
    wall_s: float,
    results: list[RequestResult],
) -> dict[str, Any]:
    successful = [item for item in results if item.ok]
    failed = [item for item in results if not item.ok]
    usage_complete = all(
        item.prompt_tokens is not None
        and item.completion_tokens is not None
        and item.total_tokens is not None
        for item in successful
    ) and bool(successful)
    prompt_tokens = (
        sum(item.prompt_tokens or 0 for item in successful) if usage_complete else None
    )
    completion_tokens = (
        sum(item.completion_tokens or 0 for item in successful) if usage_complete else None
    )
    total_tokens = (
        sum(item.total_tokens or 0 for item in successful) if usage_complete else None
    )
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "wall_time_s": wall_s,
        "request_throughput_rps": len(successful) / wall_s if wall_s else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "output_token_throughput_tps": (
            completion_tokens / wall_s
            if completion_tokens is not None and wall_s
            else None
        ),
        "total_token_throughput_tps": (
            total_tokens / wall_s if total_tokens is not None and wall_s else None
        ),
        "latency_ms": metric_block([item.latency_s for item in successful], 1000.0),
        "ttft_ms": metric_block(
            [item.ttft_s for item in successful if item.ttft_s is not None], 1000.0
        ),
        "tpot_ms": metric_block(
            [item.tpot_ms for item in successful if item.tpot_ms is not None]
        ),
        "usage_complete": usage_complete,
        "errors": [item.error for item in failed[:20]],
    }


def format_number(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def print_summary(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(
        f"并发={summary['concurrency']}  请求={summary['requests']}  "
        f"成功={summary['successful']}  失败={summary['failed']}  "
        f"墙钟时间={summary['wall_time_s']:.2f}s"
    )
    print(
        "请求吞吐="
        f"{format_number(summary['request_throughput_rps'])} req/s  "
        "输出吞吐="
        f"{format_number(summary['output_token_throughput_tps'])} tok/s  "
        "总吞吐="
        f"{format_number(summary['total_token_throughput_tps'])} tok/s"
    )
    print(
        "E2E(ms) "
        f"p50={format_number(summary['latency_ms']['p50'])} "
        f"p95={format_number(summary['latency_ms']['p95'])} "
        f"p99={format_number(summary['latency_ms']['p99'])}"
    )
    print(
        "TTFT(ms) "
        f"p50={format_number(summary['ttft_ms']['p50'])} "
        f"p95={format_number(summary['ttft_ms']['p95'])} "
        f"p99={format_number(summary['ttft_ms']['p99'])}"
    )
    print(
        "TPOT(ms) "
        f"p50={format_number(summary['tpot_ms']['p50'])} "
        f"p95={format_number(summary['tpot_ms']['p95'])} "
        f"p99={format_number(summary['tpot_ms']['p99'])}"
    )
    if not summary["usage_complete"]:
        print(
            "[WARN] 服务没有为所有成功请求返回 usage；tok/s 显示 N/A，"
            "不要用 SSE chunk 数冒充 token 数。"
        )
    for error in summary["errors"][:3]:
        print(f"[ERROR] {error}")


def main() -> int:
    args = parse_args()
    endpoint = endpoint_from_base(args.base_url)
    base_prompt = (
        Path(args.prompt_file).read_text(encoding="utf-8")
        if args.prompt_file
        else args.prompt
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_summaries: list[dict[str, Any]] = []
    any_failed = False

    print(f"endpoint={endpoint}")
    print(f"model={args.model}")
    print(f"concurrency={args.concurrency}")
    print("说明：输出 tok/s 是所有成功请求的 completion_tokens / 墙钟时间。")

    for concurrency in args.concurrency:
        total_requests = args.requests or max(20, concurrency)
        if total_requests < concurrency:
            print(
                f"[WARN] requests={total_requests} 小于 concurrency={concurrency}，"
                "无法占满该并发档位。",
                file=sys.stderr,
            )
        if not args.skip_warmup:
            print(f"[WARMUP] concurrency={concurrency}")
            warmup = send_request(
                endpoint=endpoint,
                model=args.model,
                api_key=args.api_key,
                prompt=base_prompt + "\n这是预热请求。",
                max_tokens=min(args.max_tokens, 32),
                temperature=args.temperature,
                ignore_eos=not args.allow_eos,
                timeout=args.timeout,
                request_id=-1,
            )
            if not warmup.ok:
                print(f"[FAIL] 预热失败：{warmup.error}", file=sys.stderr)
                return 2

        print(
            f"[RUN] concurrency={concurrency} requests={total_requests} "
            f"max_tokens={args.max_tokens}"
        )
        started = time.perf_counter()
        results: list[RequestResult] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = []
            for request_id in range(total_requests):
                prompt = base_prompt
                if not args.same_prompt:
                    prompt += f"\n请求编号：{request_id}。"
                futures.append(
                    executor.submit(
                        send_request,
                        endpoint=endpoint,
                        model=args.model,
                        api_key=args.api_key,
                        prompt=prompt,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        ignore_eos=not args.allow_eos,
                        timeout=args.timeout,
                        request_id=request_id,
                    )
                )
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        wall_s = time.perf_counter() - started
        results.sort(key=lambda item: item.request_id)
        summary = summarize(concurrency, wall_s, results)
        all_summaries.append(summary)
        print_summary(summary)
        any_failed = any_failed or bool(summary["failed"])

        output = {
            "created_at": datetime.now().astimezone().isoformat(),
            "endpoint": endpoint,
            "model": args.model,
            "config": {
                "concurrency": concurrency,
                "requests": total_requests,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "ignore_eos": not args.allow_eos,
                "same_prompt": args.same_prompt,
            },
            "summary": summary,
            "requests": [asdict(item) for item in results],
        }
        result_path = output_dir / f"bench-{timestamp}-c{concurrency}.json"
        result_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"结果文件：{result_path}")

    print("\n汇总：")
    print("并发\t成功/总数\t输出 tok/s\t总 tok/s\tTTFT p99(ms)\tE2E p99(ms)")
    for summary in all_summaries:
        print(
            f"{summary['concurrency']}\t"
            f"{summary['successful']}/{summary['requests']}\t"
            f"{format_number(summary['output_token_throughput_tps'])}\t"
            f"{format_number(summary['total_token_throughput_tps'])}\t"
            f"{format_number(summary['ttft_ms']['p99'])}\t"
            f"{format_number(summary['latency_ms']['p99'])}"
        )
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
