"""Concurrent load benchmark for the inference API.

Usage:
    make run                    # terminal 1: start the server
    make benchmark-concurrent   # terminal 2: run this script
"""

import argparse
import asyncio
import statistics
import sys
import time
from collections import Counter

import httpx

BASE_URL = "http://127.0.0.1:8000"
API_URL = f"{BASE_URL}/api/v1/inference"
IMAGE_PATH = "tests/fixtures/streetAndpeople.jpg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concurrent inference benchmark")
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model", default="yolo11")
    parser.add_argument("--image", default=IMAGE_PATH)
    return parser.parse_args()


def percentile(sorted_values: list[float], pct: float) -> float:
    index = min(int(len(sorted_values) * pct / 100), len(sorted_values) - 1)
    return sorted_values[index]


async def send_request(
        client: httpx.AsyncClient,
        image_bytes: bytes,
        model: str,
) -> tuple[float, int]:
    """Send one request. Returns (latency_ms, server_batch_size)."""

    started = time.perf_counter()

    response = await client.post(
        API_URL,
        params={"model": model},  # the API reads `model` from the query string
        files={"file": ("image.jpg", image_bytes, "image/jpeg")},
    )

    latency_ms = (time.perf_counter() - started) * 1000

    response.raise_for_status()

    return latency_ms, int(response.json().get("batch_size", 1))


async def worker(
        client: httpx.AsyncClient,
        image_bytes: bytes,
        model: str,
        queue: asyncio.Queue[int],
        latencies: list[float],
        batch_sizes: list[int],
        errors: Counter[str],
) -> None:
    while True:
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return  # no work left -> worker finishes cleanly

        try:
            latency, batch_size = await send_request(client, image_bytes, model)
            latencies.append(latency)
            batch_sizes.append(batch_size)
        except httpx.HTTPStatusError as exc:
            errors[f"HTTP {exc.response.status_code}"] += 1
        except httpx.HTTPError as exc:
            errors[type(exc).__name__] += 1
        finally:
            queue.task_done()


async def check_server(client: httpx.AsyncClient) -> None:
    try:
        response = await client.get(f"{BASE_URL}/health")
        response.raise_for_status()
    except httpx.HTTPError:
        print(f"ERROR: API is not reachable at {BASE_URL}.")
        print("Start it first in another terminal:  make run")
        sys.exit(1)


async def benchmark(args: argparse.Namespace) -> None:
    with open(args.image, "rb") as file:
        image_bytes = file.read()

    async with httpx.AsyncClient(timeout=60.0) as client:
        await check_server(client)

        # Warm-up: first calls load weights/kernels and are much slower.
        for _ in range(args.warmup):
            await send_request(client, image_bytes, args.model)

        queue: asyncio.Queue[int] = asyncio.Queue()
        for index in range(args.requests):
            queue.put_nowait(index)

        latencies: list[float] = []
        batch_sizes: list[int] = []
        errors: Counter[str] = Counter()

        started = time.perf_counter()

        await asyncio.gather(
            *[
                worker(client, image_bytes, args.model, queue, latencies, batch_sizes, errors)
                for _ in range(args.concurrency)
            ]
        )

        elapsed = time.perf_counter() - started

    print()
    print("--- Concurrent Benchmark ---")
    print(f"requests:    {args.requests}")
    print(f"concurrency: {args.concurrency}")
    print(f"succeeded:   {len(latencies)}")
    print(f"failed:      {sum(errors.values())} {dict(errors) if errors else ''}")

    if not latencies:
        print("No successful requests - check the server logs.")
        sys.exit(1)

    latencies.sort()

    print(f"mean:        {statistics.mean(latencies):.2f} ms")
    print(f"p50:         {percentile(latencies, 50):.2f} ms")
    print(f"p95:         {percentile(latencies, 95):.2f} ms")
    print(f"min:         {latencies[0]:.2f} ms")
    print(f"max:         {latencies[-1]:.2f} ms")
    print(f"throughput:  {len(latencies) / elapsed:.2f} req/s")
    print(f"avg batch:   {statistics.mean(batch_sizes):.2f}")


if __name__ == "__main__":
    asyncio.run(benchmark(parse_args()))