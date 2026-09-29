import asyncio
import statistics
import time

import httpx


API_URL = "http://127.0.0.1:8000/api/v1/inference"
IMAGE_PATH = "tests/fixtures/streetAndpeople.jpg"

TOTAL_REQUESTS = 40
CONCURRENCY = 8


async def send_request(
        client: httpx.AsyncClient,
        image_bytes: bytes,
) -> float:
    started = time.perf_counter()

    response = await client.post(
        API_URL,
        files={
            "file": (
                "streetAndpeople.jpg",
                image_bytes,
                "image/jpeg",
            )
        },
        data={
            "model": "yolo11",
        },
    )

    response.raise_for_status()

    return (
            time.perf_counter() - started
    ) * 1000


async def worker(
        client: httpx.AsyncClient,
        image_bytes: bytes,
        queue: asyncio.Queue[int],
        latencies: list[float],
) -> None:
    while True:
        try:
            await queue.get()
        except asyncio.CancelledError:
            return

        try:
            latency = await send_request(
                client,
                image_bytes,
            )

            latencies.append(latency)

        finally:
            queue.task_done()


async def benchmark() -> None:
    with open(IMAGE_PATH, "rb") as file:
        image_bytes = file.read()

    queue: asyncio.Queue[int] = asyncio.Queue()

    for index in range(TOTAL_REQUESTS):
        queue.put_nowait(index)

    latencies: list[float] = []

    async with httpx.AsyncClient(
            timeout=60.0,
    ) as client:
        workers = [
            asyncio.create_task(
                worker(
                    client,
                    image_bytes,
                    queue,
                    latencies,
                )
            )
            for _ in range(CONCURRENCY)
        ]

        started = time.perf_counter()

        await queue.join()

        elapsed = (
                time.perf_counter() - started
        )

        for worker_task in workers:
            worker_task.cancel()

        await asyncio.gather(
            *workers,
            return_exceptions=True,
        )

    latencies.sort()

    p50 = statistics.median(latencies)

    p95_index = min(
        int(len(latencies) * 0.95),
        len(latencies) - 1,
        )

    p95 = latencies[p95_index]

    throughput = (
            TOTAL_REQUESTS / elapsed
    )

    print()
    print("--- Concurrent Benchmark ---")
    print(f"requests: {TOTAL_REQUESTS}")
    print(f"concurrency: {CONCURRENCY}")
    print(f"mean: {statistics.mean(latencies):.2f} ms")
    print(f"p50: {p50:.2f} ms")
    print(f"p95: {p95:.2f} ms")
    print(f"min: {min(latencies):.2f} ms")
    print(f"max: {max(latencies):.2f} ms")
    print(f"throughput: {throughput:.2f} req/s")


if __name__ == "__main__":
    asyncio.run(benchmark())