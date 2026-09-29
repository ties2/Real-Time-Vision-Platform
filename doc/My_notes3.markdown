# step 9

 **Step 9 — Concurrent Benchmark**.

هدف این مرحله این نیست که فقط latency یک request را اندازه بگیریم؛ می‌خواهیم نشان دهیم **batching زیر بار concurrent چه اثری روی throughput و latency دارد.**

### 9.1 — benchmark فعلی را نگه می‌داریم

Baseline فعلی تو:

```text
requests:   20
p50:        38.91 ms
p95:        87.18 ms
throughput: 22.25 req/s
```

این را دست نمی‌زنیم؛ baseline ماست.

---

# 9.2 — فایل جدید

بساز:

```text
scripts/benchmark_concurrent.py
```

محتوا:

```python
import asyncio
import statistics
import time

import httpx


API_URL = "http://127.0.0.1:8000/api/v1/inference"
IMAGE_PATH = "streetAndpeople.jpg"

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
```

---

# 9.3 — dependency

اگر `httpx` در requirements نیست:

```bash
pip install httpx
```

و به `requirements.txt` اضافه کن:

```text
httpx
```

اگر از قبل هست، دوباره اضافه نکن.

---

# 9.4 — Makefile

به `Makefile` اضافه کن:

```makefile
benchmark-concurrent:
	python scripts/benchmark_concurrent.py
```

پس:

```bash
make benchmark-concurrent
```

---

# 9.5 — قبل از benchmark

سرور را اجرا کن:

```bash
make run
```

در terminal دیگر:

```bash
make benchmark-concurrent
```

انتظار چیزی شبیه:

```text
--- Concurrent Benchmark ---
requests: 40
concurrency: 8
mean: XX.XX ms
p50: XX.XX ms
p95: XX.XX ms
min: XX.XX ms
max: XX.XX ms
throughput: XX.XX req/s
```

**عددها را از قبل حدس نمی‌زنیم.**

---

# 9.6 — تست چند سطح concurrency

بعد از اینکه benchmark اول کار کرد، این سه حالت را اجرا کن:

```text
CONCURRENCY = 1
CONCURRENCY = 4
CONCURRENCY = 8
```

و نتایج را نگه دار.

مثلاً جدول نهایی ما:

| Concurrency | Requests | p50 | p95 | Throughput |
| ----------: | -------: | --: | --: | ---------: |
|           1 |       40 |   — |   — |          — |
|           4 |       40 |   — |   — |          — |
|           8 |       40 |   — |   — |          — |

این خیلی مهم‌تر از یک benchmark ساده است، چون نشان می‌دهد سیستم تحت load چه رفتاری دارد.

---

## یک نکته مهم

در این مرحله **هنوز ادعا نمی‌کنیم batching سریع‌تر است**.

ممکن است روی CPU/Mac تو throughput بهتر نشود یا حتی latency افزایش پیدا کند. این کاملاً قابل قبول است.

هدف engineering این مرحله:

```text
Measure
  ↓
Compare
  ↓
Understand bottleneck
  ↓
Optimize
```

نه اینکه حتماً یک عدد بزرگ‌تر تولید کنیم.

---

### بعد از اجرای benchmark

خروجی کامل `make benchmark-concurrent` را بفرست.

بعد با baseline فعلی:

```text
22.25 req/s
p50 = 38.91 ms
p95 = 87.18 ms
```

مقایسه می‌کنیم و **Step 9.2 — benchmark matrix + batching metrics** را انجام می‌دهیم.
---
