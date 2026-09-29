import statistics
import time

import requests

API_URL = "http://127.0.0.1:8000/api/v1/inference"
IMAGE_PATH = "tests/fixtures/street2.jpg"
REQUESTS = 20
WARMUP_REQUESTS = 3


def send_request(image_data: bytes) -> float:
    started = time.perf_counter()

    response = requests.post(
        API_URL,
        params={"model": "yolo11"},
        files={
            "file": (
                "streetAndpeople.jpg",
                image_data,
                "image/jpeg",
            )
        },
        timeout=30,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    response.raise_for_status()

    return elapsed_ms


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    values = sorted(values)

    index = int(len(values) * percentile_value / 100)

    index = min(
        index,
        len(values) - 1,
    )

    return values[index]


def main() -> None:
    latencies: list[float] = []

    with open(IMAGE_PATH, "rb") as image_file:
        image_data = image_file.read()

    print(f"Warm-up requests: {WARMUP_REQUESTS}")

    for index in range(WARMUP_REQUESTS):
        latency = send_request(image_data)

        print(f"warmup={index + 1:02d} latency={latency:.2f} ms")

    print(f"\nBenchmark requests: {REQUESTS}")

    for index in range(REQUESTS):
        latency = send_request(image_data)

        latencies.append(latency)

        print(f"request={index + 1:02d} latency={latency:.2f} ms")

    duration_seconds = sum(latencies) / 1000

    throughput = len(latencies) / duration_seconds

    print("\n--- Benchmark ---")

    print(f"requests: {len(latencies)}")

    print(f"mean: {statistics.mean(latencies):.2f} ms")

    print(f"p50: {statistics.median(latencies):.2f} ms")

    print(f"min: {min(latencies):.2f} ms")

    print(f"max: {max(latencies):.2f} ms")
    print(f"p95: {percentile(latencies, 95):.2f} ms")

    print(f"throughput: {throughput:.2f} req/s")


if __name__ == "__main__":
    main()
