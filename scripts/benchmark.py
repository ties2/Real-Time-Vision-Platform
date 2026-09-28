import statistics
import time

import requests


API_URL = "http://127.0.0.1:8000/api/v1/inference"
IMAGE_PATH = "street2.jpg"
REQUESTS = 20


def main() -> None:
    latencies: list[float] = []

    with open(IMAGE_PATH, "rb") as image_file:
        image_data = image_file.read()

    for index in range(REQUESTS):
        started = time.perf_counter()

        response = requests.post(
            API_URL,
            params={"model": "yolo11"},
            files={
                "file": (
                    "street2.jpg",
                    image_data,
                    "image/jpeg",
                )
            },
            timeout=30,
        )

        elapsed_ms = (
                             time.perf_counter() - started
                     ) * 1000

        response.raise_for_status()

        latencies.append(elapsed_ms)

        print(
            f"request={index + 1:02d} "
            f"latency={elapsed_ms:.2f} ms"
        )

    print("\n--- Baseline ---")

    print(
        f"requests: {len(latencies)}"
    )

    print(
        f"mean: "
        f"{statistics.mean(latencies):.2f} ms"
    )

    print(
        f"p50: "
        f"{statistics.median(latencies):.2f} ms"
    )

    print(
        f"min: "
        f"{min(latencies):.2f} ms"
    )

    print(
        f"max: "
        f"{max(latencies):.2f} ms"
    )


if __name__ == "__main__":
    main()