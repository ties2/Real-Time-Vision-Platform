# step 4
بریم سراغ **Step 4 — Model Layer**. این مرحله اولین جایی است که `YOLO11` واقعاً وارد architecture می‌شود، ولی هنوز inference API را نمی‌سازی #

هدف این است:

```text
models.yaml
     │
     ▼
ModelRegistry
     │
     ▼
UltralyticsModel
     │
     ▼
YOLO11
     │
     ▼
yolo11n.pt
```

## 4.1 — `configs/models.yaml`

این فایل را بساز:

```yaml
models:
  yolo11:
    provider: ultralytics
    task: detect
    version: "1.0.0"
    artifact: yolo11n.pt
    device: auto
    confidence_threshold: 0.5
```

چرا `provider` داریم؟

چون بعداً می‌توانیم داشته باشیم:

```yaml
models:
  yolo11:
    provider: ultralytics

  custom_classifier:
    provider: onnx

  production_detector:
    provider: triton
```

پس registry به YOLO قفل نمی‌شود.

---

# 4.2 — یک Model Specification بسازیم

به جای اینکه YAML را مستقیم همه‌جا dictionary فرض کنیم، یک schema مشخص داشته باشیم.

`app/models/registry.py` را با این نسخه جایگزین کن:

```python
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.models.base import BaseModel as ModelInterface


class ModelConfig(BaseModel):
    """Configuration describing a model artifact."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    task: str
    version: str
    artifact: str
    device: str = "auto"
    confidence_threshold: float = 0.5


class ModelRegistry:
    """Registry responsible for model configuration and instances."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._configs: dict[str, ModelConfig] = {}
        self._models: dict[str, ModelInterface] = {}

    def register_config(
        self,
        name: str,
        config: ModelConfig,
    ) -> None:
        """Register model configuration."""

        if name in self._configs:
            raise ValueError(
                f"Model configuration '{name}' is already registered."
            )

        self._configs[name] = config

    def register(
        self,
        name: str,
        model: ModelInterface,
    ) -> None:
        """Register an initialized model instance."""

        if name in self._models:
            raise ValueError(
                f"Model '{name}' is already registered."
            )

        self._models[name] = model

    def get(self, name: str) -> ModelInterface:
        """Return a registered model instance."""

        try:
            return self._models[name]
        except KeyError as exc:
            raise KeyError(
                f"Model '{name}' is not loaded."
            ) from exc

    def get_config(self, name: str) -> ModelConfig:
        """Return model configuration."""

        try:
            return self._configs[name]
        except KeyError as exc:
            raise KeyError(
                f"Model configuration '{name}' is not registered."
            ) from exc

    def exists(self, name: str) -> bool:
        """Check whether a model is loaded."""

        return name in self._models

    def list_models(self) -> list[str]:
        """Return loaded model names."""

        return list(self._models.keys())

    def list_configured_models(self) -> list[str]:
        """Return configured model names."""

        return list(self._configs.keys())

    def metadata(self) -> dict[str, dict[str, Any]]:
        """Return metadata for loaded models."""

        return {
            name: model.metadata()
            for name, model in self._models.items()
        }
```

### چرا این تغییر مهم است؟

قبلاً registry فقط:

```text
name → model instance
```

بود.

الان:

```text
name
 ├── configuration
 └── loaded model
```

داریم.

این بعداً برای model versioning و MLflow خیلی مهم می‌شود.

---

# 4.3 — `UltralyticsModel`

حالا فایل:

```text
app/models/ultralytics.py
```

را بساز:

```python
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from app.models.base import BaseModel
from app.models.registry import ModelConfig


class UltralyticsModel(BaseModel):
    """Ultralytics-backed object detection model."""

    def __init__(
        self,
        name: str,
        config: ModelConfig,
        models_dir: Path,
    ) -> None:
        self.name = name
        self.config = config
        self.models_dir = models_dir

        self._model: YOLO | None = None

        self.artifact_path = (
            self.models_dir / self.config.artifact
        )

    def load(self) -> None:
        """Load the Ultralytics model artifact."""

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {self.artifact_path}"
            )

        self._model = YOLO(str(self.artifact_path))

    def predict(
        self,
        input_data: Any,
    ) -> Any:
        """Run inference using the loaded model."""

        if self._model is None:
            raise RuntimeError(
                f"Model '{self.name}' has not been loaded."
            )

        return self._model.predict(
            source=input_data,
            device=self.config.device,
            conf=self.config.confidence_threshold,
            verbose=False,
        )

    def metadata(self) -> dict[str, Any]:
        """Return model metadata."""

        return {
            "name": self.name,
            "provider": self.config.provider,
            "task": self.config.task,
            "version": self.config.version,
            "artifact": self.config.artifact,
            "device": self.config.device,
            "confidence_threshold": (
                self.config.confidence_threshold
            ),
            "loaded": self.is_loaded(),
        }

    def is_loaded(self) -> bool:
        """Return whether the model is loaded."""

        return self._model is not None
```

---

# 4.4 — یک نکته مهم درباره model path

الان `models_dir` را به registry می‌دهیم.

ولی `yolo11n.pt` فعلاً در root پروژه است:

```text
BBAP-Sec/
└── yolo11n.pt
```

در architecture نهایی بهتر است:

```text
BBAP-Sec/
└── models/
    └── yolo11n.pt
```

باشد.

اما چون `.gitignore` تو `*.pt` را ignore می‌کند، انتقالش مشکلی برای Git ندارد.

پس:

```bash
mkdir -p models
mv yolo11n.pt models/yolo11n.pt
```

بعد YAML:

```yaml
artifact: yolo11n.pt
```

درست می‌ماند.

---

# 4.5 — Registry Loader

الان registry هنوز YAML را نمی‌خواند.

یک فایل جدید بساز:

```text
app/models/loader.py
```

```python
from pathlib import Path
from typing import Any

import yaml

from app.models.registry import ModelConfig, ModelRegistry
from app.models.ultralytics import UltralyticsModel


def load_model_registry(
    config_path: Path,
    models_dir: Path,
) -> ModelRegistry:
    """Load model configurations and initialize configured models."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file) or {}

    registry = ModelRegistry(models_dir=models_dir)

    models_config = raw_config.get("models", {})

    for name, raw_model_config in models_config.items():
        config = ModelConfig(**raw_model_config)

        registry.register_config(name, config)

        if config.provider == "ultralytics":
            model = UltralyticsModel(
                name=name,
                config=config,
                models_dir=models_dir,
            )

            model.load()
            registry.register(name, model)

        else:
            raise ValueError(
                f"Unsupported model provider: {config.provider}"
            )

    return registry
```

حالا architecture خیلی تمیزتر شده:

```text
models.yaml
     │
     ▼
loader.py
     │
     ▼
ModelConfig
     │
     ▼
ModelRegistry
     │
     ▼
UltralyticsModel
```

---

# 4.6 — Configuration را برای models آماده کنیم

در `app/core/config.py` این field را داشته باش:

```python
models_config_path: Path = Field(
    default=PROJECT_ROOT / "configs" / "models.yaml"
)
```

پس بخشی از Settings می‌شود:

```python
model_registry_path: Path = Field(
    default=PROJECT_ROOT / "models"
)

models_config_path: Path = Field(
    default=PROJECT_ROOT / "configs" / "models.yaml"
)

default_model: str = "yolo11"

model_confidence_threshold: float = 0.5
```

در واقع بعداً `confidence_threshold` را از Settings حذف می‌کنیم، چون model-specific است و در `models.yaml` قرار گرفته.

فعلاً می‌توانی حذفش کنی:

```python
model_confidence_threshold: float = 0.5
```

چون دیگر:

```yaml
confidence_threshold: 0.5
```

مرجع اصلی است.

---

# 4.7 — Model Registry را در Application Lifecycle load کنیم

حالا `app/main.py` را کمی ارتقا بده.

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.models.loader import load_model_registry

settings = get_settings()

configure_logging(settings.log_level)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""

    logger.info(
        "Starting %s v%s",
        settings.app_name,
        settings.app_version,
    )

    registry = load_model_registry(
        config_path=settings.models_config_path,
        models_dir=settings.model_registry_path,
    )

    app.state.model_registry = registry

    logger.info(
        "Loaded models: %s",
        registry.list_models(),
    )

    yield

    logger.info("Shutting down application")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production-oriented real-time "
        "computer vision model serving platform."
    ),
    lifespan=lifespan,
)

app.include_router(health_router)
```

---

# 4.8 — الان یک نکته مهم

با این تغییر، وقتی:

```bash
make run
```

می‌زنی، application در startup واقعاً این کار را انجام می‌دهد:

```text
FastAPI starts
      ↓
Read models.yaml
      ↓
Find yolo11n.pt
      ↓
Load YOLO11
      ↓
Register model
      ↓
Application ready
```

اگر model پیدا نشود، application باید fail شود.

این رفتار برای production بهتر از این است که server بالا بیاید ولی model خراب باشد.

---

# 4.9 — تست واقعی Model

یک test بساز:

```text
tests/unit/test_ultralytics.py
```

اما این تست نباید حتماً GPU داشته باشد.

```python
from pathlib import Path

import pytest

from app.models.registry import ModelConfig
from app.models.ultralytics import UltralyticsModel


def test_model_artifact_path():
    config = ModelConfig(
        provider="ultralytics",
        task="detect",
        version="1.0.0",
        artifact="yolo11n.pt",
    )

    model = UltralyticsModel(
        name="yolo11",
        config=config,
        models_dir=Path("/tmp/models"),
    )

    assert model.artifact_path == Path(
        "/tmp/models/yolo11n.pt"
    )


def test_missing_model_artifact():
    config = ModelConfig(
        provider="ultralytics",
        task="detect",
        version="1.0.0",
        artifact="missing.pt",
    )

    model = UltralyticsModel(
        name="yolo11",
        config=config,
        models_dir=Path("/tmp/models"),
    )

    with pytest.raises(FileNotFoundError):
        model.load()
```

این تست‌ها **YOLO را load نمی‌کنند** و بنابراین سریع هستند.

---

# 4.10 — تست Integration واقعی

بعد یک test integration خواهیم داشت که model واقعی را load کند.

ولی فعلاً انجامش نمی‌دهیم، چون:

```text
unit tests
    ↓
fast
    ↓
every commit

integration tests
    ↓
real model
    ↓
slower
```

را جدا نگه می‌داریم.

---

# 4.11 — تست کن

حالا:

```bash
make format
```

بعد:

```bash
make check
```

و سپس:

```bash
make run
```

اگر همه‌چیز درست باشد، log باید تقریباً این باشد:

```text
INFO | app.main | Starting BBAP-Sec Real-Time Vision Platform v0.1.0
INFO | app.main | Loaded models: ['yolo11']
INFO | uvicorn.error | Application startup complete.
```

اگر این را دیدی، یعنی:

**YOLO11 با موفقیت وارد Model Registry شده است.**

---

## یک نکته درباره CPU/GPU

فعلاً:

```yaml
device: auto
```

گذاشته‌ایم.

یعنی Ultralytics خودش device مناسب را انتخاب می‌کند.

بعداً در production می‌توانیم configuration دقیق‌تری داشته باشیم:

```yaml
device: cpu
```

یا:

```yaml
device: 0
```

یا حتی:

```text
CUDA
TensorRT
ONNX
```

ولی فعلاً abstraction را ساده نگه می‌داریم.

---

# نتیجه Step 4

الان دیگر پروژه از این:

```text
FastAPI
  +
YOLO script
```

به این تبدیل شده:

```text
                    models.yaml
                         │
                         ▼
                    Model Loader
                         │
                         ▼
                  Model Configuration
                         │
                         ▼
                   Model Registry
                         │
                         ▼
                  BaseModel interface
                         │
                         ▼
                  UltralyticsModel
                         │
                         ▼
                      YOLO11
```

این **foundation واقعی Model Serving** است.

### بعد از موفقیت این مرحله، Step 5

می‌شود جذاب‌ترین قسمت تا اینجا:

```text
POST /api/v1/inference
```

و architecture:

```text
Image
  │
  ▼
FastAPI
  │
  ▼
Inference Request Schema
  │
  ▼
Inference Engine
  │
  ├── Preprocessing
  │
  ├── Model Registry
  │       ↓
  │    YOLO11
  │
  └── Postprocessing
          │
          ▼
    Structured JSON
```

یعنی در Step 5 برای اولین بار **یک image واقعی را از API می‌گیریم، YOLO11 روی آن inference می‌گیریم و response حرفه‌ای و versioned برمی‌گردانیم.**

---
# step 5

عالی. بریم **Step 5 — Inference API**. این مرحله مهم‌ترین milestone تا اینجای پروژه است: از اینجا دیگر واقعاً یک **ML model serving API** داریم.

هدف Step 5:

```text
Image
  │
  ▼
POST /api/v1/inference
  │
  ▼
Request Validation
  │
  ▼
Inference Engine
  │
  ├── Preprocessing
  │
  ├── Model Registry
  │       │
  │       ▼
  │    YOLO11
  │
  └── Postprocessing
          │
          ▼
   Structured Response
```

نکته مهم: فعلاً **batching، tracking، RTSP و WebSocket نداریم**. آنها را بعداً روی همین engine سوار می‌کنیم.

---

# 5.1 — API schema

اول response را از implementation جدا می‌کنیم.

ساختار:

```text
app/
└── api/
    └── schemas/
        ├── __init__.py
        └── inference.py
```

بساز.

### `app/api/schemas/inference.py`

```python
from typing import Any

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Object bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """Single object detection."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class InferenceResponse(BaseModel):
    """Structured inference response."""

    model: str
    model_version: str
    inference_time_ms: float
    image_width: int
    image_height: int
    detections: list[Detection]
```

این خیلی مهم است.

ما نمی‌خواهیم API مستقیماً objectهای Ultralytics را به client برگرداند.

---

# 5.2 — Preprocessing

حالا:

```text
app/inference/preprocessing.py
```

فعلاً ساده نگه می‌داریم:

```python
from io import BytesIO

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV image."""

    buffer = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Unable to decode image.")

    return image
```

این layer عمداً مستقل از YOLO است.

بعداً می‌توانیم اینجا اضافه کنیم:

```text
resize
normalize
color conversion
validation
```

بدون اینکه API route تغییر کند.

---

# 5.3 — Postprocessing

حالا:

```text
app/inference/postprocessing.py
```

```python
from typing import Any

from app.api.schemas.inference import (
    BoundingBox,
    Detection,
)


def postprocess_results(
    results: Any,
) -> list[Detection]:
    """Convert Ultralytics results into API detections."""

    detections: list[Detection] = []

    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes

        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            confidence = float(boxes.conf[index].item())

            coordinates = boxes.xyxy[index].tolist()

            x1, y1, x2, y2 = coordinates

            class_name = result.names[class_id]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    ),
                )
            )

    return detections
```

حالا architecture:

```text
Ultralytics Result
       │
       ▼
postprocess_results()
       │
       ▼
Detection
       │
       ▼
API Response
```

این separation بعداً خیلی ارزشمند است.

---

# 5.4 — Inference Engine

حالا قلب سیستم:

```text
app/inference/engine.py
```

```python
import time
from dataclasses import dataclass

import numpy as np

from app.models.registry import ModelRegistry


@dataclass
class InferenceResult:
    """Raw inference result with timing metadata."""

    results: object
    inference_time_ms: float
    image_width: int
    image_height: int


class InferenceEngine:
    """Orchestrates model inference."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    def predict(
        self,
        model_name: str,
        image: np.ndarray,
    ) -> InferenceResult:
        """Run inference using a registered model."""

        model = self.registry.get(model_name)

        image_height, image_width = image.shape[:2]

        start_time = time.perf_counter()

        results = model.predict(image)

        elapsed = time.perf_counter() - start_time

        return InferenceResult(
            results=results,
            inference_time_ms=elapsed * 1000,
            image_width=image_width,
            image_height=image_height,
        )
```

این قسمت architectural gold است.

Route دیگر نمی‌داند:

```text
YOLO
Ultralytics
PyTorch
OpenCV
```

چی هستند.

Route فقط می‌گوید:

```text
engine.predict(...)
```

---

# 5.5 — API dependency

حالا:

```text
app/api/dependencies.py
```

را:

```python
from fastapi import Request

from app.inference.engine import InferenceEngine


def get_inference_engine(request: Request) -> InferenceEngine:
    """Return the application inference engine."""

    return InferenceEngine(
        registry=request.app.state.model_registry,
    )
```

بعداً این را optimize می‌کنیم تا engine را یک بار بسازیم، ولی فعلاً dependency ساده و clean است.

---

# 5.6 — Inference route

بساز:

```text
app/api/routes/inference.py
```

```python
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.dependencies import get_inference_engine
from app.api.schemas.inference import InferenceResponse
from app.inference.engine import InferenceEngine
from app.inference.postprocessing import postprocess_results
from app.inference.preprocessing import decode_image

router = APIRouter(
    prefix="/api/v1",
    tags=["inference"],
)


@router.post(
    "/inference",
    response_model=InferenceResponse,
)
async def inference(
    file: UploadFile = File(...),
    model: str = "yolo11",
    engine: InferenceEngine = Depends(get_inference_engine),
) -> InferenceResponse:
    """Run object detection on an uploaded image."""

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Missing content type.",
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image files are supported.",
        )

    data = await file.read()

    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    try:
        result = engine.predict(
            model_name=model,
            image=image,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    detections = postprocess_results(result.results)

    model_instance = engine.registry.get(model)
    metadata = model_instance.metadata()

    return InferenceResponse(
        model=model,
        model_version=str(metadata["version"]),
        inference_time_ms=result.inference_time_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        detections=detections,
    )
```

---

# 5.7 — Route را register کن

در `app/main.py`:

```python
from app.api.routes.inference import router as inference_router
```

و پایین:

```python
app.include_router(health_router)
app.include_router(inference_router)
```

پس:

```text
/health

/api/v1/inference
```

داریم.

---

# 5.8 — یک بهبود مهم در `BaseModel`

الان `BaseModel` خوب است، ولی type مربوط به `predict()` خیلی generic است.

فعلاً:

```python
def predict(self, input_data: Any) -> Any:
```

را نگه می‌داریم.

چون در آینده ممکن است:

```text
image
batch
video frame
tensor
```

وارد model شود.

فعلاً premature typing نمی‌کنیم.

---

# 5.9 — Test برای preprocessing

بساز:

```text
tests/unit/test_preprocessing.py
```

```python
import cv2
import numpy as np
import pytest

from app.inference.preprocessing import decode_image


def test_decode_image():
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    success, encoded = cv2.imencode(".jpg", image)

    assert success

    decoded = decode_image(encoded.tobytes())

    assert decoded.shape == (100, 200, 3)


def test_invalid_image():
    with pytest.raises(ValueError):
        decode_image(b"not-an-image")
```

---

# 5.10 — Test برای postprocessing

```text
tests/unit/test_postprocessing.py
```

```python
from types import SimpleNamespace

from app.inference.postprocessing import postprocess_results


class FakeValue:

    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class FakeBox:

    def __init__(self):
        self.cls = [FakeValue(0)]
        self.conf = [FakeValue(0.95)]
        self.xyxy = [[10.0, 20.0, 100.0, 200.0]]

    def __len__(self):
        return 1


def test_postprocess_results():
    result = SimpleNamespace(
        boxes=FakeBox(),
        names={0: "person"},
    )

    detections = postprocess_results([result])

    assert len(detections) == 1

    detection = detections[0]

    assert detection.class_id == 0
    assert detection.class_name == "person"
    assert detection.confidence == 0.95
    assert detection.bbox.x1 == 10.0
```

---

# 5.11 — Integration test واقعی

حالا مهم‌ترین تست.

در:

```text
tests/integration/test_inference.py
```

فعلاً یک test ساده می‌سازیم:

```python
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_inference():
    client = TestClient(app)

    image_path = Path("streetAndpeople.jpg")

    if not image_path.exists():
        return

    with image_path.open("rb") as image:
        response = client.post(
            "/api/v1/inference",
            files={
                "file": (
                    "streetAndpeople.jpg",
                    image,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "yolo11"
    assert data["model_version"] == "1.0.0"
    assert data["image_width"] > 0
    assert data["image_height"] > 0
    assert "detections" in data
```

اما اینجا یک مشکل داریم:

تو در `.gitignore` داری:

```gitignore
*.jpg
```

بنابراین `streetAndpeople.jpg` اگر در Git باشد tracked نیست.

برای local development فعلاً مشکلی ندارد.

---

# 5.12 — یک تست مهم‌تر برای API

در نهایت می‌خواهیم این را داشته باشیم:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/api/v1/inference \
  -F "file=@streetAndpeople.jpg"
```

و response:

```json
{
  "model": "yolo11",
  "model_version": "1.0.0",
  "inference_time_ms": 21.47,
  "image_width": 1280,
  "image_height": 720,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.93,
      "bbox": {
        "x1": 100.2,
        "y1": 80.1,
        "x2": 300.7,
        "y2": 600.4
      }
    }
  ]
}
```

مقادیر واقعی obviously بسته به image و hardware متفاوت هستند.

---

# 5.13 — یک نکته مهم: `streetAndpeople.jpg`

چون `.gitignore` تو imageها را ignore می‌کند، برای تست production-style من پیشنهاد می‌کنم **الان** یک directory مخصوص test fixtures بسازیم:

```text
tests/
├── conftest.py
├── fixtures/
│   └── streetAndpeople.jpg
├── integration/
└── unit/
```

اما چون `*.jpg` ignore است، exception اضافه کن:

```gitignore
!tests/fixtures/
!tests/fixtures/*.jpg
```

و image را:

```bash
mkdir -p tests/fixtures
mv streetAndpeople.jpg tests/fixtures/streetAndpeople.jpg
```

این باعث می‌شود test fixture واقعاً داخل repository قابل استفاده باشد.

برای پروژه portfolio/Upwork این بهتر از گذاشتن image در root است.

---

# 5.14 — اما یک مشکل کوچک دیگر

`tests/integration/test_inference.py` الان startup واقعی application را trigger می‌کند و YOLO را load می‌کند.

این **درست است** برای integration test، ولی pytest را کندتر می‌کند.

فعلاً acceptable است.

بعداً testها را به:

```text
unit
integration
e2e
```

تفکیک می‌کنیم و integration را با marker اجرا می‌کنیم:

```bash
pytest -m integration
```

این را در مرحله testing infrastructure انجام می‌دهیم.

---

# 5.15 — حالا اجرا

اول:

```bash
make format
```

بعد:

```bash
make check
```

بعد server:

```bash
make run
```

و در terminal دوم:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/api/v1/inference \
  -F "file=@tests/fixtures/streetAndpeople.jpg"
```

یا از Swagger:

```text
http://127.0.0.1:8000/docs
```

باید endpoint جدید را ببینی:

```text
POST /api/v1/inference
```

---

## اگر همه چیز درست باشد

در این مرحله architecture ما واقعاً تبدیل شده به:

```text
                       ┌──────────────┐
                       │    Client    │
                       └──────┬───────┘
                              │
                         Image Upload
                              │
                       ┌──────▼───────┐
                       │    FastAPI   │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │  Preprocess  │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │   Inference  │
                       │    Engine    │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │    Registry  │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │  YOLO11      │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │ Postprocess  │
                       └──────┬───────┘
                              │
                       ┌──────▼───────┐
                       │ JSON Response│
                       └──────────────┘
```

این **اولین vertical slice واقعی پروژه** است: از ورودی کاربر تا model و response کامل.

بعد از اینکه این مرحله سبز شد، **Step 6** را می‌رویم روی `GET /api/v1/models` + model metadata + error handling حرفه‌ای + response/request IDs. بعد از آن تازه وارد batching و performance می‌شویم.

بله، **فرق اساسی دارند** و اتفاقاً چیزی که الان داری در `BaseModel` از نظر معماری ارزشمند است. اشتباه بود اگر آن را با `app/api/schemas/inference.py` یکی کنیم.

### تفاوت اصلی

کدی که داری:

```text
app/models/base.py
```

یک **Model Abstraction / Model Contract** است.

یعنی به implementation مدل می‌گوید:

> «هر مدل ML که وارد این platform می‌شود باید این interface را داشته باشد.»

اما:

```text
app/api/schemas/inference.py
```

یک **API Contract** است.

یعنی به client می‌گوید:

> «اگر از API inference استفاده کردی، request/response چه شکلی خواهد بود.»

---

## معماری درست

این دو باید جدا بمانند:

```text
                 ┌──────────────────────┐
                 │      API Client      │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ API Schemas          │
                 │                      │
                 │ InferenceResponse    │
                 │ Detection            │
                 │ BoundingBox          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Inference Engine     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Model Interface      │
                 │                      │
                 │ BaseModel            │
                 │ ├─ preprocess()      │
                 │ ├─ predict()         │
                 │ ├─ postprocess()     │
                 │ └─ metadata()        │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ UltralyticsModel     │
                 │ YOLO11                │
                 └──────────────────────┘
```

---

# ولی یک نکته مهم‌تر

`BaseModel` فعلی تو حتی از چیزی که من در Step 4 پیشنهاد دادم **کامل‌تر است**:

```python
def preprocess(...)
def predict(...)
def postprocess(...)
def inference(...)
```

این abstraction برای model-serving platform خیلی خوب است.

بنابراین **آن را حذف نکن.**

اتفاقاً باید architecture را بر اساس همین طراحی اصلاح کنیم.

---

## یک مسئله معماری که باید حل کنیم

الان دو مفهوم داریم:

### Model-level preprocessing

مثلاً:

```python
class UltralyticsModel(BaseModel):

    def preprocess(self, input_data):
        ...
```

و:

### Pipeline-level preprocessing

مثل:

```text
bytes
 ↓
decode image
 ↓
validate
 ↓
resize
```

این دو را نباید بی‌دلیل duplicate کنیم.

من پیشنهاد می‌کنم در architecture نهایی:

```text
API
 │
 ▼
Input validation / decoding
 │
 ▼
InferenceEngine
 │
 ▼
BaseModel.inference()
 │
 ├── preprocess()
 ├── predict()
 └── postprocess()
 │
 ▼
API Response
```

یعنی **مدل مالک preprocessing مخصوص خودش باشد.**

---

# بنابراین Step 5 را کمی اصلاح می‌کنیم

به‌جای اینکه `InferenceEngine` این کار را انجام دهد:

```text
Engine
 ├── preprocessing
 ├── model.predict
 └── postprocessing
```

بهتر است:

```text
InferenceEngine
       │
       ▼
BaseModel.inference()
       │
       ├── preprocess()
       ├── predict()
       └── postprocess()
```

اما یک preprocessing خیلی ابتدایی در API می‌تواند باقی بماند:

```text
multipart bytes
       ↓
decode image
       ↓
numpy image
       ↓
BaseModel.inference()
```

---

# یک تفاوت ظریف دیگر

مثلاً API response ما:

```json
{
  "class_id": 0,
  "class_name": "person",
  "confidence": 0.94,
  "bbox": {
    "x1": 10,
    "y1": 20,
    "x2": 100,
    "y2": 200
  }
}
```

نباید الزاماً خروجی مستقیم `BaseModel.postprocess()` باشد.

بهتر است:

```text
YOLO raw output
       ↓
BaseModel.postprocess()
       ↓
Internal normalized result
       ↓
API adapter/schema
       ↓
Pydantic InferenceResponse
```

چرا؟

چون فردا اگر API تغییر کند:

```text
REST API
WebSocket
gRPC
Kafka
```

نباید model implementation را تغییر بدهیم.

---

# پس پیشنهاد نهایی من

`BaseModel` فعلی را نگه دار:

```text
app/models/base.py
```

و `schemas` را جدا نگه دار:

```text
app/api/schemas/inference.py
```

و architecture را این‌طور کنیم:

```text
app/
│
├── api/
│   ├── dependencies.py
│   ├── routes/
│   │   └── inference.py
│   └── schemas/
│       └── inference.py
│
├── inference/
│   └── engine.py
│
└── models/
    ├── base.py
    ├── registry.py
    └── ultralytics.py
```

### مسئولیت‌ها

| Component             | مسئولیت                         |
| --------------------- | ------------------------------- |
| `BaseModel`           | قرارداد مدل ML                  |
| `UltralyticsModel`    | implementation مربوط به YOLO    |
| `ModelRegistry`       | مدیریت model instances          |
| `InferenceEngine`     | orchestration                   |
| `preprocessing.py`    | decoding/validation ورودی عمومی |
| `postprocessing.py`   | در صورت نیاز adapter عمومی      |
| `api/schemas`         | قرارداد HTTP API                |
| `routes/inference.py` | HTTP layer                      |

---

## یک تغییر دیگر هم پیشنهاد می‌کنم

در `BaseModel` فعلی:

```python
def inference(self, input_data: Any) -> Any:
    preprocessed_data = self.preprocess(input_data)
    predictions = self.predict(preprocessed_data)
    return self.postprocess(predictions)
```

این قسمت را **نگه می‌داریم**.

این در واقع یک implementation از **Template Method Pattern** است و برای architecture ما بسیار مناسب است:

```text
BaseModel.inference()
       │
       ├── preprocess()
       ├── predict()
       └── postprocess()
```

هر مدل فقط این سه مرحله را implement می‌کند.

مثلاً:

```text
BaseModel
   │
   ├── UltralyticsModel
   │      ├── preprocess
   │      ├── predict
   │      └── postprocess
   │
   ├── ONNXModel
   │      ├── preprocess
   │      ├── predict
   │      └── postprocess
   │
   └── TensorRTModel
          ├── preprocess
          ├── predict
          └── postprocess
```

این دقیقاً همان چیزی است که برای یک **reusable ML serving platform** می‌خواهیم.

### نتیجه

پس **`app/models/base.py` و `app/api/schemas/inference.py رقیب هم نیستند؛ در دو لایه متفاوت‌اند و باید هر دو وجود داشته باشند.**

و چون `BaseModel` فعلی تو طراحی بهتری برای این project دارد، **Step 5 را بر اساس همین `BaseModel` ادامه می‌دهیم، نه اینکه BaseModel را با نسخه ساده قبلی جایگزین کنیم.**
---

# Step 6
عالی. پس **Step 6 بخش اصلی تمام شد**. حالا قبل از performance، یک cleanup کوچک ولی مهم انجام می‌دهیم تا foundation واقعاً حرفه‌ای باشد.

# Step 6.1 — Error Handling و Request ID را کامل کنیم

الان باید این architecture را داشته باشیم:

```text
Request
   │
   ▼
RequestIDMiddleware
   │
   ▼
FastAPI Route
   │
   ├── Validation error
   ├── ModelNotFoundError
   └── Unexpected exception
          │
          ▼
   Global Exception Handler
          │
          ▼
      JSON Error
```

## 1. `app/core/exceptions.py`

مطمئن شو این نسخه را داری:

```python
class BBAPException(Exception):
    """Base exception for application-level errors."""

    code = "APPLICATION_ERROR"
    status_code = 500

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class ModelNotFoundError(BBAPException):
    """Raised when a requested model does not exist."""

    code = "MODEL_NOT_FOUND"
    status_code = 404


class InvalidInputError(BBAPException):
    """Raised when request input is invalid."""

    code = "INVALID_INPUT"
    status_code = 400
```

---

# 2. Error response schema

`app/api/schemas/errors.py`:

```python
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard API error response."""

    code: str
    message: str
    request_id: str
```

---

# 3. Global handlers

در `app/main.py` این دو handler را داشته باش:

```python
@app.exception_handler(BBAPException)
async def bbap_exception_handler(
    request: Request,
    exc: BBAPException,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "request_id": request_id,
        },
    )
```

و برای unexpected errors:

```python
@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        "unknown",
    )

    logger.exception(
        "Unhandled exception request_id=%s",
        request_id,
    )

    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred.",
            "request_id": request_id,
        },
    )
```

### نکته

`logger.exception()` مهم است.

Client فقط این را می‌بیند:

```json
{
  "code": "INTERNAL_SERVER_ERROR",
  "message": "An unexpected error occurred.",
  "request_id": "abc-123"
}
```

ولی server log stack trace کامل را نگه می‌دارد.

این دقیقاً همان separation است که می‌خواهیم.

---

# Step 6.2 — تست واقعی Error Contract

این test را اضافه کن:

`tests/integration/test_errors.py`

```python
from fastapi.testclient import TestClient

from app.main import app


def test_unknown_model_returns_404():
    client = TestClient(app)

    response = client.post(
        "/api/v1/inference?model=does-not-exist",
        files={
            "file": (
                "test.jpg",
                b"fake-image",
                "image/jpeg",
            )
        },
        headers={
            "X-Request-ID": "test-request-404",
        },
    )

    assert response.status_code == 404

    data = response.json()

    assert data["code"] == "MODEL_NOT_FOUND"
    assert "does-not-exist" in data["message"]
    assert data["request_id"] == "test-request-404"
```

این test یک نکته خیلی خوب را هم verify می‌کند:

```text
X-Request-ID
      │
      ▼
Exception
      │
      ▼
Error response
```

---

# Step 6.3 — Test برای invalid image

```python
def test_invalid_image_returns_400():
    client = TestClient(app)

    response = client.post(
        "/api/v1/inference",
        files={
            "file": (
                "test.jpg",
                b"not-an-image",
                "image/jpeg",
            )
        },
        headers={
            "X-Request-ID": "test-invalid-image",
        },
    )

    assert response.status_code == 400

    data = response.json()

    assert data["code"] == "INVALID_INPUT"
    assert data["request_id"] == "test-invalid-image"
```

---

# Step 6.4 — Models API

حالا یک چیز کوچک ولی مهم.

`GET /api/v1/models` باید **هیچ اطلاعات implementation-specific اضافه‌ای** بیرون ندهد.

یعنی response فعلی:

```json
{
  "name": "yolo11",
  "provider": "ultralytics",
  "task": "detect",
  "version": "1.0.0",
  "artifact": "yolo11n.pt",
  "device": "auto",
  "confidence_threshold": 0.5,
  "loaded": true
}
```

برای development خوب است.

ولی برای production بعداً بهتر است `artifact` path داخلی را public نکنیم.

فعلاً نگه می‌داریم چون پروژه development/portfolio است؛ در مرحله security hardening اصلاحش می‌کنیم.

---

# Step 6.5 — تست دستی

حالا:

```bash
make check
```

باید چیزی مثل:

```text
ruff check .
All checks passed!

ruff format --check .
...

pytest
...
passed
```

بعد:

```bash
make run
```

---

### Test 1 — models

```bash
curl -i \
  http://127.0.0.1:8000/api/v1/models
```

---

### Test 2 — request ID

```bash
curl -i \
  -H "X-Request-ID: demo-001" \
  http://127.0.0.1:8000/api/v1/models
```

باید:

```text
X-Request-ID: demo-001
```

را ببینی.

---

### Test 3 — unknown model

```bash
curl -i \
  -H "X-Request-ID: error-001" \
  -X POST \
  "http://127.0.0.1:8000/api/v1/inference?model=does-not-exist" \
  -F "file=@tests/fixtures/streetAndpeople.jpg"
```

باید:

```json
{
  "code": "MODEL_NOT_FOUND",
  "message": "Model 'does-not-exist' is not loaded.",
  "request_id": "error-001"
}
```

بگیری.

---

# یک cleanup مهم قبل از Step 7

الان احتمالاً `InferenceEngine` را این‌طور ساخته‌ای:

```python
def get_inference_engine(request: Request) -> InferenceEngine:
    return InferenceEngine(
        registry=request.app.state.model_registry,
    )
```

این یعنی برای هر request یک object جدید از `InferenceEngine` می‌سازیم.

فعلاً مشکل بزرگی نیست، ولی قبل از batching بهتر است architecture را تغییر بدهیم:

```text
Startup
   │
   ▼
ModelRegistry
   │
   ▼
InferenceEngine
   │
   ▼
app.state.inference_engine
```

و بعد request:

```text
Request
   │
   ▼
Existing InferenceEngine
   │
   ▼
Model
```

یعنی engine singleton در application lifecycle.

در `main.py` بعد از registry:

```python
registry = load_model_registry(
    config_path=settings.models_config_path,
    models_dir=settings.model_registry_path,
)

app.state.model_registry = registry
app.state.inference_engine = InferenceEngine(
    registry=registry,
)
```

و dependency:

```python
def get_inference_engine(
    request: Request,
) -> InferenceEngine:
    return request.app.state.inference_engine
```

این تغییر را **الان انجام بده**.

---

# چرا این مهم است؟

چون در Step 7 می‌خواهیم:

```text
                 InferenceEngine
                       │
            ┌──────────┼──────────┐
            │          │          │
         Request 1  Request 2  Request 3
            │          │          │
            └──────────┼──────────┘
                       │
                    Batcher
                       │
                       ▼
                    YOLO11
```

و اگر engine برای هر request ساخته شود، بعداً batching و queue management سخت‌تر می‌شود.

---

# Step 7 — Performance Engineering

حالا foundation آماده است.

در Step 7 می‌رویم سراغ چیزی که پروژه را از یک API معمولی به **Real-Time ML Serving Platform** نزدیک می‌کند:

```text
Multiple Requests
       │
       ▼
   Request Queue
       │
       ▼
 Dynamic Batching
       │
       ▼
     YOLO11
       │
       ▼
 Individual Results
```

و metrics:

```text
latency
throughput
batch size
queue time
inference time
FPS
```

همچنین یک benchmark واقعی می‌سازیم تا بتوانی در README/Upwork بگویی مثلاً:

> Benchmarked inference latency and throughput under concurrent requests.

اما **قبل از شروع Step 7 فقط مطمئن شو `make check` سبز است**؛ چون از اینجا به بعد performance code روی همین foundation ساخته می‌شود.

