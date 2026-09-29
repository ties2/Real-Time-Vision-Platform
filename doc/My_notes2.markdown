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

---
# Step 7

عالی. حالا که **15 تست سبز شده**، Step 7 را شروع می‌کنیم. این مرحله را عمداً **از batching شروع نمی‌کنیم**؛ اول baseline واقعی می‌گیریم، بعد batching را روی عدد واقعی مقایسه می‌کنیم.

# Step 7 — Performance Engineering

هدف نهایی:

```text
                Requests
                   │
                   ▼
             Inference API
                   │
                   ▼
              Queue/Batcher
                   │
             ┌─────┴─────┐
             │           │
          Batch 1      Batch 2
             │           │
             └─────┬─────┘
                   ▼
                YOLO11
                   │
                   ▼
             Postprocessing
                   │
                   ▼
               Response
```

ولی اول:

```text
Step 7.1 → Baseline
Step 7.2 → Metrics
Step 7.3 → Async Batcher
Step 7.4 → Dynamic batching
Step 7.5 → Concurrency
Step 7.6 → Benchmark
```

---

# Step 7.1 — Baseline Inference Timing

فعلاً **هیچ batching اضافه نکن**.

می‌خواهیم بدانیم الان یک inference معمولی چقدر زمان می‌برد.

## 1. `app/inference/engine.py`

در `InferenceEngine`، timing را با `perf_counter()` اضافه کن.

اگر متد فعلی چیزی شبیه این است:

```python
def predict(
    self,
    model_name: str,
    image: np.ndarray,
) -> InferenceResult:
    model = self.registry.get(model_name)

    result = model.inference(image)

    ...
```

آن را به این شکل تغییر بده:

```python
from time import perf_counter
```

و:

```python
def predict(
    self,
    model_name: str,
    image: np.ndarray,
) -> InferenceResult:
    model = self.registry.get(model_name)

    started_at = perf_counter()

    result = model.inference(image)

    elapsed_ms = (perf_counter() - started_at) * 1000

    return InferenceResult(
        results=result,
        inference_time_ms=elapsed_ms,
        image_width=image.shape[1],
        image_height=image.shape[0],
    )
```

اگر `InferenceResult` را already داری و fieldهای `inference_time_ms`, `image_width`, `image_height` در آن وجود دارند، **همان را استفاده کن و schema جدید نساز.**

---

# چرا `perf_counter()`؟

برای benchmark باید از:

```python
time.time()
```

استفاده نکنیم.

برای duration:

```python
perf_counter()
```

انتخاب مناسب‌تری است.

ما دقیقاً این را اندازه می‌گیریم:

```text
model.inference()
     │
     ├── preprocess
     ├── model forward
     └── postprocess
```

فعلاً.

بعداً آن را به:

```text
queue_time
preprocess_time
inference_time
postprocess_time
total_latency
```

تفکیک می‌کنیم.

---

# Step 7.2 — یک نکته معماری مهم

من فعلاً **timing را داخل `InferenceEngine` می‌گذارم، نه route**.

یعنی این کار را نکن:

```python
@app.post(...)
async def inference(...):
    started = perf_counter()

    result = engine.predict(...)

    elapsed = ...
```

چون بعداً وقتی batching اضافه کنیم، route دیگر نمی‌تواند accurately بگوید:

```text
request waiting time
batch waiting time
actual model inference time
```

Engine باید مسئول measurement مربوط به inference باشد.

---

# Step 7.3 — Baseline Benchmark

حالا یک script واقعی بساز:

```text
scripts/benchmark.py
```

نسخه اول ساده است:

```python
import statistics
import time

import requests


API_URL = "http://127.0.0.1:8000/api/v1/inference"
IMAGE_PATH = "streetAndpeople.jpg"
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
                    "streetAndpeople.jpg",
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
```

---

# Step 7.4 — Dependency

اگر `requests` در `requirements.txt` نیست:

```bash
pip install requests
```

و در `requirements.txt` هم اضافه کن:

```text
requests
```

ولی اگر `realtime_client.py` همین الان از `requests` استفاده می‌کند و dependency موجود است، دوباره اضافه نکن.

---

# Step 7.5 — اجرای benchmark

اول server:

```bash
make run
```

بعد در terminal دوم:

```bash
python scripts/benchmark.py
```

مثلاً خروجی ممکن است چیزی شبیه:

```text
request=01 latency=92.31 ms
request=02 latency=48.72 ms
request=03 latency=46.11 ms
...
request=20 latency=45.83 ms

--- Baseline ---

requests: 20
mean: 48.72 ms
p50: 46.21 ms
min: 44.90 ms
max: 92.31 ms
```

**عددها فقط مثال هستند.** عدد واقعی سیستم تو مهم است.

---

# Step 7.6 — چرا این benchmark مهم است؟

الان یک baseline داریم:

```text
Current system
──────────────

single request
      │
      ▼
   YOLO11
      │
      ▼
   ~XX ms
```

بعد batching را اضافه می‌کنیم:

```text
4 requests
     │
     ▼
 Batch
 [1,2,3,4]
     │
     ▼
 YOLO11
     │
     ▼
4 results
```

و مقایسه می‌کنیم:

| Metric          |   Before | After |
| --------------- | -------: | ----: |
| p50 latency     |    XX ms |     ? |
| p95 latency     |    XX ms |     ? |
| throughput      | XX req/s |     ? |
| batch size      |        1 |     4 |
| queue time      |        0 |     ? |
| model inference |    XX ms |     ? |

این خیلی بهتر از این است که صرفاً بگوییم:

> Added batching support.

چون می‌توانیم در README پروژه نشان دهیم:

> **Performance benchmark: single-request inference vs dynamic batching under concurrent load.**

این برای portfolio/Upwork خیلی ارزشمندتر است.

---

# یک تغییر کوچک در `Makefile`

اضافه کن:

```makefile
benchmark:
	python scripts/benchmark.py
```

بعد:

```bash
make benchmark
```

---

# فعلاً این کارها را انجام نده

در این مرحله:

❌ `asyncio.Queue` نساز
❌ worker نساز
❌ batch size اضافه نکن
❌ multiprocessing اضافه نکن
❌ GPU optimization اضافه نکن
❌ Redis اضافه نکن

اول baseline.

---

## معیار پایان Step 7.1

باید این سه چیز را داشته باشیم:

```text
app/inference/engine.py
        │
        └── inference timing

scripts/benchmark.py
        │
        └── baseline latency

Makefile
        │
        └── make benchmark
```

و:

```bash
make check
```

باید همچنان:

```text
15 passed
```

یا بیشتر داشته باشیم.

بعد:

```bash
make benchmark
```

را اجرا کن و **خروجی benchmark را بفرست**.

از روی عدد واقعی سیستم تو، در **Step 7.2** تصمیم می‌گیریم batching را با چه `batch_size` و `max_wait_ms` طراحی کنیم؛ یعنی مثلاً کورکورانه `batch_size=8` نمی‌گذاریم.

(.venv) BBAP-Sec > make benchmark    
python scripts/benchmark.py
request=01 latency=2239.91 ms
request=02 latency=31.88 ms
...
request=20 latency=36.25 ms

--- Baseline ---
requests: 20
mean: 143.36 ms
p50: 31.97 ms
min: 27.93 ms
max: 2239.91 ms

----
عالی. این baseline **اتفاقاً اطلاعات مهمی به ما می‌دهد** و فعلاً نباید batching را اضافه کنیم.

### تحلیل baseline تو

| Metric   |          مقدار |
| -------- | -------------: |
| Requests |             20 |
| Min      |   **27.93 ms** |
| P50      |   **31.97 ms** |
| Mean     |  **143.36 ms** |
| Max      | **2239.91 ms** |

مهم‌ترین نکته:

```text
Request #1   2239.91 ms  ← cold start
Request #2      31.88 ms
Request #3      27.93 ms
...
Request #20     36.25 ms
```

یعنی inference واقعی بعد از warm-up تقریباً **28–42 ms** است؛ میانگین `143 ms` به‌خاطر همان request اول شدیداً skew شده.

پس فعلاً این را به عنوان performance baseline ثبت می‌کنیم:

```text
Cold-start latency: ~2.24 s
Warm latency:       ~28–42 ms
P50 latency:        ~32 ms
```

این خیلی خوب است، چون نشان می‌دهد احتمالاً آن `2.2s` مربوط به **اولین model execution / framework initialization / memory setup** است، نه latency معمول YOLO11.

---

# Step 7.2 — Benchmark را حرفه‌ای‌تر کنیم

Benchmark فعلی یک مشکل دارد:

```python
statistics.mean(latencies)
```

cold start را با warm requests قاطی می‌کند.

برای یک ML serving benchmark حرفه‌ای، باید warm-up را جدا کنیم.

## `scripts/benchmark.py`

قسمت configuration را تغییر بده:

```python
REQUESTS = 20
WARMUP_REQUESTS = 3
```

و ساختار `main()` را این‌طور کن:

```python
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

    elapsed_ms = (
        time.perf_counter() - started
    ) * 1000

    response.raise_for_status()

    return elapsed_ms
```

بعد `main()`:

```python
def main() -> None:
    latencies: list[float] = []

    with open(IMAGE_PATH, "rb") as image_file:
        image_data = image_file.read()

    print(
        f"Warm-up requests: {WARMUP_REQUESTS}"
    )

    for index in range(WARMUP_REQUESTS):
        latency = send_request(image_data)

        print(
            f"warmup={index + 1:02d} "
            f"latency={latency:.2f} ms"
        )

    print(
        f"\nBenchmark requests: {REQUESTS}"
    )

    for index in range(REQUESTS):
        latency = send_request(image_data)

        latencies.append(latency)

        print(
            f"request={index + 1:02d} "
            f"latency={latency:.2f} ms"
        )

    print("\n--- Benchmark ---")

    print(f"requests: {len(latencies)}")

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
```

---

# یک metric مهم دیگر: P95

برای production، `P95` خیلی مهم‌تر از `mean` است.

مثلاً:

```text
P50 = typical request
P95 = slow requests
P99 = tail latency
```

اضافه کن:

```python
def percentile(
    values: list[float],
    percentile_value: float,
) -> float:
    values = sorted(values)

    index = int(
        len(values) * percentile_value / 100
    )

    index = min(
        index,
        len(values) - 1,
    )

    return values[index]
```

و:

```python
print(
    f"p95: "
    f"{percentile(latencies, 95):.2f} ms"
)
```

---

# یک metric دیگر: Throughput

بعداً برای concurrent batching خیلی مهم می‌شود.

فعلاً sequential throughput:

```python
duration_seconds = sum(latencies) / 1000

throughput = (
    len(latencies) / duration_seconds
)
```

و:

```python
print(
    f"throughput: "
    f"{throughput:.2f} req/s"
)
```

---

# خروجی مورد انتظار

بعد از warm-up احتمالاً چیزی نزدیک این خواهی داشت:

```text
--- Benchmark ---

requests: 20
mean: 33.xx ms
p50: 32.xx ms
p95: 4x.xx ms
min: 2x.xx ms
max: 4x.xx ms
throughput: 30.xx req/s
```

عدد واقعی مهم است، نه مثال بالا.

---

# اما یک نکته مهم درباره batching

با baseline فعلی، **نباید انتظار داشته باشیم batching لزوماً latency یک request را کمتر کند.**

Batching برای این است:

```text
                 Single inference
Request 1 ───────────────► YOLO
Request 2 ───────────────► YOLO
Request 3 ───────────────► YOLO
Request 4 ───────────────► YOLO
```

در مقابل:

```text
                 Dynamic batch
Request 1 ─┐
Request 2 ─┤
Request 3 ─┼──► [1,2,3,4] ──► YOLO
Request 4 ─┘
```

هدف اصلی:

**افزایش throughput تحت concurrent load**

است، نه الزاماً کاهش single-request latency.

حتی ممکن است:

```text
single request:
~32 ms

batched request:
~35-50 ms
```

اما throughput مثلاً از:

```text
~30 req/s
```

به:

```text
~70-100 req/s
```

برسد.

این همان trade-off مهم real-time serving است.

---

# Step 7.3 بعدی

بعد از اینکه benchmark جدید را اجرا کردی، می‌رویم سراغ:

```text
DynamicBatcher
```

با دو پارامتر قابل تنظیم:

```yaml
batching:
  enabled: true
  max_batch_size: 4
  max_wait_ms: 10
```

و معماری:

```text
HTTP requests
      │
      ▼
   Queue
      │
      ├── max_batch_size = 4
      │
      └── max_wait_ms = 10ms
                │
                ▼
          DynamicBatcher
                │
                ▼
          InferenceEngine
                │
                ▼
              YOLO
```

**ولی فعلاً همین benchmark warm-up را پیاده کن و `make benchmark` را دوباره بفرست.** از آن عدد برای انتخاب `max_batch_size` و `max_wait_ms` استفاده می‌کنیم، نه از یک مقدار تصادفی.
----

عالی. این baseline الان **واقعاً قابل استفاده برای طراحی batching** است.

نتیجه فعلی:

| Metric                |        Baseline |
| --------------------- | --------------: |
| Warm-up #1            |      2725.72 ms |
| Warm-up #2–3          |          ~35 ms |
| Mean                  |    **44.94 ms** |
| P50                   |    **38.91 ms** |
| P95                   |    **87.18 ms** |
| Min                   |    **30.98 ms** |
| Max                   |    **87.18 ms** |
| Sequential throughput | **22.25 req/s** |

نکته مهم: warm-up اول را از benchmark جدا کرده‌ای، پس `44.94 ms` دیگر تحت تأثیر cold start نیست. این baseline را نگه می‌داریم.

## Step 7.3 — Dynamic Batcher

حالا batching واقعی را اضافه می‌کنیم.

اما طراحی را ساده و قابل توسعه نگه می‌داریم:

```text
                    HTTP Requests
                         │
                         ▼
                  InferenceEngine
                         │
                         ▼
                  DynamicBatcher
                    │        │
                    │        └── max_wait_ms
                    │
                    └────────── max_batch_size
                              │
                              ▼
                         Model inference
                              │
                              ▼
                       Individual results
```

### هدف اولیه

برای اولین implementation:

```yaml
max_batch_size: 4
max_wait_ms: 10
```

چرا؟

چون baseline تو حدود `39 ms P50` است. یک wait window ده‌میلی‌ثانیه‌ای هنوز نسبتاً کوچک است و برای workload real-time منطقی‌تر از مثلاً 50ms است.

**ولی این اعداد را فعلاً بهینه فرض نمی‌کنیم.** بعداً با benchmark مقایسه می‌کنیم.

---

# 7.3.1 — Config

در:

```text
configs/development.yaml
```

اضافه کن:

```yaml
batching:
  enabled: true
  max_batch_size: 4
  max_wait_ms: 10
```

و در production:

```yaml
batching:
  enabled: true
  max_batch_size: 8
  max_wait_ms: 5
```

فعلاً production فقط configuration است؛ هنوز benchmark نکرده‌ایم که `8/5` بهتر باشد.

---

# 7.3.2 — Settings

در `app/core/config.py` یک config model برای batching اضافه کن.

اگر از Pydantic Settings فعلی استفاده می‌کنی:

```python
class BatchingConfig(BaseModel):
    enabled: bool = True
    max_batch_size: int = 4
    max_wait_ms: int = 10
```

و داخل settings:

```python
batching: BatchingConfig = BatchingConfig()
```

فراموش نکن:

```python
from pydantic import BaseModel
```

اگر `BaseModel` دیگری در فایل داری، اسم import را با ساختار فعلی پروژه هماهنگ کن.

---

# 7.3.3 — Batcher

فایل:

```text
app/inference/batching.py
```

فعلاً یک implementation کوچک و تمیز می‌سازیم.

```python
import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass
class InferenceRequest:
    """Single inference request waiting for execution."""

    input_data: Any
    future: asyncio.Future


class DynamicBatcher:
    """Collect inference requests into bounded batches."""

    def __init__(
        self,
        max_batch_size: int = 4,
        max_wait_ms: int = 10,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        self._queue: asyncio.Queue[
            InferenceRequest
        ] = asyncio.Queue()

        self._worker_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background batch worker."""

        if self._running:
            return

        self._running = True

        self._worker_task = asyncio.create_task(
            self._worker()
        )

    async def stop(self) -> None:
        """Stop the background batch worker."""

        self._running = False

        if self._worker_task:
            await self._worker_task

            self._worker_task = None

    async def submit(
        self,
        input_data: Any,
    ) -> Any:
        """Submit one inference request."""

        loop = asyncio.get_running_loop()

        future = loop.create_future()

        request = InferenceRequest(
            input_data=input_data,
            future=future,
        )

        await self._queue.put(request)

        return await future

    async def _worker(self) -> None:
        """Collect and process batches."""

        while self._running:
            request = await self._queue.get()

            batch = [request]

            deadline = (
                perf_counter()
                + self.max_wait_ms / 1000
            )

            while len(batch) < self.max_batch_size:
                remaining = (
                    deadline - perf_counter()
                )

                if remaining <= 0:
                    break

                try:
                    next_request = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=remaining,
                    )

                    batch.append(next_request)

                except asyncio.TimeoutError:
                    break

            await self._process_batch(batch)

    async def _process_batch(
        self,
        batch: list[InferenceRequest],
    ) -> None:
        """Process one collected batch."""

        # Temporary implementation.
        # Model batch inference will be added next.
        for request in batch:
            if not request.future.done():
                request.future.set_result(
                    request.input_data
                )
```

### اما یک نکته مهم

این **هنوز به YOLO وصل نیست**.

عمداً.

فعلاً infrastructure batching را جدا می‌کنیم تا بتوانیم آن را مستقل test کنیم.

---

# 7.3.4 — چرا `Future`؟

این قسمت:

```python
future = loop.create_future()
```

خیلی مهم است.

مثلاً:

```text
Request A
   │
   ▼
 Future A ───────────────┐
                         │
Request B                │
   │                     │
   ▼                     │
 Future B ───────────┐   │
                     │   │
                     ▼   ▼
                  Batch [A,B]
                     │
                     ▼
                  YOLO
                     │
                ┌────┴────┐
                ▼         ▼
             Future A   Future B
                │         │
                ▼         ▼
             Response   Response
```

هر HTTP request نتیجه خودش را دریافت می‌کند، حتی اگر inference به‌صورت batch اجرا شده باشد.

این یکی از بخش‌های مهم طراحی serving system است.

---

# 7.3.5 — Test مستقل Batcher

بساز:

```text
tests/unit/test_batching.py
```

```python
import asyncio

import pytest

from app.inference.batching import DynamicBatcher


@pytest.mark.asyncio
async def test_single_request():
    batcher = DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=10,
    )

    await batcher.start()

    result = await batcher.submit("image-1")

    assert result == "image-1"

    await batcher.stop()
```

و test برای چند request:

```python
@pytest.mark.asyncio
async def test_multiple_requests():
    batcher = DynamicBatcher(
        max_batch_size=4,
        max_wait_ms=10,
    )

    await batcher.start()

    results = await asyncio.gather(
        batcher.submit("image-1"),
        batcher.submit("image-2"),
        batcher.submit("image-3"),
        batcher.submit("image-4"),
    )

    assert results == [
        "image-1",
        "image-2",
        "image-3",
        "image-4",
    ]

    await batcher.stop()
```

---

# یک مشکل کوچک در `stop()`

برای implementation فعلی، worker ممکن است در:

```python
await self._queue.get()
```

گیر کند و `stop()` نتواند به‌موقع آن را متوقف کند.

پس قبل از integration با FastAPI، این را حرفه‌ای‌تر می‌کنیم.

در `DynamicBatcher` یک sentinel اضافه کن:

```python
_STOP = object()
```

ولی چون queue type ما `InferenceRequest` است، بهتر است این بخش را در مرحله بعد با shutdown mechanism تمیز کنیم.

**فعلاً اگر testها پاس شدند، دست نزن.**

---

# خیلی مهم: فعلاً Engine را تغییر نده

در این مرحله:

```text
HTTP
 ↓
InferenceEngine
 ↓
BaseModel
 ↓
YOLO
```

هنوز همان baseline را حفظ می‌کنیم.

Batcher فعلاً isolated است:

```text
DynamicBatcher
    ↑
unit tests
```

بعد در **Step 7.4** آن را به engine وصل می‌کنیم.

این کار عمداً انجام می‌شود تا اگر batching bug داشت، inference فعلی خراب نشود.

---

## حالا اجرا کن

```bash
make check
```

باید:

```text
ruff check .
All checks passed!

pytest
...
```

و تعداد تست‌ها باید از 15 بیشتر شود.

اگر سبز شد، خروجی `make check` را بفرست.

بعد می‌رویم **Step 7.4 — اتصال واقعی DynamicBatcher به `InferenceEngine` و YOLO11**؛ آنجا بخش جذاب شروع می‌شود: batch واقعی `[N, C, H, W]` و سپس benchmark concurrent برای مقایسه throughput با baseline فعلی `22.25 req/s`.
---
عالی. حالا می‌رویم سراغ **Step 7.4 — اتصال واقعی Dynamic Batcher به YOLO11**.

اما یک نکته مهم: چون `DynamicBatcher` فعلی با `Any` کار می‌کند، اگر مستقیم آن را به YOLO وصل کنیم احتمالاً دوباره مجبور به refactor می‌شویم. این بار interface را درست می‌سازیم:

```text
HTTP request
     │
     ▼
InferenceEngine
     │
     ▼
DynamicBatcher
     │
     ├── request 1 ─┐
     ├── request 2 ─┤
     ├── request 3 ─┼──► batch
     └── request 4 ─┘
                       │
                       ▼
                  YOLO11 batch
                       │
                 ┌─────┴─────┐
                 ▼           ▼
              result 1    result 2 ...
```

## 7.4.1 — اول interface مدل را batch-aware کنیم

در `app/models/base.py`، **متد فعلی `inference()` را حذف نکن**. یک متد جدید اضافه کن:

```python
from typing import Any


class BaseModel(ABC):
    ...

    def inference_batch(
        self,
        inputs: list[Any],
    ) -> list[Any]:
        """Run inference on a batch of inputs."""

        return [
            self.inference(item)
            for item in inputs
        ]
```

این نکته مهم است:

### مدل‌هایی که batch واقعی را پشتیبانی نمی‌کنند

خودکار fallback می‌گیرند:

```text
inference_batch([A, B, C])
        │
        ├── inference(A)
        ├── inference(B)
        └── inference(C)
```

اما YOLO11 را بعداً override می‌کنیم تا واقعاً batch را یکجا اجرا کند.

این باعث می‌شود abstraction ما framework-specific نشود.

---

# 7.4.2 — YOLO11 را batch-aware کنیم

در:

```text
app/models/ultralytics.py
```

در کلاس `UltralyticsModel` یک متد:

```python
def inference_batch(
    self,
    inputs: list[Any],
) -> list[Any]:
```

اضافه کن.

اما اینجا یک مسئله مهم داریم:

**`inputs` باید بعد از preprocessing چه شکلی باشند؟**

اگر الان `preprocess()` یک `numpy.ndarray` برمی‌گرداند، می‌توانیم:

```python
images = [
    self.preprocess(item)
    for item in inputs
]
```

بعد YOLO را با list تصاویر صدا بزنیم.

ساختار:

```python
def inference_batch(
    self,
    inputs: list[Any],
) -> list[Any]:
    images = [
        self.preprocess(item)
        for item in inputs
    ]

    outputs = self.model(
        images,
        verbose=False,
    )

    return [
        self.postprocess(output)
        for output in outputs
    ]
```

**ولی این قسمت را فعلاً کورکورانه paste نکن.**

چون implementation فعلی `UltralyticsModel` تو مشخص می‌کند `preprocess()` و `postprocess()` دقیقاً چه typeهایی دارند.

اگر `preprocess()` الان `numpy.ndarray` می‌دهد، همین pattern مناسب است؛ اگر مستقیماً `Results` یا چیز دیگری می‌دهد، باید با implementation خودت هماهنگش کنیم.

---

# 7.4.3 — Batcher باید executor داشته باشد

الان batcher این کار را می‌کند:

```python
await self._process_batch(batch)
```

ولی خودش نباید بداند YOLO چیست.

بهتر است یک callback به آن بدهیم:

```python
Batcher
   │
   └── batch_handler
             │
             ▼
       InferenceEngine
             │
             ▼
           Model
```

پس constructor:

```python
def __init__(
    self,
    batch_handler: Callable[
        [list[Any]],
        Awaitable[list[Any]],
    ],
    max_batch_size: int = 4,
    max_wait_ms: int = 10,
) -> None:
```

و import:

```python
from collections.abc import Awaitable, Callable
```

بعد:

```python
self._batch_handler = batch_handler
```

و `_process_batch()`:

```python
async def _process_batch(
    self,
    batch: list[InferenceRequest],
) -> None:
    inputs = [
        request.input_data
        for request in batch
    ]

    results = await self._batch_handler(inputs)

    if len(results) != len(batch):
        error = RuntimeError(
            "Batch handler returned an invalid "
            "number of results."
        )

        for request in batch:
            if not request.future.done():
                request.future.set_exception(error)

        return

    for request, result in zip(
        batch,
        results,
        strict=True,
    ):
        if not request.future.done():
            request.future.set_result(result)
```

این قسمت خیلی مهم است:

```python
zip(..., strict=True)
```

چون اگر YOLO مثلاً برای 4 input فقط 3 result بدهد، silently خراب نمی‌شویم.

---

# 7.4.4 — Engine مسئول اجرای batch

در `InferenceEngine` یک متد:

```python
async def predict_batch(
    self,
    model_name: str,
    inputs: list[Any],
) -> list[Any]:
    model = self.registry.get(model_name)

    return model.inference_batch(inputs)
```

اما اینجا یک نکته architecture داریم:

**batching باید per-model باشد.**

یعنی:

```text
yolo11 requests ──► YOLO batch
resnet requests ──► ResNet batch
```

نباید این اتفاق بیفتد:

```text
YOLO + ResNet + YOLO
          │
          ▼
       one batch ❌
```

پس batch queue در نهایت باید key داشته باشد:

```text
(model_name, input)
```

ولی برای اولین implementation فقط `yolo11` را batch می‌کنیم.

---

# 7.4.5 — اولین تست واقعی

قبل از اتصال HTTP، یک unit test می‌نویسیم.

`tests/unit/test_batching.py`:

```python
@pytest.mark.asyncio
async def test_batch_handler_receives_multiple_inputs():
    received_batches: list[list[str]] = []

    async def handler(
        inputs: list[str],
    ) -> list[str]:
        received_batches.append(inputs)

        return [
            f"{item}-result"
            for item in inputs
        ]

    batcher = DynamicBatcher(
        batch_handler=handler,
        max_batch_size=4,
        max_wait_ms=20,
    )

    await batcher.start()

    try:
        results = await asyncio.gather(
            batcher.submit("image-1"),
            batcher.submit("image-2"),
            batcher.submit("image-3"),
            batcher.submit("image-4"),
        )

        assert results == [
            "image-1-result",
            "image-2-result",
            "image-3-result",
            "image-4-result",
        ]

        assert received_batches == [
            [
                "image-1",
                "image-2",
                "image-3",
                "image-4",
            ]
        ]
    finally:
        await batcher.stop()
```

این تست به ما ثابت می‌کند:

```text
4 requests
    ↓
ONE handler call
    ↓
4 inputs
    ↓
4 individual responses
```

---

## اما یک اصلاح مهم در طراحی فعلی

من پیشنهاد می‌کنم **فعلاً `DynamicBatcher` را مستقیماً داخل route نگذاریم.**

یعنی این architecture را نمی‌خواهیم:

```text
Route
  ↓
Batcher
  ↓
YOLO
```

بلکه:

```text
Route
  ↓
InferenceEngine
  ↓
Batcher
  ↓
Model
```

چون در آینده:

```text
REST
WebSocket
RTSP
Camera
   │
   ▼
InferenceEngine
   │
   ▼
Batcher
```

همه از یک serving layer استفاده خواهند کرد.

این برای پروژه‌ای که می‌خواهی در Upwork ارائه کنی، architecture بسیار قابل دفاع‌تری است.

---

## فعلاً فقط این بخش را انجام بده

ترتیب کار:

1. `BaseModel.inference_batch()`
2. `DynamicBatcher` را به `batch_handler` مجهز کن
3. `test_batch_handler_receives_multiple_inputs`
4. `make check`

**فعلاً `UltralyticsModel` و HTTP route را تغییر نده.**

اگر تست جدید سبز شد، مرحله بعدی را می‌رویم روی **YOLO11 واقعی** و آنجا batch را به `Ultralytics` وصل می‌کنیم؛ بعد benchmark concurrent می‌سازیم تا ببینیم آیا از baseline فعلی **22.25 req/s** بهتر شده یا نه.
---
