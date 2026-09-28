This is a fantastic goal! Upgrading a local batch-processing script into a production-ready **Real-Time API with MLOps pipelines (MLflow + DVC)** is exactly how modern computer vision systems are built.

Looking at your old script, here are a few immediate observations:
1. **Unused Preprocessing:** You imported `torchvision.transforms` and wrote a `preprocess_image` function, but you never actually used it. This is completely fine because Ultralytics YOLOv8 automatically handles resizing, normalization, and tensor conversion under the hood!
2. **Double I/O operations:** You are passing the `image_path` to the model (which reads it from the disk), and then you use `cv2.imread(img_path)` to read it *again* to draw the boxes. This double-read is a major bottleneck for real-time applications.

Here is a comprehensive guide and the refactored code to hit your target architecture.

---

### Phase 1 & 2: Building the FastAPI Service
To make this an API, we need to move away from reading local folders (`os.listdir`) and instead accept images over HTTP, run inference in memory, and return a JSON payload with the bounding box coordinates.

**1. Install required packages:**
```bash
pip install fastapi uvicorn python-multipart ultralytics opencv-python-headless
```

**2. Create `../app/main.py` (The FastAPI App):**
```python
from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Real-time object detection using YOLOv8",
    version="1.0.0",
)

# Load the model globally so it only loads once when the server starts
# Replace 'yolov8n.pt' with your fine-tuned model path later (e.g., 'models/best.pt')
model = YOLO("yolov8n.pt")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read the uploaded image file into memory
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run inference directly on the OpenCV image (No need to read from disk!)
    results = model(img)

    # 3. Parse the results into a JSON-friendly format
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        detections.append(
            {
                "class": label,
                "confidence": round(conf, 2),
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            }
        )

    return {"filename": file.filename, "object_count": len(detections), "detections": detections}


# To run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

### Phase 3: MLOps Pipeline (DVC + MLflow)
To achieve that **85% mAP**, you will likely need to fine-tune the model on custom data.

**1. DVC (Data Version Control):**
Instead of storing gigabytes of images in Git, you use DVC to track your datasets and model weights, while Git only tracks the lightweight `.dvc` pointer files.
```bash
# Initialize DVC
dvc init

# Track your dataset and model weights
dvc add yolo8/data/images
dvc add models/yolov8n.pt

# Push data to remote storage (e.g., AWS S3)
dvc remote add -d myremote s3://my-yolo-bucket/dvcstore
dvc push
```

**2. MLflow (Experiment Tracking):**
Ultralytics has **built-in support** for MLflow. When you are training your model to reach that 85% mAP, you just need to ensure MLflow is installed and configured.
```bash
pip install mlflow
```
In your training script (`train.py`):
```python
import mlflow
from ultralytics import YOLO

# Tell Ultralytics to log to MLflow
# It will automatically log metrics like mAP50, mAP50-95, loss, and the final model weights!
model = YOLO("yolov8n.pt")
results = model.train(
    data="your_dataset.yaml", epochs=50, imgsz=640, project="YOLOv8_API_Project", name="run_1"
)
```

---

### Phase 4: AWS EC2 Deployment Architecture

To deploy this reliably on AWS EC2, you shouldn't just run the python script. You should containerize it.

**1. Create a `Dockerfile`:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 8000
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**2. Deployment Steps on EC2:**
1. Provision an **AWS EC2 Ubuntu Instance** (a `t3.medium` or `t3.large` is usually sufficient for CPU inference on the YOLOv8-nano model. If you use a heavier model, you may need a `g4dn` instance with an NVIDIA GPU).
2. SSH into your EC2 instance and install Docker.
3. Clone your GitHub repository and pull your dataset/models using `dvc pull`.
4. Build and run the docker container:
   ```bash
   docker build -t yolo-api .
   docker run -d -p 80:8000 yolo-api
   ```
5. You can now send POST requests with images directly to your EC2 instance's Public IP!
```bash
# Test your new live API
curl -X POST "http://<YOUR_EC2_IP>/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

```</YOUR_EC2_IP>