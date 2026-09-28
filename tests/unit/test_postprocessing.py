from types import SimpleNamespace

import numpy as np

from app.inference.postprocessing import postprocess_results


class FakeValue:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


# class FakeBox:
#     def __init__(self):
#         self.cls = [FakeValue(0)]
#         self.conf = [FakeValue(0.95)]
#         self.xyxy = [[10.0, 20.0, 100.0, 200.0]]
#
#     def __len__(self):
#         return 1
class FakeBox:
    """Mocking Ultralytics Boxes behavior."""

    def __init__(self):
        self.cls = np.array([0])  # Class ID
        self.conf = np.array([0.95])  # Confidence score
        self.xyxy = np.array([[10.0, 20.0, 100.0, 200.0]])  # Bounding box

    def __len__(self):
        return len(self.cls)


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
