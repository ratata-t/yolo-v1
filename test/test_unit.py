from utils.data import Bbox
from yolo.loss import IoU


def test_iou():
    assert IoU(Bbox(0, 0, 10, 10), Bbox(0, 0, 10, 10)) == 1
    assert IoU(Bbox(0, 0, 5, 5), Bbox(5, 5, 10, 10)) == 0
    assert IoU(Bbox(0, 0, 10, 10), Bbox(5, 5, 15, 15)) == 1 / 7
    assert IoU(Bbox(0, 0, 5, 5), Bbox(0, 5, 5, 10)) == 0
