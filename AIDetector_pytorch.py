import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.BaseDetector import baseDet
from utils.torch_utils import select_device
from utils.datasets import letterbox
from models.common import DetectMultiBackend
# from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                        increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.augmentations import letterbox

class Detector(baseDet):
    def __init__(self, weights, imgsz = (640, 640), half = False, reset_id=True):
        super(Detector, self).__init__()
        self.build_config()
        self.flag = reset_id
        self.weights = weights
    
        # Initialize
        self.device = select_device('0')
        self.model = DetectMultiBackend(weights, device=self.device, dnn=False)
        self.stride, names, pt, jit, onnx, engine = \
            self.model.stride, self.model.names, self.model.pt, self.model.jit, self.model.onnx, self.model.engine
        self.imgsz = check_img_size(imgsz, s=self.stride)

        # Half
        self.half = half
        half &= (pt or jit or onnx or engine) and self.device.type != 'cpu'  # FP16 supported on limited backends with CUDA
        if pt or jit:
            self.model.model.half() if half else self.model.model.float()

    def detect(self, img):
        im0 = img.copy()
        h, w = im0.shape[:2]
        
        # Load model 
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

        # Run inference
        img = letterbox(img, new_shape=(640, 640), stride=self.stride)[0]
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=0, agnostic=False, max_det=1000)

        pred_boxes = []
        for det in pred:
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                for *x, conf, cls_id in det:
                    lbl = "person"
                    x1, y1 = int(x[0]), int(x[1])
                    x2, y2 = int(x[2]), int(x[3])
                    pred_boxes.append(
                        (x1, y1, x2, y2, lbl, conf))

        return img, pred_boxes

    # def detect(self, im):

    #     im0, img = self.preprocess(im)

    #     pred = self.m(img, augment=False)[0]
    #     pred = pred.float()
    #     pred = non_max_suppression(prediction=pred, conf_thres=self.threshold, iou_thres=0.4, classes=0)

    #     pred_boxes = []
    #     for det in pred:

    #         if det is not None and len(det):
    #             det[:, :4] = scale_coords(
    #                 img.shape[2:], det[:, :4], im0.shape).round()

    #             for *x, conf, cls_id in det:
    #                 lbl = self.names[int(cls_id)]
    #                 x1, y1 = int(x[0]), int(x[1])
    #                 x2, y2 = int(x[2]), int(x[3])
    #                 pred_boxes.append(
    #                     (x1, y1, x2, y2, lbl, conf))

    #     return im, pred_boxes

