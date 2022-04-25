import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image

from com_ineuron_apparel.com_ineuron_utils.utils import encodeImageIntoBase64

import sys
sys.path.insert(0, 'com_ineuron_apparel/predictor_yolo_detector')

from com_ineuron_apparel.predictor_yolo_detector.models.common import DetectMultiBackend
from com_ineuron_apparel.predictor_yolo_detector.utils.datasets import LoadStreams, LoadImages
from com_ineuron_apparel.predictor_yolo_detector.utils.general import (
    LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)

from com_ineuron_apparel.predictor_yolo_detector.utils.plots import Annotator, colors, save_one_box
from com_ineuron_apparel.predictor_yolo_detector.utils.torch_utils import select_device, time_sync


class Detector():
    def __init__(self, filename):
        self.weights = "./com_ineuron_apparel/predictor_yolo_detector/best.pt"
        self.conf = float(0.5)
        self.source = "./com_ineuron_apparel/predictor_yolo_detector/inference/images/"
        self.img_size = (416, 416)
        self.save_dir = "./com_ineuron_apparel/predictor_yolo_detector/inference/output"
        self.save_conf = False
        self.view_img = False
        self.save_txt = False
        self.device = 'cpu'
        self.augment = True
        self.agnostic_nms = True
        self.conf_thres = float(0.5)
        self.iou_thres = float(0.45)
        self.classes = None
        self.save_conf = True
        self.update = False
        self.dnn=False
        self.save_crop = False
        self.hide_conf = False,  # hide confidences
        self.hide_labels = False
        self.half = False,  # use FP16 half-precision inference
        self.filename = filename

    def detect(self, save_img=False):
        out, source, weights, view_img, save_txt, imgsz = \
            self.save_dir, self.source, self.weights, self.view_img, self.save_txt, self.img_size
        line_thickness = 3
        webcam = source.isnumeric() or source.startswith(('rtsp://', 'rtmp://', 'http://')) or source.endswith('.txt')

        # Initialize
        #LOGGER
        device = select_device(self.device)
        if os.path.exists(out):  # output dir
            shutil.rmtree(out)  # delete dir
        os.makedirs(out)  # make new dir
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = DetectMultiBackend(weights, device=device)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz)
        else:
            save_img = True
            dataset = LoadImages(source, img_size=imgsz)
            bs = 1

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            t1 = time_sync()
            pred = model(im, augment=self.augment)
            t3 = time_sync()
            dt[1] += t3 - t2

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            dt[2] += time_sync() - t3

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(Path(out) / 'output.jpg')
                txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if self.save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or self.save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if self.hide_labels else (names[c] if self.hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                            if self.save_crop:
                                save_one_box(xyxy, imc, file=out / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                # Stream results
                im0 = annotator.result()

                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                    else:  # 'video' or 'stream'
                        if vid_path[i] != save_path:  # new video
                            vid_path[i] = save_path
                            if isinstance(vid_writer[i], cv2.VideoWriter):
                                vid_writer[i].release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                            vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer[i].write(im0)



            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                # Print results
        t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
        if save_txt or save_img:
            s = f"\n{len(list(out.glob('labels/*.txt')))} labels saved to {out / 'labels'}" if save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', out)}{s}")
        if self.update:
            strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


    def detect_action(self):
        with torch.no_grad():
            self.detect()
        bgr_image = cv2.imread("./com_ineuron_apparel/predictor_yolo_detector/inference/output/output.jpg")
        im_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite('color_img.jpg', im_rgb)
        opencodedbase64 = encodeImageIntoBase64("color_img.jpg")
        result = {"image": opencodedbase64.decode('utf-8')}
        return result

