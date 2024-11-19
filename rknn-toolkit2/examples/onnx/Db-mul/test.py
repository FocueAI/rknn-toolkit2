import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2, math
from rknn.api import RKNN
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import Polygon
import pyclipper

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = '/usr/projects/explore/rknn-toolkit2/rknn-toolkit2/examples/onnx/Db-mul/final_2024_05_05_17_02.onnx'
RKNN_MODEL = '/usr/projects/explore/rknn-toolkit2/rknn-toolkit2/examples/onnx/Db-mul/final_2024_05_05_17_02.rknn'
IMG_PATH = '/usr/projects/explore/rknn-toolkit2/rknn-toolkit2/examples/onnx/Db-mul/IMG_20191014_161409.jpg'
DATASET = '/usr/projects/explore/rknn-toolkit2/rknn-toolkit2/examples/onnx/Db-mul/dataset.txt'

QUANTIZE_ON =  False

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    old_h, old_w = shape
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) # 看看谁变化最小

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        new_h, new_w = im.shape[:2]
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    img_info = {
        "im": im,
        'w_ratio': old_w/new_w,
        'h_ratio': old_h/new_h,
        'old_w': old_w,
        'old_h': old_h,
        'new_w': new_w,
        'new_h': new_h,
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right
    }
    return img_info


class SegDetector:
    def __init__(self, **kargs):
        self.thresh = kargs["thresh"]
        self.max_candidates = kargs["max_candidates"]
        self.min_size = kargs["min_size"]
        self.box_thresh = kargs["box_thresh"]
        
        
    def binarize(self, pred):
        return pred > self.thresh
    
    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])
    
    def unclip(self, box, unclip_ratio=1.5):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
    
    def box_from_bitmap(self, pred, img_infoes):
        bitmap = self.binarize(pred)[0][0]
        pred = pred[0]
        height, width = pred.shape[1:]
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        output = np.copy(bitmap)*255.0
        output = np.ones_like(output)[:,:,None] * 255
        output =output.repeat(3, axis=-1)
        output = output.astype(np.uint8)
        cv2.drawContours(output, contours, -1, (255, 255, 0), 2)
        cv2.imwrite('contours.jpg', output)
        
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred[0], points.reshape(-1, 2))     
            if score < self.box_thresh:
                continue
            box = self.unclip(points, unclip_ratio=1.2).reshape(-1, 1, 2)
            ##########################################
            box_ = np.squeeze(box)
            if Polygon(box_).area < 12000:
                   continue
            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            if sside < self.min_size + 2:
                continue
            ######################################### img_infoes
            box[:, 0] = np.clip( (np.round(box[:, 0] - img_infoes["left"] ) / img_infoes["new_w"] * img_infoes["old_w"] ) , 0, img_infoes["old_w"])
            box[:, 1] = np.clip( (np.round(box[:, 1] - img_infoes["top"]  ) / img_infoes["new_h"] * img_infoes["old_h"]), 0, img_infoes["old_h"])
            ####################################
            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            if sside < self.min_size + 2:
                continue
            ####################################
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score     
        return boxes, scores          
            
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int16), 0, w - 1) # np.int ---> np.int16 hou
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int16), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int16), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int16), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)

        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def represent(self, pred, img_infoes):
       boxes, scores = self.box_from_bitmap(pred, img_infoes)
       return boxes, scores





if __name__ == '__main__':
    rk_model_gen = True
    config={
        "thresh": 0.5,
        "max_candidates": 999999,
        "min_size": 0.002,
        "box_thresh": 0
    }
    deteor = SegDetector(**config)
    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ########################## 增加一些其他配置 ########################
    # image_leng_side = 1728  # 由于该平台的局限性，先这样设置
    image_leng_side = 1280
    dynamic_input = [
    # [[1,3,1280,1728]],    # 第一个输入形状
    # [[1,3,1728,1280]],    # 第二个输入形状
    [[1,3,image_leng_side,image_leng_side]] 
    ]
    rknn.config(#mean_values=[103.94, 116.78, 123.68], 
            #std_values=[58.82, 58.82, 58.82], 
            quant_img_RGB2BGR=True, 
            dynamic_input=dynamic_input, target_platform='rk3588')
    ##################################################################
    if rk_model_gen:
        ret = rknn.load_onnx(model=ONNX_MODEL,)
        if ret != 0:
            print('Load model failed!')
            exit(ret)
        print('done')

    # Build model
    print('--> Building model')
    if rk_model_gen:
        ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
        if ret != 0:
            print('Build model failed!')
            exit(ret)
        print('done')

        # Export RKNN model
        print('--> Export rknn model')
        ret = rknn.export_rknn(RKNN_MODEL)
        if ret != 0:
            print('Export rknn model failed!')
            exit(ret)
        print('done')
    else:
        ret = rknn.build(do_quantization=QUANTIZE_ON, dataset=DATASET)
        rknn.load_rknn("/usr/projects/explore/rknn-toolkit2/rknn-toolkit2/examples/onnx/Db-mul/final_2024_05_05_17_02.rknn")
    #################################################  编写具体的图像处理流程 ############################################
    # Init runtime environment
    ret = rknn.init_runtime()
    if ret != 0:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    # Set inputs
    img = cv2.imread(IMG_PATH)
    
    img_h, img_w = img.shape[:2]
    
    if img_h > img_w: # 1280,1728
        new_height = image_leng_side
        new_width = int(math.ceil(new_height / img_h * img_w / 32) * 32)
    else:  
        new_width = image_leng_side
        new_height = int(math.ceil(new_width / img_h * img_w / 32) * 32)
        
    img_infoes = letterbox(img,new_shape=(image_leng_side,image_leng_side),color=(125,125,125))
    # im, ratio, (top, _, left,  _) = img_infoes
    im = img_infoes["im"] - np.array([122.67891434, 116.66876762, 104.00698793])
    img2 = np.expand_dims(im, 0)
    img2 = img2/255.
    img2 = img2.astype(np.float32)
    rknn_infere_result = rknn.inference(inputs=[img2], data_format=['nhwc'])
    # print(f'outputs:{rknn_infere_result}')
    pred, mcls = rknn_infere_result[0], rknn_infere_result[1]
    print(f'pred:{pred}')
    detor_info = deteor.represent(pred, img_infoes)
    detect_boxes = detor_info[0]
    
    for box in detect_boxes:
        bbox = np.array(box,np.int32)
        print(f'bbox:{bbox}')
        cv2.polylines(img, [bbox], True, (0, 255, 0), 3) 
    cv2.imwrite('res.jpg', img)
    # print(f'detor_info:{detor_info}')
    rknn.release()
