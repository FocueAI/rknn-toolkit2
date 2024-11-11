import os
import urllib
import traceback
import time
import sys
import numpy as np
import cv2, math
from rknn.api import RKNN

# Model from https://github.com/airockchip/rknn_model_zoo
ONNX_MODEL = 'final_2024_05_05_17_02.onnx'
RKNN_MODEL = 'final_2024_05_05_17_02.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'

QUANTIZE_ON =  True

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # pre-process config
    print('--> Config model')
    # rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]], target_platform='rk3588')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    ########################## 增加一些其他配置 ########################
    dynamic_input = [
    # [[1,3,1280,1728]],    # 第一个输入形状
    # [[1,3,1728,1280]],    # 第二个输入形状
    [[1,3,1728,1728]] 
    ]
    rknn.config(mean_values=[103.94, 116.78, 123.68], 
            std_values=[58.82, 58.82, 58.82], 
            quant_img_RGB2BGR=True, 
            dynamic_input=dynamic_input, target_platform='rk3588')
    ##################################################################
    ret = rknn.load_onnx(model=ONNX_MODEL,)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
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
    image_leng_side = 1728  # 由于该平台的局限性，先这样设置
    if img_h > img_w: # 1280,1728
        new_height = image_leng_side
        new_width = int(math.ceil(new_height / img_h * img_w / 32) * 32)
    else:  
        new_width = image_leng_side
        new_height = int(math.ceil(new_width / img_h * img_w / 32) * 32)
        
    img_infoes = letterbox(img,new_shape=(1728,1728))
    im, ratio, (dw, dh) = img_infoes
    img2 = np.expand_dims(im, 0)
    rknn_infere_result = rknn.inference(inputs=[img2], data_format=['nhwc'])
    print(f'outputs:{rknn_infere_result}')
    pred, mcls = rknn_infere_result[0], rknn_infere_result[1]
    # ---------------------- ----------------------------------------
    # def softmax(x, axis):
    #     x -= np.max(x, axis=axis, keepdims=True)
    #     f_x = np.exp(x) / np.sum(np.exp(x), axis=axis, keepdims=True)
    #     return f_x
    # b, _, w, h = mcls.shape
    # out_class = mcls.transpose(0, 2, 3, 1).reshape(-1, self.class_num)
    # out_class = softmax(out_class, -1)
    # out_class = np.expand_dims(out_class.max(1).reshape(b, w, h),axis=1)
    # boxes, scores, classes = self.structure.representer.represent(batch, pred, out_class,
    #                                                                 is_output_polygon=poly_flag)
    
    
    
    
    ####################################################################################################################
    # # Init runtime environment
    # print('--> Init runtime environment')
    # ret = rknn.init_runtime()
    # if ret != 0:
    #     print('Init runtime environment failed!')
    #     exit(ret)
    # print('done')

    # # Set inputs
    # img = cv2.imread(IMG_PATH)
    # # img, ratio, (dw, dh) = letterbox(img, new_shape=(IMG_SIZE, IMG_SIZE))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # # Inference
    # print('--> Running model')
    # img2 = np.expand_dims(img, 0)
    # outputs = rknn.inference(inputs=[img2], data_format=['nhwc'])
    # np.save('./onnx_yolov5_0.npy', outputs[0])
    # np.save('./onnx_yolov5_1.npy', outputs[1])
    # np.save('./onnx_yolov5_2.npy', outputs[2])
    # print('done')

    # # post process
    # input0_data = outputs[0]
    # input1_data = outputs[1]
    # input2_data = outputs[2]

    # input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
    # input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
    # input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))

    # input_data = list()
    # input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    # input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))

    # boxes, classes, scores = yolov5_post_process(input_data)

    # img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # if boxes is not None:
    #     draw(img_1, boxes, scores, classes)
    #     cv2.imwrite('result.jpg', img_1)
    #     print('Save results to result.jpg!')

    rknn.release()
