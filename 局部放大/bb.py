#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：bb.py
@Author  ：Xiaodong
@Function: 左键实时绘制矩形框及坐标；右键撤销上一个框；中键滚动后，移动鼠标动态放大光标处；点击中键退出
@Date    ：2022/10/23 16:49 
'''


from cv2 import EVENT_LBUTTONDOWN, EVENT_FLAG_LBUTTON, EVENT_LBUTTONUP, EVENT_RBUTTONDOWN, EVENT_MOUSEWHEEL, EVENT_MOUSEMOVE, EVENT_MBUTTONDOWN
from cv2 import namedWindow, destroyWindow, destroyAllWindows, setMouseCallback, FONT_HERSHEY_COMPLEX_SMALL
from cv2 import imread, imshow, waitKey, resize, rectangle, circle, putText, getTextSize, INTER_CUBIC, cvtColor, COLOR_BGR2RGB

from copy import deepcopy
from os import listdir, makedirs
from os.path import join as opjoin

from matplotlib.pyplot import figure, axis, xticks, yticks, savefig
from matplotlib.pyplot import imshow as pimshow


def Partial_magnification(pic, target, location='lower_right', ratio=1):
    '''
    :param pic: input pic
    :param target: Intercept area, for example [target_x, target_y, target_w, target_h]
    :param location: lower_right,lower_left,top_right,top_left,center
    :param ratio: gain
    :return: oringal pic, pic
    '''

    w, h = pic.shape[1], pic.shape[0],

    target_x, target_y = target[0], target[1]
    target_w, target_h = target[2], target[3]
    rectangle(pic, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 2)
    new_pic = pic[target_y:target_y + target_h, target_x:target_x + target_w]
    new_pic = resize(new_pic, (target_w*ratio, target_h*ratio), interpolation=INTER_CUBIC)
    if location == 'lower_right':
        pic[h-1-target_h*ratio:h-1, w-1-target_w*ratio:w-1] = new_pic
        # cv2.line(pic, (target_x + target_w, target_y + target_h), (w-1-target_w*ratio, h-1-target_h*ratio), (255, 0, 0),2)
    elif location == 'lower_left':
        pic[h-1-target_h*ratio:h-1, 0:target_w*ratio] = new_pic
    elif location == 'top_right':
        pic[0:target_h*ratio, w-1-target_w*ratio:w-1] = new_pic
    elif location == 'top_left':
        pic[0:target_h*ratio, 0:target_w*ratio] = new_pic
    elif location == 'center':
        pic[int(h/2-target_h*ratio/2):int(h/2+target_h*ratio/2),
            int(w/2-target_w*ratio/2):int(w/2+target_w*ratio/2)] = new_pic
    return img, pic


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):  # 鼠标事件函数
    img = param[0]
    flag = param[1]-1
    if event == EVENT_LBUTTONDOWN:  # 左键按下，获得起始点坐标
        try:
            destroyWindow('bpart')
        except:
            pass
        setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img, 1])
        point.append((x, y))
        target.append((x, y))
    elif flags == EVENT_FLAG_LBUTTON:  # 左键长按，实时显示框与坐标
        point.append((x, y))
        for i in range(0, len(point), 2):
            rectangle(img, point[i], point[i+1], (0, 255, 0), thickness=1, lineType=8, shift=0)
            circle(img, target[i], 1, (255, 0, 0), thickness=-1)
            circle(img, point[i], 1, (255, 0, 0), thickness=-1)
            putText(img, str(point[i]), (point[i][0], point[i][1]), FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), thickness=1)
            putText(img, str(point[i+1]), point[i+1], FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), thickness=1)
        imshow("image", img)
        point.pop(-1)
        added = img - copyImg
        setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 1])
    elif event == EVENT_LBUTTONUP:  # 左键弹起，保存坐标
        point.append((x, y))
        target.append((x, y))
        for i in range(0, len(target), 2):
            rectangle(img, point[i], point[i+1], (0, 255, 0), thickness=1, lineType=8, shift=0)
            circle(img, point[i], 1, (255, 0, 0), thickness=-1)
            putText(img, str(point[i]), (point[i][0], point[i][1]), FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), thickness=1)
            putText(img, str(point[i+1]), point[i+1], FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), thickness=1)
        imshow("image", img)
    elif event == EVENT_RBUTTONDOWN:  # 右键按下，撤销前一个框的数据
        try:
            destroyWindow('bpart')
        except:
            pass
        added = img - copyImg
        img = img - added
        if len(target) > 0:
            for _ in range(2):
                target.pop(-1)
                point.pop(-1)
        for i in range(0, len(target), 2):
            rectangle(img, point[i], point[i+1], (0, 255, 0), thickness=1, lineType=8, shift=0)
            circle(img, point[i], 1, (255, 0, 0), thickness=-1)
            putText(img, str(point[i]), (point[i][0], point[i][1]), FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0), thickness=1)
            putText(img, str(point[i+1]), point[i+1], FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), thickness=1)
        imshow("image", img)
        setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 1])
    elif event == EVENT_MOUSEWHEEL:  # 中键前滚，标志位置1
        added = img - copyImg
        imshow("image", img - added)
        setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 0])
    elif event == EVENT_MOUSEMOVE and flag:  # 标志位置为1时候，动态放大
        cop = deepcopy(copyImg)
        shape = cop.shape
        w = 30
        x1 = 0 if x-w < 0 else x-w
        y1 = 0 if y-w < 0 else y-w
        x2 = x + w if x + w < shape[1] else shape[1]
        y2 = y + w if y + w < shape[0] else shape[0]
        if x1 == 0: x2 = 2*w
        if y1 == 0: y2 = 2*w
        if x2 == shape[1]: x1 = x2-2*w
        if y2 == shape[0]: y1 = y2-2*w
        rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=8, shift=0)
        part = cop[y1:y2, x1:x2]
        bpart = resize(part, (10*w, 10*w))
        imshow("image", img)
        imshow("bpart", bpart)
        added = img - copyImg
        point.clear()
        target.clear()
        setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 0])
    elif event == EVENT_MBUTTONDOWN:  # 中键按下，退出
        destroyAllWindows()


if __name__ == '__main__':
    pics_path = 'picss'
    save_path = 'bpart'

    if pics_path is not None:
        pics = listdir(pics_path)

    for idx, ppic in enumerate(pics):
        point = []
        target = []
        path = opjoin(pics_path, ppic)
        img = imread(path)
        if max(img.shape[0], img.shape[1]) > 800:
            img = resize(img, (800, 800*img.shape[0]//img.shape[1]))
        if img is not None:
            copyImg = deepcopy(img)
            namedWindow("image")
            if idx == 0:
                note = 'Please frame the area from\nleft to right and\nfrom top to bottom.'
                text_line = note.split("\n")
                fontScale = 0.4
                thickness = 1
                fontFace = FONT_HERSHEY_COMPLEX_SMALL
                text_size, baseline = getTextSize(str(text_line), fontFace, fontScale, thickness)
                for i, text in enumerate(text_line):
                    if text:
                        draw_point = [img.shape[0]//8, img.shape[0]//4 + (text_size[1] + 10 + baseline) * i]
                        putText(img, text, draw_point, fontFace, 1.0, (255, 0, 0), thickness=thickness)
                        imshow("image", img)
                        waitKey(666)

            setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img-(img-copyImg), 1])
            imshow("image", img)
            waitKey(0)
            destroyAllWindows()

            location = ['top_left', 'top_right', 'lower_right', 'lower_left', 'center']
            img = imread(path)
            if max(img.shape[0], img.shape[1]) > 800:
                img = resize(img, (800, 800 * img.shape[0] // img.shape[1]))

            for i in range(len(target)//2):
                if target[2*i+1][0]-target[2*i][0] > 0:
                    target1 = [target[2*i][0], target[2*i][1], target[2*i+1][0]-target[2*i][0], target[2*i+1][1]-target[2*i][1]]
                else:
                    raise ValueError('请从左往右，从下往上 框取区域')
                if i == 0:
                    pic, pic1 = Partial_magnification(img, target1, location=location[i], ratio=2)
                else:
                    pic, pic1 = Partial_magnification(pic, target1, location=location[i], ratio=2)
            try:
                fig = figure(figsize=(5, 5))  # figsize 尺寸
                axis('off')  # 去坐标轴
                xticks([])  # 去 x 轴刻度
                yticks([])  # 去 y 轴刻度
                pimshow(cvtColor(pic, COLOR_BGR2RGB))
                makedirs(opjoin(save_path, 'save'), exist_ok=True)
                savefig(opjoin(save_path, 'save', ppic), dpi=300, bbox_inches='tight')  # dpi 分辨率
                del pic
            except Exception as e:
                pass
