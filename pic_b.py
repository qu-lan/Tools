#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@File    ：pic_b.py
@Author  ：Xiaodong Qian
@Function:
@Date    ：2022/7/6 21:01 
'''

import cv2
import numpy as np
import copy
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
import shutil

def Partial_magnification(pic, target, location='lower_right', ratio=1):
    '''
    :param pic: input pic
    :param target: Intercept area, for example [target_x, target_y, target_w, target_h]
    :param location: lower_right,lower_left,top_right,top_left,center
    :param ratio: gain
    :return: oringal pic, pic
    '''
    # img = copy.copy(pic)

    w, h = pic.shape[1], pic.shape[0],

    target_x, target_y = target[0], target[1]
    target_w, target_h = target[2], target[3]
    cv2.rectangle(pic, (target_x, target_y), (target_x + target_w, target_y + target_h), (0, 255, 0), 2)
    new_pic = pic[target_y:target_y + target_h, target_x:target_x + target_w]
    new_pic = cv2.resize(new_pic, (target_w*ratio, target_h*ratio), interpolation=cv2.INTER_CUBIC)
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


def Partial_magnification_2(pic, target1, target2, *args):
    '''
    :param pic: input pic
    :param target: Intercept area, for example [target_x, target_y, target_w, target_w]
    :param location: lower_right,lower_left,top_right,top_left,center
    :param ratio: gain
    :return: oringal pic, pic
    '''
    # imgg = copy.copy(pic)
    w, h = pic.shape[1], pic.shape[0],

    assert target1[0]+target1[2] < w and target1[1]+target1[3] < h,\
        'The target1 area is too large and exceeds the image size'
    assert target2[0]+target2[2] < w and target2[1]+target2[3] < h,\
        'The target2 area is too large and exceeds the image size'

    assert target1[2] > 10 or target1[3] > 10, \
        'The target1 area is too small, not recommended'
    assert target2[2] > 10 or target2[3] > 10, \
        'The target2 area is too small, not recommended'

    assert target1[0] > 0 and target1[0] < w, \
        'The starting point of the target1 area is beyond the scope of the image'
    assert target2[0] > 0 and target2[0] < w, \
        'The starting point of the target2 area is beyond the scope of the image'

    if target2[2] / target2[3] == target1[2] / target1[3]:  #
        R = target2[2]/target2[3]
        if target1[1] > target2[1]:
            target1_x, target1_y = target2[0], target2[1]
            target1_w, target1_h = target2[2], target2[3]

            target2_x, target2_y = target1[0], target1[1]
            target2_w, target2_h = target1[2], target1[3]

        else:
            target1_x, target1_y = target1[0], target1[1]
            target1_w, target1_h = target1[2], target1[3]

            target2_x, target2_y = target2[0], target2[1]
            target2_w, target2_h = target2[2], target2[3]

        cv2.rectangle(pic, (target1_x, target1_y), (target1_x + target1_w, target1_y + target1_h), (255, 250, 255), 2)
        cv2.rectangle(pic, (target2_x, target2_y), (target2_x + target2_w, target2_y + target2_h), (255, 252, 255), 2)

        new_pic1 = pic[target1_y:target1_y + target1_h, target1_x:target1_x + target1_w]
        new_pic1 = cv2.resize(new_pic1, (int(h//2 * R), h//2), cv2.INTER_CUBIC)

        new_pic2 = pic[target2_y:target2_y + target2_h, target2_x:target2_x + target2_w]
        new_pic2 = cv2.resize(new_pic2, (int(h//2 * R), h//2))

        img = np.zeros((h, int(h//2 * R), 3), np.uint8)
        img[0:h//2, 0:int(h//2 * R)] = new_pic1
        img[h//2:h, 0:int(h//2 * R)] = new_pic2

        hmerge = np.hstack((pic, img))
        cv2.line(hmerge, (target1_x + target1_w, target1_y), (w, 0), (255, 255, 255), 2)
        cv2.line(hmerge, (target2_x + target2_w, target2_y + target2_h), (w, h), (255, 255, 255), 2)
        return hmerge
    else:
        raise ValueError('Make sure the aspect ratio of target is consistent !')


def copy_pics(images_path, target_path, num, *args):
    images = os.listdir(images_path)
    assert num < len(images), 'Expect num smaller than items of images file!'
    os.makedirs(target_path, exist_ok=True)
    for i, pic in enumerate(tqdm(images, total=num)):
        if i == num:
            break
        if not os.path.exists(os.path.join(target_path, pic)):
            pic_path = os.path.join(images_path, pic)
            shutil.copy(pic_path, target_path)
        else:
            pass
    print(f'Copy {num} pics from <<{images_path}>> to <<{target_path}>> over!')

def crop_pic(pic,data):
    return pic[data[2]:data[3], data[0]:data[1]]

# if __name__ == '__main__':
#     path = r'D:\w\SR-TEST\temp\t'
#     for pic in os.listdir(path):
#         # img = cv2.imread(os.path.join(path, pic))
#         # left = [140, 240, 220, 270]
#         # left = crop_pic(img, left)
#         # left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
#         # right = [285, 285+100, 220, 270]
#         # right = crop_pic(img, right)
#         # right = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)
#         #
#         # mouth = [205, 205+100, 335, 435]
#         # mouth = crop_pic(img, mouth)
#         # mouth = cv2.cvtColor(mouth, cv2.COLOR_BGR2RGB)
#         # hmerge = np.vstack((left, right))
#         # hmerge = np.hstack((hmerge, mouth))
#
#         img = cv2.imread(os.path.join(path, pic))
#         dic = [0,128,256, 512-128, 512]
#         for i in range(4):
#             for j in range(4):
#                 cv2.line(img, (dic[i], dic[j]), (dic[i], 512), (255, 255, 255), 2)
#
#         for i in range(4):
#             for j in range(4):
#                 cv2.line(img, (dic[j],dic[i]), (512,dic[i]), (255, 255, 255), 2)
#
#         left = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         plt.imshow(left)
#         plt.show()

'''左键起始与释放绘制矩形及坐标，右键回退，中键滚动动态放大，中键退出'''
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    img = param[0]
    flag = param[1]-1
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键按下，获得起始点坐标
        try:
            cv2.destroyWindow('bpart')
        except:
            pass
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img, 1])
        point.append((x, y))
        target.append((x, y))
    elif flags == cv2.EVENT_FLAG_LBUTTON:  # 左键长按，事实显示框与坐标
        point.append((x, y))
        for i in range(0, len(point), 2):
            cv2.rectangle(img, point[i], point[i+1], (0, 255, 0), thickness=1, lineType=8, shift=0)
            cv2.circle(img, target[i], 1, (255, 0, 0), thickness=-1)
            cv2.circle(img, point[i], 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, str(point[i]), (point[i][0], point[i][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0),thickness=1)
            cv2.putText(img, str(point[i+1]), point[i+1], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), thickness=1)
        cv2.imshow("image", img)
        point.pop(-1)
        added = img - copyImg
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 1])
    elif event == cv2.EVENT_LBUTTONUP:  # 左键弹起，保存坐标
        point.append((x, y))
        target.append((x, y))
        for i in range(0, len(target), 2):
            cv2.rectangle(img, point[i], point[i+1], (0, 255, 0), thickness=1, lineType=8, shift=0)
            cv2.circle(img, point[i], 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, str(point[i]), (point[i][0], point[i][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0),thickness=1)
            cv2.putText(img, str(point[i+1]), point[i+1], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), thickness=1)
        cv2.imshow("image", img)
    elif event == cv2.EVENT_RBUTTONDOWN:  # 右键按下，撤销前一个框的数据
        try:
            cv2.destroyWindow('bpart')
        except:
            pass
        added = img - copyImg
        img = img - added
        if len(target)>0:
            for _ in range(2):
                target.pop(-1)
                point.pop(-1)
        for i in range(0, len(target), 2):
            cv2.rectangle(img, point[i], point[i+1], (0, 255, 0), thickness=1, lineType=8, shift=0)
            cv2.circle(img, point[i], 1, (255, 0, 0), thickness=-1)
            cv2.putText(img, str(point[i]), (point[i][0], point[i][1]), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 0),thickness=1)
            cv2.putText(img, str(point[i+1]), point[i+1], cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), thickness=1)
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 1])
    elif event == cv2.EVENT_MOUSEWHEEL:  # 中键前滚，标志位置1
        added = img - copyImg
        cv2.imshow("image", img - added)
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 0])
    elif event == cv2.EVENT_MOUSEMOVE and flag:  # 标志位置为1时候，动态放大
        cop = copy.deepcopy(copyImg)
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=1, lineType=8, shift=0)
        part = cop[y1:y2, x1:x2]
        bpart = cv2.resize(part, (10*w, 10*w))
        cv2.imshow("image", img)
        cv2.imshow("bpart", bpart)
        added = img - copyImg
        point.clear()
        target.clear()
        cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img - added, 0])
    elif event == cv2.EVENT_MBUTTONDOWN:  # 中键按下，退出
        cv2.destroyAllWindows()

'''point、target、img、"image"不能修改，和前文定义函数绑定'''
if __name__ == '__main__':
    # events = [i for i in dir(cv2) if 'EVENT'in i]
    # print(events)
    # pics_path = input("Enter pics_path:")
    # save_path = input("Enter save_path:")
    pics_path = 'picss'
    save_path = 'bpart'

    if pics_path is not None:
        pics = os.listdir(pics_path)

    for idx, ppic in enumerate(pics):
        point = []
        target = []
        path = os.path.join(pics_path, ppic)
        img = cv2.imread(path)
        if max(img.shape[0], img.shape[1]) > 800:
            img = cv2.resize(img, (800, 800*img.shape[0]//img.shape[1]))
        if img is not None:
            copyImg = copy.deepcopy(img)
            cv2.namedWindow("image")
            if idx == 0:
                note = 'Please frame the area from\nleft to right and\nfrom top to bottom.'
                text_line = note.split("\n")
                fontScale = 0.4
                thickness = 1
                fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL
                text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
                for i, text in enumerate(text_line):
                    if text:
                        draw_point = [img.shape[0]//8, img.shape[0]//4 + (text_size[1] + 10 + baseline) * i]
                        cv2.putText(img, text, draw_point, fontFace, 1.0, (255, 0, 0), thickness=thickness)
                        cv2.imshow("image", img)
                        cv2.waitKey(666)

            cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN, param=[img-(img-copyImg), 1])
            cv2.imshow("image", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            location = ['top_left', 'top_right', 'lower_right', 'lower_left', 'center']
            img = cv2.imread(path)
            if max(img.shape[0], img.shape[1]) > 800:
                img = cv2.resize(img, (800, 800 * img.shape[0] // img.shape[1]))

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
                fig = plt.figure(figsize=(5, 5))  # figsize 尺寸
                plt.imshow(cv2.cvtColor(pic, cv2.COLOR_BGR2RGB))
                os.makedirs(os.path.join(save_path, 'save'), exist_ok=True)
                plt.savefig(os.path.join(save_path, 'save', ppic), dpi=300, bbox_inches='tight')  # dpi 分辨率
                # plt.show()
                del pic
            except Exception as e:
                pass



    # cv2.imwrite('aaa_bbb11.png',pic)

    # hmerge = np.hstack((pic1, pic2))
    # # hmerge = np.vstack((pic1, pic2))
    # if max(hmerge.shape[0], hmerge.shape[1]) > 1000:
    #     # cv2.namedWindow('merge', 0)
    #     # cv2.resizeWindow('merge', 1000, int(1000*hmerge.shape[0]/hmerge.shape[1]))
    #     # cv2.imshow('merge', hmerge)
    #     # cv2.waitKey(0)
    #     # cv2.destroyAllWindows()
    #     hmerge = cv2.cvtColor(hmerge, cv2.COLOR_BGR2RGB)
    #     fig = plt.figure(figsize=(40, 20))  # figsize 尺寸
    #     plt.imshow(hmerge)
    #     plt.savefig('aaa_bbb.png', dpi=300, bbox_inches='tight')  # dpi     分辨率
    #     plt.show()
    # else:
    #     plt.imshow(hmerge)
    #     plt.show()

# if __name__ == '__main__': #Partial_magnification_2
    # img = cv2.imread(r'C:\Users\Administrator\Desktop\dataset\1\test/Image_11L.jpg')  # C:\Users\Administrator\Desktop\dd.jpg
    # target1 = [250, 650, 100, 100]
    # target2 = [450, 400, 100, 100]
    # pic1 = Partial_magnification_2(img, target1, target2)
    #
    # hmerge = cv2.cvtColor(pic1, cv2.COLOR_BGR2RGB)
    # fig = plt.figure(figsize=(30, 20))
    # fig.patch.set_facecolor('gray')
    # plt.imshow(hmerge)
    # plt.savefig('aaa_bbb.png', dpi=300, bbox_inches='tight')
    # plt.show()

    # path1 = 'D:\Study\Datasets\FFHQ\masks512'
    # path2 = 'D:\Study\Datasets\FFHQ\masks512_10000'
    # copy_pics(path1, path2, 10000)

# flags
# cv2.EVENT_FLAG_ALTKEY   32按Alt不放
# cv2.EVENT_FLAG_CTRLKEY  8按Ctrl不放
# cv2.EVENT_FLAG_LBUTTON  1左键拖拽
# cv2.EVENT_FLAG_MBUTTON  4中键拖拽
# cv2.EVENT_FLAG_RBUTTON  2右键拖拽
# cv2.EVENT_FLAG_SHIFTKEY 16按住Shift不放

# event
# cv2.EVENT_LBUTTONDBLCLK 7左键双击
# cv2.EVENT_LBUTTONDOWN   1左键按下
# cv2.EVENT_LBUTTONUP     4左键释放
# cv2.EVENT_MBUTTONDBLCLK 8中键双击
# cv2.EVENT_MBUTTONDOWN   2中键按下
# cv2.EVENT_MBUTTONUP     5中键释放
# cv2.EVENT_MOUSEHWHEEL   11横向滚轮滚动
# cv2.EVENT_MOUSEMOVE     0鼠标移动
# cv2.EVENT_MOUSEWHEEL    10滚轮滚动
# cv2.EVENT_RBUTTONDBLCLK 8右键双击
# cv2.EVENT_RBUTTONDOWN   2右键按下
# cv2.EVENT_RBUTTONUP     5右键释放
