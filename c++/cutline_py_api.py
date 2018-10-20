# _*_ coding: utf-8 _*_

'''
            用法

import cutline_py_api as cla

# img：待切行的图像
# ms：切出来的每一行小图像的list集合
# 每一个list元素是一个二元组，其中第一个元素是bool型，代表该行前面有没有空格；第二个元素是该行的图像
# 若没有检测到，则返回空list
# ms = cla.run(img)

'''

#THE PATH OF SO
SO_PATH="cutline_plus.so"



import numpy as np
import ctypes as ct
import cv2

# resize
def __src_resize(src):
    r0, c0 = src.shape[:2]
    if (r0 > c0 and r0 > 900):
        ra = float(r0) / float(c0)
        dst = cv2.resize(src, (int(900 / ra), 900))
    elif (c0 >= r0 and c0 > 900):
        ra = float(c0) / float(r0)
        dst = cv2.resize(src, (900, int(900 / ra)))
    else:
        return src

    return dst

def show_mats(mats):
    s = len(mats)
    if s == 0:
        dst = np.zeros((100, 100), np.uint8)
        return dst
    mr, mc = mats[0][1].shape[:2]
    gap = 3
    dst = np.zeros(((mr + gap) * s - gap, mc), np.uint8)
    dst[::] = 0
    for i in range(s):
        have_blk, mat = mats[i]
        dst[i * (mr + gap):i * (mr + gap) + mr, 0:mc] = mat
        if have_blk:
            cv2.circle(dst, (15, (mr + gap) * i + int(mr / 2)), 3, 0, 3)

    h, w = dst.shape[:2]
    if h > 1010:
        h_ = 1010
        w_ = int(1010 * w / h)
        dst = cv2.resize(dst, (w_, h_))
        cv2.putText(dst, 'Resized', (2, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, 100)
    return dst

def run(m):
    m=__src_resize(m)
    if not m.flags['C_CONTIGUOUS']:
        print("df")
        m = np.ascontiguous(m, dtype=m.dtype)
    if len(m.shape)!=2:
        m=cv2.cvtColor(m,cv2.COLOR_BGR2GRAY)

    info = np.zeros((1, 2), np.uint8)
    so = ct.cdll.LoadLibrary(SO_PATH)
    so.get_info(m.ctypes.data_as(ct.POINTER(ct.c_ubyte)), info.ctypes.data_as(ct.POINTER(ct.c_ubyte)), ct.c_int(m.shape[0]),
                ct.c_int(m.shape[1]))
    # print(info)

    num, hig = info[0]
    wit = m.shape[1]
    res = []
    for i in range(num):
        res.append([False, np.zeros((hig, wit), np.uint8)])
        res[i][0] = so.get_data(res[i][1].ctypes.data_as(ct.POINTER(ct.c_ubyte)), i)

    return res

if __name__ == '__main__':
    m = cv2.imread('binimg/3.bmp',0)
    res = run(m)
    dst = show_mats(res)
    cv2.imshow('dst', dst)
    cv2.waitKey()
