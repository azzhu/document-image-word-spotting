
import numpy as np
import ctypes as ct
import cv2
import cutline_py_api as cla
import time

def myresize(src):
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


for i in range(1,55):
    fn='./binimg/D'+str(i)+'.jpg'
    img=cv2.imread(fn)
    # img=cv2.transpose(img)
    # cv2.imshow("src",img)

    #print img
    if not img.flags['C_CONTIGUOUS']:
        print("not C_CONTIGUOUS")
        a = np.ascontiguous(img, dtype=img.dtype)
    # _, img=cv2.threshold(img,175,255,cv2.THRESH_BINARY)

    a=time.time()
    res=cla.run(img)
    print 'time:',time.time()-a

    dst=cla.show_mats(res)
    cv2.imshow("dst",dst)

    cv2.waitKey()

