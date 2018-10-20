# _*_ coding: utf-8 _*_

from __future__ import print_function
import segment as sg
import cv2
import numpy as np

# print (cv2.__version__<'3.1')


imgpath = 'data/D26.jpg'
img = cv2.imread(imgpath)
img_slices = sg.run(img)
cv2.imshow('src', img)
cv2.imshow('dst', sg.show_mats(img_slices))
cv2.waitKey()
