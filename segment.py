# _*_ coding: utf-8 _*_

'''
            用法

import segment as sg

#img：待切行的图像
#ms：切出来的每一行小图像的list集合
#每一个list元素是一个二元组，其中第一个元素是bool型，代表该行前面有没有空格；第二个元素是该行的图像
#若没有检测到，则返回空list
ms = sg.run(img)

#默认返回灰度图，若需要返回二值图，则：
ms = sg.run(img, True)

'''

# 切行算法V2.0

import cv2
import numpy as np
import math
import copy


# 全局变量，行距，行高
HJ = -1
KD = -1
# 图像高和宽
H = 1
W = 1
# heatmap
IMG_HM = np.zeros((H, W), np.uint8)
# 完成切行后，每一行的纵坐标（rect左上点）信息保存在这里面
MSPOS = []


# 类似于C++里的Pair类
class Pair:
    def __init__(self, first, second):
        self.first = first
        self.second = second


# 用来保存小区域（平行四边形）的四个顶点，以便后续仿射变换
class WarpPs:
    def __init__(self, p1, p2, p3, p4):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4


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


# 计算书写方向，输入二值图像
def __calDir(imgbin):
    img_bin = copy.deepcopy(imgbin)
    edges = cv2.Canny(img_bin, 200, 100)
    param = 100
    while True:
        ls = cv2.HoughLines(edges, 1, np.pi / 180, param)
        if np.any(ls):
            if len(ls) <= 5:
                param -= 5
            elif len(ls) >= 500:
                param += 5
            else:
                break
        else:
            param -= 5
        if param <= 0 or param >= 300:
            return 1.5708  # 返回1.5708，即水平方向，不旋转
    # print param
    roate_list = []
    for i in range(len(ls)):
        abs_roate = abs(ls[i][0][1] - 1.57075)
        if abs_roate < 0.2:
            #   print abs_roate
            roate_list.append(ls[i][0][1])
    # print roate_list

    if len(roate_list) > 0:
        roate = np.mean(roate_list)
    else:
        roate = 1.5708
    return roate


# 根据书写方向旋转图像
def __warpImg(img, dir):
    if abs(dir - 1.57075) < 0.05 or abs(dir - 1.57075) > 0.5:
        # 角度过小时不需要旋转处理，过大时不符合常理，很有可能是干扰因素
        # 造成的方向计算错误，也不予处理
        return img
    r, c = img.shape[:2]
    theta = 3.1416 - dir
    # warp_dst = np.zeros(img.shape, np.uint8)
    M1 = np.float32([[c / 2, r / 2], [c / 2, r - 1], [c - 1, r / 2]])
    M2 = np.float32([[c / 2, r / 2], [c / 2 + (r / 2) * math.cos(theta), r / 2 + (r / 2) * math.sin(theta)],
                     [c / 2 + (c / 2) * math.sin(theta), r / 2 - (c / 2) * math.cos(theta)]])
    M = cv2.getAffineTransform(M1, M2)
    # dst = cv2.warpAffine(img, M, (c, r), None, cv2.WARP_FILL_OUTLIERS, cv2.BORDER_CONSTANT)
    dst = cv2.warpAffine(img, M, (c, r), None, cv2.WARP_FILL_OUTLIERS, cv2.BORDER_WRAP)

    return dst


# 计算heatmap,heatmap特点：对上下不敏感，对左右敏感
# 输入二值图像
def __get_heatmap_old(img):
    # param
    th = 0.2
    ker = 30
    ker2 = 10

    th_num = int((2 * ker + 1) * th)
    th_num2 = int((2 * ker2 + 1) * th)

    r, c = img.shape[:2]
    heatmap = np.zeros((r, c, 1), np.uint8)
    # 边界附近小于ker的区域不作处理
    for i in range(r):
        for j in range(ker, c - ker):
            roi = img[i: i + 1, j - ker:j + ker + 1]
            heatval = 2 * ker + 1 - cv2.countNonZero(roi)
            if (heatval > th_num):
                heatmap[i, j] = 255
            # 使用双核
            roi2 = img[i:i + 1, j - ker2:j + ker2 + 1]
            heatval2 = 2 * ker2 + 1 - cv2.countNonZero(roi2)
            if (heatval2 > th_num2):
                heatmap[i, j] = 255
    pass

    heatmap = cv2.medianBlur(heatmap, 5)
    kernel = np.ones((5, 5), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel)

    return heatmap


# 求行距和宽度
def __get_hangnju_kuandu_v2(heatmap):
    src = cv2.transpose(heatmap)
    r, c = src.shape[:2]
    hj = []
    kd = []

    # 画十条竖线，找交叉点
    bu = int(r / 10)
    for k in range(1, 10):
        pos = bu * k
        st = -1
        ed = -1
        for i in range(1, c):
            v0 = src[pos, i - 1]
            v1 = src[pos, i]
            if (v0 == 0 and v1 == 255):
                if (st > 0 and ed > 0):
                    if (i - st >= 5 and ed - st >= 5 and ed - st < 60):
                        hj.append(i - st)
                        kd.append(ed - st)
                    pass
                st = i
                pass
            elif (v0 == 255 and v1 == 0):
                ed = i
        pass
    pass

    # 排序，减两头，再平均
    hj.sort()
    kd.sort()
    size = len(hj)
    tem = int(size / 7)
    if (tem == 0): tem = 1
    sum_hj, sum_kd = 0, 0
    for i in range(tem, size - tem):
        sum_hj += hj[i]
        sum_kd += kd[i]
        pass
    res = Pair(0, 0)
    if (int(size - tem * 2) == 0):
        res.first = 50
    else:
        res.first = int(sum_hj / (size - tem * 2))
    if (int(size - tem * 2) == 0):
        res.second = 20
    else:
        res.second = int(sum_kd / (size - tem * 2))
    if (res.first < 15 and res.second < 10):
        res.first = 50
        res.second = 20

    return res


def __get_heatmap_v3(img):
    # param  th = 0.15  ker = 30
    th = 0.14
    ker = 40
    h = 1
    th_num = int((2 * ker + 1) * (2 * h + 1) * th)

    # 原图加pad
    r0, c0 = img.shape[:2]
    padimg = np.ones((r0 + h * 2, c0 + ker * 2), np.uint8)
    padimg *= 255
    padimg[h:h + r0, ker:ker + c0] = img
    img = padimg

    def get_sum(img_heat):
        hight, width = img_heat.shape[:2]
        # d = np.random.rand(hight,width)    ST = time.time()
        # d = (d>0.5).astype(int)
        p = np.zeros((hight + 1, width + 1), int)
        p[1:, 1:] = img_heat
        # d = p    #print d.shape    #print d
        for i in range(1, img_heat.shape[0] + 1):
            p[i, :] += p[i - 1, :]
        for i in range(1, img_heat.shape[1] + 1):
            p[:, i] += p[:, i - 1]
        hw = 2 * h + 1
        dw = 2 * ker + 1
        ans = p[hw:, dw:] + p[:hight - hw + 1, :width - dw + 1] - p[:hight - hw + 1, dw:] - p[hw:, :width - dw + 1]
        return ans

    r, c = img.shape[:2]
    heatmap = np.zeros((r, c), np.uint8)
    kernel = np.ones((2 * h + 1, 2 * ker + 1))
    img_heat = (img > 0.5).astype(np.uint8)
    #   print time.time()-a
    # #  grad = signal.convolve2d(img_heat, kernel, boundary='symm', mode='valid')
    #   print("hellow")
    grad = get_sum(img_heat)
    # print time.time()-a
    heatval = ((2 * ker + 1) * (2 * h + 1) - grad) > th_num
    heatmap[1:-1, ker:-ker] = np.uint8(heatval * 255)
    heatmap = cv2.medianBlur(heatmap, 5)
    kernel = np.ones((5, 5), np.uint8)
    heatmap = cv2.dilate(heatmap, kernel)

    heatmap = heatmap[h:h + r0, ker:ker + c0]

    return heatmap


# 优化heatmap
def __heatmap_opt(heatmap):
    '''
    思想：如果一个点是黑的，且这个点下面第th个点也是黑的，
    则把这两个点中间所有点都置黑。
    '''

    # param
    th = 10  # 窗口高度

    r, c = heatmap.shape[:2]
    if r < th + 2:
        return heatmap
    img = heatmap.copy()
    imgpad = np.zeros((r + 2, c), np.uint8)
    imgpad[1:-1, :] = img
    imgpad = (imgpad > 0).astype(np.uint8)
    imgpad[:-th, :] += imgpad[th:, :]
    imgpad = (imgpad > 0).astype(np.uint8)
    # 生成mask
    for t in range(th - 1):
        for i in range(r + 1):
            i = r + 1 - i
            imgpad[i, :] *= imgpad[i - 1, :]
    mask = imgpad[1:-1, :]
    mask = (mask > 0).astype(np.uint8)
    dst = mask * img
    return dst


# 去掉小的区域
def __heatmap_opt2(heatmap):
    # param
    min_size = 600

    heatmap = heatmap.astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(heatmap)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2


# 获取特征点，用画竖线的方法，
def __get_featurePs_old(heatmap):
    # 为便于处理，先对图像进行转置
    img = cv2.transpose(heatmap)
    r, c = img.shape[:2]

    # param
    ls = 19  # 特征点密集度，即画几根竖线
    stp = int(r / (ls + 1))

    # 下面开始，基于每根线提取特征点
    fps = []
    for l in range(1, ls + 1):
        pos = stp * l
        temp = []
        st, ed = -1, -1
        for i in range(1, c):
            v0 = img[pos, i - 1]
            v1 = img[pos, i]
            if (v0 == 0 and v1 == 255):
                st = i
            elif (v0 == 255 and v1 == 0):
                ed = i
                if (st > 0 and ed > 0): temp.append((pos, int((st + ed) / 2)))
            pass
        pass
        fps.append(temp)

    # 去掉空的
    fps_ = []
    for it in fps:
        if (len(it) != 0):
            fps_.append(it)

    return fps_


def __get_featurePs_v2(heatmap):
    # 为便于处理，先对图像进行转置
    img = cv2.transpose(heatmap)
    r, c = img.shape[:2]

    # param
    ls = 19  # 特征点密集度，即画几根竖线
    stp = int(r / (ls + 1))

    # 下面开始，基于每根线提取特征点
    fps = []
    for l in range(1, ls + 1):
        pos = stp * l
        temp = []
        st, ed = -1, -1
        for i in range(1, c):
            v0 = img[pos, i - 1]
            v1 = img[pos, i]
            if (v0 == 0 and v1 == 255):
                st = i
            elif (v0 == 255 and v1 == 0):
                ed = i
                if st > 0 and ed > 0 and ed - st >= 10:
                    if ed - st > HJ:
                        ls = int((ed - st) / HJ)
                        for m in range(ls + 1):
                            # pp = (pos, int(st + KD / 2 + HJ * m))
                            # 加上一个判断，防止越界（越界原因：当HJ、KD计算不出来时，它们还是会被赋给一个经验值，这个经验值可能会导致越界）
                            if int(st + KD / 2 + HJ * m) < ed and heatmap[
                                int(st + KD / 2 + HJ * m), pos] == 255:
                                temp.append((pos, int(st + KD / 2 + HJ * m)))
                    else:
                        temp.append((pos, int((st + ed) / 2)))

                    tes = len(temp)
                    if tes >= 2:
                        y1 = temp[tes - 1][1]
                        y2 = temp[tes - 2][1]
                        if y1 - y2 < KD:
                            temp.pop()
                            temp.pop()
                            temp.append((pos, int((y1 + y2) / 2)))
            pass
        pass
        fps.append(temp)

    # 去掉空的
    fps_ = []
    for it in fps:
        if (len(it) != 0):
            fps_.append(it)

    return fps_


# 整理特征点，对特征点基于行归类，传入src就用了一下src.cols，其他信息都没用
def __zl_featurePs(fps, src):
    '''
    算法思想，先从第一列得出的点作为起点进行第一遍归类，归类后的点删除，
    再从第二列得出的点作为起点归类，
    最后基于每一类的第一个点坐标值排序;
    最后，去掉去掉一个点两个点的集合，如果不在左边
    '''

    # param
    # th = 8  # 下一个点纵坐标偏差范围 def:9
    # th = int(HJ * 0.25)
    th = int(HJ * 0.5)
    if th < 3: th = 3

    # 先找每个点的下一点，没下一点则设为（-1，-1）点
    # 第一步先把每个点的下一点设为（-1，-1）
    fps_ = []
    s0 = len(fps)
    for i in range(s0):
        tem = []
        s1 = len(fps[i])
        for j in range(s1):
            tem.append(Pair(Pair(fps[i][j], (-1, -1)), Pair(-1, -1)))
        fps_.append(tem)

    # 第二步，求每一点的下一点，有可能多个点公用同一个下一点
    for i in range(s0 - 1):
        s1 = len(fps[i])
        for j in range(s1):
            y = fps[i][j][1]
            flg = True
            for k in range(1, 5):
                if flg == False:  break
                # bbb = i + k >= s0
                if (i + k >= s0): break
                s3 = len(fps[i + k])
                if s3 < 2:
                    if flg == False:  break
                    y_ = fps[i + k][0][1]
                    if (abs(y - y_) < th):
                        fps_[i][j].first.second = fps[i + k][0]
                        fps_[i][j].second.first = i + k
                        fps_[i][j].second.second = 0
                        flg = False
                else:
                    for m in range(s3 - 1):
                        if flg == False:  break
                        y_ = fps[i + k][m][1]
                        y_n = fps[i + k][m + 1][1]
                        if (abs(y - y_) < th and abs(y - y_) < abs(y - y_n)):
                            fps_[i][j].first.second = fps[i + k][m]
                            fps_[i][j].second.first = i + k
                            fps_[i][j].second.second = m
                            flg = False
                    if (abs(y - fps[i + k][s3 - 1][1]) < th):
                        if flg == False:  break
                        fps_[i][j].first.second = fps[i + k][s3 - 1]
                        fps_[i][j].second.first = i + k
                        fps_[i][j].second.second = s3 - 1
                        flg = False

    # 第2.5步（新加），当多个点公用同一个下一点的时候，有可能先从下一行开始连接行，这样可能会出问题
    # 修复方法：保证一个点只有一个上一点，有多个时，只保留距离最近的。
    # 1，先找出有多个上一点的点
    fps_lps = {}
    for i in range(s0):
        s_i = len(fps_[i])
        for j in range(s_i):
            fps_lps[(i, j)] = []
    for i in range(s0):
        s_i = len(fps_[i])
        for j in range(s_i):
            x = fps_[i][j].second.first
            y = fps_[i][j].second.second
            if (x, y) in fps_lps:
                fps_lps[(x, y)].append((i, j))
    # 2, 删除其他距离不是最近的点
    for x, y in fps_lps:
        l_xy = len(fps_lps[(x, y)])
        if l_xy > 1:
            y0 = fps_[x][y].first.first[1]
            dis = 10000
            xx, yy = 0, 0
            for m in range(l_xy):
                i_, j_ = fps_lps[(x, y)][m]
                ym = fps_[i_][j_].first.first[1]
                if dis > abs(ym - y0):
                    dis = abs(ym - y0)
                    xx, yy = i_, j_
            for m in range(l_xy):
                i_, j_ = fps_lps[(x, y)][m]
                if i_ != xx or j_ != yy:
                    fps_[i_][j_].first.second = (-1, -1)
                    fps_[i_][j_].second = (-1, -1)

    # 第三步，根据上面求出的下一点，连接成行，每个点只会使用一次
    lpss = []
    for i in range(s0):
        s1 = len(fps_[i])
        for j in range(s1):
            lps = []
            ii, jj = i, j
            while (True):
                p1 = fps_[ii][jj].first.first
                p2 = fps_[ii][jj].first.second
                if (p2 == (0, 0)):
                    break
                elif (p2 == (-1, -1)):
                    lps.append(p1)
                    fps_[ii][jj].first.second = (0, 0)
                    break
                else:
                    lps.append(p1)
                    fps_[ii][jj].first.second = (0, 0)
                    ii_ = fps_[ii][jj].second.first
                    jj_ = fps_[ii][jj].second.second
                    ii, jj = ii_, jj_
            if len(lps) != 0:
                lpss.append(lps)

    # 去掉不在最后两行的小点集
    lpss.sort(key=lambda x: x[0][1])
    lpss_ = []
    l = len(lpss)
    if l > 2:
        for i in range(l - 2):
            if len(lpss[i]) > 1:
                lpss_.append(lpss[i])
        lpss_.append(lpss[l - 2])
        lpss_.append(lpss[l - 1])
    else:
        lpss_ = lpss

    # 每一行左右两边各增加一个特征点，如果不越界的话
    r, c = src.shape[:2]
    ldjj = int(c / 20)  # 每一行两个相邻特征点之间的间距
    lpss__ = []
    lpss_size = len(lpss_)
    for i in range(lpss_size):
        lps__ = []
        if lpss_[i][0][0] - ldjj >= 0:
            lps__.append((lpss_[i][0][0] - ldjj, lpss_[i][0][1]))
        lps_size = len(lpss_[i])
        for j in range(lps_size):
            lps__.append(lpss_[i][j])
        if lpss_[i][lps_size - 1][0] + ldjj < c:
            lps__.append((lpss_[i][lps_size - 1][0] + ldjj, lpss_[i][lps_size - 1][1]))
        lpss__.append(lps__)
    pass
    return lpss__


# 为仿射变换准备，收集所有待变换的小区域的四点
def __get_WarpPs(heatmap, lpss):
    res = []
    if len(lpss) == 0:    return res

    # param,lasttime:1.8
    th = 2.0  # 每一行在计算出来的宽度上放大多少倍

    r, c = heatmap.shape[:2]
    lj = int(c / 10)
    for i in range(len(lpss)):
        if len(lpss[i]) >= 2:
            lj = lpss[i][1][0] - lpss[i][0][0]
            break
    hj, kd = HJ, KD
    ker = int(kd / 2 * th)

    s0 = len(lpss)
    for i in range(s0):
        v_wps = []
        # 前边特殊处理
        if lpss[i][0][0] - lj < 0:
            x1 = 0
        else:
            x1 = lpss[i][0][0] - lj
        x2 = lpss[i][0][0]
        if lpss[i][0][1] - ker < 0:
            y1 = 0
        else:
            y1 = lpss[i][0][1] - ker
        if lpss[i][0][1] + ker > r - 1:
            y2 = r - 1
        else:
            y2 = lpss[i][0][1] + ker
        v_wps.append(WarpPs((x1, y1), (x2, y1), (x1, y2), (x2, y2)))
        # 中间部分
        s1 = len(lpss[i])
        for j in range(s1 - 1):
            p = lpss[i][j]
            p_next = lpss[i][j + 1]
            if p[1] - ker < 0:
                p1 = (p[0], 0)
            else:
                p1 = (p[0], p[1] - ker)
            if p_next[1] - ker < 0:
                p2 = (p_next[0], 0)
            else:
                p2 = (p_next[0], p_next[1] - ker)
            if p[1] + ker > r - 1:
                p3 = (p[0], r - 1)
            else:
                p3 = (p[0], p[1] + ker)
            if p_next[1] + ker > r - 1:
                p4 = (p_next[0], r - 1)
            else:
                p4 = (p_next[0], p_next[1] + ker)
            v_wps.append(WarpPs(p1, p2, p3, p4))
        # 后边特殊处理
        x1 = lpss[i][len(lpss[i]) - 1][0]
        if lpss[i][len(lpss[i]) - 1][0] + lj > c - 1:
            x2 = c - 1
        else:
            x2 = lpss[i][len(lpss[i]) - 1][0] + lj
        if lpss[i][len(lpss[i]) - 1][1] - ker < 0:
            y1 = 0
        else:
            y1 = lpss[i][len(lpss[i]) - 1][1] - ker
        if lpss[i][len(lpss[i]) - 1][1] + ker > r - 1:
            y2 = r - 1
        else:
            y2 = lpss[i][len(lpss[i]) - 1][1] + ker
        v_wps.append(WarpPs((x1, y1), (x2, y1), (x1, y2), (x2, y2)))

        res.append(v_wps)

    return res


# 遍历所有小区域，仿射变换，得出转换后的每一行
def __warp_imgs(src, wps):
    lr_, lc_ = src.shape[:2]
    lr, lc = 0, lc_
    l_wps = len(wps)
    for i in range(l_wps):
        l_wpsi = len(wps[i])
        for j in range(l_wpsi):
            y2 = wps[i][j].p3[1]
            y1 = wps[i][j].p1[1]
            if y2 - y1 + 1 > lr:
                lr = y2 - y1 + 1

    res = []
    s0 = len(wps)
    if s0 == 0: return res

    global MSPOS
    MSPOS = []  # clear()
    tm = []
    for i in range(s0):
        lineimg = np.zeros((lr, lc), np.uint8)
        lineimg[::] = 255
        tm = wps[i]

        MSPOS.append((tm[0].p1[1], tm[0].p3[1]))

        s1 = len(tm)
        for j in range(s1):
            p1, p2, p3, p4 = tm[j].p1, tm[j].p2, tm[j].p3, tm[j].p4
            smalldst = np.zeros((p3[1] - p1[1] + 1, p2[0] - p1[0] + 1, 1), np.uint8)
            # srcTri = [p1, p2, p3]
            # dstTri = [(0, 0), (p2[0] - p1[0], 0), (0, p3[1] - p1[1])]
            srcTri = np.float32([[p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]])
            dstTri = np.float32([[0, 0], [p2[0] - p1[0], 0], [0, p3[1] - p1[1]]])
            warp_mat = cv2.getAffineTransform(srcTri, dstTri)
            if cv2.__version__ < '3':
                smalldst = cv2.warpAffine(src, warp_mat, (smalldst.shape[1], smalldst.shape[0]), None)
            else:
                smalldst = cv2.warpAffine(src, warp_mat, (smalldst.shape[1], smalldst.shape[0]), None,
                                          cv2.WARP_FILL_OUTLIERS, cv2.BORDER_CONSTANT)

            if (len(smalldst.shape) == 3 and smalldst.shape[2] == 3):
                smalldst = cv2.cvtColor(smalldst, cv2.COLOR_BGR2GRAY)
            # cv2.imshow('test', smalldst)
            # cv2.waitKey()
            x1 = p1[0]
            x2 = smalldst.shape[1] + p1[0] + 1
            y1 = 0
            y2 = smalldst.shape[0] + 1
            # roiii=lineimg[p1[0]:smalldst.shape[1] + p1[0] + 1, 0:smalldst.shape[0] + 1]
            # roiii=smalldst.copy()
            rl, cl = lineimg.shape[:2]
            rs, cs = smalldst.shape[:2]
            lineimg[0:smalldst.shape[0], p1[0]:smalldst.shape[1] + p1[0]] = smalldst
        # lineimg = cv2.adaptiveThreshold(lineimg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 5)
        res.append(lineimg)
        del lineimg

    return res


# pos1, pos2:该行的上下边界坐标
def __get_blank_len_v2(pos1, pos2):
    r, c = pos2 + 1 - pos1, W
    res = c - 1
    hm = IMG_HM[pos1:pos2 + 1, :]
    for i in range(c):
        colimg = hm[:, i:i + 1]
        num = cv2.countNonZero(colimg)
        if num > 5:
            res = i
            break
        pass

    return res


def __blank_judge_v2_old(ms):
    th = 100  # 默认阈值
    temp = []  # 存每一行前面的空白的长度
    s0 = len(ms)
    if s0 == 0:
        return temp
    for i in range(s0):
        temp.append(__get_blank_len_v2(MSPOS[i][0], MSPOS[i][1]))
    temp_sorted = sorted(temp)

    # 计算最优阈值
    # 算法：去掉3/10个最大值和1/10个最小值，然后求平均
    min_n = int(s0 / 10)
    max_n = int(s0 * 3 / 10)
    sum = 0
    for i in range(min_n, s0 - max_n):
        sum += temp_sorted[i]
    avr = sum / (s0 - max_n - min_n)
    th = avr

    # 判断
    res = []
    for i in range(s0):
        if temp[i] > th + 10:
            res.append((True, ms[i]))
        else:
            res.append((False, ms[i]))

    return res


# 输入ms，判断哪一行前面有空格,,new
def __blank_judge_v2(ms):
    th = 25  # 默认阈值
    hl = []  # 存每一行前面的空白的长度
    s0 = len(ms)
    if s0 == 0:
        return hl
    if s0 == 1:
        return [(True, ms[0])]
    c = ms[0].shape[1]
    for i in range(s0):
        hl.append(__get_blank_len_v2(MSPOS[i][0], MSPOS[i][1]))
    res = []
    # try:
    for i in range(s0):
        if hl[i] > c / 2:
            res.append((True, ms[i]))
        elif i == 0:
            if hl[0] - hl[1] >= th:
                res.append((True, ms[i]))
            else:
                res.append((False, ms[i]))
        elif i == s0 - 1:
            if hl[i] - hl[i - 1] >= th:
                res.append((True, ms[i]))
            else:
                res.append((False, ms[i]))
        elif hl[i] - hl[i - 1] >= th or hl[i] - hl[i + 1] >= th:
            res.append((True, ms[i]))
        else:
            res.append((False, ms[i]))
    # except:
    #     pass
    return res


# 对二值化之后的图像去下划线
# "D:\\img\\1.jpg" 2386*1408 haoshi:350ms c++:16ms
def __cleanUnderline(imgbin):
    # param
    cd = 15  # 应设为奇数。在x轴方向连续多少个像素值为0时，把这连续这么多的像素都去掉
    ker = int(cd / 2)
    if imgbin.shape[1] <= cd:
        return imgbin

    imgbin = (imgbin != 0).astype(int)
    h, w = imgbin.shape[:2]
    # 为求x轴方向积分图，向左边扩出来一列零向量
    imgbinpad = np.zeros([h, w + 1], int)
    imgbinpad[0:, 1:] = imgbin
    for i in range(1, w):
        imgbinpad[:, i] += imgbinpad[:, i - 1]
    integralImg = imgbinpad[0:, 1:]

    # cout为比较结果，仅保存了连续cd个像素为零时的中心像素坐标，
    # 后面还要再处理，再往外扩cd/2个像素
    cont = integralImg[:, cd - 1:] - integralImg[:, :w - cd + 1]
    cont = (cont != 0).astype(int)
    # 再往外扩cd/2个像素，执行ker遍
    w_c = cont.shape[1]
    for k in range(ker):
        for i in range(w_c - 1):
            cont[:, i] *= cont[:, i + 1]
        for i in range(1, w_c):
            i = w_c - i
            cont[:, i] *= cont[:, i - 1]
    roi = imgbin[:, cd - 1 - ker:w - ker]
    roi = (roi == 0).astype(int)
    # 相乘去掉图像中的下划线
    roi = roi * cont
    roi = (roi == 0).astype(int)
    imgbin[:, cd - 1 - ker:w - ker] = roi
    imgbin *= 255
    imgbin = imgbin.astype(np.uint8)
    return imgbin
    pass


def __cleanVerticalLine(imgbin):
    # param
    cd = 30  # 应设为奇数。在x轴方向连续多少个像素值为0时，把这连续这么多的像素都去掉
    if imgbin.shape[0] <= cd:
        return imgbin

    imgbin = cv2.transpose(imgbin)
    imgbin = (imgbin != 0).astype(int)
    h, w = imgbin.shape[:2]
    # 为求x轴方向积分图，向左边扩出来一列零向量
    imgbinpad = np.zeros([h, w + 1], int)
    imgbinpad[0:, 1:] = imgbin
    for i in range(1, w):
        imgbinpad[:, i] += imgbinpad[:, i - 1]
    integralImg = imgbinpad[0:, 1:]
    ker = int(cd / 2)
    # cout为比较结果，仅保存了连续cd个像素为零时的中心像素坐标，
    # 后面还要再处理，再往外扩cd/2个像素
    cont = integralImg[:, cd - 1:] - integralImg[:, :w - cd + 1]
    cont = (cont != 0).astype(int)
    # 再往外扩cd/2个像素，执行ker遍
    w_c = cont.shape[1]
    for k in range(ker):
        for i in range(w_c - 1):
            cont[:, i] *= cont[:, i + 1]
        for i in range(1, w_c):
            i = w_c - i
            cont[:, i] *= cont[:, i - 1]
    roi = imgbin[:, cd - 1 - ker:w - ker]
    roi = (roi == 0).astype(int)
    # 相乘去掉图像中的下划线
    roi = roi * cont
    roi = (roi == 0).astype(int)
    imgbin[:, cd - 1 - ker:w - ker] = roi
    imgbin *= 255
    imgbin = imgbin.astype(np.uint8)
    return cv2.transpose(imgbin)


# huatu
def __drawvec(v):
    c = len(v)
    r = v.max() + 10
    img = np.ones([r, c], int)
    for i in range(c):
        val = v[i]
        if val != 0: img[-val:, i] = 0

    def rotate(m):
        h, w = m.shape[:2]
        m_ = np.zeros([w, h], int)
        for i in range(w):
            m_[i, ::-1] = m[:, i]
        return m_

    img = rotate(img)
    img *= 255
    pjimg = img.astype(np.uint8)
    return pjimg


def get_binm_hm(img):
    src = __src_resize(img)
    global H, W
    H, W = src.shape[:2]
    bin0 = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    bin = __cleanUnderline(bin0)
    bin = __cleanVerticalLine(bin)
    heatmap = __get_heatmap_v3(bin)
    heatmap = __heatmap_opt(heatmap)
    heatmap = __heatmap_opt2(heatmap)
    return bin0, heatmap


# 总的接口
def run(img, isbin=False):
    # 预处理
    src = __src_resize(img)
    global H, W
    H, W = src.shape[:2]

    if (len(src.shape) == 3 and src.shape[2] != 1):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    else:
        bin = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    bin = __cleanUnderline(bin)
    bin = __cleanVerticalLine(bin)

    di = __calDir(bin)

    src_warp = __warpImg(src, di)
    bin = __warpImg(bin, di)
    # src_warp = src

    # 执行
    heatmap = __get_heatmap_v3(bin)

    heatmap = __heatmap_opt(heatmap)
    heatmap = __heatmap_opt2(heatmap)

    global IMG_HM
    IMG_HM = heatmap.copy()
    temp = __get_hangnju_kuandu_v2(heatmap)
    global HJ, KD
    HJ, KD = temp.first, temp.second

    fps = __get_featurePs_v2(heatmap)

    lpss = __zl_featurePs(fps, bin)
    # __visualization(bin, heatmap, fps, lpss)

    wps = __get_WarpPs(heatmap, lpss)

    if isbin:
        ms = __warp_imgs(bin, wps)
    else:
        ms = __warp_imgs(src_warp, wps)

    ms_blks = __blank_judge_v2(ms)

    return ms_blks


# 把得到的每一行的小图像整合到一张图像
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
    # 如果图像过大，屏幕显示不完，要resize一下
    h, w = dst.shape[:2]
    if h > 1010:
        h_ = 1010
        w_ = int(1010 * w / h)
        dst = cv2.resize(dst, (w_, h_))
        cv2.putText(dst, 'Resized', (2, 50), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, 100)
    return dst


def __visualization(imgbin, heatmap, fps, lpss):
    # pj = ((imgbin == 0).astype(int)).sum(axis=1)
    # pjimg = __drawvec(pj)
    imgbin = (imgbin > 10).astype(np.uint8)
    imgbin *= 100
    imgbin += 155
    imgbin = cv2.cvtColor(imgbin, cv2.COLOR_GRAY2BGR)
    ###########jiashuxian###################
    h, w = imgbin.shape[:2]
    st = int(w / 20.0)
    for i in range(1, 20):
        imgbin = cv2.line(imgbin, (i * st, 0), (i * st, h), (200, 16, 147), 1)
    ###########jiashuxian###################
    heatmap = (heatmap != 0).astype(np.uint8)
    heatmap *= 255
    # imgbin[:, :, 1] = (imgbin[:, :, 1] / 2 + heatmap / 2).astype(int)
    imgbin[:, :, 2] -= heatmap
    imgbin[:, :, 0] -= heatmap
    for i in range(len(fps)):
        for j in range(len(fps[i])):
            cv2.circle(imgbin, fps[i][j], 3, [0, 0, 255], 2)
            # cv2.putText(imgbin, "(" + str(fps[i][j][0]) + ",",
            #             (fps[i][j][0] - 20, fps[i][j][1] + 25),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.4, [0, 0, 255])
            # cv2.putText(imgbin, str(fps[i][j][1]) + ")",
            #             (fps[i][j][0] - 20, fps[i][j][1] + 40),
            #             cv2.FONT_HERSHEY_DUPLEX, 0.4, [0, 0, 255])
    # 两点之间连线
    for i in range(len(lpss)):
        for j in range(len(lpss[i]) - 1):
            p1, p2 = lpss[i][j], lpss[i][j + 1]
            cv2.line(imgbin, p1, p2, (0, 0, 255), 2)

    # # 显示投影图
    # rtimg = np.ones([pjimg.shape[0], pjimg.shape[1], 3], int)
    # rtimg[:, :, 0] = pjimg
    # rtimg[:, :, 1] = pjimg
    # res = np.zeros([pjimg.shape[0], pjimg.shape[1] + imgbin.shape[1], 3], int)
    # res[:, 0:imgbin.shape[1], :] = imgbin
    # res[:, imgbin.shape[1]:, :] = rtimg
    # imgbin = res.astype(np.uint8)

    # # 加护眼色
    # rr = 199 / 255
    # rg = 237 / 255
    # rb = 204 / 255
    # imgbin = imgbin.astype(np.float)
    # imgbin[:, :, 0] *= rb
    # imgbin[:, :, 1] *= rg
    # imgbin[:, :, 2] *= rr
    # imgbin = imgbin.astype(np.uint8)

    cv2.imshow('__visualization', imgbin)
    pass


