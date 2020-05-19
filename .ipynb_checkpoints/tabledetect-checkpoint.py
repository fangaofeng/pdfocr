import cv2
import numpy as np

import json


class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img
        self.mask = None
        self.joints_img = None
        self.gray_img = None
        if len(self.src_img.shape) == 2:  # 灰度图
            self.gray_img = self.src_img
        elif len(self.src_img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

    def getmask(self, scale=15):

        thresh_img = cv2.adaptiveThreshold(
            ~self.gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()

        h_size = int(h_img.shape[1]/scale)

        h_structure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (h_size, 1))  # 形态学因子
        h_erode_img = cv2.erode(h_img, h_structure, 1)

        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)
        # cv2.imshow("h_erode",h_dilate_img)
        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        self.mask = h_dilate_img+v_dilate_img
        self.joints_img = cv2.bitwise_and(h_dilate_img, v_dilate_img)
        # cv2.imshow("joints", joints_img)
        cv2.imwrite("joints.jpg", self.mask)
        cv2.imwrite("mask.jpg", self.joints_img)
        # cv2.imshow("mask", mask_img)

        return self.mask, self.joints_img
        # 将生成的json数据显示在图像上

    def getHmask(self, scale=15):

        thresh_img = cv2.adaptiveThreshold(
            ~self.gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        h_img = thresh_img.copy()

        h_size = int(h_img.shape[1]/scale)

        h_structure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (h_size, 1))  # 形态学因子
        h_erode_img = cv2.erode(h_img, h_structure, 1)

        h_dilate_img = cv2.dilate(h_erode_img, h_structure, 1)
        # cv2.imshow("h_erode",h_dilate_img)

        # cv2.imshow("mask", mask_img)

        return h_dilate_img

    def getVmask(self, scale=15):

        thresh_img = cv2.adaptiveThreshold(
            ~self.gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

        v_img = thresh_img.copy()

        v_size = int(v_img.shape[0] / scale)

        v_structure = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, v_size))  # 形态学因子
        v_erode_img = cv2.erode(v_img, v_structure, 1)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure, 1)

        return v_dilate_img

    def verticalShadowSplite(self, img, splitthreas, threashold=1):
        '''
        :param img:
        :param threashold: 计数阀值
        :return:
        '''
        h, w = img.shape
        x_count = [0 for z in range(0, w)]
        x_segmentation = []
        startX = 0
        endX = 0
        for x in range(w):
            for y in range(h):
                if img[y, x] >= threashold:
                    x_count[x] += 1
            if x_count[x] > splitthreas:
                if startX == 0:
                    startX = x
            else:
                if startX != 0:
                    endX = x
                    x_segmentation.append([startX, endX])
                    startX = 0
                    endX = 0
        return x_count, x_segmentation

    def horizontalShadowSplite(self, img, splitthreas, threashold=1):
        '''
        :param img:
        :param threashold: 阀值
        :return:
        '''
        h, w = img.shape

        y_count = [0 for z in range(0, h)]
        y_segmentation = []
        startY = 0
        endY = 0
        for y in range(h):
            for x in range(w):
                if img[y, x] >= threashold:
                    y_count[y] += 1
            if y_count[y] > splitthreas:
                if startY == 0:
                    startY = y
            else:
                if startY != 0:
                    endY = y
                    y_segmentation.append([startY, endY])
                    startY = 0
                    endY = 0
        return y_count, y_segmentation

    def getShadowimg(self, img, line_count, HorV='H'):
        h, w = img.shape
        b = 255
        emptyImage = np.zeros(img.shape, np.uint8)
        if HorV == "H":
            for y in range(h):
                for x in range(line_count[y]):
                    emptyImage[y, x] = b
        else:
            for x in range(w):  # 遍历每一列
                # 从该列应该变黑的最顶部的点开始向最底部涂黑
                for y in range(line_count[x]):
                    emptyImage[h-y-1, x] = b
        return emptyImage

    def getGrid(self, img, x_segmentation, y_segmentation):
        h, w = img.shape

        rows = []
        colums = []
        #green = (255, 255, 255)

        for i, (startx, endx) in enumerate(x_segmentation):

            colums.append(startx+round((endx-startx)/2))

            #cv2.line(img, (self.rows[i], 0), (self.rows[i], h), green)

        for i, (starty, endy) in enumerate(y_segmentation):
            rows.append(starty+round((endy-starty)/2))
            #cv2.line(img, (0, self.colums[i]), (w, self.colums[i]), green)

            #cv2.imwrite("gride.jpg", img)

        return rows, colums

    def getGridabc(self, img, x_segmentation, y_segmentation):
        h, w = img.shape

        rows = []
        colums = []
        # abc 定制

        for i, (startx, endx) in enumerate(x_segmentation):
            if i+1 < len(x_segmentation):
                next_x = w-1
            else:
                next_x = x_segmentation[i+1]
            colums.append(endx+round((endx-next_x)/2))

            #cv2.line(img, (self.rows[i], 0), (self.rows[i], h), green)

        for i, (starty, endy) in enumerate(y_segmentation):

            if i+1 < len(y_segmentation):
                next_y = h-1
            else:
                next_y = y_segmentation[i+1]

            rows.append(starty+round((endy-next_y)/2))
            #cv2.line(img, (0, self.colums[i]), (w, self.colums[i]), green)

            #cv2.imwrite("gride.jpg", img)

        return rows, colums

    def getLineCor(self, long, segmentation, next=False):
        line = []

        if next:
            for i, (start, end) in enumerate(segmentation):
                if i+1 < len(segmentation):
                    next_ = segmentation[i+1][0]
                else:
                    next_ = long-1
                if (next_ - end) > 30:  # 过滤小的
                    line.append(end+round((next_ - end)/2))
        else:
            for i, (start, end) in enumerate(segmentation):
                line.append(start+round((end-start)/2))

        return line
    # 返回多个table区域

    def getTableRois(self, mask,minarea = 1000):

        image, contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        regroi = []
        for i in range(len(contours)):
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if area < minarea:
                continue
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            # print(epsilon)
            approx = cv2.approxPolyDP(cnt, 10, True)
            # print(approx)
            regroi.append(cv2.boundingRect(approx))
        return regroi

    def getRoiImg(self, roi, img=None,  border=0):
        x, y, w, h = roi
        if img is None:
            img = self.gray_img
        roi = img[y-border:(y+h)+border,
                  x-border: (x+w)+border]
        return roi
    # 编程单点

    def isolate(self, img):
        idx = np.argwhere(img < 1)
        rows, cols = img.shape

        for i in range(idx.shape[0]):
            c_row = idx[i, 0]
            c_col = idx[i, 1]
            if c_col+1 < cols and c_row+1 < rows:
                img[c_row, c_col+1] = 1
                img[c_row+1, c_col] = 1
                img[c_row+1, c_col+1] = 1
            if c_col+2 < cols and c_row+2 < rows:
                img[c_row+1, c_col+2] = 1
                img[c_row+2, c_col] = 1
                img[c_row, c_col+2] = 1
                img[c_row+2, c_col+1] = 1
                img[c_row+2, c_col+2] = 1
        return img


if __name__ == '__main__':

    print(pytesseract.get_tesseract_version())
    img = cv2.imread('2.jpg')
    cv2.imshow("img", img)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask, joint = detectTable(img).run()
    x_structure = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3))
    x_mask = cv2.dilate(mask, x_structure, 1)
    not_output = np.zeros(gray_img.shape, np.uint8)
    not_output = cv2.bitwise_not(gray_img, not_output, mask=~x_mask)
    bin_threshold = 100
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3))
    iterations = 1
    areaRange = [100, 10000]
    cells = cutImage(not_output, mask, bin_threshold, kernel,
                     iterations, areaRange, 'outjson.json', border=2).getRes()

    mask2 = np.zeros((img.shape), np.uint8)
    for x1, y1, x2, y2 in cells:
        cv2.rectangle(mask2, (x1, y1), (x2, y2), (255, 255, 0), 1)

    cv2.imwrite("tabledetect_res.png", mask2)
    cv2.waitKey()
