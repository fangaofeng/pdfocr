# -*- coding: utf-8 -*-
"""
Image processing functions.

Created on Wed Dec 14 09:51:20 2016

@author: mkonrad
"""

from logging import warning
from math import degrees

import numpy as np
import cv2
from numba import jit
#from imgproc.common import ROTATION, SKEW_X, SKEW_Y, DIRECTION_HORIZONTAL, DIRECTION_VERTICAL
from skewManager.geom import normalize_angle, project_polarcoord_lines


ROTATION = 'r'
SKEW_X = 'sx'
SKEW_Y = 'sy'

DIRECTION_HORIZONTAL = 'h'
DIRECTION_VERTICAL = 'v'
PIHLF = np.pi / 2
PI4TH = np.pi / 4


def pt_to_tuple(p):
    return (int(round(p[0])), int(round(p[1])))


class ImageProc:
    """
    Class for image processing. Methods for detecting lines in an image and clustering them. Helper methods for
    drawing.
    """
    DRAW_LINE_WIDTH = 1

    def __init__(self, imgfile, img=None):
        """
        Create a new image processing object for <imgfile>.
        """
        if not imgfile:
            raise ValueError(
                "parameter 'imgfile' must be a non-empty, non-None string")

        self.imgfile = imgfile
        self.input_img = img
        self.img_w = None
        self.img_h = None

        self.gray_img = None  # grayscale version of the input image
        self.edges = None  # edges detected by Canny algorithm

        # contains tuples (rho, theta, theta_norm, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL)
        self.lines_hough = []
        if img is None:
            self.input_img = self._cv_imread(self.imgfile)

        if len(self.input_img.shape) == 2:  # 灰度图
            self.gray_img = self.input_img
        elif len(self.input_img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)
        if self.input_img is None:
            raise IOError("could not load file '%s'" % self.imgfile)

        self.img_h, self.img_w = self.input_img.shape[:2]

    def detect_lines(self,
                     canny_low_thresh,
                     canny_high_thresh,
                     canny_kernel_size,
                     hough_rho_res,
                     hough_theta_res,
                     hough_votes_thresh,
                     gray_conversion=cv2.COLOR_BGR2GRAY):
        """
        Detect lines in input image using hough transform.
        Return detected lines as list with tuples:
        (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
        """

        #self.gray_img = cv2.cvtColor(self.input_img, gray_conversion)

        self.edges = cv2.Canny(
            self.gray_img,
            canny_low_thresh,
            canny_high_thresh,
            apertureSize=canny_kernel_size)

        # detect lines with hough transform
        lines = cv2.HoughLines(self.edges, hough_rho_res, hough_theta_res,
                               hough_votes_thresh)
        if lines is None:
            lines = []

        self.lines_hough = self._generate_hough_lines(lines)

        return self.lines_hough

    def ab_lines_from_hough_lines(self, lines_hough):
        """
        From a list of lines <lines_hough> in polar coordinate space, generate lines in cartesian coordinate space
        from points A to B in image dimension space. A and B are at the respective opposite borders
        of the line projected into the image.
        Will return a list with tuples (A, B, DIRECTION_HORIZONTAL or DIRECTION_VERTICAL).
        """

        projected = project_polarcoord_lines([l[:2] for l in lines_hough],
                                             self.img_w, self.img_h)
        return [(p1, p2, line_dir)
                for (p1, p2), (_, _, _,
                               line_dir) in zip(projected, lines_hough)]

    def find_rotation_or_skew2(self,
                               rot_thresh,
                               rot_same_dir_thresh,
                               omit_on_rot_thresh=PI4TH/3,
                               only_direction=None):
        """
        Find page rotation or horizontal/vertical skew using detected lines in <lines>. The lines list must consist
        of arrays with the line rotation "theta" at array index 1 like the returned list from detect_lines().
        <rot_thresh> is the minimum threshold in radians for a rotation to be counted as such.
        <rot_same_dir_thresh> is the maximum threshold for the difference between horizontal and vertical line
        rotation.
        <omit_on_rot_thresh> is an optional threshold to filter out "stray" lines whose angle is too far apart from
        the median angle of all other lines that go in the same direction.
        <only_direction> optional parameter: only use lines in this direction to find out the rotation/skew
        """
        if len(self.lines_hough) < 1:
            return ROTATION, 0

        if only_direction is not None:
            if only_direction not in (DIRECTION_HORIZONTAL,
                                      DIRECTION_VERTICAL):
                only_direction = None

        # get the deviations

        hori_deviations = []  # deviation from unit vector in x-direction
        vert_deviations = []  # deviation from unit vector in y-direction

        lines_w_deviations = [] if omit_on_rot_thresh is not None else None

        for rho, theta, theta_norm, line_dir in self.lines_hough:
            #print("----", theta, theta_norm, line_dir)
            if line_dir == DIRECTION_VERTICAL and (
                    only_direction is None
                    or only_direction == DIRECTION_VERTICAL):
                deviation = -theta_norm
                if deviation < -PIHLF:
                    deviation += np.pi
                # print(deviation)
                if deviation > omit_on_rot_thresh:
                    continue
                vert_deviations.append(-deviation)
            elif line_dir == DIRECTION_HORIZONTAL and (
                    only_direction is None
                    or only_direction == DIRECTION_HORIZONTAL):
                deviation = PIHLF - theta_norm
                # print(deviation)
                if deviation > omit_on_rot_thresh:
                    continue
                hori_deviations.append(-deviation)
            else:
                deviation = None

            if omit_on_rot_thresh is not None and deviation is not None:
                assert abs(deviation) <= PI4TH
                # print('dsdf', abs(deviation))
                if abs(deviation) > omit_on_rot_thresh:
                    continue
                lines_w_deviations.append((rho, theta, theta_norm, line_dir,
                                           -deviation))

        # get the medians
        #print(len(hori_deviations), sum(hori_deviations))
        #print(len(vert_deviations), sum(vert_deviations),sum(vert_deviations)/len(vert_deviations))
        if hori_deviations:
            median_hori_dev = np.median(hori_deviations)
            hori_rot_above_thresh = abs(median_hori_dev) > rot_thresh
        else:
            if only_direction is None:
                warning('no horizontal lines found')
            median_hori_dev = None
            hori_rot_above_thresh = False

        if vert_deviations:
            median_vert_dev = np.median(vert_deviations)
            vert_rot_above_thresh = abs(median_vert_dev) > rot_thresh
        else:
            if only_direction is None:
                warning('no vertical lines found')
            median_vert_dev = None
            vert_rot_above_thresh = False

        #print(median_hori_dev, median_vert_dev, omit_on_rot_thresh)

        if omit_on_rot_thresh is not None:
            # if only_direction is None:
            #     assert len(lines_w_deviations) == len(self.lines_hough)
            # else:
            #     assert len(lines_w_deviations) == len(
            #         [l for l in self.lines_hough if l[3] == only_direction])
            lines_filtered = []
            for rho, theta, theta_norm, line_dir, deviation in lines_w_deviations:
                dir_dev = median_hori_dev if line_dir == DIRECTION_HORIZONTAL else median_vert_dev
                # print(abs(dir_dev), abs(deviation))
                if dir_dev is None or abs(abs(dir_dev) -
                                          abs(deviation)) < omit_on_rot_thresh:
                    # print("----", theta, theta_norm,
                    #      abs(abs(dir_dev) - abs(deviation)), omit_on_rot_thresh)
                    lines_filtered.append((rho, theta, theta_norm, line_dir))
            assert len(lines_filtered) <= len(self.lines_hough)
            self.lines_hough = lines_filtered

        if hori_rot_above_thresh and vert_rot_above_thresh:
            if abs(median_hori_dev - median_vert_dev) < rot_same_dir_thresh:
                return ROTATION, (median_hori_dev + median_vert_dev) / 2
            else:
                warning(
                    'horizontal / vertical rotation not in same direction (%f / %f)'
                    % (degrees(median_hori_dev), degrees(median_vert_dev)))

        print(median_hori_dev, median_vert_dev, hori_rot_above_thresh,
              vert_rot_above_thresh)
        if (len(hori_deviations) / (len(vert_deviations) + 0.1)) > 3:
            if hori_rot_above_thresh:
                return SKEW_X, median_hori_dev
            else:
                return SKEW_X, 0.0
            return SKEW_Y, median_vert_dev
        if (len(vert_deviations) / (len(hori_deviations) + 0.1)) > 3:
            if vert_rot_above_thresh:
                return SKEW_Y, median_vert_dev
            else:
                return SKEW_Y, 0.0
            return SKEW_X, median_hori_dev
        if hori_rot_above_thresh:
            return SKEW_Y, median_hori_dev
        if vert_rot_above_thresh:
            return SKEW_X, median_vert_dev

        return ROTATION, 0

    def find_rotation_or_skew(self,
                              rot_thresh,
                              rot_same_dir_thresh,
                              omit_on_rot_thresh=None,
                              only_direction=None):
        """
        Find page rotation or horizontal/vertical skew using detected lines in <lines>. The lines list must consist
        of arrays with the line rotation "theta" at array index 1 like the returned list from detect_lines().
        <rot_thresh> is the minimum threshold in radians for a rotation to be counted as such.
        <rot_same_dir_thresh> is the maximum threshold for the difference between horizontal and vertical line
        rotation.
        <omit_on_rot_thresh> is an optional threshold to filter out "stray" lines whose angle is too far apart from
        the median angle of all other lines that go in the same direction.
        <only_direction> optional parameter: only use lines in this direction to find out the rotation/skew
        """
        if len(self.lines_hough) < 1:
            return ROTATION, 0

        if only_direction is not None:
            if only_direction not in (DIRECTION_HORIZONTAL,
                                      DIRECTION_VERTICAL):
                only_direction = None

        # get the deviations

        hori_deviations = []  # deviation from unit vector in x-direction
        vert_deviations = []  # deviation from unit vector in y-direction

        lines_w_deviations = [] if omit_on_rot_thresh is not None else None

        for rho, theta, theta_norm, line_dir in self.lines_hough:
            #print("----", theta, theta_norm, line_dir)
            if line_dir == DIRECTION_VERTICAL and (
                    only_direction is None
                    or only_direction == DIRECTION_VERTICAL):
                deviation = -theta_norm
                if deviation < -PIHLF:
                    deviation += np.pi
                vert_deviations.append(-deviation)
            elif line_dir == DIRECTION_HORIZONTAL and (
                    only_direction is None
                    or only_direction == DIRECTION_HORIZONTAL):
                deviation = PIHLF - theta_norm
                hori_deviations.append(-deviation)
            else:
                deviation = None

            if omit_on_rot_thresh is not None and deviation is not None:
                assert abs(deviation) <= PI4TH
                lines_w_deviations.append((rho, theta, theta_norm, line_dir,
                                           -deviation))

        # get the medians
        #print(len(hori_deviations), sum(hori_deviations))
        #print(len(vert_deviations), sum(vert_deviations),sum(vert_deviations)/len(vert_deviations))
        if hori_deviations:
            median_hori_dev = np.median(hori_deviations)
            hori_rot_above_thresh = abs(median_hori_dev) > rot_thresh
        else:
            if only_direction is None:
                warning('no horizontal lines found')
            median_hori_dev = None
            hori_rot_above_thresh = False

        if vert_deviations:
            median_vert_dev = np.median(vert_deviations)
            vert_rot_above_thresh = abs(median_vert_dev) > rot_thresh
        else:
            if only_direction is None:
                warning('no vertical lines found')
            median_vert_dev = None
            vert_rot_above_thresh = False

        #print(median_hori_dev, median_vert_dev, omit_on_rot_thresh)

        if omit_on_rot_thresh is not None:
            # if only_direction is None:
            #     assert len(lines_w_deviations) == len(self.lines_hough)
            # else:
            #     assert len(lines_w_deviations) == len(
            #         [l for l in self.lines_hough if l[3] == only_direction])
            lines_filtered = []
            for rho, theta, theta_norm, line_dir, deviation in lines_w_deviations:
                dir_dev = median_hori_dev if line_dir == DIRECTION_HORIZONTAL else median_vert_dev
                #print(abs(dir_dev), abs(deviation))
                if dir_dev is None or abs(abs(dir_dev) -
                                          abs(deviation)) < omit_on_rot_thresh:
                    print("----", theta, theta_norm,
                          abs(abs(dir_dev) - abs(deviation)), omit_on_rot_thresh)
                    lines_filtered.append((rho, theta, theta_norm, line_dir))
            assert len(lines_filtered) <= len(self.lines_hough)
            self.lines_hough = lines_filtered

        if hori_rot_above_thresh and vert_rot_above_thresh:
            if abs(median_hori_dev - median_vert_dev) < rot_same_dir_thresh:
                return ROTATION, (median_hori_dev + median_vert_dev) / 2
            else:
                warning(
                    'horizontal / vertical rotation not in same direction (%f / %f)'
                    % (degrees(median_hori_dev), degrees(median_vert_dev)))

        print(median_hori_dev, median_vert_dev, hori_rot_above_thresh,
              vert_rot_above_thresh)
        if (len(hori_deviations) / (len(vert_deviations) + 0.1)) > 3:
            if hori_rot_above_thresh:
                return SKEW_X, median_hori_dev
            else:
                return SKEW_X, 0.0
            return SKEW_Y, median_vert_dev
        if (len(vert_deviations) / (len(hori_deviations) + 0.1)) > 3:
            if vert_rot_above_thresh:
                return SKEW_Y, median_vert_dev
            else:
                return SKEW_Y, 0.0
            return SKEW_X, median_hori_dev
        if hori_rot_above_thresh:
            return SKEW_Y, median_hori_dev
        if vert_rot_above_thresh:
            return SKEW_X, median_vert_dev

        return ROTATION, 0

    def draw_lines(self, orig_img_as_background=True, draw_line_num=False):
        """
        Draw detected lines and return the rendered image.
        <orig_img_as_background>: if True, draw on top of input image
        <draw_line_num>: if True, draw line number
        """
        lines_ab = self.ab_lines_from_hough_lines(self.lines_hough)

        baseimg = self._baseimg_for_drawing(orig_img_as_background)

        for i, (p1, p2, line_dir) in enumerate(lines_ab):
            line_color = (0, 255,
                          0) if line_dir == DIRECTION_HORIZONTAL else (0, 255,
                                                                       255)
            #print(pt_to_tuple(p1), pt_to_tuple(p2))
            cv2.line(baseimg, pt_to_tuple(p1), pt_to_tuple(p2), line_color,
                     self.DRAW_LINE_WIDTH)

            if draw_line_num:
                p_text = pt_to_tuple(p1 + (p2 - p1) * 0.5)
                cv2.putText(baseimg, str(i), p_text, cv2.FONT_HERSHEY_SIMPLEX,
                            1, line_color, 3)

        return baseimg

    def draw_line_clusters(self,
                           direction,
                           clusters_w_vals,
                           orig_img_as_background=True):
        """
        Draw detected clusters of lines in direction <direction> using <clusters_w_vals>.
        """
        if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
            raise ValueError("invalid value for 'direction': '%s'" % direction)

        baseimg = self._baseimg_for_drawing(orig_img_as_background)

        n_colors = len(clusters_w_vals)
        color_incr = max(1, round(255 / n_colors))

        for i, (_, vals) in enumerate(clusters_w_vals):
            i += 2

            line_color = (
                (color_incr * i) % 256,
                (color_incr * i * i) % 256,
                (color_incr * i * i * i) % 256,
            )

            self.draw_lines_in_dir(baseimg, direction, vals, line_color)

        return baseimg

    @staticmethod
    def draw_lines_in_dir(baseimg,
                          direction,
                          line_positions,
                          line_color,
                          line_width=None):
        """
        Draw a list of lines <line_positions> on <baseimg> in direction <direction> using <line_color> and <line_width>.
        """
        if direction not in (DIRECTION_HORIZONTAL, DIRECTION_VERTICAL):
            raise ValueError("invalid value for 'direction': '%s'" % direction)

        if not line_width:
            line_width = ImageProc.DRAW_LINE_WIDTH

        h, w = baseimg.shape[:2]

        for pos in line_positions:
            pos = int(round(pos))

            if direction == DIRECTION_HORIZONTAL:
                p1 = (0, pos)
                p2 = (w, pos)
            else:
                p1 = (pos, 0)
                p2 = (pos, h)

            cv2.line(baseimg, p1, p2, line_color, line_width)

    def _baseimg_for_drawing(self, use_orig):
        """
        Get a base image for drawing: Either the input image if <use_orig> is True or an empty (black) image.
        """
        if use_orig:
            return np.copy(self.input_img)
        else:
            return np.zeros((self.img_h, self.img_w, 3), np.uint8)

    def _cv_imread(self, img_path=""):
        img_mat = cv2.imdecode(
            np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return img_mat

    
    def removeRedStamp(self, image=None, debug=False):
        # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# timg.jpeg
        #imgSkin = np.zeros(image.shape, np.uint8)
        if image is None:
            image = self.input_img
        img = image.copy()    
        if len(self.input_img.shape) != 3:
            return img, None
        rows, cols = image.shape
        imgSkin = image.copy()
        
        imgHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for r in range(rows):
            for c in range(cols):

                # get values of hue, saturation and value
                # standard -- h range: [0,360]; s range: [0,1]; v range: [0,255]
                # opencv -- h range: [0,180]; s range: [0,255]; v range: [0,255]
                H = imgHsv.item(r, c, 0)
                S = imgHsv.item(r, c, 1)
                V = imgHsv.item(r, c, 2)

                # non-skin area if skin equals 0, skin area otherwise
                skin = 0

                if ((H >= 0) and (H <= 50)) or ((H >= 250 / 2) and
                                                (H <= 360 / 2)):
                    if ((S >= 0.2 * 255) and
                            (S <= 0.7 * 255)) and (V >= 0.3 * 255):
                        skin = 1

                if 1 == skin:
                    img.itemset((r, c, 0), 255)
                    img.itemset((r, c, 1), 255)
                    img.itemset((r, c, 2), 255)
                if debug and 0 == skin :
                    imgSkin.itemset((r, c, 0), 0)
                    imgSkin.itemset((r, c, 1), 0)
                    imgSkin.itemset((r, c, 2), 0)

        self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if debug:
            return img, imgSkin
        else:
            return img, None

    def _load_imgfile(self):
        """Load the image file self.imgfile to self.input_img. Additionally set the image width and height (self.img_w
        and self.img_h)"""
        self.input_img = self._cv_imread(self.imgfile)
        if len(self.input_img.shape) == 2:  # 灰度图
            self.gray_img = self.input_img
        elif len(self.input_img.shape) == 3:
            self.gray_img = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2GRAY)
        if self.input_img is None:
            raise IOError("could not load file '%s'" % self.imgfile)

        self.img_h, self.img_w = self.input_img.shape[:2]

    def _generate_hough_lines(self, lines):
        """
        From a list of lines in <lines> detected by cv2.HoughLines, create a list with a tuple per line
        containing:
        (rho, theta, normalized theta with 0 <= theta_norm < np.pi, DIRECTION_VERTICAL or DIRECTION_HORIZONTAL)
        """
        lines_hough = []
        for line in lines:
            # they come like this from OpenCV's hough transform
            rho, theta = line[0]
            theta_norm = normalize_angle(theta)

            if abs(PIHLF - theta_norm) > PI4TH:  # vertical
                line_dir = DIRECTION_VERTICAL
            else:
                line_dir = DIRECTION_HORIZONTAL

            lines_hough.append((rho, theta, theta_norm, line_dir))

        return lines_hough

    def rotate_image(self, image=None, preangle=None):
        """
        find the whole contours and make a numpy array
            make a rotated bounding box, and rotate the image
        """
        if image is None:
            image = self.gray_img
        if abs(preangle) < 1e-6:
            return image
        angle = 0.0

        edges = cv2.Canny(image, 100, 200)
        _, contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_NONE)
        points = []
        for h, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area >= 500:
                for p in cnt:
                    points.append(p[0])

        points = np.array(points)
        # print(points)

        # DEBUG
        #mask = np.zeros((image.shape),np.uint8)
        print("points.length()={}".format(points.size))

        if points.size > 10:
            rect = cv2.minAreaRect(points)
            angle = abs(rect[2])

            #if angle > 45: angle = angle - 90
            if preangle:
                angle = preangle
            print("校正角度{}-{}".format(preangle, angle))
            if abs(angle) < 1e-6:
                return image
            mat = cv2.getRotationMatrix2D(rect[0], -angle, 1)

            image = cv2.warpAffine(
                image, mat, (image.shape[1], image.shape[0]), image.size,
                cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, (255, 255, 255))

        return image

    def rotate_image2(self, image=None, angle=None):
        """
        find the whole contours and make a numpy array
            make a rotated bounding box, and rotate the image
        """
        if image is None:
            image = self.gray_img
        print("校正角度{}".format(angle))
        if abs(angle) < 1e-6:
            return image
        height, width = image.shape[:2]
        RotateMatrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle,
                                               1)

        image = cv2.warpAffine(
            image, RotateMatrix, (image.shape[1], image.shape[0]), image.size,
            cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, (255, 255, 255))

        return image

     
    def removeEllipseStamp(self, dstimg=None, pointsTheshold=100, areathreshold=1000.0, color=(255, 255, 255), debug=False):

        if dstimg is None:
            dstimgT = self.gray_img
        else:
            dstimgT = dstimg
        # if maskimg is None:

        imgrayT = cv2.Canny(dstimg, 100, 200, 3)  # Canny边缘检测，参数可更改
        ret, maskimgT = cv2.threshold(imgrayT, 10, 255, cv2.THRESH_BINARY)

        image, contours, hierarchy = cv2.findContours(
            maskimgT, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # contours为轮廓集，可以计算轮廓的长度、面积等
        #emptyImageTT = np.zeros(maskimgT.shape, np.uint8)
        # print(len(contours))
        # print(3)
        # plt.gcf().set_size_inches(60.0,40.0)
        # plt.subplot(2,1,1),plt.imshow(imgrayT,'gray')
        # plt.subplot(2,1,2),plt.imshow(maskimgT,'gray')

        # plt.show()
        for cnt in contours:
            # print(len(cnt))
            if len(cnt) > pointsTheshold:
                S1 = cv2.contourArea(cnt)
                ell = cv2.fitEllipse(cnt)
                S2 = math.pi*ell[1][0]*ell[1][1]
                # print(S2)
                if S1 > areathreshold and (S1/S2) > 0.005:  # 面积比例，可以更改，根据数据集。。。
                    print(S1, S2, ell)
                    #cv2.drawContours(emptyImageTT, cnt, -1, (255, 255, 255), 2)
                    cv2.ellipse(dstimgT, ell, color, 15)
                    cv2.ellipse(dstimgT, ell, color, cv2.FILLED)

        self.gray_img = dstimgT
        return dstimgT

     
    def denoise(self, dstimg=None, debug=False):
        if dstimg is None:
            dstimgT = self.gray_img

        imgThreshT = cv2.bilateralFilter(dstimg, 21, 21 * 2, 25 / 2)
        imgThresh = cv2.fastNlMeansDenoising(imgThreshT, None, 21, 11, 21)
        return imgThresh
