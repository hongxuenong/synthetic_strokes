'''
Generate 4-instance synthetic training images with bursh inputs.
'''
import os
import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
import math

import os
import sys
import argparse

import time
import multiprocessing
from multiprocessing import Pool, Queue, Lock
from joblib import Parallel, delayed
from numpy import random


def overlay_two_image(image, overlay, ignore_color=[0, 0, 0]):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay == ignore_color).all(-1, keepdims=True)
    out = np.where(mask, image,
                   (image * 0.5 + overlay * 0.5).astype(image.dtype))
    return out


def main():
    imglist = glob.glob('data/DUTS-TR/DUTS-TR-Image/*.jpg')
    print(len(imglist))
    for collection in ["train", "test"]:
        if (collection == "train"):
            num = 5000
        else:
            num = 100
        for i in range(0, num + 1):
            print('{} out of {}'.format(i + 1, num))
            indexes = np.random.randint(0, len(imglist), 4)

            c_img = np.zeros((1000, 1000, 4))

            img = cv2.imread(imglist[indexes[0]], cv2.IMREAD_UNCHANGED)
            label_name = imglist[indexes[0]].replace('Image', "Mask")
            label_name = label_name.replace(".jpg", ".png")
            label = cv2.imread(label_name, 0)
            img = cv2.resize(img, (500, 500))
            label = cv2.resize(label, (500, 500))
            c_img[:500, :500, :3] = img
            c_img[:500, :500, 3] = label
            img = cv2.imread(imglist[indexes[1]], cv2.IMREAD_UNCHANGED)
            img = cv2.resize(img, (500, 500))
            label_name = imglist[indexes[1]].replace('Image', "Mask")
            label_name = label_name.replace(".jpg", ".png")
            label = cv2.imread(label_name, 0)
            label = cv2.resize(label, (500, 500))
            c_img[500:, :500, :3] = img
            c_img[500:, :500, 3] = label
            img = cv2.imread(imglist[indexes[2]], cv2.IMREAD_UNCHANGED)
            label_name = imglist[indexes[2]].replace('Image', "Mask")
            label_name = label_name.replace(".jpg", ".png")
            label = cv2.imread(label_name, 0)
            img = cv2.resize(img, (500, 500))
            label = cv2.resize(label, (500, 500))
            c_img[:500, 500:, :3] = img
            c_img[:500, 500:, 3] = label
            img = cv2.imread(imglist[indexes[3]], cv2.IMREAD_UNCHANGED)
            label_name = imglist[indexes[3]].replace('Image', "Mask")
            label_name = label_name.replace(".jpg", ".png")
            label = cv2.imread(label_name, 0)
            img = cv2.resize(img, (500, 500))
            label = cv2.resize(label, (500, 500))
            c_img[500:, 500:, :3] = img
            c_img[500:, 500:, 3] = label
            cv2.imwrite('multi_instance/{}/raw/{}.jpg'.format(collection, i),
                        c_img[:, :, :3])

            mask = np.zeros((1000, 1000))
            rand_num = np.random.randint(0, 4, 1)
            if rand_num == 0:
                mask[:500, :500] = c_img[:500, :500, 3]
            elif rand_num == 1:
                mask[:500, 500:] = c_img[:500, 500:, 3]
            elif rand_num == 2:
                mask[500:, :500] = c_img[500:, :500, 3]
            else:
                mask[500:, 500:] = c_img[500:, 500:, 3]

            ## initialize random variables
            # n_samples = np.random.randint(5,10)
            n_samples = 1
            num_fg_stroke_pts = np.random.randint(3, 9, n_samples)
            num_bg_stroke_pts = np.random.randint(6, 15, n_samples)
            num_edge_seeds_fg_n_samples = np.random.randint(0, 4, n_samples)
            num_edge_seeds_bg_n_samples = np.random.randint(0, 4, n_samples)

            rand_parms = zip(num_fg_stroke_pts, num_bg_stroke_pts,
                             num_edge_seeds_fg_n_samples,
                             num_edge_seeds_bg_n_samples)

            positive_brush = [0, 255, 0]
            negative_brush = [0, 0, 255]
            for iter, (num_fg_seeds, num_bg_seeds, num_edge_seeds_fg,
                       num_edge_seeds_bg) in enumerate(rand_parms):
                image = c_img[:, :, :3]
                foreground_pixels = []
                background_pixels = []
                for x in range(mask.shape[0]):
                    for y in range(mask.shape[1]):
                        if mask[x][y] >= 128:
                            foreground_pixels.append([x, y])
                        else:
                            background_pixels.append([x, y])
                fg_pt_idxs = np.random.randint(0, len(foreground_pixels),
                                               num_fg_seeds)
                bg_pt_idxs = np.random.randint(0, len(background_pixels),
                                               num_bg_seeds)

                fg_stroke_pts = [foreground_pixels[idx] for idx in fg_pt_idxs]
                bg_stroke_pts = [background_pixels[idx] for idx in bg_pt_idxs]

                fg_stroke_pts = np.array(fg_stroke_pts)
                bg_stroke_pts = np.array(bg_stroke_pts)
                #draw on empty images
                fg_map_img = np.zeros(image.shape, np.uint8)
                bg_map_img = np.zeros(image.shape, np.uint8)
                temp = fg_stroke_pts[:, 0].copy()
                fg_stroke_pts[:, 0] = fg_stroke_pts[:, 1]
                fg_stroke_pts[:, 1] = temp.copy()
                fg_stroke_pts = fg_stroke_pts.reshape((-1, 1, 2))

                fg_map_img = cv2.polylines(fg_map_img, [fg_stroke_pts],
                                           False,
                                           positive_brush,
                                           thickness=15)

                temp = bg_stroke_pts[:, 0].copy()
                bg_stroke_pts[:, 0] = bg_stroke_pts[:, 1]
                bg_stroke_pts[:, 1] = temp.copy()
                bg_stroke_pts = bg_stroke_pts.reshape((-1, 1, 2))
                bg_map_img = cv2.polylines(bg_map_img, [bg_stroke_pts],
                                           False,
                                           negative_brush,
                                           thickness=15)
                #draw original image

                ##stroke calibrate
                _, fg_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
                _, bg_mask = cv2.threshold(mask, 0, 128, cv2.THRESH_BINARY_INV)

                for c in range(3):
                    fg_map_img[:, :, c] = np.multiply(fg_map_img[:, :, c],
                                                      fg_mask / 255)
                    bg_map_img[:, :, c] = np.multiply(bg_map_img[:, :, c],
                                                      bg_mask / 255)
                    # bg_map_img = cv2.bitwise_and(bg_map_img,bg_map_img,mask=bg_mask)

                #combine fg_edges with fg and bg_edges with bg
                image = overlay_two_image(image, fg_map_img)
                image = overlay_two_image(image, bg_map_img)

                fg_map_img = cv2.cvtColor(fg_map_img, cv2.COLOR_BGR2GRAY)
                bg_map_img = cv2.cvtColor(bg_map_img, cv2.COLOR_BGR2GRAY)

                _, fg_mask = cv2.threshold(fg_map_img, 128, 255,
                                           cv2.THRESH_BINARY_INV)
                _, bg_mask = cv2.threshold(bg_map_img, 20, 255,
                                           cv2.THRESH_BINARY_INV)

                dismap_fg = fg_mask
                dismap_bg = bg_mask
                # dismap_fg = cv2.distanceTransform(fg_mask, cv2.DIST_L2,
                #                                   cv2.DIST_MASK_PRECISE)

                # dismap_bg = cv2.distanceTransform(bg_mask, cv2.DIST_L2,
                #                                   cv2.DIST_MASK_PRECISE)
                cv2.imwrite(
                    'multi_instance_v2/' + collection +
                    '/Images_with_strokes/' + str(i) + '.png', image)
                cv2.imwrite(
                    'multi_instance_v2/' + collection +
                    '/InteractionMaps/fg/' + str(i) + '.png', dismap_fg)
                cv2.imwrite(
                    'multi_instance_v2/' + collection +
                    '/InteractionMaps/bg/' + str(i) + '.png', dismap_bg)
                cv2.imwrite(
                    'multi_instance_v2/' + collection + '/label/' + str(i) +
                    '.png', mask)


if __name__ == '__main__':
    main()