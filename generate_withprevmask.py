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


def arg_parse():
    parser = argparse.ArgumentParser(description="arguments parser")
    parser.add_argument(
        "--i",
        dest="Image_Dir",
        default='../data/train_with_prev_mask/DUTS-TR-Image/',
        help="Image directory for Datasets.")
    parser.add_argument(
        "--m",
        dest="Mask_Dir",
        default='../data/train_with_prev_mask/DUTS-TR-Mask/',
        help="Mask directory for Datasets.")
    parser.add_argument(
        "--o",
        dest="Out_Dir",
        default='../data/train_with_prev_mask/Output/',
        help="Output directory")

    parser.add_argument(
        "--positive_brush",
        dest="positive_brush",
        default=[0, 255, 0],
        help="positive_brush")
    parser.add_argument(
        "--negative_brush",
        dest="negative_brush",
        default=[0, 0, 255],
        help="negative_brush")
    return parser.parse_args()


def overlay_two_image(image, overlay, ignore_color=[0, 0, 0]):
    ignore_color = np.asarray(ignore_color)
    mask = (overlay == ignore_color).all(-1, keepdims=True)
    out = np.where(mask, image,
                   (image * 0.5 + overlay * 0.5).astype(image.dtype))
    return out


def process_image(f, iteration, image_num, num_cores):
    #     for f in images:
    start_time = time.time()
    prog_args = arg_parse()
    Out_Dir = prog_args.Out_Dir
    positive_brush = prog_args.positive_brush
    negative_brush = prog_args.negative_brush
    print('processing:', iteration)
    print("progress:{:.2f}%".format(iteration / image_num * 100))
    fid = open("Output.txt", "a")
    fid.write("progress:{:.2f}%\n".format(iteration / image_num * 100))
    fid.close()

    ##check if this image is already processed
    current_interaction_maps = [
        f for f in glob.glob(
            prog_args.Out_Dir + 'fg/' + '*.png', recursive=True)
    ]

    Map_filename = f.split('/')[-1]
    Map_filename = prog_args.Out_Dir + 'fg/' + Map_filename
    Map_filename = Map_filename.replace('.jpg', '_0.png')

    if any(Map_filename in s for s in current_interaction_maps):
        print(Map_filename, ' is already processed. Skipping...')
        return

    ## initialize random variables
    n_samples = np.random.randint(1, 4)
    num_fg_stroke_pts = np.random.randint(2, 9, n_samples)
    num_bg_stroke_pts = np.random.randint(0, 9, n_samples)
    num_edge_seeds_fg_n_samples = np.random.randint(0, 4, n_samples)
    num_edge_seeds_bg_n_samples = np.random.randint(0, 4, n_samples)

    rand_parms = zip(num_fg_stroke_pts, num_bg_stroke_pts,
                     num_edge_seeds_fg_n_samples, num_edge_seeds_bg_n_samples)

    for iter, (num_fg_seeds, num_bg_seeds, num_edge_seeds_fg,
               num_edge_seeds_bg) in enumerate(rand_parms):

        mask_filename = f.replace('jpg', 'png')
        mask_filename = mask_filename.replace('Image', 'Mask')

        image = cv2.imread(f)
        ori_image = image.copy()
        mask = cv2.imread(mask_filename, 0)
        _, mask_bin = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
        # find largest contour:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        mode = np.random.rand()
        if mode < 0.6:
            maskofmask = np.zeros_like(mask_bin)
            maskofmask[y:int(y + h / 2), x:int(x + w / 2)] = 255
            fg_area = cv2.bitwise_and(maskofmask, mask_bin)
            pre_mask = mask.copy()
            pre_mask[y:int(y + h / 2), x:int(x + w / 2)] = 0
            pre_mask = cv2.blur(pre_mask, (10, 10))
        else:
            maskofmask = mask.copy()
            maskofmask[y:int(y + h / 2), x:int(x + w / 2)] = 255
            bg_area = cv2.bitwise_xor(maskofmask, mask_bin)
            pre_mask = mask.copy()
            pre_mask[y:int(y + h / 2), x:int(x + w / 2)] = 255
            pre_mask = cv2.blur(pre_mask, (10, 10))
            # cv2.imwrite("bg_area.png", bg_area)

        #

        #generate fg,bg stroke points
        foreground_pixels = []
        background_pixels = []
        if (mode < 0.6):
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if fg_area[x][y] >= 128:
                        foreground_pixels.append([x, y])
                    elif (mask[x][y] <= 128):
                        background_pixels.append([x, y])
        else:
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if bg_area[x][y] >= 128:
                        background_pixels.append([x, y])
                    elif (mask[x][y] >= 128):
                        foreground_pixels.append([x, y])
                if (np.random.rand() < 0.5):
                    num_fg_seeds = 0
        if (len(foreground_pixels) == 0):
            return
        fg_pt_idxs = np.random.randint(0, len(foreground_pixels), num_fg_seeds)
        bg_pt_idxs = np.random.randint(0, len(background_pixels), num_bg_seeds)
        p1 = np.random.rand()
        if ((mode < 0.6) &
            (p1 < 0.1)):  # 30% chance random drop all background brush
            bg_pt_idxs = []

        fg_stroke_pts = [foreground_pixels[idx] for idx in fg_pt_idxs]
        bg_stroke_pts = [background_pixels[idx] for idx in bg_pt_idxs]
        fg_stroke_pts = np.array(fg_stroke_pts)
        bg_stroke_pts = np.array(bg_stroke_pts)
        #draw on empty images

        fg_map_img = np.zeros(image.shape, np.uint8)
        bg_map_img = np.zeros(image.shape, np.uint8)

        print(len(bg_stroke_pts), len(fg_stroke_pts))
        if ((len(bg_stroke_pts) <= 1) & (len(fg_stroke_pts) <= 1)):

            return
        print("test..")
        if (len(fg_stroke_pts) > 0):
            temp = fg_stroke_pts[:, 0].copy()
            fg_stroke_pts[:, 0] = fg_stroke_pts[:, 1]
            fg_stroke_pts[:, 1] = temp.copy()
            fg_stroke_pts = fg_stroke_pts.reshape((-1, 1, 2))

        fg_map_img = cv2.polylines(
            fg_map_img, [fg_stroke_pts], False, positive_brush, thickness=15)

        if (len(bg_stroke_pts) > 0):
            temp = bg_stroke_pts[:, 0].copy()
            bg_stroke_pts[:, 0] = bg_stroke_pts[:, 1]
            bg_stroke_pts[:, 1] = temp.copy()
            bg_stroke_pts = bg_stroke_pts.reshape((-1, 1, 2))
            bg_map_img = cv2.polylines(
                bg_map_img, [bg_stroke_pts],
                False,
                negative_brush,
                thickness=15)

        #draw original image

        ##stroke calibrate
        _, fg_mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        _, bg_mask = cv2.threshold(mask, 0, 128, cv2.THRESH_BINARY_INV)

        fg_map_img = cv2.bitwise_and(fg_map_img, fg_map_img, mask=fg_mask)
        bg_map_img = cv2.bitwise_and(bg_map_img, bg_map_img, mask=bg_mask)
        #combine fg_edges with fg and bg_edges with bg
        # stroke_fg = stroke_fg + stroke_edge_fg
        # stroke_bg = stroke_bg + stroke_edge_bg
        image = overlay_two_image(image, fg_map_img)
        image = overlay_two_image(image, bg_map_img)
        # image = cv2.addWeighted(image,0.4,bg_map_img,0.1,0)

        #         cp3 = timeit.timeit()
        #         fid.write("Checkpoint3:{}".format(cp3)+"\n")

        # generate L2 distance map
        # fg_mask_invert = 1-fg_mask
        fg_map_img = cv2.cvtColor(fg_map_img, cv2.COLOR_BGR2GRAY)
        bg_map_img = cv2.cvtColor(bg_map_img, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(fg_map_img, 128, 255, cv2.THRESH_BINARY_INV)
        _, bg_mask = cv2.threshold(bg_map_img, 50, 255, cv2.THRESH_BINARY_INV)

        if (not os.path.exists(Out_Dir + "/Images_with_strokes/")):
            Dir = Out_Dir + "Images_with_strokes/"
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'fg/')):
            Dir = Out_Dir + 'fg/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'bg/')):
            Dir = Out_Dir + 'bg/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'pre_mask/')):
            Dir = Out_Dir + 'pre_mask/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'raw/')):
            Dir = Out_Dir + 'raw/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'label/')):
            Dir = Out_Dir + 'label/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'brush_feature/')):
            Dir = Out_Dir + 'brush_feature/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)

        filename = mask_filename.split('/')[-1]
        name = filename.split('.')[0]
        extension = filename.split('.')[1]
        name = name + '_{}'.format(iter)
        filename = name + "." + extension

        brush_feature = np.zeros_like(ori_image)
        brush_feature[:, :, 0] = bg_mask
        brush_feature[:, :, 1] = fg_mask
        brush_feature[:, :, 2] = pre_mask
        cv2.imwrite(Out_Dir + "Images_with_strokes/" + filename, image)
        cv2.imwrite(Out_Dir + "raw/" + name + '.jpg', ori_image)
        cv2.imwrite(Out_Dir + 'fg/' + filename, fg_mask)
        cv2.imwrite(Out_Dir + 'bg/' + filename, bg_mask)
        cv2.imwrite(Out_Dir + 'pre_mask/' + filename, pre_mask)
        cv2.imwrite(Out_Dir + 'label/' + filename, mask)
        cv2.imwrite(Out_Dir + 'brush_feature/' + filename, brush_feature)


#         print("saved image_with_strokes to: "+Out_Dir+"Images_with_strokes/"+filename)
#         print("saved foreground interaction map to: "+Out_Dir+'InteractionMaps/fg/'+filename)
#         print("saved background interaction map to: "+Out_Dir+'InteractionMaps/bg/'+filename)

    end = time.time()
    time_elap = end - start_time
    print("Time Elapsed:{:.2f} seconds".format(time_elap))
    time_remain = time_elap * (image_num - iteration) / num_cores
    hours, rem = divmod(time_remain, 3600)
    minutes, seconds = divmod(rem, 60)

    print("Estimated time remaining:{:.2f}:{:.2f}:{:.2f}, ".format(
        hours, minutes, seconds))
    fid = open("Output.txt", "a")
    fid.write("Estimated time remaining:{:.2f}:{:.2f}:{:.2f}, \n".format(
        hours, minutes, seconds))
    fid.close()


def main():

    prog_args = arg_parse()

    Image_Dir = prog_args.Image_Dir
    Mask_Dir = prog_args.Mask_Dir
    Out_Dir = prog_args.Out_Dir

    # Read in images
    print(Image_Dir)
    images = [f for f in glob.glob(Image_Dir + '*.jpg', recursive=True)]
    image_num = len(images)
    total_images = np.zeros(image_num)
    print("Found total {} images to process".format(image_num))
    masks = [f for f in glob.glob(Mask_Dir + '*.png', recursive=True)]
    #     current_interaction_maps = [f for f in glob.glob(Out_Dir+'InteractionMaps/fg/'+'*.png', recursive=True)]

    # num_cores = multiprocessing.cpu_count()
    num_cores = 1
    print("number of core used:{}".format(num_cores))
    Parallel(n_jobs=num_cores)(
        delayed(process_image)(i, iter, image_num, num_cores)
        for iter, i in enumerate(images))


if __name__ == '__main__':
    main()