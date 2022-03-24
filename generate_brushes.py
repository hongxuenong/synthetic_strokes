'''
generate synthetic brushes.
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


def add_stroke_pixels(center, width, stroke, gt, Is_fg):
    for i in range(-(width - 2), width - 1):
        for j in range(-(width - 2), width - 1):
            #check target pixel is within image range
            if ((center[0] + i < gt.shape[0]) & (center[1] + j < gt.shape[1])):
                #check target pixel is a fg pixel or background pixel
                if ((stroke.count([center[0] + i, center[1] + j]) == 0) &
                    (gt[center[0] + i][center[1] + j] == Is_fg * 255)):
                    stroke.append([center[0] + i, center[1] + j])
    return stroke


def edge_(mask, num_edge_seeds_fg, num_edge_seeds_bg):

    edges = cv2.Canny(mask, 100, 200)

    kernel = np.ones((10, 10), np.uint8)

    edge = cv2.dilate(edges, kernel, iterations=1)
    edge_pixels_fg = []
    edge_pixels_bg = []
    stroke_edge_fg = []
    stroke_edge_bg = []

    for x in range(edge.shape[0]):
        for y in range(edge.shape[1]):
            if (edge[x][y] == 255) & (mask[x][y] > 128):
                edge_pixels_fg.append([x, y])
            if (edge[x][y] == 255) & (mask[x][y] < 128):
                edge_pixels_bg.append([x, y])
    rand_edge = np.random.randint(0, len(edge_pixels_fg), num_edge_seeds_fg)
    edge_seed_fg = [edge_pixels_fg[sd] for sd in rand_edge]

    rand_edge = np.random.randint(0, len(edge_pixels_bg), num_edge_seeds_bg)
    edge_seed_bg = [edge_pixels_bg[sd] for sd in rand_edge]

    #generate edge fg and bg pixels from edge seeds
    for seed in edge_seed_fg:
        x1 = seed[0] - 30 if (seed[0] - 30 >= 0) else 0
        x2 = seed[0] + 30 if (seed[0] + 30 < mask.shape[0]) else mask.shape[0]
        y1 = seed[1] - 30 if (seed[1] - 30 >= 0) else 0
        y2 = seed[1] + 30 if (seed[1] + 30 < mask.shape[1]) else mask.shape[1]
        window = edge[x1:x2, y1:y2]
        mask_window = mask[x1:x2, y1:y2]
        for index, i in np.ndenumerate(window):
            if ((window[index] == 255) & (mask_window[index] > 200)):
                stroke_edge_fg.append(
                    [seed[0] - 30 + index[0], seed[1] - 30 + index[1]])

    for seed in edge_seed_bg:
        x1 = seed[0] - 30 if (seed[0] - 30 >= 0) else 0
        x2 = seed[0] + 30 if (seed[0] + 30 < mask.shape[0]) else mask.shape[0]
        y1 = seed[1] - 30 if (seed[1] - 30 >= 0) else 0
        y2 = seed[1] + 30 if (seed[1] + 30 < mask.shape[1]) else mask.shape[1]
        window = edge[x1:x2, y1:y2]
        mask_window = mask[x1:x2, y1:y2]
        for index, i in np.ndenumerate(window):
            if ((window[index] == 255) & (mask_window[index] < 100)):
                stroke_edge_bg.append(
                    [seed[0] - 30 + index[0], seed[1] - 30 + index[1]])

    # margin_p = mask + edge
    # margin_n = mask - edge

    return stroke_edge_fg, stroke_edge_bg


def arg_parse():
    parser = argparse.ArgumentParser(description="arguments parser")
    parser.add_argument("--i",
                        dest="Image_Dir",
                        default='./Test/Image/',
                        help="Image directory for Datasets.")
    parser.add_argument("--m",
                        dest="Mask_Dir",
                        default='./Test/Mask/',
                        help="Mask directory for Datasets.")
    parser.add_argument("--o",
                        dest="Out_Dir",
                        default='./Test/Output/',
                        help="Output directory")

    parser.add_argument("--positive_brush",
                        dest="positive_brush",
                        default=[0, 255, 0],
                        help="positive_brush")
    parser.add_argument("--negative_brush",
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
    Image_Dir = prog_args.Image_Dir
    Mask_Dir = prog_args.Mask_Dir
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
        f
        for f in glob.glob(prog_args.Out_Dir + 'InteractionMaps/fg/' + '*.png',
                           recursive=True)
    ]

    Map_filename = f.split('/')[-1]
    Map_filename = prog_args.Out_Dir + 'InteractionMaps/fg/' + Map_filename
    Map_filename = Map_filename.replace('.jpg', '_0.png')

    if any(Map_filename in s for s in current_interaction_maps):
        print(Map_filename, ' is already processed. Skipping...')
        return

    ## initialize random variables
    n_samples = np.random.randint(5, 10)
    num_fg_stroke_pts = np.random.randint(3, 9, n_samples)
    num_bg_stroke_pts = np.random.randint(6, 15, n_samples)
    num_edge_seeds_fg_n_samples = np.random.randint(0, 4, n_samples)
    num_edge_seeds_bg_n_samples = np.random.randint(0, 4, n_samples)

    rand_parms = zip(num_fg_stroke_pts, num_bg_stroke_pts,
                     num_edge_seeds_fg_n_samples, num_edge_seeds_bg_n_samples)

    for iter, (num_fg_seeds, num_bg_seeds, num_edge_seeds_fg,
               num_edge_seeds_bg) in enumerate(rand_parms):

        mask_filename = f.replace('jpg', 'png')
        mask_filename = mask_filename.replace('Image', 'Mask')

        image = cv2.imread(f)
        mask = cv2.imread(mask_filename, 0)

        #generate fg,bg stroke points
        foreground_pixels = []
        background_pixels = []
        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                if mask[x][y] >= 128:
                    foreground_pixels.append([x, y])
                else:
                    background_pixels.append([x, y])
        fg_pt_idxs = np.random.randint(0, len(foreground_pixels), num_fg_seeds)
        bg_pt_idxs = np.random.randint(0, len(background_pixels), num_bg_seeds)

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

        dismap_fg = cv2.distanceTransform(fg_mask, cv2.DIST_L2,
                                          cv2.DIST_MASK_PRECISE)

        dismap_bg = cv2.distanceTransform(bg_mask, cv2.DIST_L2,
                                          cv2.DIST_MASK_PRECISE)

        if (not os.path.exists(Out_Dir + "/Images_with_strokes/")):
            Dir = Out_Dir + "Images_with_strokes/"
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'InteractionMaps/fg/')):
            Dir = Out_Dir + 'InteractionMaps/fg/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)
        if (not os.path.exists(Out_Dir + 'InteractionMaps/bg/')):
            Dir = Out_Dir + 'InteractionMaps/bg/'
            os.makedirs(Dir)
            print("Directory '% s' created" % Dir)

        filename = mask_filename.split('/')[-1]
        name = filename.split('.')[0]
        extension = filename.split('.')[1]
        name = name + '_{}'.format(iter)
        filename = name + "." + extension

        cv2.imwrite(Out_Dir + "Images_with_strokes/" + filename, image)
        cv2.imwrite(Out_Dir + 'InteractionMaps/fg/' + filename, dismap_fg)
        cv2.imwrite(Out_Dir + 'InteractionMaps/bg/' + filename, dismap_bg)


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