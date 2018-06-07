import numpy as np
import os
import pdb
import cv2

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from skimage.measure import label, regionprops
from skimage.morphology import closing, square


def find_biggest_bbox(binary_img):
    label_list = label(binary_img)
    bbox = None
    max_area = 0
    for region in regionprops(label_list):
        if region.area > max_area:
            minr, minc, maxr, maxc = region.bbox
            bbox = (minc, minr, maxc - minc, maxr - minr)
            max_area = region.area
    return bbox


def binarize(img, thresh):
    # img = np.dot(img[...,:3], [0.299, 0.587, 0.114])  # RGB to Grey-scale
    return closing(img > thresh, square(3))


def find_location_by_cam(cam, thresh=0.2):
    # cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
    binary_img = binarize(cam, thresh)
    bbox = find_biggest_bbox(binary_img)
    return bbox


def draw_bounding_box(ax, bbox, fill=False, color='red', linewidth=2):
    x, y, width, height = bbox
    rect = mpatches.Rectangle(
        (x, y), width, height,
        fill=fill, color=color, linewidth=linewidth)
    ax.add_patch(rect)

# def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
def create_heatmap(im_cloud, kernel_size=(5, 5), colormap=cv2.COLORMAP_JET, a1=0.5, a2=0.5):
    '''
    img is numpy array
    kernel_size must be odd ie. (5,5)
    '''

    # create blur image, kernel must be an odd number
    im_cloud_blur = cv2.GaussianBlur(im_cloud, kernel_size, 0)

    pdb.set_trace()

    # If you need to invert the black/white data image
    # im_blur = np.invert(im_blur)
    # Convert back to BGR for cv2
    #im_cloud_blur = cv2.cvtColor(im_cloud_blur,cv2.COLOR_GRAY2BGR)

    # Apply colormap
    im_cloud_clr = cv2.applyColorMap(im_cloud_blur, colormap)

    # blend images 50/50
    # return (a1*im_map + a2*im_cloud_clr).astype(np.uint8)
    return a2*im_cloud_clr.astype(np.uint8)


def visualize(X, cam_list, bbox_list, nb_samples, save_plot=None):
    fig, axs = plt.subplots(nb_samples, 2, figsize=(6, 18))
    for i in range(nb_samples):
        axs[i][0].imshow(X[i])
        draw_bounding_box(axs[i][0], bbox_list[i], color='red')

        axs[i][1].imshow(X[i])
        axs[i][1].imshow(cam_list[i],
                         cmap=plt.cm.jet,
                         alpha=0.5,
                         interpolation='nearest')
        draw_bounding_box(axs[i][1], bbox_list[i], color='red')

        if save_plot != None:
           plt.savefig(os.path.join(save_plot, "frames_%d.jpg" %i))

        plt.show()

    return fig, axs

def save_result_cam(result_list, bbox_list, nb_samples, save_plot=None):
    for i in range(nb_samples):
        left_top = (bbox_list[i][0], bbox_list[i][1])
        right_bottom = (bbox_list[i][2], bbox_list[i][3])
        result = cv2.rectangle(result_list[i], left_top, right_bottom, (0, 255, 0), 3)
        if save_plot != None:
            cv2.imwrite(os.path.join(save_plot, "frames_%d.jpg" %i), result)
