import numpy as np
import pdb

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

    # pdb.set_trace()
    # plt.imshow(img * 255); plt.show()
    # plt.imshow(closing(img > thresh, square(3))); plt.show()


    return closing(img > thresh, square(3))


def find_location_by_cam(cam, thresh=0.4):
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


def visualize(X, cam_list, bbox_list, nb_samples):
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
    plt.show()

    return fig, axs

