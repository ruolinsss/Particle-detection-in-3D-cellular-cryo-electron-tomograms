import random
import numpy as np
import scipy
import scipy.ndimage
from scipy.ndimage.morphology import distance_transform_edt, distance_transform_cdt, grey_closing
from math import pi as PI
from skimage.measure import label

def random_rotation_3d(image, gt, max_angle, order=1):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    image: 3d image (z,y,x)
    gt: 3d groundtruth (z,y,x)
    max_angle: `float`. The maximum rotation angle.
    order: whether use interpolation or not:
           1: use linear interpolation
           0: not use interpolation (for instance label)

    Returns:
    imagez: rotated image
    gtz: rotated gt
    """    
    # rotate along x-axis
    angle = random.uniform(-max_angle, max_angle)
#     angle=45
    imagex = scipy.ndimage.interpolation.rotate(image, angle,  axes=(0, 1), reshape=False, order=1)
    gtx = scipy.ndimage.interpolation.rotate(gt, angle,  axes=(0, 1), reshape=False, order=order)

#     # rotate along y-axis
#     angle=45
    angle = random.uniform(-max_angle, max_angle)
    imagey = scipy.ndimage.interpolation.rotate(imagex, angle, axes=(0, 2), reshape=False, order=1)
    gty = scipy.ndimage.interpolation.rotate(gtx, angle,  axes=(0, 2), reshape=False, order=order)

#     #rotate along z-axis
    angle = random.uniform(-max_angle, max_angle)
#     angle=45
    imagez = scipy.ndimage.interpolation.rotate(imagey, angle, axes=(1, 2), reshape=False, order=1)
    gtz = scipy.ndimage.interpolation.rotate(gty, angle,  axes=(1, 2), reshape=False, order=order)
    
    return imagez,gtz



def dist_label(label, neighbor_radius=None, apply_grayscale_closing=True):
    """ 
    Cell center label creation (Euclidean distance). Modified from https://bitbucket.org/t_scherr/cell-segmentation-and-tracking/src/master/segmentation/training/train_data_representations.py#lines-167

    :param label: Intensity-coded instance segmentation label image.
        :type label:
    :param neighbor_radius: Defines the area to look for neighbors (smaller radius in px decreases the computation time)
        :type neighbor_radius: int
    :param apply_grayscale_closing: close gaps in between neighbor labels.
        :type apply_grayscale_closing: bool
    :return: Cell distance label image, neighbor distance label image.
    """
    # Relabel label to avoid some errors/bugs
    label_dist = np.zeros(shape=label.shape, dtype=np.float)
    label_dist_neighbor = np.zeros(shape=label.shape, dtype=np.float)

    mean_diameter = []
    for i in np.unique(label):
        mean_diameter.append((6 * np.sum(label==i) / PI) ** (1 / 3))
    mean_diameter = np.mean(np.array(mean_diameter))
    neighbor_radius = 3 * mean_diameter

    # Find centroids, crop image, calculate distance transform
    for i in np.unique(label)[1:]:

        # Get nucleus and Euclidean distance transform for each nucleus
        nucleus = (label == i)
        centroid, diameter = np.mean(np.where(nucleus),axis=-1), ((6 * np.sum(nucleus) / PI) ** (1 / 3))
        nucleus_crop = nucleus[
                       int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
                       int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
                       int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
                       ].astype(np.float32)
        nucleus_crop_dist = distance_transform_edt(nucleus_crop)
        nucleus_crop_dist[nucleus_crop_dist==1] = 0
        if np.max(nucleus_crop_dist) > 0:
            nucleus_crop_dist = nucleus_crop_dist / np.max(nucleus_crop_dist)

        label_dist[
        int(max(centroid[0] - neighbor_radius, 0)):int(min(centroid[0] + neighbor_radius, label.shape[0])),
        int(max(centroid[1] - neighbor_radius, 0)):int(min(centroid[1] + neighbor_radius, label.shape[1])),
        int(max(centroid[2] - neighbor_radius, 0)):int(min(centroid[2] + neighbor_radius, label.shape[2]))
        ] += nucleus_crop_dist

        # Get crop containing neighboring nuclei
        nucleus_neighbor_crop = np.copy(label[
                                int(max(centroid[0] - neighbor_radius, 0)):int(
                                    min(centroid[0] + neighbor_radius, label.shape[0])),
                                int(max(centroid[1] - neighbor_radius, 0)):int(
                                    min(centroid[1] + neighbor_radius, label.shape[1])),
                                int(max(centroid[2] - neighbor_radius, 0)):int(
                                    min(centroid[2] + neighbor_radius, label.shape[2]))
                                ])

    return label_dist