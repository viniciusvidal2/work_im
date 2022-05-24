import cv2
import numpy as np
import os
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import img_as_float, data
import matplotlib.pyplot as plt
from matplotlib import cm
import image_pkg


if __name__ == '__main__':
    print('Lets begin our algorithm!')

    # Read the image
    print('Reading image!')
    im = cv2.imread(os.path.join(os.getenv('HOME'), 'teste.JPG'))

    # Create our class object
    ip = image_pkg.ImageProcessor()

    # Resize and apply blur
    print('Resizing and blurring image ...')
    im = ip.resize_and_blur(im, show=False)

    # Apply gamma correction and grayscale
    print('Applying gamma correction and grayscale ...')
    im = ip.adjust_gamma(im, 0.5)
    gray = ip.rgb2gray(im)

    # Apply thresholds and define background
    print('Applying threshold ...')
    markers = ip.define_scene(gray, show=True)

    # Find the max with scikit-image
    print('Finding peaks in the image ...')

    # Test watershed algorithm
    # peak_coordinates = relevant_centroids
    # color_table = create_color_table('tab10')
    # if len(peak_coordinates) > 0:
    #     print('Lets do watershed ...')
    #     n_markers = len(peak_coordinates)
    #     markers = np.zeros_like(im_gray, dtype=np.int32)
    #     for i, pc in enumerate(peak_coordinates):
    #         cv2.circle(markers, tuple(pc), 5, (i), -1)
    #
    #     # im_copy = im.copy()
    #     cv2.watershed(im, light_scene)
    #     segments = np.zeros_like(im, dtype=np.uint8)
    #     for nm in range(n_markers):
    #         segments[markers == nm] = color_table[nm % len(color_table)]
    #     cv2.imshow('watershed', light_scene)
    #     cv2.waitKey(0)
