import cv2
import numpy as np
import os
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import img_as_float, data
import matplotlib.pyplot as plt
from matplotlib import cm


class ImageProcessor:
    def __int__(self):
        # Initialize default color map
        self.cmap = cm.get_cmap('tab10')
        # Default gamma = 1 table
        self.table = np.array([i for i in np.arange(0, 256)]).astype("uint8")

    def create_color_table(self, name):
        self.cmap = cm.get_cmap(name)
        return np.array([tuple(np.array(self.cmap(i)[:3]) * 255)
                         for i in range(len(self.cmap.colors))]).astype(int)

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        inv_gamma = 1.0 / gamma
        self.table = np.array([((i / 255.0) ** inv_gamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")
        # apply gamma correction using the lookup table
        return cv2.LUT(image, self.table)

    def resize_and_blur(self, im, resize_factor=5, blur_size=21, show=False):
        # Set dimensions and reshape
        dims = (int(im.shape[1] / resize_factor), int(im.shape[0] / resize_factor))
        im = cv2.resize(im, dims, cv2.INTER_AREA)
        # Apply blur
        im = cv2.medianBlur(im, blur_size)
        if show:
            plt.imshow(im, cmap='rgb')
            plt.show()
        return im

    def rgb2gray(self, im):
        return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    def define_scene(self, im, thresh_pct=0.5, filter_area=True, show=False):
        '''
        This function will receive an image (im) and apply two thresholds: regular and OTSU.
        It will detect shiny spots, and classify foreground and background
        with OTSU method.
        It will take relevant regions as bright spots in the scene
        :return:
        Brightness image, with relevant segmented areas
        '''
        # Defining background and foreground with threshold methods
        foreground_value = 255
        _, background = cv2.threshold(im, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Find the foreground centers with the distance operation
        dist_transform = cv2.distanceTransform(background, cv2.DIST_L2, maskSize=5)
        # _, foreground = cv2.threshold(im, 0.7*dist_transform.max(),
        #                               255, cv2.THRESH_BINARY)
        _, foreground = cv2.threshold(im, thresh_pct*im.max(), 255, cv2.THRESH_BINARY)
        # Apply simple erosion to better segment and filter the foreground image
        kernel = np.ones((3, 3), dtype=np.uint8)
        foreground = cv2.erode(foreground, kernel=kernel, iterations=1)
        if show:
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(background, cmap=plt.cm.gray)
            ax[0].axis('off')
            ax[0].set_title('Background')
            ax[1].imshow(dist_transform, cmap=plt.cm.gray)
            ax[1].axis('off')
            ax[1].set_title('Dist transform')
            ax[2].imshow(foreground, cmap=plt.cm.gray)
            ax[2].autoscale(False)
            ax[2].axis('off')
            ax[2].set_title('Foreground')
            fig.tight_layout()
            plt.show()
        # Estimate the unknown region
        unknown = cv2.subtract(background, foreground)
        if show:
            plt.imshow(unknown, cmap='gray')
            plt.show()
        # Split the foreground into regions representing different classes
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 4, cv2.CV_8U)
        # Lets make the unknown region as the black part of the classified image
        labels = (labels + 1).astype(np.uint8)
        labels[unknown == unknown.max()] = 0
        if show:
            plt.imshow(labels, cmap='gray')
            plt.show()
        # source_areas = stats[:, 4]
        # relevant_sources = source_areas > 0.1 * source_areas.max()
        # relevant_centroids = []
        # for i, rs in enumerate(relevant_sources):
        #     if rs:
        #         relevant_centroids.append(centroids[i])
        # light_sources = np.zeros_like(labels, dtype=int)
        # for i in range(labels.shape[0]):
        #     for j in range(labels.shape[1]):
        #         if relevant_sources[labels[i, j]] != 0:
        #             light_sources[i, j] = int((labels[i, j] + 1) * 255 / num_labels)
        # light_scene = background + light_sources
        # if show:
        #     plt.imshow(light_scene, cmap='gray')
        #     plt.show()

    def find_max_intensity_coordinates(self, im, filter_size=100, show=False):
        im_float = img_as_float(im)
        im_max = ndi.maximum_filter(im_float, size=filter_size, mode='reflect')
        peak_coordinates = peak_local_max(im_float, min_distance=filter_size)
        # Display results if needed
        if show:
            fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
            ax = axes.ravel()
            ax[0].imshow(im, cmap=plt.cm.gray)
            ax[0].axis('off')
            ax[0].set_title('Original')
            ax[1].imshow(im_max, cmap=plt.cm.gray)
            ax[1].axis('off')
            ax[1].set_title('Maximum filter')
            ax[2].imshow(im, cmap=plt.cm.gray)
            ax[2].autoscale(False)
            ax[2].plot(peak_coordinates[:, 1], peak_coordinates[:, 0], 'r*')
            ax[2].axis('off')
            ax[2].set_title('Peak local max')
            fig.tight_layout()
            plt.show()
        return peak_coordinates

