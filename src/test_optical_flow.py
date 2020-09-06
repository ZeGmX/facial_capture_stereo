"""
Testing optical flow with opencv
Using https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
"""

from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2


def mean_on_mask(img, mask):
    """
    Computes the mean intensity of a gray scale image on the pixels
    corresponding to the mask
    ----
    input:
        img: int array of shape (M, N) -> the gray scale image
        mask: int or bool array of shape (M, N) -> True or 1 if the pixel must
            be taken into account for the mean value
    ----
    output:
        res: float -> the mean value on the mask
    """
    assert img.shape == mask.shape, "Mask and image have different shapes"
    res = (img * mask).sum() / mask.sum()
    return res


def std_on_mask(img, mask):
    """
    Computes the standard deviation of the intensity of a gray scale image on
    the pixels corresponding to the mask
    ----
    input:
        img: int array of shape (M, N) -> the gray scale image
        mask: int or bool array of shape (M, N) -> True or 1 if the pixel must
            be taken into account for the standard deviation
    ----
    output:
        res: float -> the standard deviation on the mask
    """
    assert img.shape == mask.shape, "Mask and image have different shapes"
    sq_diff = (img - mean_on_mask(img, mask)) ** 2
    res = np.sqrt((sq_diff * mask).sum() / mask.sum())
    return res


def custom_standardization(img1, img2, mask):
    """
    Modify the two gray scale images so that they have the same mean and
    standard deviation on the mask and that each pixel has a value in [0, 255]
    ----
    input:
        img1, img2: int arrays of shape (M, N) -> the gray scale images
        mask: int or bool array of shape (M, N) -> True or 1 if the pixel must
            be taken into account
    ----
    output:
        img1, img2: float arrays of shape (M, N) -> the gray scale modified
            images
    """
    assert img1.shape == img2.shape, "The two images have different shapes"
    assert img1.shape == mask.shape, "Mask and image have different shapes"

    mean1 = mean_on_mask(img1, mask)
    mean2 = mean_on_mask(img2, mask)
    std1 = std_on_mask(img1, mask)
    std2 = std_on_mask(img2, mask)

    # Standardization -> once transformed mean = 0 and std = 1
    img1 = (img1 - mean1) / std1
    img2 = (img2 - mean2) / std2

    # Normalization -> once transformed, the values are in [0, 255]
    # Both image have the same mean and std
    mini = min(img1.min(), img2.min())
    maxi = max(img1.max(), img2.max())
    img1 = (img1 - mini) / (maxi - mini) * 255
    img2 = (img2 - mini) / (maxi - mini) * 255

    # Cleaning
    img1[np.logical_not(mask)] = 0
    img2[np.logical_not(mask)] = 0

    return img1, img2


def get_flow(img1, img2, method="hist", mask=None):
    """
    Computes the optical flow between the two image. Two preprocessing
    methods can be used, histogram equalization and  standardization
    ----
    input:
        img1, img2: int arrays of shape (M, N, 3)
        method: string -> "hist" for histogram equalization, "standard" for
            standardization, and anything else for no preprocessing
        mask: int or bool array of shape (M, N) -> The delimitation of the face
            It is only used if hist = "standard". Then the mean and standard
            deviation are only computed on the mask. if mask = None, the whole
            image is used
    ----
    output:
        flow: float array of shape (M, N, 2) -> the optical flow between the
            two images
        img1_gray, img2_gray: float arrays of shape (M, N) -> the preprocessed
            images in gray scale
    """
    # Computing the optical flow reauires a gray image
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if method == "standard":
        img1_gray, img2_gray = custom_standardization(img1_gray, img2_gray,
                                                      mask)
    elif method == "hist":
        img1_gray = cv2.equalizeHist(img1_gray)
        img2_gray = cv2.equalizeHist(img2_gray)

    flow = cv2.calcOpticalFlowFarneback(prev=img1_gray, next=img2_gray,
                                        flow=None, pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flags=0)

    return flow, img1_gray, img2_gray


def reconstruct_from_flow(img, flow):
    """
    Given an optical flow and an initial image, reconstructs the other image
    ----
    input:
        img: float array of shape (M, N, 3) -> the initial image
        flow: float array of shape (M, N, 2) -> the optical flow from img to
            the image we want to reconstruct
    ----
    output:
        reconstruct: float array of shape (M, N, 3) -> The reconstructed image
    """
    M, N, _ = img1.shape
    reconstruct = np.zeros_like(img1)
    for x in range(M):
        for y in range(N):
            xx, yy = np.round(flow[x, y] + [x, y]).astype(np.int)
            if 0 <= xx < M and 0 <= yy < N:
                reconstruct[x, y] = img2[xx, yy]
    return reconstruct


if __name__ == '__main__':
    # Either "standard" for standardization or "hist" for histogram
    # equalization or anything else for no pre processing
    method = "standard"

    path1 = "data/warped_images/warped_image_BEE10_onto_emily.png"
    path2 = "../data/Wikihuman_project/unpolarized/png/cam2_mixed_w.png"
    path_mask = "data/masks/mask_Emily.png"

    img1 = io.imread(path1)[..., :3]
    img2 = io.imread(path2)[..., :3]
    mask = io.imread(path_mask) / 255
    img2[np.logical_not(mask)] = 0

    M, N, _ = img1.shape

    flow, img1_gray, img2_gray = get_flow(img1, img2, method=method, mask=mask)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros_like(img1)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    fig = plt.figure()
    fig.add_subplot(1, 3, 1)
    plt.imshow(bgr)
    plt.axis("off")
    fig.add_subplot(1, 3, 2)
    plt.imshow(img1_gray, cmap="gray")
    plt.axis("off")
    fig.add_subplot(1, 3, 3)
    plt.imshow(img2_gray, cmap="gray")
    plt.axis("off")
    plt.show()
    # The optical flow (the color represents the direction and the intensity
    # represents the norm of the vector) and the two gray scale images

    plt.figure()
    plt.imshow(mag, cmap="gray")
    plt.axis("off")
    plt.show()
    # A gray map of the intensity of the flow between the two images

    reconstruct = reconstruct_from_flow(img2, flow)

    plt.figure()
    plt.imshow(reconstruct)
    plt.axis("off")
    plt.show()
    # The reconstructed image using optical flow
