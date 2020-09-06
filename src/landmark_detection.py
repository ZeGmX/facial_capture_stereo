"""""""""""""""
Finds the landmarks on each image and adds a plot to show it for both the dlib
and sfd method

Part of this comes from the example code file detect_landmarks_in_image.py
from https://github.com/1adrianb/face-alignment

To run, use python3 landmark_detection.py [3D] [line]
Use the 3D argument if you want a 3D plot of the face
Use the line argument if you want the landmarks to be connected
"""""""""""""""


import face_alignment
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
from sys import argv
from os import listdir
from time import time
import collections


def plot_image_and_landmarks(img, preds, file_index, nb_files,
                             dimension, fig, pred_types, plot_style):
    """
    plots the image with its corresponding landmark predictions, and eventually
    a 3D plot
    ----
    input:
        img: array with shape (M, N, 3) -> the image to be displayed
        preds: array of shape (68, dimension) -> the prediction of the
            landmark positions
        file_index: int -> position of the image on the line
        nb_files: int -> number of columns to be displayed
        dimension: int (2 or 3) -> if 2, only plots the face, if 3 plots in the
            3D space
        fig: plt.Figure -> where the image will be displayed
        pred_types: dictionary -> contains the informations of color and
            indices to plot the landmarks
        plot_style: dictionary -> style to be used when ploting the landmarks
        ----
        output:
            None
    """

    # 2D-Plot
    ax = fig.add_subplot(dimension - 1, nb_files, file_index + 1)
    ax.imshow(img)

    for pred_type in pred_types.values():
        ax.plot(preds[pred_type.slice, 0],
                preds[pred_type.slice, 1],
                color=pred_type.color, **plot_style)

    ax.axis('off')

    if dimension == 3:
        # 3D-Plot

        ax = fig.add_subplot(2, nb_files, file_index + nb_files + 1,
                             projection='3d')
        ax.scatter(preds[:, 0] * 1.2,
                   preds[:, 1],
                   preds[:, 2],
                   c='cyan',
                   alpha=1.0,
                   edgecolor='b')

        for pred_type in pred_types.values():
            ax.plot3D(preds[pred_type.slice, 0] * 1.2,
                      preds[pred_type.slice, 1],
                      preds[pred_type.slice, 2], color='blue')

        ax.view_init(elev=90., azim=90.)
        ax.set_xlim(ax.get_xlim()[::-1])


def compute_mean_distance(pred_sfd, pred_dlib):
    """
    Returns the mean distance (sqrt of the sum of the suqares of the
    differences divided by the numner of points) between the sfd prediction and
    the dlib prediction
    ----
    input:
        pred_sfd, pred_dlib: float arrays of shape (68, 2 or 3)
    ----
    output:
        mean: float
    """

    assert pred_sfd.shape == pred_dlib.shape, "The two predictions have \
different shape"
    mean = np.sqrt(((pred_sfd - pred_dlib) ** 2).sum() / len(pred_sfd))
    return mean


if "3D" in argv:
    # face_dector = sfd or dlib
    fa_sfd = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                          device='cpu', flip_input=True,
                                          face_detector="sfd")
    fa_dlib = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                           device='cpu', flip_input=True,
                                           face_detector="dlib")
    dimension = 3
else:
    fa_sfd = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                          device='cpu', flip_input=True,
                                          face_detector="sfd")
    fa_dlib = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                           device='cpu', flip_input=True,
                                           face_detector="dlib")
    dimension = 2

if "line" in argv:
    linestyle = '-'
else:
    linestyle = ' '


plot_style = dict(marker='o',
                  markersize=3,
                  linestyle=linestyle,
                  lw=2)

# Indices for each part of the face and a color for the points
pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
              'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
              'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
              'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
              'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
              'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
              'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
              'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
              'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
              }

path_image_folder = "../data/Wikihuman_project/unpolarized/png/"
folder = listdir(path_image_folder)
# Remove other files/directory
folder = sorted([filename for filename in folder if filename[-4:] == ".png"])
nb_files = len(folder)

fig_sfd = plt.figure()
fig_dlib = plt.figure()


for file_index in range(nb_files):
    filename = folder[file_index]
    path_image = path_image_folder + filename

    # Just a sample to limit the execution time and memory use
    # [,,:3] because get_landmarks expects a RGB image
    input_img = io.imread(path_image)[::4, ::4, :3]
    t1 = time()
    preds_sfd = fa_sfd.get_landmarks(input_img)[0]
    t2 = time()
    preds_dlib = fa_dlib.get_landmarks(input_img)[0]
    t3 = time()
    print("Computation time: {:.2f}s using SFD and {:.2f}s using Dlib".format(
            t2 - t1, t3 - t2))

    plot_image_and_landmarks(input_img, preds_sfd, file_index, nb_files,
                             dimension, fig_sfd, pred_types, plot_style)
    plot_image_and_landmarks(input_img, preds_dlib, file_index, nb_files,
                             dimension, fig_dlib, pred_types, plot_style)

    print("For the camera {}, mean difference of {:.2f}".format(file_index,
                             compute_mean_distance(preds_sfd, preds_dlib)))

plt.show()
