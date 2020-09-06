"""
Projecting the original mesh and the laplacian edited mesh in order to see if
it fits well
"""

import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from skimage import io
import triangulation_from_model as tfm


def projection_overlay(image, M, path_model):
    m, n, _ = image.shape
    r = 9
    with open(path_model, 'r') as file:
        for line in file:
            if line[:2] == "v ":
                coords = line.split()[1:4]
                point = np.array(coords, dtype=np.double)
                proj_point = tfm.project_point(point, M)
                y = int(round(proj_point[0]))
                x = int(round(proj_point[1]))
                # x and y are inverted because the axes are different
                if 0 <= x < m and 0 <= y < n:
                    image[x - r:x + r + 1, y - r:y + r + 1, :] = [255, 0, 0]

            elif line[:2] == "f ":
                break


if __name__ == "__main__":

    """""""""""""""""""""""""""
    "         Parsing         "
    """""""""""""""""""""""""""

    path_to_model = "../data/Wikihuman_project/obj/Corrected_Emily_2_1_closed_color.obj"
    path_to_laplacian = "../data/Wikihuman_project/obj/laplacian_edition_closed.obj"

    print("Recovered the positions of the landmarks on the mesh")

    path_to_cameras = "../data/Wikihuman_project/calibration/"
    camera_matrices, K_matrices, r_vectors, t_vectors, dist_vectors = \
        tfm.get_info_cameras(path_to_cameras, dataset="Emily")

    print("Recovered the camera matrices")

    """""""""""""""""""""""""""
    "  Computing projections  "
    """""""""""""""""""""""""""

    path_images = "../data/Wikihuman_project/unpolarized/png/"
    image_filenames = listdir(path_images)
    image_filenames = sorted([filename for filename in image_filenames
                             if filename[-4:] == ".png"])
    nb_views = len(image_filenames)

    fig = plt.figure(figsize=(17, 4))

    for k in range(nb_views):
        print("Computing projections on the camera {}".format(k))

        M = camera_matrices[k]

        # The face_alignment module requires a RGB image and not RGBA
        image = io.imread(path_images + image_filenames[k])[..., :3]
        image_original = image.copy()
        image_laplacian = image.copy()

        projection_overlay(image_original, M, path_to_model)
        projection_overlay(image_laplacian, M, path_to_laplacian)

        fig.add_subplot(2, nb_views, k + 1)
        plt.imshow(image_original)
        plt.axis("off")

        fig.add_subplot(2, nb_views, k + nb_views + 1)
        plt.imshow(image_laplacian)
        plt.axis("off")

    plt.show()

    print("Projections done")
