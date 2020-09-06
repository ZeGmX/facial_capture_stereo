"""
Triangulation of the keypoints from the BEE10 database, transformation to fit
the Emily mesh and laplacian editing
"""

import numpy as np
import matplotlib.pyplot as plt
import triangulation_from_model as tfm
from os import listdir
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
from laplacian_meshes import LaplacianMesh
from io_keypoints import recover_keypoints, save_keypoints
from mesh_helper import procrustes, procrustes_apply
import face_alignment
import laplacian_editing
import igl


if __name__ == "__main__":

    """""""""""""""""""""""""""
    "         Parsing         "
    """""""""""""""""""""""""""

    path_to_model = "../data/Wikihuman_project/obj/Corrected_Emily_2_1_closed_color.obj"
    path_to_cameras = "../data/BEE10_ReleasePack/BEE10_ReleasePack/cameras/"

    camera_matrices, *_ = tfm.get_info_cameras(path_to_cameras, "BEE10")

    print("Recovered the camera matrices")

    """""""""""""""""""""""""""
    "   Computing landmarks   "
    """""""""""""""""""""""""""

    path_images = "../data/BEE10_ReleasePack/BEE10_ReleasePack/data/jpg/"

    image_filenames = listdir(path_images)
    image_filenames = sorted([filename for filename in image_filenames if
                                    filename[-4:] == ".jpg"])

    nb_views = len(image_filenames)

    # Computed 2D landmarks on each image
    computed_landmarks_2d = np.zeros((nb_views, 2, 68), dtype=np.double)

    # Face alignment object
    fa_dlib = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                           device='cpu', flip_input=True,
                                           face_detector="dlib")

    fa_sfd = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D,
                                          device='cpu', flip_input=True,
                                          face_detector="sfd")

    fig = plt.figure(figsize=(17, 4))
    for k in range(nb_views):
        print("Computing landmarks on the camera {}".format(k))

        # The face_alignment module requires a RGB image and not RGBA
        image = io.imread(path_images + image_filenames[k])[..., :3]

        if k == 3:
            image_computed_pts, *_ = tfm.find_2d_landmarks(image, fa_sfd, 8)
        else:
            image_computed_pts, *_ = tfm.find_2d_landmarks(image, fa_dlib, 8)
        computed_landmarks_2d[k] = image_computed_pts.T[:2, :]

        fig.add_subplot(1, nb_views, k + 1)
        plt.imshow(image)
        plt.scatter(image_computed_pts[:17, 0], image_computed_pts[:17, 1],
                    c=(0, 0, 1), label="Dlib", alpha=.4)

    plt.show()

    print("2D landmarks found")

    """""""""""""""""""""
    "   Triangulation   "
    """""""""""""""""""""

    nb_pts = len(computed_landmarks_2d[0][0])
    visibility = np.ones((nb_pts, nb_views), dtype=np.bool)
    visibility[:17, :] = False
    visibility[:5, 0] = visibility[:5, 1] = True
    visibility[12:17, 6] = visibility[12:17, 3] = True
    visibility[5:12, 1:3] = visibility[5:12, 5:7] = True

    mean_landmarks_BEE10 = tfm.mean_triangulation(camera_matrices,
                                                  computed_landmarks_2d,
                                                  visibility)
    path_kpt_Emily = "data/triangulated_keypoints_Emily.txt"
    path_kpt_BEE10 = "data/triangulated_keypoints_BEE10.txt"
    save_keypoints(mean_landmarks_BEE10, None, path_kpt_BEE10)

    mean_landmarks_Emily = recover_keypoints(path_kpt_Emily, False)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(mean_landmarks_BEE10[:, 0], mean_landmarks_BEE10[:, 1],
               mean_landmarks_BEE10[:, 2], c=(0, 0, 1))
    ax.scatter(mean_landmarks_Emily[:, 0], mean_landmarks_Emily[:, 1],
               mean_landmarks_Emily[:, 2], c=(1, 0, 0))
    plt.show()
    # In blue the triangulated keypoints from the BEE10 database
    # In red the same but for Emily

    print("Computed 3D landmarks using triangulation")

    """""""""""""""""""""""""""""""""
    "     Maping the other mesh     "
    """""""""""""""""""""""""""""""""

    # Using only the eybrows and the edges of the face to fit
    dist, Emily_fit_sample, tform = procrustes(mean_landmarks_BEE10[:27, :],
                                               mean_landmarks_Emily[:27, :])

    Emily_fit = procrustes_apply(mean_landmarks_Emily, tform)

    fit_path = "fit.obj"
    v, f = igl.read_triangle_mesh(path_to_model)
    new_v = procrustes_apply(v, tform, scaling=True)
    igl.write_obj(fit_path, new_v, f)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(Emily_fit[:, 0], Emily_fit[:, 1], Emily_fit[:, 2], c=(0, 0, 1))
    ax.scatter(mean_landmarks_BEE10[:, 0], mean_landmarks_BEE10[:, 1],
               mean_landmarks_BEE10[:, 2], c=(1, 0, 0))
    plt.show()
    # In red the triangulated keypoints from the BEE10 database
    # In blue the triangulated keypoints from the Emily database transformed to
    # best fit the red points

    print("Transformed the Emily mesh to best fit the triangulated keypoints")

    """""""""""""""""""""""""""""""""
    "     Creating new obj file     "
    """""""""""""""""""""""""""""""""

    line_indices, _, anchors_id, _ = \
        tfm.find_closest_vertices(Emily_fit, fit_path)
    # Should be the same with mean_landmarks_Emily, path_to_model

    """
    # Using the laplacian_meshes module
    mesh = LaplacianMesh.PolyMesh()
    mesh.loadFile(path_to_model)

    mesh = LaplacianMesh.solveLaplacianMesh(mesh, BEE10_fit,
                                            np.array(anchors_id),
                                            cotangent=True)
    mesh.saveFile("laplacian_edition.obj")
    """

    # Using laplacian8editing.py from this project
    v, f = laplacian_editing.laplacian_editing("fit.obj",
                                               mean_landmarks_BEE10,
                                               anchors_id, tetra=False)
    laplacian_editing.save_mesh("test2.obj", v, f)

    print("New .obj file created")
