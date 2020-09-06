"""
Applies laplacian mesh editing to deform the original mesh using the
triangulated landmarks as anchors

For the module laplacian_meshes, credit to Chris Tralie and Brooks Mershon
I just modified a few things in order to make it compatible with python3
See https://github.com/bmershon/laplacian-meshes for more info
"""

from laplacian_meshes import LaplacianMesh
import numpy as np
import laplacian_editing
from io_keypoints import recover_keypoints


if __name__ == "__main__":

    path_to_model = "../data/Wikihuman_project/obj/Corrected_Emily_2_1_closed_color.obj"
    path_to_vertex_keypoints = "data/vertex_keypoints_emily.txt"
    path_to_triangulated_keypoints = "data/triangulated_keypoints_emily.txt"

    """""""""""""""""""""""""""""""""
    "     Creating new obj file     "
    """""""""""""""""""""""""""""""""

    _, anchors_id = recover_keypoints(path_to_vertex_keypoints,
                                      have_indices=True)
    mean_landmarks_dlib = recover_keypoints(path_to_triangulated_keypoints,
                                            have_indices=False)
    """
    # Using the laplacian_meshes module
    mesh = LaplacianMesh.PolyMesh()
    mesh.loadFile(path_to_model)

    mesh = LaplacianMesh.solveLaplacianMesh(mesh, mean_landmarks_dlib,
                                            np.array(anchors_id),
                                            cotangent=True)
    mesh.saveFile("laplacian_edition.obj")
    """

    # Using laplacian8editing.py from this project
    v, f = laplacian_editing.laplacian_editing(path_to_model,
                                               mean_landmarks_dlib,
                                               anchors_id)
    laplacian_editing.save_mesh("data/meshes/test.obj", v, f)

    print("New .obj file created")
