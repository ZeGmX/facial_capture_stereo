"""
Some experiments to see the difference laplacian editing makes if we use a
surfacic mesh or a volumic one
"""

import igl
import laplacian_editing as lp_ed
import triangulation_from_model as tfm
import numpy as np
from time import time
from io_keypoints import recover_keypoints
from mesh_helper import procrustes


# Exp 1, 1 point with different weights
def exp1(W, pt_index, mvt, pathes):
    """
    Moving one point with different weights
    ----
    input:
        W: float list of length N -> the different weights
        pt_index: int -> the index of the vertex we want to move
        mvt: float list of length 3 -> the shift we want to apply to the vertex
        pathes: 3-tuple -> path to the surfacic mesh, the volumetric .obj and
            the volunetric .mesh, in this order
    ----
    output:
        times: float array of shape (3, N) -> computation time for each mesh
            and for each weight
    """
    times = [[], [], []]

    v, f = igl.read_triangle_mesh(pathes[0])
    anchors = np.array([v[pt_index] + mvt], dtype=np.double)
    anchors_id = np.array([pt_index], dtype=np.int)

    for w in W:
        print("Experience 1, w = {}".format(w))

        t1 = time()
        v, f = lp_ed.laplacian_editing(pathes[0], anchors, anchors_id,
                                       tetra=False, custom_weight=[w])
        lp_ed.save_mesh("exp1_tri_w={}.obj".format(w), v, f)
        t2 = time()
        v, f = lp_ed.laplacian_editing(pathes[1], anchors, anchors_id,
                                       tetra=False, custom_weight=[w])
        lp_ed.save_mesh("exp1_tet_tri_w={}.obj".format(w), v, f)
        t3 = time()
        v, f = lp_ed.laplacian_editing(pathes[2], anchors, anchors_id,
                                       tetra=True, custom_weight=[w])
        lp_ed.save_mesh("exp1_tet_tet_w={}.obj".format(w), v, f)
        t4 = time()

        times[0].append(t2 - t1)
        times[1].append(t3 - t2)
        times[2].append(t4 - t3)
        dt1, dt2, dt3 = times[0][-1], times[1][-1], times[2][-1]
        print(f"tri: {dt1}s, tet_tri: {dt2}s, tet_tet: {dt3}s")

    print(times)
    print("Total time using tri:", sum(times[0]))
    print("Total time using tet_tri:", sum(times[1]))
    print("Total time using tet_tet:", sum(times[2]))

    return times


# Exp 2, 2 points with different weights
def exp2(W, pt_index, anchor_index, mvt, pathes):
    """
    Moving one point with different weights, with one other anchor
    ----
    input:
        W: float list of length N -> the different weights
        pt_index: int -> the index of the vertex we want to move
        anchor_index: int -> the index of the vertex we want not to move
        mvt: float list of length 3 -> the shift we want to apply to the vertex
        pathes: 3-tuple -> path to the surfacic mesh, the volumetric .obj and
            the volunetric .mesh, in this order
    ----
    output:
        times: float array of shape (3, N) -> computation time for each mesh
            and for each weight
    """
    times = [[], [], []]

    v, f = igl.read_triangle_mesh(pathes[0])
    anchors = np.array([v[anchor_index], v[pt_index] + mvt], dtype=np.double)
    anchors_id = np.array([anchor_index, pt_index], dtype=np.int)

    for w1 in W:
        for w2 in W:
            print("Experience 2, w1 = {}, w2 = {}".format(w1, w2))

            t1 = time()
            v, f = lp_ed.laplacian_editing(pathes[0], anchors, anchors_id,
                                           tetra=False, custom_weight=[w1, w2])
            lp_ed.save_mesh("exp2_tri_w1={}_w2={}.obj".format(w1, w2), v, f)
            t2 = time()
            v, f = lp_ed.laplacian_editing(pathes[1], anchors, anchors_id,
                                           tetra=False, custom_weight=[w1, w2])
            lp_ed.save_mesh("exp2_tet_tri_w1={}_w2={}.obj".format(w1, w2), v, f)
            t3 = time()
            v, f = lp_ed.laplacian_editing(pathes[2], anchors, anchors_id,
                                           tetra=True, custom_weight=[w1, w2])
            lp_ed.save_mesh("exp2_tet_tet_w1={}_w2={}.obj".format(w1, w2), v, f)
            t4 = time()

            times[0].append(t2 - t1)
            times[1].append(t3 - t2)
            times[2].append(t4 - t3)
            dt1, dt2, dt3 = times[0][-1], times[1][-1], times[2][-1]
            print(f"tri: {dt1}s, tet_tri: {dt2}s, tet_tet: {dt3}s")

    print(times)
    print("Total time using tri:", sum(times[0]))
    print("Total time using tet_tri:", sum(times[1]))
    print("Total time using tet_tet:", sum(times[2]))

    return times


# Exp 3, a whole neighborhood of points, no other anchors
def exp3(W, pt_index, radius, mvt, pathes):
    """
    Moving every vertex around one by the same amount
    ----
    input:
        W: float list of length N -> the different weights
        pt_index: int -> the index of the central vertex we want to move
        radius: float -> each vertex whose distance to the center is less than
            radius will have to move
        mvt: float list of length 3 -> the shift we want to apply to the vertex
        pathes: 3-tuple -> path to the surfacic mesh, the volumetric .obj and
            the volunetric .mesh, in this order
    ----
    output:
        times: float array of shape (3, N) -> computation time for each mesh
            and for each weight
    """

    times = [[], [], []]

    v, f = igl.read_triangle_mesh(pathes[0])
    center = v[pt_index]
    indices = np.array([i for i in range(len(v))
                       if np.linalg.norm(v[i] - center) < radius],
                       dtype=np.int)
    anchors = v[indices] + mvt

    for w in W:
        print("Experience 3, w = {}".format(w))

        w_list = [w] * len(anchors)

        t1 = time()
        v, f = lp_ed.laplacian_editing(pathes[0], anchors, indices,
                                       tetra=False, custom_weight=w_list)
        lp_ed.save_mesh("exp3_tri_w={}.obj".format(w), v, f)
        t2 = time()
        v, f = lp_ed.laplacian_editing(pathes[1], anchors, indices,
                                       tetra=False, custom_weight=w_list)
        lp_ed.save_mesh("exp3_tet_tri_w={}.obj".format(w), v, f)
        t3 = time()
        v, f = lp_ed.laplacian_editing(pathes[2], anchors, indices,
                                       tetra=True, custom_weight=w_list)
        lp_ed.save_mesh("exp3_tet_tet_w={}.obj".format(w), v, f)
        t4 = time()

        times[0].append(t2 - t1)
        times[1].append(t3 - t2)
        times[2].append(t4 - t3)
        dt1, dt2, dt3 = times[0][-1], times[1][-1], times[2][-1]
        print(f"tri: {dt1}s, tet_tri: {dt2}s, tet_tet: {dt3}s")

    print(times)
    print("Total time using tri:", sum(times[0]))
    print("Total time using tet_tri:", sum(times[1]))
    print("Total time using tet_tet:", sum(times[2]))

    return times


# Exp 4, a whole neighborhood of points, with an anchor
def exp4(W, pt_index, radius, anchor_index, mvt, pathes):
    """
    Moving every vertex around one by the same amount
    ----
    input:
        W: float list of length N -> the different weights
        pt_index: int -> the index of the central vertex we want to move
        radius: float -> each vertex whose distance to the center is less than
            radius will have to move
        mvt: float list of length 3 -> the shift we want to apply to the vertex
        pathes: 3-tuple -> path to the surfacic mesh, the volumetric .obj and
            the volunetric .mesh, in this order
    ----
    output:
        times: float array of shape (3, N) -> computation time for each mesh
            and for each weight
    """

    times = [[], [], []]

    v, f = igl.read_triangle_mesh(pathes[0])
    center = v[pt_index]
    indices = np.array([i for i in range(len(v))
                       if np.linalg.norm(v[i] - center) < radius]
                       + [anchor_index], dtype=np.int)
    anchors = v[indices] + mvt
    anchors[-1] = v[anchor_index]

    for w1 in W:
        for w2 in W:
            print("Experience 4, w1 = {}, w2 = {}".format(w1, w2))

            w_list = [w1] * (len(anchors) - 1) + [w2]

            t1 = time()
            v, f = lp_ed.laplacian_editing(pathes[0], anchors, indices,
                                           tetra=False, custom_weight=w_list)
            lp_ed.save_mesh("exp4_tri_w1={}_w2={}.obj".format(w1, w2), v, f)
            t2 = time()
            v, f = lp_ed.laplacian_editing(pathes[1], anchors, indices,
                                           tetra=False, custom_weight=w_list)
            lp_ed.save_mesh("exp4_tet_tri_w1={}_w2={}.obj".format(w1, w2), v, f)
            t3 = time()
            v, f = lp_ed.laplacian_editing(pathes[2], anchors, indices,
                                           tetra=True, custom_weight=w_list)
            lp_ed.save_mesh("exp4_tet_tet_w1={}_w2={}.obj".format(w1, w2), v, f)
            t4 = time()

            times[0].append(t2 - t1)
            times[1].append(t3 - t2)
            times[2].append(t4 - t3)
            dt1, dt2, dt3 = times[0][-1], times[1][-1], times[2][-1]
            print(f"tri: {dt1}s, tet_tri: {dt2}s, tet_tet: {dt3}s")

    print(times)
    print("Total time using tri:", sum(times[0]))
    print("Total time using tet_tri:", sum(times[1]))
    print("Total time using tet_tet:", sum(times[2]))

    return times


# Exp 5, close the eyes
def exp5(W, path_kpt_Emily, path_kpt_BEE10, pathes):
    """
    We try to close the eyes. To do so we use the BEE10 database since the
    person has here eyes closed
    ----
    input:
        path_kpt_Emily: str -> path to the txt file where Emily's keypoints
            are stored
        path_kpt_BEE10:  str -> path to the txt file where BEE10's keypoints
            are stored
        pathes: 3-tuple -> path to the surfacic mesh, the volumetric .obj and
            the volunetric .mesh, in this order
    ----
    output:
        times: float array of shape (3, N) -> computation time for each mesh
            and for each weight
    """
    times = [[], [], []]

    kpt_Emily = recover_keypoints(path_kpt_Emily, have_indices=False)
    kpt_BEE10 = recover_keypoints(path_kpt_BEE10, have_indices=False)
    _, BEE10_fit, _ = procrustes(kpt_Emily, kpt_BEE10)
    *_, anchors_id, _ = tfm.find_closest_vertices(kpt_Emily, pathes[0])

    anchors = BEE10_fit[36:48]  # Eyes
    anchors_id = anchors_id[36:48]

    for w in W:
        print("Experience 5, w = {}".format(w))

        w_list = [w] * len(anchors)

        t1 = time()
        v, f = lp_ed.laplacian_editing(pathes[0], anchors, anchors_id,
                                       tetra=False, custom_weight=w_list)
        lp_ed.save_mesh("exp5_tri_w={}.obj".format(w), v, f)
        t2 = time()
        v, f = lp_ed.laplacian_editing(pathes[1], anchors, anchors_id,
                                       tetra=False, custom_weight=w_list)
        lp_ed.save_mesh("exp5_tet_tri_w={}.obj".format(w), v, f)
        t3 = time()
        v, f = lp_ed.laplacian_editing(pathes[2], anchors, anchors_id,
                                       tetra=True, custom_weight=w_list)
        lp_ed.save_mesh("exp5_tet_tet_w={}.obj".format(w), v, f)
        t4 = time()

        times[0].append(t2 - t1)
        times[1].append(t3 - t2)
        times[2].append(t4 - t3)
        dt1, dt2, dt3 = times[0][-1], times[1][-1], times[2][-1]
        print(f"tri: {dt1}s, tet_tri: {dt2}s, tet_tet: {dt3}s")

    print(times)
    print("Total time using tri:", sum(times[0]))
    print("Total time using tet_tri:", sum(times[1]))
    print("Total time using tet_tet:", sum(times[2]))

    return times


if __name__ == "__main__":
    path_tri = "../data/Wikihuman_project/obj/Corrected_Emily_2_1_closed.obj"
    path_tet_tri = "tet_color.obj"
    path_tet_tet = "test_scale.mesh"
    pathes = path_tri, path_tet_tri, path_tet_tet

    path_kpt_Emily = "triangulated_keypoints_emily.txt"
    path_kpt_BEE10 = "triangulated_keypoints_BEE10.txt"

    W = [1, 2, 5]
    pt_index = 7614  # Top of the head
    mvt = [0, 1, 0]
    anchor_index = 1488  # Nose
    radius = 4

    times1 = exp1(W, pt_index, mvt, pathes)
    times2 = exp2(W, pt_index, anchor_index, mvt, pathes)
    times3 = exp3(W, pt_index, radius, mvt, pathes)
    times4 = exp4(W, pt_index, radius, anchor_index, mvt, pathes)
    times5 = exp5(W, path_kpt_Emily, path_kpt_BEE10, pathes)

    print(times1)
    print(times2)
    print(times3)
    print(times4)
    print(times5)
