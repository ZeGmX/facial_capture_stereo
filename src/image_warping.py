"""
Warping an image onto another using both meshes
"""

from skimage import io
from math import floor, ceil
import triangulation_from_model as tfm
import matplotlib.pyplot as plt
import numpy as np
import mesh_helper
import io_keypoints
import igl
import cv2


def naive_approach(pathes1, pathes2, r=9):
    """
    A naive approch at image warping. We justproject the vertices on both
    images and associate the corresponding pixels
    This methods only recover a small amount of pixels
    ----
    input:
        pathes1, pathes2: tuple containing 4 strings corresponding, in this
            order, to the path to the image, the keypoints .txt file, the mesh
            .obj file and the camera calibration folder
        r: int -> radius of the patch of pixels we copy when we find a match
    ----
    output:
        None
    """
    path_img1, path_kpt1, path_mesh1, path_cameras1 = pathes1
    path_img2, path_kpt2, path_mesh2, path_cameras2 = pathes2
    db1 = "BEE10" if "BEE10" in path_img1 else "Emily"
    db2 = "BEE10" if "BEE10" in path_img2 else "Emily"

    img1 = io.imread(path_img1)[..., :3]
    img2 = io.imread(path_img2)[..., :3]

    kpt1 = io_keypoints.recover_keypoints(path_kpt1, have_indices=False)
    kpt2 = io_keypoints.recover_keypoints(path_kpt2, have_indices=False)

    *_, tform_2_1 = mesh_helper.procrustes(kpt1[:27], kpt2[:27])

    v1, _ = igl.read_triangle_mesh(path_mesh1)
    v2, _ = igl.read_triangle_mesh(path_mesh2)

    cams1, *_ = tfm.get_info_cameras(path_cameras1, dataset=db1)
    cams2, *_ = tfm.get_info_cameras(path_cameras2, dataset=db2)

    cam1 = cams1[2]
    cam2 = cams2[5]

    M1, N1, _ = img1.shape
    M2, N2, _ = img2.shape
    new_img1 = np.zeros((M1, N1, 3), dtype=np.int)

    for k in range(len(v1)):
        pt1 = v1[k]
        pt2 = v2[k]

        y1, x1 = np.round(tfm.project_point(pt1, cam1)).astype(int)
        y2, x2 = np.round(tfm.project_point(pt2, cam2)).astype(int)

        if 0 <= x1 - r and x1 + r < M1 and 0 <= y1 - r and y1 + r < N1 and\
           0 <= x2 - r and x2 + r < M2 and 0 <= y2 - r and y2 + r < N2:
            new_img1[x1 - r: x1 + r + 1, y1 - r: y1 + r + 1] =\
                    img2[x2 - r: x2 + r + 1, y2 - r: y2 + r + 1]

    plt.figure()
    plt.imshow(new_img1 / new_img1.max())
    plt.axis("off")
    plt.show()


# Step 2: associating pixels and triangles


def in_bound(p, bounds):
    """
    Checks if a point is within certain bounds. Lower bounds are assumed to
    always be 0
    ----
    input:
        p: int array of length N -> the coordinates of the point
        bounds: itn array of length N -> the upper bounds for each coordinate
    ----
    output:
        boolean -> True if the point is in bound, False otherwise
    """
    n = len(p)
    assert n == len(bounds), "Both parameters have different dimensions"
    for k in range(n):
        if not (0 <= p[k] < bounds[k]):
            return False
    return True


def inside_triangle(p, triangle):
    """
    Checks if a 2D point is inside of a 2D triangle
    See https://mathworld.wolfram.com/TriangleInterior.html
    ----
    input:
        p: float array of length 2 -> the point
        triangle: float array of shape (3, 2) -> the coordinates of the 3
            vertices of the triangle
    ----
    output:
        boolean: True if the point is inside of the triangle, else otherwise
    """
    p1, p2, p3 = triangle
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x, y = p

    d = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    # Determinant of p1p3, p2p3
    assert d != 0, "This is not a triangle, found two colinear edges"

    a = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / d
    b = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / d
    c = 1 - a - b

    if in_bound([a], [1]) and in_bound([b], [1]) and in_bound([c], [1]):
        return True
    return False


def pixels_inside_triangle(triangle, bounds, r=4):
    """
    Returns all points with integer coordinate inside of a triangle
    ----
    input:
        triangle: float array of shape (3, 2) -> the coordinates of the 3
            vertices of the triangle
        bounds: int array of length 2
        r: int -> Only checks integer coordinates that are a multiple of r
    ----
    output:
        res: int array of shape (N, 2) -> every point with integer coordinates,
            in bounds and inside of the triangle
    """
    p1, p2, p3 = triangle
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    max_x = floor(max(x1, x2, x3))
    min_x = ceil(min(x1, x2, x3))
    max_y = floor(max(y1, y2, y3))
    min_y = ceil(min(y1, y2, y3))

    res = []

    for x in range(ceil(min_x / r) * r, floor(max_x / r) * r + 1, r):
        for y in range(ceil(min_y / r) * r, floor(max_y / r) * r + 1, r):
            # The bounds could be improved on y
            p = (x, y)
            if inside_triangle(p, triangle) and in_bound(p, bounds):
                res.append(p)

    return np.array(res, dtype=np.int)


def compute_association_pixels_triangles(triangles_2d, shape, r=4):
    """
    For each pixel, computes the the indices of the faces whose projection
    contains the pixel
    ----
    input:
        triangles_2d: float array of shape (n, 3, 3) -> each line is a face,
            each face is 3 3D vertices
        shape: tuple (M, N) -> the shape of the image
        r: int -> Only checks integer coordinates that are a multiple of r
    ----
    output:
        associated_triangles: int set array of shape (M, N)
    """
    M, N = shape
    associated_triangles = np.zeros((M // r + 1, N // r + 1), dtype="object")

    print("Initializing the association between pixels and faces")

    for i in range(M // r + 1):
        for j in range(N // r + 1):
            associated_triangles[i, j] = set()

    print("Computing the association between pixels and faces")

    for k in range(len(triangles_2d)):
        if k % 1000 == 0:
            print("Progress : {:.2f}%".format(k / len(triangles_2d) * 100))

        triangle = triangles_2d[k]
        pts_inside = pixels_inside_triangle(triangle, (N, M), r)
        # We use (N1, M1) instead of (M1, N1) because the axis are different
        for y, x in pts_inside:
            associated_triangles[x // r, y // r].add(k)

    return associated_triangles


def save_associated_triangles(associated_triangles,
                              path="associated_triangles.npy"):
    """
    Saves the associated_triangles array as a .npy file
    ----
    input:
        associated_triangles: int set array
        path: str -> where the file will be saved
    ----
    output:
        None
    """
    with open(path, "wb") as file:
        np.save(file, associated_triangles, allow_pickle=True)


def get_association_from_file(path):
    """
    Recovers the associated_triangles array from a .npy file
    ----
    input:
        path: str -> where the file is located
    ----
    output:
        associated_triangles: int set array
    """
    with open(path, "rb") as file:
        associated_triangles = np.load(file, allow_pickle=True)
    return associated_triangles


def save_mask(associated_triangles, path="mask.png"):
    """
    Saves a mask from the associated_triangles array
    ----
    input:
        associated_triangles: int set array
        path: str -> where the file will be written
    ----
    output:
        None
    """
    M, N = associated_triangles.shape
    mask = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            if len(associated_triangles[i, j]) > 0:
                mask[i, j] = 1
    io.imsave(path, mask)


# Step 3: intersecting point and local coordinate


def find_intersection_line_plane(triangle, l0, dir):
    """
    Finds the coordinate of the intersection point between a line and a plane
    Will raise an error if there is no intersection or if the line is contained
    in the plane
    See https://en.wikipedia.org/wiki/Line-plane_intersection
    ----
    input:
        triangle: float array of shape (3, 2) -> the coordinates of the 3
            vertices of the triangle
        l0: float array of length 3 -> a point on the line
        dir: float array of lenfth 3 -> the direction vector of the line
    ----
    output:
        res: float array of length 3 -> the coordinates of the intersection
            point
    """
    p0, p1, p2 = triangle
    n = np.cross(p1 - p0, p2 - p0)  # Normal vector
    denom = dir @ n
    assert denom != 0, "There is either no intersection point or the line is \
contained in the plane"
    d = ((p0 - l0) @ n) / denom
    res = l0 + dir * d
    return res


def find_the_right_face(v, f, tri_indices, l0, dir):
    """
    Selecting the right face (the one that is rendered camera) amongst the
    possible match
    ----
    input:
        v: float array of shape (M, 3) -> the coordinates of the vertices
        f: int array of shape (N, 3) -> the indices of the vertices forming
            each face
        tri_indices: int set -> the indices of the faces whose projection
            contains the pixel emiting the ray
        l0: float array of length 3 -> the woorld coordinates of a point on the
            ray
        dir: float array of length 3 -> a direction vector (in world
            coordinates) of the ray
    ----
    output:
        triangle_index: int -> index of the right face
        argmin: float array of length 3 -> the coordinates of the intersection
            point of the plane (formed by the triangle) and the ray
    """
    min_abs_d = float("inf")
    argmin = None
    triangle_index = None
    for index in tri_indices:
        triangle = v[f[index]]
        inter = find_intersection_line_plane(triangle, l0, dir)
        d = (inter - l0)[0] / dir[0]
        if np.abs(d) < min_abs_d:
            min_abs_d = np.abs(d)
            argmin = inter
            triangle_index = index
    return triangle_index, argmin


def better_approach(pathes1, pathes2, view_indices, r=4, path_assoc_tri=None):
    """
    A better approach to image warping using the following steps:
    1) Each face is projected on the first image
    2) We associate to each pixel the set of faces it is in
    3) For every pixel having at least one matching face, we compute the ray
        passing through it and find the intersection point to the closest
        matching face. We the compute its barycentric coordinates
    4) We find the point on the other mesh that has the same barycentric
        coordinates in the corresponding faces and project it onto its image
        plane
    5) The color of the projected point will be the one used for the initial
        pixel
    It also saves the mask of the first image as mask.png and the
    associated_triangles matrix as a .npy file
    ----
    input:
        pathes1, pathes2:tuple containing 4 strings corresponding, in this
            order, to the path to the image, the keypoints .txt file, the mesh
            .obj file and the camera calibration folder
        view_indices: two ints -> the indices of the views used
        r: int -> sampling ration on the image we want to tompute
        path_assoc_tri: str tuple of length 2 -> if the pixel-triangle
            association has already been computed for the two images, this is
            the path to the .npy files. If None, it is computed and saved
    ----
    output:
        new_img: a float array with shape image1.shape / r -> The face on
            image2 warped on image1
    """
    path_img1, path_kpt1, path_mesh1, path_cameras1 = pathes1
    path_img2, path_kpt2, path_mesh2, path_cameras2 = pathes2
    db1 = "BEE10" if "BEE10" in path_img1 else "Emily"
    db2 = "BEE10" if "BEE10" in path_img2 else "Emily"

    view_index1, view_index2 = view_indices

    img1 = io.imread(path_img1)[..., :3]
    img2 = io.imread(path_img2)[..., :3]

    M1, N1, _ = img1.shape
    M2, N2, _ = img2.shape

    cams1, _, r_vectors1, t_vectors1, _ = tfm.get_info_cameras(path_cameras1,
                                                               dataset=db1)
    cams2, _, r_vectors2, t_vectors2, _ = tfm.get_info_cameras(path_cameras2,
                                                               dataset=db2)

    cam1 = cams1[view_index1]
    cam2 = cams2[view_index2]

    rvec1, tvec1 = r_vectors1[view_index1], t_vectors1[view_index1]
    rvec2, tvec2 = r_vectors2[view_index2], t_vectors2[view_index2]

    # Step 1: projecting the faces

    v1, f1 = igl.read_triangle_mesh(path_mesh1)
    v2, f2 = igl.read_triangle_mesh(path_mesh2)

    v1_proj, *_ = tfm.project_landmarks(v1, cam1)
    v2_proj, *_ = tfm.project_landmarks(v2, cam2)

    triangles_2d1 = v1_proj[f1]
    triangles_2d2 = v2_proj[f2]

    # Step 2: associating pixels and triangles

    if path_assoc_tri is None:
        assoc_tri1 = compute_association_pixels_triangles(triangles_2d1,
                                                          (M1, N1), r)
        assoc_tri2 = compute_association_pixels_triangles(triangles_2d2,
                                                          (M2, N2), 1)

        print("Saving the file")

        path_save1 = f"assoc_{db1}_view{view_index1}_r={r}.npy"
        path_save2 = f"assoc_{db2}_view{view_index2}_r={1}.npy"
        save_associated_triangles(assoc_tri1, path_save1)
        save_associated_triangles(assoc_tri2, path_save2)
    else:
        assoc_tri1 = get_association_from_file(path_assoc_tri[0])
        assoc_tri2 = get_association_from_file(path_assoc_tri[1])

    save_mask(assoc_tri1)

    # Step 3: intersecting point and local coordinate

    # Some pre-processing

    cam_square1 = np.vstack((cam1, [0, 0, 0, 1]))
    cam_square2 = np.vstack((cam2, [0, 0, 0, 1]))
    cam_square_inv1 = np.linalg.inv(cam_square1)
    cam_square_inv2 = np.linalg.inv(cam_square2)

    R1 = cv2.Rodrigues(rvec1)[0]
    R2 = cv2.Rodrigues(rvec2)[0]
    camera_pos1 = - R1.T @ tvec1
    camera_pos2 = - R2.T @ tvec2

    new_img = np.zeros_like(img1[::r, ::r])

    print("Creating the warped image")

    for i in range(0, M1, r):
        if i % 500 == 0:
            print("Progress : {:.2f}%".format(i / M1 * 100))

        for j in range(0, N1, r):
            tri_indices = assoc_tri1[i // r, j // r]

            if len(tri_indices) > 0:
                projected_point_h = np.array([j, i, 1, 1], dtype=np.double)
                unprojected_h = cam_square_inv1 @ projected_point_h
                unprojected = tfm.from_homogeneous(unprojected_h)
                dir = unprojected - camera_pos1

                # Selecting the right face amongst the possible match

                triangle_index1, inter = find_the_right_face(v1, f1, tri_indices, camera_pos1, dir)

                # Barycentric coordinates
                # See https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates
                p0, p1, p2 = v1[f1[triangle_index1]]
                denom = np.linalg.norm(np.cross(p1 - p0, p2 - p0))
                u = np.linalg.norm(np.cross(inter - p0, inter - p2)) / denom
                v = np.linalg.norm(np.cross(inter - p0, inter - p1)) / denom

                # Step 4: projecting the corresponding point on the other
                #           view/mesh onto its image plane

                p0_2, p1_2, p2_2 = v2[f2[triangle_index1]]
                inter_2 = p0_2 + u * (p1_2 - p0_2) + v * (p2_2 - p0_2)
                proj2 = np.round(tfm.project_point(inter_2, cam2))
                y, x = proj2.astype(np.int)

                projected_point_h = np.array([y, x, 1, 1], dtype=np.double)
                unprojected_h = cam_square_inv2 @ projected_point_h
                unprojected = tfm.from_homogeneous(unprojected_h)
                dir = unprojected - camera_pos2
                tri_indices = assoc_tri2[x, y]
                triangle_index2, _ = find_the_right_face(v2, f2, tri_indices, camera_pos2, dir)

                # Step 5: coloring :)

                if triangle_index1 == triangle_index2 and in_bound(proj2, (N2, M2)):
                    new_img[i // r, j // r] = img2[x, y]

    return new_img


if __name__ == "__main__":
    # 0 0 or 2 5
    view_index1 = 0
    view_index2 = 0
    view_indices = view_index1, view_index2

    path_img1 = f"../data/Wikihuman_project/unpolarized/png/cam{view_index1}_mixed_w.png"
    path_img2 = f"../data/BEE10_ReleasePack/BEE10_ReleasePack/data/jpg/cam{view_index2}.jpg"

    path_kpt1 = "data/triangulated_keypoints_emily.txt"  # Emily
    path_kpt2 = "data/triangulated_keypoints_BEE10.txt"  # BEE10

    path_mesh1 = "../data/Wikihuman_project/obj/Corrected_Emily_2_1_closed_color.obj"
    path_mesh2 = "data/meshes/BEE10_tri.obj"
    # /!\ the tri and tet files don't have the faces in the same order
    # /!\ (and for the same face, the three indices are nt necesseraly in the
    # /!\ same order)

    path_cameras1 = "../data/Wikihuman_project/calibration/"
    path_cameras2 = "../data/BEE10_ReleasePack/BEE10_ReleasePack/cameras/"

    pathes1 = path_img1, path_kpt1, path_mesh1, path_cameras1
    pathes2 = path_img2, path_kpt2, path_mesh2, path_cameras2

    # naive_approach(pathes1, pathes2, r=9)
    new_image = better_approach(pathes2, pathes1, view_indices[::-1], 8, None)

    plt.figure()
    plt.imshow(new_image)
    plt.show()
