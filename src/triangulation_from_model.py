"""
Retrieves the information about each camera
Retrieves the position of the red vertices on the mesh corresponding to
landmarks
Projects the landmarks onto each image plane
Uses dlib and sfd to find every landmark on each image
Triangulates the landmarks on each view pair in order to find an approximate 3D
position
Finds their closest vertex on the mesh and creates a new obj file where they
are represented in blue
Evaluates the best of the two methods (dlib / sfd) with these points
"""

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from skimage import io
from mpl_toolkits.mplot3d import Axes3D
import face_alignment
import cv2
import io_keypoints


def to_homogeneous(vect):
    """
    Transforms a vector into its homogeneous form
    ----
    input:
        vect: array of shape (N,)
    ----
    output:
        array of shape (N + 1,)
    """
    return np.append(vect, 1)


def from_homogeneous(vect):
    """
    Transforms a homogeneous vector into its normal form
    ----
    input:
        vect: array of shape (N,)
    ----
    output:
        array of shape (N - 1,)
    """
    assert vect[-1] != 0, "homogeneous vector with w coordinate equals to 0"
    return vect[:-1] / vect[-1]


def nose_indices():
    """
    Returns the indices of the landmarks corresponding to the nose
    ----
    input:
        void
    ----
    output:
        int list
    """
    return [3, 11, 2, 14, 18, 15, 13, 10, 9]


def lips_indices():
    """
    Returns the indices of the landmarks corresponding to the lower lips
    ----
    input:
        void
    ----
    output:
        int list
    """
    return [0, 12, 5, 1, 4, 6, 7, 8, 17, 16]


def mesh_to_detection_index(i):
    """
    Return the index of the computed landmark corresponding to the index of the
    real landmark
    ----
    input:
        i: int -> the index of the landmark in the real_landmarks_position_3d
                  list
    ----
    output:
        int ->the index of the landmark in the real_landmarks_position_2d list
    """
    d = {
         0: 67, 1: 64, 2: 29, 3: 27, 4: 55, 5: 65, 6: 56, 7: 57, 8: 58, 9: 35,
         10: 34, 11: 28, 12: 66, 13: 33, 14: 30, 15: 32, 16: 48, 17: 59, 18: 31
        }
    assert i in d, "Index out of bound, got {}, should be in [0, 18]".format(i)
    return d[i]


def detection_to_mesh_index(i):
    """
    Return the index of the real landmark corresponding to the index of the
    computed landmark
    ----
    input:
        i: int -> the index of the landmark in the real_landmarks_position_2d
                  list
    ----
    output:
        int ->the index of the landmark in the real_landmarks_position_3d list
    """
    d = {
         27: 3, 28: 11, 29: 2, 30: 14, 31: 18, 32: 15, 33: 13, 34: 10, 35: 9,
         48: 16, 55: 4, 56: 6, 57: 7, 58: 8, 59: 17, 64: 1, 65: 5, 66: 12,
         67: 0
        }
    assert i in d, "Index out of bound, got {}, should be in [27, 35], [48], \
    [55, 59] or [64, 67]".format(i)
    return d[i]


"""""""""""""""""""""""""""
"         Parsing         "
"""""""""""""""""""""""""""


def get_info_red_landmarks(path_to_model):
    """
    Parsing to get the position of the red points on a mesh
    ----
    input:
        path_to_model: string -> the relative path to the .obj file
    ----
    output:
        real_landmarks_position_3d: float array of shape (N, 3) -> the 3D
            position of each red point
    """
    real_landmarks_position_3d = []
    with open(path_to_model, 'r') as file:
        for line in file:
            if line[:2] == "v " and line[-27:-19] == "1.000000":  # Red vertex
                str_coords = line.split()[1:4]
                x, y, z = [float(val) for val in str_coords]
                real_landmarks_position_3d.append(np.array((x, y, z)))

    real_landmarks_position_3d = np.array(real_landmarks_position_3d,
                                          dtype=np.double)

    return real_landmarks_position_3d


def get_info_cameras_Emily(path_to_cameras):
    """
    Parsing to get the camera matrices for the Emily dataset
    ----
    input:
        path_to_cameras: string -> the relative path to the calibration file
    ----
    output:
        camera_matrices: float array of shape (nb_views, 3, 4) -> the camera
            matrix for each view
        K_matrices: float array of shape (nb_views, 3, 3) -> the intrinsic
            matrix for each view
        r_vectors, t_vectors: float array of shape (nb_views, 3) -> the
            rotation angles (in radian) and translation distance
        dist_vectors: float array of shape (nb_views, 4) -> the distortion
            coefficients for each view
    """
    calibration_filenames = sorted(listdir(path_to_cameras))
    camera_matrices = []
    K_matrices = []
    r_vectors = []
    t_vectors = []
    dist_vectors = []
    for filename in calibration_filenames:
        with open(path_to_cameras + filename, 'r') as file:
            L = file.readlines()

            f_line = L[1].split()  # Focal lenghts
            fx, fy = float(f_line[0]), float(f_line[1])

            res_line = L[3].split()  # Resolution
            w, _ = float(res_line[0]), float(res_line[1])

            c_line = L[5].split()  # Principal point
            cx, cy = float(c_line[0]), float(c_line[1])

            d_line = L[7].split()  # Distortion
            dist = np.array([float(val) for val in d_line], dtype=np.double)
            dist_vectors.append(dist)

            # K[I 0] matrix
            intrinsic_matrix = np.array([
                [-fx, 0.0, w-cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0]
            ], dtype=np.double)
            # We use -fx, w-cx instead of -fx, cx because the axis from imshow
            # are not the same as the ones for a normal image
            # The projected is flipped but using imshow only flips the y axis
            K_matrices.append(intrinsic_matrix[:, :3])

            # R t matrix
            transfo_matrix = np.zeros((4, 4), dtype=np.double)
            coeffs_matrix_transfo = L[-5:-1]
            for k in range(4):
                line = coeffs_matrix_transfo[k]
                str_split = line.split('\t')
                transfo_matrix[k] = [float(str_val) for str_val in str_split]
            transfo_matrix = np.linalg.inv(transfo_matrix)
            t_vectors.append(transfo_matrix[:3, 3])
            r_vectors.append(cv2.Rodrigues(transfo_matrix[:3, :3])[0])
            camera_matrices.append(intrinsic_matrix @ transfo_matrix)

    return camera_matrices, K_matrices, r_vectors, t_vectors, dist_vectors


def get_info_cameras_BEE10(path_to_cameras):
    """
    Parsing to get the camera matrices for the BEE10 dataset
    ----
    input:
        path_to_cameras: string -> the relative path to the calibration file
    ----
    output:
        camera_matrices: float array of shape (nb_views, 3, 4) -> the camera
            matrix for each view
        K_matrices: float array of shape (nb_views, 3, 3) -> the intrinsic
            matrix for each view
        r_vectors, t_vectors: float array of shape (nb_views, 3) -> the
            rotation angles (in radian) and translation distance
        dist_vectors: float array of shape (nb_views, 4) -> the distortion
            coefficients for each view
    """
    calibration_filenames = sorted(listdir(path_to_cameras))
    camera_matrices = []
    K_matrices = []
    r_vectors = []
    t_vectors = []
    dist_vectors = []
    for filename in calibration_filenames:
        with open(path_to_cameras + filename, 'r') as file:
            L = file.readlines()

            assert len(L) == 2, "Calibration file with size {} instead of 2"\
                .format(len(L))

            names = L[0].split()[1:]
            values = [float(val) for val in L[1].split()[1:]]
            d = dict(zip(names, values))

            fx, fy = d["fx"], d["fy"]  # Focal lenghts
            cx, cy = d["cx"], d["cy"]  # Principal point
            tx, ty, tz = d["tx"], d["ty"], d["tz"]
            rx, ry, rz = d["rx"], d["ry"], d["rz"]
            # Translation and Rotation

            # K[I 0] matrix
            intrinsic_matrix = np.array([
                [fx, 0.0, cx, 0.0],
                [0.0, fy, cy, 0.0],
                [0.0, 0.0, 1.0, 0.0]], dtype=np.double)
            K_matrices.append(intrinsic_matrix[:, :3])

            t_vec = np.array([tx, ty, tz], dtype=np.double)
            r_vec = np.array([rx, ry, rz], dtype=np.double)

            transfo_matrix = np.zeros((4, 4), dtype=np.double)
            transfo_matrix[3, 3] = 1
            transfo_matrix[:3, :3] = cv2.Rodrigues(np.array([rx, ry, rz]))[0]
            transfo_matrix[:3, 3] = t_vec

            t_vectors.append(t_vec)
            r_vectors.append(r_vec)
            dist_vectors.append(np.zeros(5, dtype=np.double))
            camera_matrices.append(intrinsic_matrix @ transfo_matrix)

    return camera_matrices, K_matrices, r_vectors, t_vectors, dist_vectors


def get_info_cameras(path_to_cameras, dataset="Emily"):
    """
    Parsing to get the camera matrices
    ----
    input:
        path_to_cameras: string -> the relative path to the calibration file
        dataset: string -> which type of calibration file. Shoulb be either
            "Emily" or "BEE10"
    ----
    output:
        camera_matrices: float array of shape (nb_views, 3, 4) -> the camera
            matrix for each view
        K_matrices: float array of shape (nb_views, 3, 3) -> the intrinsic
            matrix for each view
        r_vectors, t_vectors: float array of shape (nb_views, 3) -> the
            rotation angles (in radian) and translation distance
        dist_vectors: float array of shape (nb_views, 4) -> the distortion
            coefficients for each view
    """
    if dataset == "Emily":
        return get_info_cameras_Emily(path_to_cameras)
    elif dataset == "BEE10":
        return get_info_cameras_BEE10(path_to_cameras)
    else:
        raise ValueError("Dataset sould be either Emily or BEE10")


"""""""""""""""""""""""""""""""""""""""""""""
"    Computing projections and landmarks    "
"""""""""""""""""""""""""""""""""""""""""""""


def project_point(point, M=None, K_matrix=None, r_vector=None, t_vector=None,
                  dist_vector=None):
    """
    Projects a point onto the image plane and returns its 2D (non homogeneous)
    coordinates
    ----
    input:
        point: float array of shape (, 3) -> 3D point (non homogeneous)
        M: float array of shape (3, 4) -> the camera matrix
            if M is None, we use the opencv function to project
            Thus, the following parameters must not be None
        K_matrix: float array of shape (3, 3) -> intrinsic matrix
        r_vector, t_vector: float arrays of shape (1, 3) -> the rotation
            angles (in radian) and translation distances
        dist_vector: float array of shape (nb_views, 4) -> the distortion
            coefficients
    ----
    output:
        proj_point: float array of shape (1, 2) -> the projected point in 2D
            coordinates (non homogeneous)
    """
    if M is not None:
        h_point = to_homogeneous(point)
        h_proj_point = M @ h_point
        proj_point = from_homogeneous(h_proj_point)
    else:
        assert K_matrix is not None, "Expected non null K_matrices"
        assert r_vector is not None, "Expected non null r_vectors"
        assert t_vector is not None, "Expected non null t_vectors"
        proj_point = cv2.projectPoints(point, r_vector, t_vector, K_matrix,
                                       dist_vector)[0][0, 0]

    return proj_point


def project_landmarks(real_landmarks_position_3d, M=None, K_matrix=None,
                      r_vector=None, t_vector=None, dist_vector=None):
    """
    Projects every 3D landmark onto the image plane and returns their 2D (non
    homogeneous) coordinate, and 2 arrays corresponding to the nose and lips
    landmarks
    ----
    input:
        real_landmarks_position_3d: float array of shape (N, 3) -> the 3D
            points coordinates (non homogeneous)
        M: float array of shape (3, 4) -> the camera matrix
            if M is None, we use the opencv function to project
            Thus, the following parameters must not be None
        K_matrix: float array of shape (3, 3) -> intrinsic matrix
        r_vector, t_vector: float arrays of shape (1, 3) -> the rotation
            angles (in radian) and translation distances
        dist_vector: float array of shape (nb_views, 4) -> the distortion
            coefficients
    ----
    output:
        image_real_pts: float array of shape (N, 2) -> the 2D coordinates of
            the points on the image plane
        nose_real_pts, lips_real_pts: float arrays of shape (., 2) -> the 2D
            coordinates of the landmarks corresponding to the nose or the lips
    """
    nb_real_landmarks = len(real_landmarks_position_3d)
    image_real_pts = np.zeros((nb_real_landmarks, 2), dtype=np.double)

    for i in range(nb_real_landmarks):
        point = real_landmarks_position_3d[i]
        image_real_pts[i] = project_point(point, M)

    nose_real_pts = image_real_pts[nose_indices(), :]
    lips_real_pts = image_real_pts[lips_indices(), :]

    return image_real_pts, nose_real_pts, lips_real_pts


def find_2d_landmarks(image, fa, r=8):
    """
    Find the 2D coordinates (non homogeneous) of the 68 landmarks on the image
    ----
    input:
        image: int array of shape (M, N, 3) or (M, N, 4) -> the image we want
            to analyze
        fa: FaceAlignment -> the object using to find the landmarks. it
            contains the method that should be used (dblib / sfd)
        r: int -> the inverse of the sampling -> we analyze an image composed
            of 1 line and column out of r (in order to reduce the computation
            time)
    ----
    output:
        image_computed_pts: float array of shape (68, 2) -> the 2D coordinates
            of the landmarks on the image plane
        nose_landmarks, lips_landmarks: float arrays of shape (., 2) -> the 2D
            coordinates of the landmarks corresponding to the nose or the lips
    """
    image_computed_pts = r * fa.get_landmarks(image[::r, ::r, :3])[0].\
        astype(np.double)
    nose_landmarks = image_computed_pts[27:36]
    lips_landmarks = np.concatenate((np.array([image_computed_pts[48, :]]),
                                    image_computed_pts[55:60],
                                    image_computed_pts[64:68]))

    return image_computed_pts, nose_landmarks, lips_landmarks


"""""""""""""""""""""""""""""""""""""""
"     Triangulation and comparison    "
"""""""""""""""""""""""""""""""""""""""


def mean_triangulation(camera_matrices, landmarks_2d, visibility=None):
    """
    Triangulate over every pair of views and returns the mean between all these
    pairs
    ----
    input:
        camera_matrices: float array of shape (nb_views, 3, 4) -> the camera
            matrix for each view
        landmarks_2d: float array of shape (nb_views, 2, nb_pts) -> the 2D
            coordinates of every landmark
        visibility: bool array of shape (nb_pts, nb_views) -> wich views should
            be used to triangulate. If None, all views are used for every point
    ----
    output:
        mean_landmarks: float array of shape (nb_pts, 3) -> the 3D
            reconstruction of the landmarks
    """
    nb_pts = len(landmarks_2d[0][0])
    nb_views = len(camera_matrices)
    mean_landmarks = np.zeros((nb_pts, 3), dtype=np.double)

    if visibility is None:
        visibility = np.ones((nb_pts, nb_views), dtype=np.bool)

    assert all(visibility[k].sum() >= 2 for k in range(nb_pts)), "Every point \
needs at least two views for triangulation"

    for i in range(nb_views):
        for j in range(i + 1, nb_views):
            reconstructed_h = np.zeros((4, nb_pts), dtype=np.double)
            cv2.triangulatePoints(camera_matrices[i], camera_matrices[j],
                                  landmarks_2d[i], landmarks_2d[j],
                                  reconstructed_h)

            for k in range(nb_pts):
                if visibility[k][i] and visibility[k][j]:
                    landmark_h = reconstructed_h[:, k]
                    mean_landmarks[k] += from_homogeneous(landmark_h)

    for k in range(nb_pts):
        n = visibility[k].sum()
        nb_pair_views = n * (n - 1) / 2
        mean_landmarks[k] /= nb_pair_views

    return mean_landmarks


"""""""""""""""""""""""""""""""""
"     Creating new obj file     "
"""""""""""""""""""""""""""""""""


def find_closest_vertex(point, path_to_model):
    """
    Find the closest (in terms of the euclidian norm) vertex to the point in
    the obj file
    ----
    input:
        point: float array of shape (, 3) -> the 3D computed point
        path_to_model: string -> path to the obj file
    ----
    output:
        best_index: int -> line index of the closest vertex in the obj file
        best_point: float array of shape (, 3) -> the 3D coordinate of the
            closest vertex in the obj file
        best_vertex_index: int -> the index of the vertex in the obj file
        min_dist2: float -> the distance squared betwenn the point and the
            closest vertex
    """
    min_dist2 = float("inf")
    best_index = best_vertex_index = vertex_index = index = 0
    best_point = np.zeros(3, dtype=np.double)
    with open(path_to_model, 'r') as file:
        for line in file:
            if line[:2] == "v ":  # Vertex
                str_coords = line.split()[1:4]
                point2 = np.array(str_coords, dtype=np.double)
                dist2 = ((point - point2) ** 2).sum()

                if dist2 < min_dist2:
                    min_dist2 = dist2
                    best_index = index
                    best_point = point2
                    best_vertex_index = vertex_index

                vertex_index += 1
            index += 1
    return best_index, best_point, best_vertex_index, min_dist2


def find_closest_vertices(points, path_to_model):
    """
    Find the closest (in terms of the euclidian norm) vertex to the each point
    in the obj file
    ----
    input:
        points: float array of shape (N, 3) -> the 3D computed points
        path_to_model: string -> path to the obj file
    ----
    output:
        line_indices: int list of length N -> line index of the closest vertex
            in the obj file
        points_coords: float array of shape (N, 3) -> the 3D coordinate of the
            closest vertices in the obj file
        vertex_indices: int list of length N -> the indices of the vertices
            (first declared vertex in the obj file has index 0 and so on)
        dist2_list: float array of length N -> the distance squared betwenn the
            points and their closest vertex
    """
    line_indices = []
    points_coords = []
    vertex_indices = []
    dist2_list = []
    for point in points:
        best_index, best_point, best_vertex_index, min_dist2 = \
                                    find_closest_vertex(point, path_to_model)
        line_indices.append(best_index)
        points_coords.append(best_point)
        vertex_indices.append(best_vertex_index)
        dist2_list.append(min_dist2)

    return line_indices, points_coords, vertex_indices, dist2_list


def write_blue_points(line_indices, path_to_model,
                      path_to_new_mesh="data/meshes/closest_vertices.obj"):
    """
    Writing the new obj file, changing only the color of the vertices with line
    index in line_indices
    ----
    input:
        line_indices: int array of length N -> the indices of the lines of the
            old obj file that have a vertex that will become blue/magenta
            Other lines will remain the same
        path_to_model, path_to_new_mesh: strings -> path to the old and new obj
            files
    ----
    output:
        None
    """
    line_indices = sorted(line_indices)
    with open(path_to_model, 'r') as initial_file:
        init_lines = initial_file.readlines()

    i = 0
    last_change = -1  # Last vertex processed
    new_change = -1  # Next vertex to process
    with open(path_to_new_mesh, 'w') as new_file:
        while i < len(line_indices):
            last_change = new_change
            new_change = line_indices[i]
            i += 1

            new_file.writelines(init_lines[last_change + 1:new_change])
            vertex_line = init_lines[new_change]
            L = vertex_line.split()
            L[-1] = "1.000000"
            if L[-3] != "1.000000":
                # Gray vertices become blue and red ones become magenta
                L[-2] = L[-3] = "0.000000"
            corrected_line = " ".join(L) + '\n'
            new_file.writelines(corrected_line)

        new_file.writelines(init_lines[new_change + 1:])


"""""""""""""""
"     Main    "
"""""""""""""""


if __name__ == "__main__":

    """""""""""""""""""""""""""
    "         Parsing         "
    """""""""""""""""""""""""""

    path_to_model = "../data/Wikihuman_project/obj/Corrected_Emily_2_1_closed_color.obj"
    real_landmarks_position_3d = get_info_red_landmarks(path_to_model)
    nb_real_landmarks = len(real_landmarks_position_3d)

    print("Recovered the positions of the landmarks on the mesh")

    path_to_cameras = "../data/Wikihuman_project/calibration/"
    camera_matrices, K_matrices, r_vectors, t_vectors, dist_vectors = \
        get_info_cameras(path_to_cameras)

    print("Recovered the camera matrices")

    """""""""""""""""""""""""""""""""""""""""""""
    "    Computing projections and landmarks    "
    """""""""""""""""""""""""""""""""""""""""""""

    path_images = "../data/Wikihuman_project/unpolarized/png/"
    image_filenames = listdir(path_images)
    image_filenames = sorted([filename for filename in image_filenames
                             if filename[-4:] == ".png"])
    nb_views = len(image_filenames)

    fig = plt.figure(figsize=(17, 4))

    # Each real landmark on each image
    real_landmarks_position_2d = np.zeros((nb_views, 2, nb_real_landmarks),
                                          dtype=np.double)
    # Computed 2D landmarks on each image
    computed_landmarks_2d_dlib = np.zeros((nb_views, 2, 68), dtype=np.double)
    computed_landmarks_2d_sfd = np.zeros((nb_views, 2, 68), dtype=np.double)

    # Face alignment objects for the two methods
    fa_dlib = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                           device='cpu', flip_input=True,
                                           face_detector="dlib")
    fa_sfd = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                          device='cpu', flip_input=True,
                                          face_detector="sfd")

    for k in range(nb_views):
        print("Computing projections and landmarks on the camera {}".format(k))

        image_real_pts, nose_real_pts, lips_real_pts =\
            project_landmarks(real_landmarks_position_3d, camera_matrices[k])

        # The face_alignment module requires a RGB image and not RGBA
        image = io.imread(path_images + image_filenames[k])[..., :3]

        image_computed_pts_dlib, nose_landmarks_dlib, lips_landmarks_dlib = \
            find_2d_landmarks(image, fa_dlib)
        image_computed_pts_sfd, nose_landmarks_sfd, lips_landmarks_sfd = \
            find_2d_landmarks(image, fa_sfd)

        fig.add_subplot(1, nb_views, k + 1)
        plt.imshow(image)
        plt.scatter(nose_landmarks_dlib[:, 0], nose_landmarks_dlib[:, 1],
                    c=(0, 0, 1), label="Dlib")
        plt.scatter(lips_landmarks_dlib[:, 0], lips_landmarks_dlib[:, 1],
                    c=(0, 0, 1), label="Dlib")
        plt.scatter(nose_landmarks_sfd[:, 0], nose_landmarks_sfd[:, 1],
                    c=(0, 1, 0), label="Sfd")
        plt.scatter(lips_landmarks_sfd[:, 0], lips_landmarks_sfd[:, 1],
                    c=(0, 1, 0), label="Sfd")
        plt.scatter(nose_real_pts[:, 0], nose_real_pts[:, 1],
                    c=(1, 0, 0), label="Real points")
        plt.scatter(lips_real_pts[:, 0], lips_real_pts[:, 1],
                    c=(1, 0, 0), label="Real points")

        real_landmarks_position_2d[k] = image_real_pts.T
        computed_landmarks_2d_dlib[k] = image_computed_pts_dlib.T
        computed_landmarks_2d_sfd[k] = image_computed_pts_sfd.T

    plt.show()
    # View from each camera with the projected real landmarks and the computed
    # ones
    # red: the projected real landmarks, green: the computed ones using sfd
    # blue: the computed ones using dlib

    print("Projections and 2D landmarks found")

    """""""""""""""""""""""""""""""""""""""
    "     Triangulation and comparison    "
    """""""""""""""""""""""""""""""""""""""

    nb_pts = len(computed_landmarks_2d_dlib[0][0])
    visibility = np.ones((nb_pts, nb_views), dtype=np.bool)
    visibility[:5, :] = False
    visibility[12:17, :] = False
    visibility[:5, 0] = visibility[:5, 2] = True
    visibility[12:17, 3] = visibility[12:17, 5] = True

    mean_landmarks_dlib = mean_triangulation(camera_matrices,
                                             computed_landmarks_2d_dlib,
                                             visibility)
    mean_landmarks_sfd = mean_triangulation(camera_matrices,
                                            computed_landmarks_2d_sfd,
                                            visibility)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(mean_landmarks_dlib[:, 0], mean_landmarks_dlib[:, 1],
               mean_landmarks_dlib[:, 2], c=(0, 0, 1))
    ax.scatter(mean_landmarks_sfd[:, 0], mean_landmarks_sfd[:, 1],
               mean_landmarks_sfd[:, 2], c=(0, 1, 0))
    ax.scatter(real_landmarks_position_3d[:, 0],
               real_landmarks_position_3d[:, 1],
               real_landmarks_position_3d[:, 2], c=(1, 0, 0))
    plt.show()
    # 3D plot of the real landmark positions (red) and the computed ones (green
    # using sfd and blue using dlib)

    print("Computed 3D landmarks using triangulation")

    """""""""""""""""""""""""""""""""
    "     Creating new obj file     "
    """""""""""""""""""""""""""""""""

    line_indices, vertex_coords, vertex_indices, _ = find_closest_vertices(mean_landmarks_dlib,
                                                        path_to_model)

    write_blue_points(line_indices, path_to_model)

    io_keypoints.save_keypoints(vertex_coords, vertex_indices,
                                "data/vertex_keypoints_emily.txt")
    io_keypoints.save_keypoints(mean_landmarks_dlib, None,
                                "data/triangulated_keypoints_emily.txt")

    print("New .obj file created")

    """""""""""""""""""""""""""
    "   Computing closeness   "
    """""""""""""""""""""""""""

    # Square of the distances between the computed landmarks and the real ones
    dist_dlib = np.zeros((nb_real_landmarks, nb_views), dtype=np.double)
    dist_sfd = np.zeros((nb_real_landmarks, nb_views), dtype=np.double)

    for view_index in range(nb_views):
        for i in range(nb_real_landmarks):
            real_pt = real_landmarks_position_2d[view_index, :, i]
            computed_pt_dlib = computed_landmarks_2d_dlib[view_index, :,
                                                    mesh_to_detection_index(i)]
            computed_pt_sfd = computed_landmarks_2d_sfd[view_index, :,
                                                    mesh_to_detection_index(i)]
            dist_dlib[i, view_index] = np.linalg.norm(real_pt - computed_pt_dlib)
            dist_sfd[i, view_index] = np.linalg.norm(real_pt - computed_pt_sfd)
    var_table_dlib = np.mean(dist_dlib ** 2, axis=1)
    var_table_sfd = np.mean(dist_sfd ** 2, axis=1)

    print(var_table_dlib)
    print(var_table_sfd)
    print(var_table_sfd > var_table_dlib)
    print(var_table_dlib.sum())
    print(var_table_sfd.sum())

    print("Closeness computed")
