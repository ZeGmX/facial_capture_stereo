"""
Implementing laplacian mesh editing using igl's cotangent matrix
"""

from scipy import sparse
from scipy.sparse.linalg import lsqr
import numpy as np
import igl


def laplacian_editing(path, anchors, anchors_id, tetra=False,
                      custom_weight=None):
    """
    Applies laplacian mesh editing to a mesh using igl's cotangent weights
    ----
    input:
        path: str -> path to the triangle mesh
        anchors: float array of shape (n, 3) -> position of the anchors
        anchors_id: float array of shape (n,) -> index of each anchor vertex
        tetra: bool -> if the mesh is tetraedral. If False, it should be with
            triangular faces
        custom_weight: float list of length n. If None, every weight is
            considered to be 1
    ----
    output:
        v_res: float array of shape (N, 3) -> the new vertices
        f: float array of shape (P, 3) -> the faces (same as before)
    """
    nb_anchors = len(anchors)
    assert nb_anchors == len(anchors_id), "anchors and anchors_id have \
different size"

    extension = path.split('.')[-1]

    weight_anchors = 1.0

    if extension == "mesh":
        v, vo, f = igl.read_mesh(path)
        f = f - 1
        # When using read_mesh, vertex indices start at 1 instead of 0
        # We could use read_triangle_mesh which returns vertices with indices
        # starting at 0
    elif extension == "obj":
        v, f = igl.read_triangle_mesh(path)
    else:
        raise ValueError("Currently, only .obj and .mesh files are supported")

    if tetra:
        assert extension == "mesh", "Laplacian editing for tetraedral mesh is\
 currently only supported for .mesh files"
        L = igl.cotmatrix(v, vo)
    else:
        L = igl.cotmatrix(v, f)

    M, N = L._shape  # M should be equal to N

    delta = np.array(L.dot(v))

    # The lines that will be appened to L and delta
    anchors_L = sparse.csc_matrix((nb_anchors, N))
    anchors_delta = np.zeros((nb_anchors, 3), dtype=np.double)
    for k in range(nb_anchors):
        if custom_weight is None:
            anchors_L[k, anchors_id[k]] = weight_anchors
            anchors_delta[k] = weight_anchors * anchors[k]
        else:
            anchors_L[k, anchors_id[k]] = custom_weight[k]
            anchors_delta[k] = custom_weight[k] * anchors[k]

    L = sparse.vstack((L, anchors_L), format="coo")
    delta = np.vstack((delta, anchors_delta))

    # Solving for least squares
    v_res = np.zeros((M, 3), dtype=np.double)
    for k in range(3):
        v_res[:, k] = lsqr(L, delta[:, k], x0=v[:, k])[0]

    return v_res, f


def save_mesh(path, v, f):
    """
    Saves a mesh as a obj file with the same color on each vertex
    ----
    input:
        path: str -> the path to the location where the file will be written
        v: float array of shape (N, 3) -> the vertices position
        f: float array of shape (P, 3) -> the 3 verticies index for each
            triangle face
    ----
    output:
        None
    """
    if f.min() == 0:
        newf = f + 1
    else:
        newf = f.copy()

    color = [str(0.752941)] * 3

    with open(path, 'w') as file:
        for pt in v:
            L = ['v'] + list(pt.astype(str)) + color
            file.writelines(" ".join(L) + '\n')
        file.writelines("#{} vertices\n\n".format(len(v)))

        for face in newf:
            L = ['f'] + list(face.astype(str))
            file.writelines(" ".join(L) + '\n')
        file.writelines("#{} faces\n".format(len(f)))
