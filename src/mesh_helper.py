"""
Helper functions for meshes
"""

import sys
import math
import numpy as np
import numpy.matlib as matlib
import scipy
from scipy import sparse


# ----------------
def normalize_row(X):
    """
    Inputs:
      X: a numpy array

    Outputs:
      X_normalized: row normalized numpy array

    From https://github.com/HTDerekLiu/Paparazzi
    """
    l2Norm = np.sqrt((X * X).sum(axis=1))
    X_normalized = X / (l2Norm.reshape(X.shape[0], 1) + 1e-7)
    return X_normalized


# ----------------
def face_normals(V, F):
    """
    Compute face normals

    Inputs:
      V: n-by-3 numpy ndarray of vertex positions
      F: m-by-3 numpy ndarray of face indices

    Outputs:
      face normals: m-by-3 numpy ndarray

    From https://github.com/HTDerekLiu/Paparazzi
    """

    vec1 = V[F[:, 1], :] - V[F[:, 0], :]
    vec2 = V[F[:, 2], :] - V[F[:, 0], :]
    FN = np.cross(vec1, vec2) / 2
    l2Norm = np.sqrt((FN * FN).sum(axis=1))
    FN_normalized = FN / (l2Norm.reshape(FN.shape[0], 1) + 1e-15)
    return FN_normalized


# ----------------
def vertex_normals(V, F):
    """
    Compute vertex normal

    Inputs:
      V: n-by-3 numpy ndarray of vertex positions
      F: m-by-3 numpy ndarray of face indices

    Outputs:
      vertex normals: n-by-3 numpy ndarray

    From https://github.com/HTDerekLiu/Paparazzi
    """
    vec1 = V[F[:, 1], :] - V[F[:, 0], :]
    vec2 = V[F[:, 2], :] - V[F[:, 0], :]
    FN = np.cross(vec1, vec2) / 2
    faceArea = np.sqrt(np.power(FN, 2).sum(axis=1))
    FN_normalized = normalize_row(FN+sys.float_info.epsilon)

    VN = np.zeros(V.shape)
    rowIdx = F.reshape(F.shape[0]*F.shape[1])
    colIdx = matlib.repmat(np.expand_dims(np.arange(F.shape[0]),axis=1),1,3).reshape(F.shape[0]*F.shape[1])
    weightData = matlib.repmat(np.expand_dims(faceArea,axis=1),1,3).reshape(F.shape[0]*F.shape[1])
    W = scipy.sparse.csr_matrix((weightData, (rowIdx, colIdx)), shape=(V.shape[0],F.shape[0]))
    vertNormal = W*FN_normalized
    vertNormal = normalize_row(vertNormal)
    return vertNormal


# ----------------
def rotation_matrix(angle, direction, point=None):
    """
    Return matrix to rotate about axis defined by point and
    direction.

    Inputs:
      angle     : float
                  Angle, in radians
      direction : (3,) float
                  Unit vector along rotation axis
      point     : (3, ) float, or None
                  Origin point of rotation axis

    Outputs:
      matrix : (4, 4) float
               Homogenous transformation matrix

    From https://github.com/mikedh/trimesh/blob/master/trimesh/transformations.py
    """

    #direction = unit_vector(direction[:3])
    direction = direction / np.linalg.norm(direction)

    # rotation matrix around unit vector
    sina = math.sin(angle)
    cosa = math.cos(angle)
    M = np.diag([cosa, cosa, cosa, 1.0])
    M[:3, :3] += np.outer(direction, direction) * (1.0 - cosa)

    direction = direction * sina
    M[:3, :3] += np.array([[0.0, -direction[2], direction[1]],
                           [direction[2], 0.0, -direction[0]],
                           [-direction[1], direction[0], 0.0]])

    # if point is specified, rotation is not around origin
    if point is not None:
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(M[:3, :3], point)

    return M


# ----------------
def translation_matrix(t):
    res = np.eye(4)
    res[:3, 3] = t
    return res


# ----------------
def transform_mesh(V, R=None, s=None, center=False):
    """
    Transform mesh coordinates by:
        V' = center ( sR*V )

    Input:
      V       :  Numpy array of size nv x 3.
                 Vertex coordinates on each row.
      R       :  Array of size 3 with euler angles, or 9 with rotation matrix.
                 Rotation angle in degrees, for x, y and z axis.
      s       :  Scalar.
                 Scaling factor
      center  :  Bool.
                 Set to True to move center of mass to origin.

    Output:
      V       :  numpy array of size nv x 3, with transformed coordinates
      T       :  transformation matrix
    """


    T = np.eye(4)

    # 1. center
    ctr = np.mean(V,0)
    T = translation_matrix(-ctr)
    T[:3, 3] = -ctr

    # 2. rotate
    if R is not None:
        #~ if isinstance(R, np.ndarray):
        if isinstance(R, list):
            if len(R) == 3:
                # convert angles to radians
                alpha = R[0] * math.pi / 180.0
                beta = R[1] * math.pi / 180.0
                gamma = R[2] * math.pi / 180.0
                # convert to rotation matrix
                Rx = rotation_matrix(alpha, [1,0,0])
                Ry = rotation_matrix(beta, [0,1,0])
                Rz = rotation_matrix(gamma, [0,0,1])

                R = np.matmul(Rx, Ry)
                R = np.matmul(R, Rz)
                R = R[:3,:3]
            else:
                R = np.array(R)
                R = np.reshape(R, (3,3))

        T[:3,:3] = R

    # 3. scale
    if s is not None:
        T[:3,:3] *= s

    # 4. un-do centering
    tmp = translation_matrix(ctr)
    T = np.matmul(tmp, T)

    # 5. apply
    V = np.matmul(T[:3, :3], V.T)
    V = V.T
    V += T[:3, 3]

    # 6. center
    if center:
        V -= V.mean(0)

    return V, T


# ----------------
#~ def procrustes(X, Y, scaling=True, reflection='best'):
def procrustes(X, Y, scaling=True, reflection=False):
    """
    https://stackoverflow.com/questions/18925181/procrustes-analysis-with-numpy
    A port of MATLAB's `procrustes` function to Numpy.

    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.

        d, Z, [tform] = procrustes(X, Y)

    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.

    scaling
        if False, the scaling component of the transformation is forced
        to 1

    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.

    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()

    Z
        the matrix of transformed Y-values

    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
        (keys: 'rotation', 'scale', translation')

    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centered Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection is not 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    # tform = {'rotation':T, 'scale':b, 'translation':c}
    tform = {
        'rotation':T,
        'scale':b,
        'translation':c,
        'muX': muX,
        'muY': muY,
        'normY': normY,
        'traceTA': traceTA
    }

    return d, Z, tform


# ----------------
def procrustes_apply(V, T, scaling=True):
    """ TODO this applies according to prev function... """

    V0 = V - T['muY']
    V0 /= T['normY']

    if scaling:
        # b = traceTA * normX / normY
        # Z = normX*traceTA*np.dot(Y0, T) + muX
        out = T['normY'] * T['scale'] * np.dot(V0, T['rotation']) + T['muX']
    else:
        out = T['normY']*np.dot(V0, T['rotation']) + T['muX']

    return out


# ----------------
def normalize_to_unit_diag(V, inplace=False):
    """
    Normalize vertices so that the center of mass is in the origin,
    and the bounding box diagonal is of size 1.

    The transformation is performed in-place, unless inplace is set to False.

    Input:
        V  :  2D array of size nv x 3 with vertex coordinates

    Output:
        tr  :  Applied translation (1D array of size 3)
        s   :  Applied scale (scalar)
    """

    tr = -np.mean(V,0)

    #m = V.min(0)        # [ min_x, min_y, min_z ]
    #M = V.max(0)        # [ max_x, max_y, max_z ]
    xyz_range = V.max(0) - V.min(0)
    s  = 1.0 / np.sqrt(np.sum(xyz_range ** 2))

    # apply
    if inplace:
        V += tr
        V *= s

    return tr, s


# ----------------
def remove_unreferenced_vertices(V, F):
    """
    Remove un-referenced vertices from V.

    Input:
        V  :  2D numpy array of size nv x 3, with vertices on each row
        F  :  2D numpy long array of size nf x 3, with triangle indices on each row
    """

    # make set of all vertices that are referenced by some face
    ref_vs = set()
    for i in range(F.shape[0]):
        ref_vs.add(F[i,0])
        ref_vs.add(F[i,1])
        ref_vs.add(F[i,2])

    if len(ref_vs) == V.shape[0]:
        return V,F

    # remove un-referenced and save in output matrix
    old2new = {}
    nv = len(ref_vs)
    outV = np.zeros((nv, 3), dtype=V.dtype)
    j = 0

    for i in range(V.shape[0]):
        if i in ref_vs:
            outV[j] = V[i]
            old2new[i] = j
            j += 1

    # re-do face matrix with new vertex indices
    outF = np.zeros_like(F)

    for i in range(F.shape[0]):
        for c in range(3):
            outF[i,c] = old2new[ F[i,c] ]

    return outV, outF


# ----------------
def get_directed_edges(F):
    """
    Get matrix of directed edges from a list of triangular faces.
    The direction of each edge goes from the lowest to the highest vertex index.

    Input:
        F  :  2D array of size nf x 3, with vertex indices for each triangle face.

    Output:
        E  :  2D array of size ne x 2, with vertex indices on each row,
              and where E[i, 0] <= E[i,1].
    """

    edges = set()

    for f in range(F.shape[0]):

        # get directed edges
        v1,v2,v3 = F[f, :]

        if v1 <= v2:
            e1_start = v1
            e1_end = v2
        else:
            e1_start = v2
            e1_end = v1

        if v2 <= v3:
            e2_start = v2
            e2_end = v3
        else:
            e2_start = v3
            e2_end = v2

        if v3 <= v1:
            e3_start = v3
            e3_end = v1
        else:
            e3_start = v1
            e3_end = v3

        edges.add((e1_start, e1_end))
        edges.add((e2_start, e2_end))
        edges.add((e3_start, e3_end))

    # convert to numpy array
    ne = len(edges)
    E = np.zeros((ne, 2), dtype=np.int32)
    i = 0

    for edge in edges:
        E[i,0] = edge[0]
        E[i,1] = edge[1]
        i += 1

    assert i == ne, "Something went wrong"
    return E


# ----------------
def build_node_arc_matrix(V, E, G=None):
    """
    *** NOT TESTED!! **

    Build node-arc incidence matrix M, needed for the stifness term.
    The matrix contains
        - one row for each edge (arc)
        - one column for each vertex (node)
        - edges are directed from lowest to highest index
        - if an edge r connects vertices (i,j), with i <= j, then
          M[r,i] = -1 and M[r,j] = 1

    If weighting matrix G = (1,1,1,\lambda) is given, then instead of M
    it builds M \kron G, with \kron the Kroncecker product.
    Note: G is given as \lambda

    Input:
        V  :  2D array of size nv x 3, with vertex coordinates
        E  :  2D array of size ne x 2, with vertex indices on each row,
              and where E[i, 0] <= E[i,1]
              See get_directed_edges(F)
        G  :  Optional: weight for translation part of transformation matrix

    Output:
        M  :  2D array of size ne x nv
              Node-arc incidence matrix
    """

    # get sparse matrix entries
    rows = []
    cols = []
    data = []

    for r in range(E.shape[0]):

        i,j = E[r,0], E[r,1]

        # M[r,i]
        rows.append(r)
        cols.append(i)
        data.append(-1)

        # M[r,j]
        rows.append(r)
        cols.append(j)
        data.append(1)

    # make sparse matrix
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)

    M = sparse.coo_matrix((data, (rows, cols)))
    return M.tocsr()


# ----------------
# https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
def R_from_vecs(vec1, vec2, normalized=False):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """

    a, b = vec1, vec2

    if not normalized:
        a = a/np.linalg.norm(a)
        b = a/np.linalg.norm(b)

    #~ a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    a, b = a.reshape(3), b.reshape(3)

    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return R

# ----------------
def face_front(scanV, refV):

    ctr = scanV.mean(0)
    scanV -= ctr
    _ref = refV-refV.mean(0)

    # 1. scale
    scan_s = scanV.max()
    ref_s = _ref.max()

    scan_dirs = scanV / scan_s
    ref_dirs  = refV / ref_s

    # convert scan and reference into sphere of angles
    scan_norms = np.linalg.norm(scan_dirs, 2, 1)
    ref_norms = np.linalg.norm(ref_dirs, 2, 1)

    scan_dirs = scan_dirs / scan_norms[:, None]
    ref_dirs = ref_dirs / ref_norms[:, None]

    # find mean directions
    def _avg_dir(d):

        avg_dir = d.mean(0)
        avg_dir /= np.linalg.norm(avg_dir)
        min_ind = np.square(d-avg_dir).sum(1).argmin()

        return avg_dir, min_ind

    scan_avg_dir, scan_avg_ind = _avg_dir(scan_dirs)
    ref_avg_dir, ref_avg_ind = _avg_dir(ref_dirs)

    # use actual vertex instead (?)
    scan_avg_dir = scan_dirs[scan_avg_ind]
    ref_avg_dir = ref_dirs[ref_avg_ind]

    # find rotation between both directions
    R = R_from_vecs(scan_avg_dir, ref_avg_dir, normalized=True)

    # rotate scan vertices
    scanV = scanV @ R.T         # (they are row vectors..)
    scanV += ctr

    return scanV, R


# ----------------
def crop(V, F, center, radius):
    # TODO batch!
    # center mesh
    V -= center

    print ("///", center)

    # keep vertices that fall withing radius
    inds = []
    old2new = {}

    for i in range(V.shape[0]):
        dist = np.linalg.norm(V[i])
        if dist <= radius:
            old2new[i] = len(inds)
            inds.append(i)

    inds_ = np.array(inds)
    inds = set(inds)
    newV = V[inds_,:]

    # make output face matrix
    # this requires both removing faces with un-references vertices,
    # and changing vertex indices

    newF = []

    for i in range(F.shape[0]):
        if F[i,0] in inds and F[i,1] in inds and F[i,2] in inds:
            newF.append([
                old2new[ F[i,0] ],
                old2new[ F[i,1] ],
                old2new[ F[i,2] ]
            ])

    newF = np.array(newF, dtype=F.dtype)

    # put mesh back in place
    #~ newV += center
    return newV, newF
