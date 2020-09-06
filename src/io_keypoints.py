"""
Functions to save and recover data about keypoints
"""

import numpy as np


def save_keypoints(pts, indices=None, path="keypoints.txt"):
    """
    Saves an the coordinates of keypoints and the vertex indices in a file as
    plain text
    The data can be recovered using recover_keypoints(path)
    ----
    input:
        pts: float array of shape (N, P) (often (68, 3)) -> the coordinates of
            the keypoints
        indices: int list of length N (often 68) -> the indices corresponding
            to the vertices. if None, no index is saved
        path: string -> where the file will be written
    ----
    output:
        None
    """
    if indices is not None:
        N = len(pts)
        assert len(indices) == N, "The number of indices does not correspond \
to the shape of the array."
    with open(path, 'w') as file:
        for pt in pts:
            str_pt = np.array(pt, dtype=str)
            line = " ".join(str_pt)
            file.writelines(line + '\n')
        if indices is not None:
            str_indices = np.array(indices, dtype=str)
            line = " ".join(str_indices)
            file.writelines(line)


def recover_keypoints(path, have_indices=False):
    """
    Recovers the coordinates of keypoints and the vertex indices from a plain
    text file
    ----
    input:
        path: string -> where the file is. Every line must have the same amount
            of elements
        have_indices: bool -> if there is a line (the last one) corresponding
            to the vertex indices
    ----
    output:
        pts: float array of shape (N, P)
        [indices]: int array of shape (N,) -> only returnes if have_indices is
            set to True
    """
    with open(path, 'r') as file:
        lines = file.readlines()
        pts_lines = lines[:-1] if have_indices else lines
        pts = [np.array(pt.split(), dtype=np.double) for pt in pts_lines]
        pts = np.array(pts, dtype=np.double)
        if have_indices:
            indices = np.array(lines[-1].split(), dtype=np.int)
            return pts, indices
        return pts
