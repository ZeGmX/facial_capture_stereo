"""
Doing some on-screen and off-screen rendering of the Emily mesh using pyrender
"""

import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    path = "../data/Wikihuman_project/obj/Corrected_Emily_2_1.obj"
    trimesh = trimesh.load(path)
    scene = pyrender.Scene()
    mesh = pyrender.Mesh.from_trimesh(trimesh)
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)

    light = pyrender.light.DirectionalLight(color=np.ones(3), intensity=5)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2.0, aspectRatio=1.0)
    fig = plt.figure()
    for k in range(5):
        theta = np.pi / 2 - k * np.pi / 4  # 90, 45, 0, -45, -90
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        rad = 25

        camera_pose = np.array([
            [cos_t,  0.0, sin_t, rad * sin_t],
            [0.0,    1.0, 0.0,   0.0],
            [-sin_t, 0.0, cos_t, rad * cos_t],
            [0.0,    0.0, 0.0,   1.0]
        ])

        scene = pyrender.Scene()
        scene.add(mesh, pose=np.eye(4)),
        scene.add(camera, pose=camera_pose)
        scene.add(light, pose=camera_pose)

        r = pyrender.OffscreenRenderer(400, 400)
        color, depth = r.render(scene)
        fig.add_subplot(1, 5, k + 1)
        plt.axis('off')
        plt.imshow(color)
    plt.show()
