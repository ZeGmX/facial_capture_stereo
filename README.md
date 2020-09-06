<div style="text-align: justify">

# **Implementing a multi-view stereo method for temporally consistent facial capture**

The purpose of this project is to implement the facial capture method exposed in [1]. We would then be able to create a mesh of the entire head with details up to pores scale, from from a stereo video.

### Getting started

Get the project using one of the two following commands:
```
git clone git@github.com:ZeGmX/facial_capture_stereo.git
git clone https://github.com/ZeGmX/facial_capture_stereo.git
```

Then you need to install the modules using:
```
cd facial_capture_stereo
pip install -r requirements.txt
```

### How to use

The `src/_index.txt` file contains a small summary of what each file does. We will here give a more specific explanation of the files in the `src` folder.  
For more details on the functions defined, please refer to their documentation.

- `test_render.py`: Does an on-screen rendering of a mesh and an off-screen one with differents viewpoints. **The file does not contain any function but it can be used on a specific mesh by changing the `path` variable.**


- `modif_obj.py`: Modify the initial obj file corresponding to Emily (Wikihuman project) to get only the main part -and thus excluding the eyeballs and eyelashes. **The file does not contain any function but it can be used on a specific mesh by changing the `path` variable.**

- `landmark_detection.py`: Finds the 68 landmarks on every image in a folder using both the dlib method and the SFD face detector in order to compare them. **The file contains functions that you can use. To run it on your image folder, just
change the `path_image_folder` variable.** Part of this comes from the file `detect_landmarks_in_image.py` of the [face-alignment Github](https://github.com/1adrianb/face-alignment).

- `triangulation_from_model.py`: One of the main part of the project. It implements the whole process from getting the calibration information to finding the landmarks on each image in a folder and to get a 3D triangulation. It also compares the computed landmarks to some manually written landmarks on the mesh - represented by red vertices. **The file contains functions that you can use. To run it with your own data just change the path variables you need** (`path_to_model` for the .obj file, `path_to_cameras` for the folder containing the calibration files (you may need to create a parser if it not a format we took into account yet) and `path_images` for the folder containing the images.

- `test_laplacian_mesh_editing`: Applies laplacian mesh editing techniques to deform the original mesh to better fit the triangulated landmarks. **The file does not contain any function but it can be used by changing the three path variables**. You can either run it using the laplacian_editing module from this project or the laplacian_meshes one from [Github](https://github.com/bmershon/laplacian-meshes). You can find in src/laplacian_meshes a copy of their repository with a few changes to make the laplacian editing compatible with python 3.

- `checking_laplacian_mesh.py`: It projects two mesh -the ground truth and one produced after laplacian editing- on a set of images to compare them. **The file does not contain any function but it can be used on your data by changing the four pathes**.

- `laplacian_editing.py`: An implementation of laplacian editing using the cotangent matrix from igl. **The file contains functions that you can use**. It currently only supports .obj and .mesh files.

- `triangulation_other_database.py`: Implements the whole process, from recovering the camera matrices to triangulating and using the laplacian edition but on a database different from the Emily one. **This file does not contain any function but it can be used on your data by changing the different path variables**.

- `mesh_helper.py`: Different useful functions to work with meshes. **This file contains functions that you can use**.

- `io_keypoints.py`: Just two functions to save keypoints coordinates in a .txt file and recover them. **This file contains functions that you can use**.

- `experiments_laplacian.py`: A few experiments to see the effects of laplacian editing if we use a surfacic or volumic mesh. **This file contains functions that you can use**.

- `image_warping.py`: We warp the face of one image onto the face of another image in order to compute the optical flow afterward. The optical flow is currently computed using opencv, in the end, a deep learning method could improve the results. **This file contains functions that you can use. To run it with your own data just change the path variables you need**.

- `test_optical_flow.py`: Computing the optical flow between two images using opencv. Two different preprocessing methods are possible to improve the quality of the flow. **This file contains functions that you can use. To run it with your own data just change the path variables you need**.


### Authors

- [**Lendy Mulot**](https://github.com/ZeGmX) - *Initial work during an internship at Inria in June-July 2020, supervised by* ***Adnane Boukhayma***.


### License

This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

### Acknowledgments

-  [1] [Multi-View Stereo on Consistent Face Topology](http://www.hao-li.com/publications/papers/eg2017MVSCFT.pdf): The paper explaining the method we are implementing.

The data we used:  
- [The Wikihuman Project](https://vgl.ict.usc.edu/Data/DigitalEmily2/)  
- [Dynamic 3D Facial Action Coding System](http://www.cs.bath.ac.uk/~dpc/D3DFACS/)  
- [High-Quality Single-Shot Capture of Facial Geometry](https://cgl.ethz.ch/publications/papers/paperBee10.php)  

Thanks to all of them for allowing us to use their data.

</div>
