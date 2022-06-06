# cv2_assignment3
//////////// FILES INCLUDED ///////////////////////

- main.py                    : Contains code to run all of 4.2
- energy_optimization.py     : Find hyperparameters of face
- face_swapping.py           : Run the face swapping algorithm and attached functions.
- supplemental_code.py       : Contains utility functions

///////////////////////////////////////////////////

//////////// energy_optimization.py ///////////////////////

To run the energy optimization simply execute the file and the model will train.

//////////// face_swapping.py ///////////////////////

Contains several functions.

Use find_g() to obtain the point cloud G with given parameters. If no parameters are given these
are set at random.

get_landmarks() finds and returns the landmarks of a given point cloud G, and is able to plot them 
as well.

The pinhole() function takes a point cloud G with some rotation and transformation and projects this
onto a 2D plane with some set width, height and fov. All these parameters can be set seperately.

interpolate() calculates the bilinear interpolation for a point x, y to obtain its R, G or B value.

texturize() is used to find the RGB values of the newly found image mesh.

//////////// DATA/data FILES INCLUDED ///////////////////////

- ##########.jpeg file       : RGB images
- ##########.png file        : RGB images
- ##########.anl file        : Landmark file
- ##########.obj file        : Object mesh file

