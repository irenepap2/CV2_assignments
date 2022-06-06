# cv2_assignment3
//////////// FILES INCLUDED ///////////////////////

- main.py                    : Run all experiments
- energy_optimization.py     : Find hyperparameters of face
- face_swapping.py           : Run the face swapping algorithm and attached functions.
- supplemental_code.py       : Contains utility functions

///////////////////////////////////////////////////

//////////// main.py ///////////////////////

Running the main function will simply execute all experiments. Additionally plots can be uncommented to show the obtained results.

//////////// face_swapping.py ///////////////////////

Contains several functions.

Use find_g() to obtain the point cloud G with given parameters. If no parameters are given these
are set at random.

get_landmarks() finds and returns the landmarks of a given point cloud G, and is able to plot them 
as well.

The pinhole() function takes a point cloud G with some rotation and transformation and projects this
onto a 2D plane with some set width, height and fov. All these parameters can be set seperately.

//////////// energy_optimization.py ///////////////////////

Run to train to predict parameters for landmark positions for a single image and compare against ground truth.


//////////// DATA/data FILES INCLUDED ///////////////////////

- ##########.jpeg file       : RGB images.
- ##########.png file        : RGB images
- ##########.anl file        : Landmark file
- ##########.obj file        : Object mesh file
