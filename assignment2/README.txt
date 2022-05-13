//////////// Code FILES INCLUDED ///////////////////////

- fundamental_matrix.py      : Find the fundamental matrix, and point matching
- pointview_matrix.py        : Obtain and plot pointview matrix
- structure_from_motion.py   : Create dense blocks and stitch together images for 3D object.
- utils.py                   : Contains utility functions

///////////////////////////////////////////////////

//////////// fundamental_matrix.py ///////////////////////



//////////// pointview_matrix.py ///////////////////////

Has 2 main options. You can set improv to false if you want to obtain the base PVM, or set it to True if you want the denser
PVM, keep in mind that this has a longer execution time. To compare the results simply uncomment the test_methods() function.

//////////// structure_from_motion.py ///////////////////////

To obtain the blocks, run compute_structures(pvm) with the loaded pvm. Different PVMs can be selected 
using the np.loadtxt(FILENAME) function. compute_structures(pvm) contains 2 other options, the argument frame_step can be
given to change the frame steps from 3 to whatever number required. Aside from this the argument one_block can be set to true
to simply return the first block found. Using the function visualize(pvm.T) you can visualize the found point cloud, although
scaling on the z axis might be necessary. The factorize_and_stich() function allows for inters to be set to False, if done so,
0 padding will be used instead of using the intersections.

//////////// Data/house FILES INCLUDED ///////////////////////

- frame000000##.png file     : 2D images of the House

