//////////// DATA/data FILES INCLUDED ///////////////////////

- ##########.jpeg file       : RGB images recorded
- ##########.pcd file        : point clouds recorded
- ##########_camera.xml file : camera parameters
- ##########_depth.png  file : depth images recorded
- ##########_normal.pcd file : normals extracted
- ##########_mask.jpeg  file : object masks

//////////// SupplementalCode FILES INCLUDED ///////////////////////

- experiments.py             : Run the subsampling experiments
- global_registration.py     : Run the Point Cloud stitching
- icp.py                     : Contains the ICP algorithm
- experiments.py             : Run the subsampling experiments
- process_results.py         : Create plots of obtained data from experiments.py
- utils.py                   : Contains utility functions

///////////////////////////////////////////////////

To test the subsampling, you can run the experiments.py. In the main section of this file you can use the subsampling variable to set which methods you would like to be tested

//////////// Methods ///////////////////////

- uniform                     : uniform subsampling
- random                      : random subsampling
- multi_res                   : multi-resolution subsampling
- info_reg                    : convex-hull subsampling
- none                        : kd-tree subsampling

///////////////////////////////////////////////////

To obtain the results either uncomment obtain_results(source, target, samplings, noise=False) for the non noised versions or obtain_results(source, target, samplings, noise=True) for the noised versions. Keep in mind that running this will take significant time, especially for random subsampling on the noised tests, so it might be better to only test half of the methods on each version. These will then be saved to /SupplementalCode/textfiles with their mean data and raw data seperately.

In order to obtain the graphs, simply run process_results.py to save the graphs to /SupplementalCode/figures/plots for each tested method. If you want to simply view these plots but not save them you can set save_fig to False, and the results will be plotted and shown directly instead.
