from icp import icp
from utils import *
import time
import numpy as np

DATA_DIR = './Data/textfiles/'

def obtain_avgs(array_set):
    '''
    Obtain avg rms of each index
    '''
    lengths = [len(arr) for arr in array_set]
    averages = []

    for i in range(max(lengths)):
        cur_avg = []

        for j in range(len(lengths)):
            if lengths[j] > i:
                cur_avg.append(array_set[j][i])

        averages.append(np.mean(cur_avg))

    return averages, np.mean(lengths)


def icp_results(source, target, sample, ratio=None, noise=False, noise_max=0):
    '''
    Obtain average ICP results.
    '''
    results = []
    times = []

    for i in range(1):
        time1 = time.time()

        if not ratio:
            _, _, rms = icp(source, target, sampling=sample, epsilon=1e-8, max_iters=100, ratio=0.01, mode='kd_tree', print_rms=False, noise=noise, noise_max=noise_max)
        else:
            _, _, rms = icp(source, target, sampling=sample, epsilon=1e-8, max_iters=100, ratio=ratio, mode='kd_tree', print_rms=False)

        time2 = time.time()
        results.append(rms)
        times.append(time2-time1)

    averages, mean_l = obtain_avgs(results)

    return results, times, np.mean(times), averages, mean_l


def obtain_results(source, target, samplings, noise=False, z_buff=False):
    '''
    Obtain results and write them to file
    '''
    for sampling in samplings:
        total_res = []
        total_times = []
        total_mean_times = []
        total_avgs = []
        total_mean_l = []
        print(sampling)
        file = str(sampling) + '.txt'

        if noise:
            file = "noise" + file

        if z_buff:
            file = "z_buff_" + file

        # obtain noised results
        if noise:
            noise_max = [0.1, 0.25, 0.5, 0.75]

            for noise_v in noise_max:
                results, times, mean_times, averages, mean_l = icp_results(source, target, sampling, noise=noise, noise_max=noise_v)
                total_res.append(results)
                total_times.append(times)
                total_mean_times.append(mean_times)
                total_avgs.append(averages)
                total_mean_l.append(mean_l)

        # obtain results of non random/uniform subsampling
        elif sampling != 'uniform' and sampling != 'random':
            results, times, mean_times, averages, mean_l = icp_results(source, target, sampling)
            total_res.append(results)
            total_times.append(times)
            total_mean_times.append(mean_times)
            total_avgs.append(averages)
            total_mean_l.append(mean_l)

        # obtain subsampling results with different ratios
        else:
            ratios = [0.01, 0.1, 0.25, 0.5, 0.75]
            for ratio in ratios:
                results, times, mean_times, averages, mean_l = icp_results(source, target, sampling, ratio)
                total_res.append(results)
                total_times.append(times)
                total_mean_times.append(mean_times)
                total_avgs.append(averages)
                total_mean_l.append(mean_l)
        result_l = [total_res, total_times, total_mean_times, total_avgs, total_mean_l]

        write_files(file, sampling, result_l, noise)


def write_files(file, sampling, result_l, noise):
    '''
    Write results to file.
    '''
    with open(DATA_DIR + file, 'w') as f:
        with open(DATA_DIR + 'raw' + file, 'w') as g:

            # obtain noised results
            if noise:
                noise_max = [0.1, 0.25, 0.5, 0.75]

                for i, noise_v in enumerate(noise_max):
                    f.write("Noise: " + str(noise_v) +'\n')
                    g.write("Noise: " + str(noise_v) +'\n')
                    f.write('Average time: ' + str(result_l[2][i]) + '\n')
                    g.write('Total times: ' + str(result_l[1][i]) + '\n')
                    f.write('Average length: ' + str(result_l[4][i]) + '\n')
                    f.write('Average rms: ' + str(result_l[3][i]) + '\n')
                    g.write('Total rms: ' + str(result_l[0][i]) + '\n')
                    f.write('\n')
                    g.write('\n')

            # obtain results of non random/uniform subsampling
            elif sampling != 'uniform' and sampling != 'random':
                f.write('Average time: ' + str(result_l[2][0]) + '\n')
                g.write('Total times: ' + str(result_l[1][0]) + '\n')
                f.write('Average length: ' + str(result_l[4][0]) + '\n')
                f.write('Average rms: ' + str(result_l[3][0]) + '\n')
                g.write('Total rms: ' + str(result_l[0][0]) + '\n')

            # obtain subsampling results with different ratios
            else:
                ratios = [0.01, 0.1, 0.25, 0.5, 0.75]
                for i, ratio in enumerate(ratios):
                    f.write("Ratio: " + str(ratio) + '\n')
                    g.write("Ratio: " + str(ratio) + '\n')
                    f.write('Average time: ' + str(result_l[2][i]) + '\n')
                    g.write('Total times: ' + str(result_l[1][i]) + '\n')
                    f.write('Average length: ' + str(result_l[4][i]) + '\n')
                    f.write('Average rms: ' + str(result_l[3][i]) + '\n')
                    g.write('Total rms: ' + str(result_l[0][i]) + '\n')
                    f.write('\n')
                    g.write('\n')

            g.close()

        f.close()


if __name__ == "__main__":
    '''
    Use this to generate the data.
    WARNING: Data generation may take a while.
    Uncomment any of the 2 methods to obtain the results and save to text file.
    '''
    # source, target = open_wave_data()
    source, target = open_bunny_data()
    samplings = ['uniform', 'random', 'multi_res', 'info_reg', 'none']
    # obtain_results(source, target, samplings, noise=False)
    # obtain_results(source, target, samplings, noise=True)
