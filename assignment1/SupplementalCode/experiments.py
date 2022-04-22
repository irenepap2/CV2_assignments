from icp import *
import time
import numpy as np

DATA_DIR = './textfiles/'


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


def write_results(source, target, sample, f, g, ratio=None, noise=False, noise_max=0):
    '''
    Obtain average ICP results and write them to file.
    '''
    results = []
    times = []

    for i in range(25):
        time1 = time.time()

        if not ratio:
            _, _, rms = icp(source, target, sampling=sample, epsilon=1e-8, max_iters=100, ratio=0.01, mode='kd_tree', print_rms=False, noise=noise, noise_max=noise_max)
        else:
            _, _, rms = icp(source, target, sampling=sample, epsilon=1e-8, max_iters=100, ratio=ratio, mode='kd_tree', print_rms=False)

        time2 = time.time()
        results.append(rms)
        times.append(time2-time1)

    f.write('Average time: ' + str(np.mean(times)) + '\n')
    g.write('Total times: ' + str(times) + '\n')
    averages, mean_l = obtain_avgs(results)
    f.write('Average length: ' + str(mean_l) + '\n')
    f.write('Average rms: ' + str(averages) + '\n')
    g.write('Total rms: ' + str(results) + '\n')

    return


def obtain_results(source, target, samplings, noise=False, mode='kd_tree'):
    '''
    Obtain results and write them to file
    '''
    for sampling in samplings:
        file = str(sampling) + '.txt'

        if noise:
            file = "noise" + file

        if z_buff:
            file = "z_buff_" + file

        with open(DATA_DIR + file, 'w') as f:
            with open(DATA_DIR + 'raw' + file, 'w') as g:

                # obtain noised results
                if noise:
                    noise_max = [0.1, 0.25, 0.5, 0.75]

                    for noise_v in noise_max:
                        f.write("Noise: " + str(noise_v) +'\n')
                        g.write("Noise: " + str(noise_v) +'\n')
                        write_results(source, target, sampling, f, g, noise, noise_v)
                        f.write('\n')
                        g.write('\n')

                # obtain results of non random/uniform subsampling
                elif sampling != 'uniform' and sampling != 'random':
                    write_results(source, target, sampling, f, g)

                # obtain subsampling results with different ratios
                else:
                    ratios = [0.01, 0.1, 0.25, 0.5, 0.75]
                    for ratio in ratios:
                        f.write("Ratio: " + str(ratio) + '\n')
                        g.write("Ratio: " + str(ratio) + '\n')
                        write_results(source, target, sampling, f, g, ratio)
                        f.write('\n')
                        g.write('\n')

                g.close()

            f.close()


if __name__ == "__main__":
    '''
    Use this to generate the data.
    WARNING: Data generation will take close to 12 hours in total.
    If you want to reduce this dont run the noised versions (especially of
    random sampling).
    '''
    # source, target, _ = open_wave_data()
    source, target, _ = open_bunny_data()
    # samplings = ['uniform', 'random', 'multi_res', 'info_reg', 'none']
    # obtain_results(source, target, samplings, noise=False)
    # obtain_results(source, target, samplings, noise=True)
