import matplotlib.pyplot as plt

DATA_DIR = './textfiles/'


def plot_results(sampling, save_fig, idxs, vals, noise):
    '''
    Generate results and either plot them in console or save them.
    '''
    title = str(sampling)

    if sampling == 'none':
        title = 'kd-tree'

    if noise:
        title = "noised " + title

    if len(idxs) > 1:
        if noise:
            vars = [1 - 0.1, 1 - 0.25, 1 - 0.5, 1 - 0.75]
        else:
            vars = [0.01, 0.1, 0.25, 0.5, 0.75]

        for i in range(len(idxs)):
            cur_vals = vals[i][:idxs[i]]
            x = [i for i in range(len(cur_vals))]
            plt.plot(x, cur_vals, label=str(vars[i]))

    else:
        cur_vals = vals[0][:idxs[0]]
        x = [i for i in range(len(cur_vals))]
        plt.plot(x, cur_vals, label=title)

    plt.title(title)
    plt.xlabel("iteration")
    plt.ylabel("RMS")
    plt.legend()
    plt.xlim([0, 100])

    if save_fig:
        save_name = './figures/plots/' + title + '.png'
        plt.savefig(save_name)
    else:
        plt.show()

    plt.close()


def find_results(sampling, save_fig=False, noise=False):
    '''
    Find all results for the selected methods and plot them accordingly.
    '''
    file = str(sampling) + ".txt"

    if noise:
        file = DATA_DIR + 'noise' + file
    else:
        file = DATA_DIR + file

    with open(file, 'r') as f:
        avg_lens = []
        all_vals = []

        for line in f:
            stripped_line = line.strip()

            if 'Average length:' in stripped_line:
                avg_len = int(float(stripped_line.split(': ')[1]) + 0.5)
                avg_lens.append(avg_len)

            if 'Average rms:' in stripped_line:
                line_vals = stripped_line.split(': [')[1][:-1]
                vals = [float(val.strip()) for val in line_vals.split(',')]
                all_vals.append(vals)

        plot_results(sampling, save_fig, avg_lens, all_vals, noise)
        f.close()


if __name__ == "__main__":
    samplings = ['uniform', 'random', 'multi_res', 'info_reg', 'none']
    for sampling in samplings:
        find_results(sampling, save_fig=True)
        find_results(sampling, save_fig=True, noise=True)
