import numpy as np
from scipy.interpolate import interp1d

full = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
full_detachment_time = [347, 318, 338, 312, 325, 328, 356, 306, 259, 288]

def read_COVAR(indices):
    """
    READ COLVAR file under runX/steer_no_NVE/
    where X is the index of the run
    """
    pairwise_time = []
    pairwise_disy = []
    pairwise_freeE = []
    for id in indices:
        print("Reading run id: ", id)
        filename = f'300K/run{id}/steer_no_NVE/COLVAR'
    
        COLVAR = filename
        with open(COLVAR, 'r') as f:
            f.readline()  # skip header
            current_run_time = []
            current_run_disy = []
            current_run_freeE = []
            for line in f:
                if not line.strip():
                    continue
                data = line.split()
                current_run_time.append(float(data[0]))
                current_run_disy.append(float(data[2]))
                current_run_freeE.append(float(data[-1]))
            current_run_disy = np.array(current_run_disy).astype(float)
            current_run_freeE = np.array(current_run_freeE).astype(float)
            current_run_disy -= current_run_disy[0]
            pairwise_time.append(current_run_time)
            pairwise_disy.append(current_run_disy)
            # cutoff the reference (disy[0]) so that it starts at 0
            pairwise_freeE.append(current_run_freeE)
    return pairwise_time, pairwise_disy, pairwise_freeE

full_time, full_disy, full_freeE = read_COVAR(full)
full_detach_pos = []
full_detach_FE = []
for i, time in enumerate(full_detachment_time):
    curr_run_time = full_time[i]
    for _, t in enumerate(curr_run_time):
        if abs(t - time) < 1e-3:
            full_detach_pos.append(full_disy[i][_])
            full_detach_FE.append(full_freeE[i][_])
full_detach_pos = np.round(full_detach_pos, 3)
full_detach_FE = np.round(full_detach_FE, 3)

# make a common grid for all runs and interpolate
def make_common_grid(disy_lists, freeE_lists):
    y_min = min([i.min() for i in disy_lists])
    y_max = max([i.max() for i in disy_lists])
    y_common = np.linspace(y_min, y_max, 1000)
    interpolated_freeE = []
    for dis_list, freeE_list in zip(disy_lists, freeE_lists):
        # Interpolate the free energy to the common grid
        # print(len(dis_list), len(freeE_list))
        interp_func = interp1d(dis_list, freeE_list, bounds_error=False, fill_value='extrapolate')
        # print(len(dis_list), len(freeE_list))
        interpolated_freeE.append(interp_func(y_common))
    return y_common, np.array(interpolated_freeE)

def moving_avg(data, window_size):
    """Calculate moving average with a specified window size."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

y_full_common, inter_full_freeE = make_common_grid(full_disy, full_freeE)
avg_full_freeE = np.mean(inter_full_freeE, axis=0)
avg_full_freeE = moving_avg(avg_full_freeE, window_size=10)
std_full_freeE = np.std(inter_full_freeE, axis=0)
std_full_freeE = moving_avg(std_full_freeE, window_size=10)
y_full_common = y_full_common[:len(avg_full_freeE)]

avg_full_detach_FE = np.mean(full_detach_FE)
avg_full_detach_pos = np.mean(full_detach_pos)

with open("4P_full_freeE_mean_std_300K.csv", "w") as f:
    f.write("4P_full_freeE_mean_std_300K\n")
    f.write("disx detached: {}, the free energy: {} \n".format(avg_full_detach_pos, avg_full_detach_FE))
    f.write("Displacement,Full_Average_Free_Energy,Full_Std_Dev\n")
    for i in range(len(y_full_common)):
        f.write(f"{y_full_common[i]},{avg_full_freeE[i]},{std_full_freeE[i]}\n")