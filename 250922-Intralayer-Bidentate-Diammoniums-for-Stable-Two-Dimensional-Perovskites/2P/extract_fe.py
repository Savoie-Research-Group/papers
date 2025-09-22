import numpy as np
import os, sys
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

d = os.listdir()
d = sorted([i for i in d if os.path.isdir(i) and len(i) < 5 and i[0] != '.'])

FE_profile = {}
disx_profile = {}  
time_profile = {}
FE_unbound = {}

for dir in d:
    # for each dir, create a dict to store the data
    FE_profile[dir] = {}
    disx_profile[dir] = {}
    time_profile[dir] = {}
    

    for run_id in range(1, 11, 1):
        # Save the run specific data in the list
        FE = []
        disx = [] 
        time = []
        
        if not os.path.exists("{}/run{}/steer_no_NVE/COLVAR".format(dir, run_id)):
            continue
        
        with open("{}/run{}/steer_no_NVE/COLVAR".format(dir, run_id), 'r') as f:
            f.readline() # read the title line
            for l in f:
                FE.append(l.split()[-1])
                disx.append(l.split()[1])
                time.append(l.split()[0])
        FE = np.array(FE, dtype=float)
        disx = np.array(disx, dtype=float)
        time = np.array(time, dtype=float)
        
        FE_profile[dir]["run"+str(run_id)] = FE
        disx_profile[dir]["run"+str(run_id)] = disx
        time_profile[dir]["run"+str(run_id)] = time

for key in disx_profile.keys():
    for run in disx_profile[key].keys():
        disx_profile[key][run] = disx_profile[key][run] - disx_profile[key][run][0]
        
interp1d_runs = {}
new_disx_runs = {}
record = []
count = 0
mean_FE_runs = {}
std_FE_runs = {}
FE_unbound = {}
disx_unbound = {}


for dir in d:
    interp1d_runs[dir] = []
    max_x = []
    min_x = []
    for run_id in range(1, 11):
        temp_disx = disx_profile[dir]
        max_x.append(max(temp_disx["run"+str(run_id)]))
        min_x.append(min(temp_disx["run"+str(run_id)]))

    # find the min in the max and max in the min
    # so that the new disx can be the same for all the runs
    max_x = min(max_x)
    min_x = max(min_x)
    new_disx = np.linspace(min_x, max_x, 1000)
    new_disx_runs[dir] = new_disx
    
    # do the interpolation for each run
    for run_id in range(1, 11, 1):
        x = disx_profile[dir]["run" + str(run_id)]
        fe = FE_profile[dir]["run" + str(run_id)]
        f = interp1d(x, fe)
        # print(f)
        interp1d_runs[dir].append(f(new_disx))
    interp1d_runs[dir] = np.array(interp1d_runs[dir])
    
    # Calculate the mean and std of the FE
    mean_FE = np.mean(interp1d_runs[dir], axis=0)
    std_FE = np.std(interp1d_runs[dir], axis=0)
    mean_FE = moving_average(mean_FE, 10)
    mean_FE_runs[dir] = mean_FE
    
    std_FE_runs[dir] = std_FE
    
    # print(mean_FE[::100], std_FE[::100])
    
    # open the disx_record file to read the time (fs) where the ligand is unbound
    with open("Xan_disx_record_1_11_no_NVE.txt", 'r') as f:
        lines = f.readlines()
    f.close()
    
    time_unbound_list = []
    for l in lines:
        if l.split()[0] == dir:
            time_unbound_list.append(l.split()[2][1:-1])
    time_unbound_list = np.array(time_unbound_list, dtype=float) / 1000
    # print(time_unbound_list)

    disx_unbound_list = []
    FE_unbound_list = []
    # use the time, go into the time_profile dict to find the corresponding disx
    for c, i in enumerate(time_unbound_list):
        disx_unbound_list.append(disx_profile[dir]["run"+str(c+1)][np.where(time_profile[dir]["run"+str(c+1)] == i)][0])
        FE_unbound_list.append(FE_profile[dir]["run"+str(c+1)][np.where(time_profile[dir]["run"+str(c+1)] == i)][0])
    
    if dir == "300K":    
        for count_i, i in enumerate(FE_unbound_list):
            print(dir, time_unbound_list[count_i], disx_unbound_list[count_i], i)
    
    FE_unbound[dir] = np.array(FE_unbound_list)
    disx_unbound[dir] = np.array(disx_unbound_list)
    mean_disx_unbound = np.mean(disx_unbound_list)
    std_disx_unbound = np.std(disx_unbound_list)
    # print(dir, disx_unbound_list)
    # print(std_disx_unbound)

print(FE_unbound["300K"], FE_unbound["300K"].mean(), FE_unbound["300K"].std())

for key in new_disx_runs.keys():
    with open("MeX_FE_mean_std_{}.csv".format(key), "w") as f:
        f.write("MeXPbI4\n")
        f.write("disx_detach: {}, FE_unbound: {}\n".format(disx_unbound[key].mean(), FE_unbound[key].mean()))
        f.write("disx,FE_mean,FE_std\n")
        for c, i in enumerate(mean_FE_runs[key]):
            f.write("{},{},{}\n".format(new_disx_runs[key][c], i, std_FE_runs[key][c]))
    f.close()