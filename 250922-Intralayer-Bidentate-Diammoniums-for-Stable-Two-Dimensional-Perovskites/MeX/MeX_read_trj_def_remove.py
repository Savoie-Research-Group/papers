import os, sys, shutil, subprocess, math
import numpy as np
import random
import re


# read the starting line of the lammpstrj file

def  read_starting_line(f):
    """
    Reads and parses the header information from a LAMMPS trajectory file.
    
    This function reads the initial lines of a LAMMPS trajectory file frame to extract
    the timestamp, number of atoms, and prepares for reading atomic coordinates.
    
    Args:
        f (file object): An open file object pointing to a LAMMPS trajectory file
    
    Returns:
        tuple: A tuple containing:
            - TS (float): The timestep value from the trajectory frame
            - atom_num (int): The total number of atoms in the system
            - data (list): An empty list initialized for storing atomic data
    """
    TS = float(f.readline()) 
    
    f.readline() # read the number of atoms line
    atom_num = int(f.readline())
    data = []
    
    #read the next 4 lines, which are box information, not particular useful
    for i in range(4):
        f.readline()
        
    f.readline() # read the Atoms line
    
    return TS, atom_num, data

def main(*args):
    """
    Main function that analyzes LAMMPS trajectory files to determine when ligands are completely 
    removed from perovskite structures during steered molecular dynamics simulations.
    
    This function:
    1. Scans through temperature directories and multiple runs
    2. Extracts ligand atom indices from plumed.dat files
    3. Analyzes trajectory files to track ligand positions relative to the perovskite structure
    4. Identifies the timestep when a ligand is completely separated from the perovskite
    5. Records and saves the separation timesteps to an output file
    
    The separation criterion is based on comparing the minimum x-coordinate of ligand atoms
    with the maximum x-coordinate of all other atoms in the system.
    
    
    Output:
        Creates a text file named "{current_directory}_disx_record_1_11_no_NVE.txt" containing
        the separation timesteps for each temperature and run combination.
    """
    submitted_jobs= []
    dir = os.listdir(".")
    temp_dir = [d for d in dir if os.path.isdir(d) and len(d) < 5 ]

    disx_record = {}
    
    for d in temp_dir: 
        disx_record[d] = {}       
        for run_i in range(1, 11, 1):
                disx_record[d][run_i] = []
                atom_index = []
                # open the dir/run_i/steer/plumed.dat file, extract the ligand information of the perovskite.
                with open(d+"/run"+str(run_i)+"/steer_no_NVE/plumed.dat", "r") as f:
                    for l in f:
                        if l.split()[0] == "LC:":
                            # split the line by comma and equal sign
                            atom_in_ligand = re.split(',|=', l)
                            atom_index = np.array([int(_) for _ in atom_in_ligand if _.isdigit()]) - 1

                # open the dir/run_i/steer/plumed.0.lammpstrj file, extract the overall trajectory information.
                with open(d+"/run"+str(run_i)+"/steer_no_NVE/plumed.0.nvt.lammpstrj", "r") as f:
                    f.readline() # read the Time step line
                    TS = float(f.readline()) 
                    
                    f.readline() # read the number of atoms line
                    atom_num = int(f.readline())
                    data = []
                    
                    #read the next 4 lines, which are box information, not particular useful
                    for i in range(4):
                        f.readline()
                        
                    f.readline() # read the Atoms line
                    
                    # read the coordinates of all the atoms
                    for l in f:
                        if len(data) < atom_num:
                            data.append(l.split())
                        elif len(data) == atom_num:
                            data = np.array(data).astype(float)
                            # print(atom_index)
                            ligand_geo = data[atom_index]
                            other_geo = np.delete(data, atom_index, axis=0)
                            
                            # find the min of the ligand x coordinate
                            # and the max in the other atoms x coordinate
                            min_ligand_geo_x = min(ligand_geo[:,2])
                            max_other_geo_x = max(other_geo[:,2])
                            
                            # if the min of the ligand x coordinate is greater than the max in the other atoms x coordinate
                            # consider this ligand as completely removed from the perovskite
                            # and print this time step, stop the loop
                            if min_ligand_geo_x - max_other_geo_x > 0:
                                print("For {}, run{}, the Time Step that ligand is completely removed is {}".format(d, run_i, TS))
                                disx_record[d][run_i].append(TS)
                                break
                            else:
                                TS, atom_num, data = read_starting_line(f)
                            data = []
                                
    current_dir = os.getcwd().split("/")[-1]                            
    with open(f"{current_dir}_disx_record_1_11_no_NVE.txt", "w") as o:                                    
        for temp in disx_record:
            for run_i in disx_record[temp]:
                o.write(f"{temp} {run_i} {disx_record[temp][run_i]} \n")
                            
                        

                       
if __name__ == "__main__":
    main(sys.argv[1:])                

    


