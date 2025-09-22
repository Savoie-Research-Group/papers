#!/usr/bin/env python3
"""
Molecular Dynamics Trajectory Analysis Script for Ligand Detachment Detection

This script analyzes LAMMPS trajectory files to detect when organic ligands 
completely detach from perovskite structures during molecular dynamics simulations.

Author: Zhichen Nian
"""

import os, sys, shutil, subprocess, math
import numpy as np
import random
import re


def read_starting_line(f):
    """
    Read and parse the header information of a LAMMPS trajectory frame.
    
    This function reads the standard LAMMPS dump file format header which includes:
    - Timestep information
    - Number of atoms
    - Box dimensions (4 lines)
    - Atoms header line
    
    Args:
        f (file object): Open file handle positioned at the start of a trajectory frame
        
    Returns:
        tuple: (TS, atom_num, data)
            - TS (float): Timestep value
            - atom_num (int): Number of atoms in the system
            - data (list): Empty list for storing atom data
    """
    # Read timestep value (first line after ITEM: TIMESTEP)
    TS = float(f.readline()) 
    
    # Skip the "ITEM: NUMBER OF ATOMS" line
    f.readline() 
    # Read the actual number of atoms
    atom_num = int(f.readline())
    
    # Initialize empty data container
    data = []
    
    # Skip the next 4 lines which contain box dimension information
    # These lines typically contain ITEM: BOX BOUNDS and the box coordinates
    for i in range(4):
        f.readline()
        
    # Skip the "ITEM: ATOMS" header line
    f.readline() 
    
    return TS, atom_num, data


def main(*args):
    """
    Main function to analyze ligand detachment from perovskite structures.
    
    This function:
    1. Scans the current directory for temperature subdirectories
    2. For each temperature, processes multiple simulation runs (1-10)
    3. Reads PLUMED files to identify ligand atoms
    4. Analyzes trajectory files to detect complete ligand detachment
    5. Records detachment timesteps in a summary file
    
    The detachment criterion is based on y-coordinate separation:
    - Ligand is considered detached when its minimum y-coordinate
      exceeds the maximum y-coordinate of all other atoms (excluding inorganic layers)
    
    Args:
        *args: Command line arguments (currently unused)
    """
    # List to track submitted jobs (currently unused)
    submitted_jobs = []
    
    # Get list of all items in current directory
    dir = os.listdir(".")
    
    # Filter for directories with names as temperatures, use re to ensure they are valid
    # These are expected to be temperature directories (e.g., "300K", "270K")
    temp_dir = [d for d in dir if os.path.isdir(d) and re.match(r'^\d{2,3}K$', d)]
    print("Temperature directories found:", temp_dir)
    
    # Dictionary to store detachment records
    # Structure: {temperature: {run_number: [detachment_timesteps]}}
    disx_record = {}
    
    # Process each temperature directory
    for d in temp_dir:
        # Initialize dictionary for current temperature
        disx_record[d] = {}       
        
        # Debug output
        print("Current temperature directory:", d)
        
        # Process runs 1 through 10 for each temperature
        for run_i in range(1, 11, 1):
            # Initialize list to store detachment timesteps for this run
            disx_record[d][run_i] = []
            
            # List to store atom indices of the ligand
            atom_index = []
            
            # Read PLUMED steering file to extract ligand atom information
            plumed_file = f"{d}/run{run_i}/steer_no_NVE/plumed.dat"
            
            try:
                with open(plumed_file, "r") as f:
                    print(f"Reading plumed.dat file: {plumed_file}")
                    
                    # Parse PLUMED file to find ligand coordination (LC) definition
                    for l in f:
                        if l.split()[0] == "LC:":
                            # Extract atom indices from the LC line
                            # Format: LC: COORDINATION GROUPA=atom1,atom2,...
                            atom_in_ligand = re.split(',|=', l)
                            
                            # Convert string indices to integers and adjust for 0-based indexing
                            # (LAMMPS uses 1-based indexing, Python uses 0-based)
                            atom_index = np.array([int(_) for _ in atom_in_ligand if _.isdigit()]) - 1
                            
            except FileNotFoundError:
                print(f"Warning: Could not find plumed.dat file: {plumed_file}")
                continue
            
            # If no ligand atoms found, skip this run
            if len(atom_index) == 0:
                print(f"Warning: No ligand atoms found in {plumed_file}")
                continue
                           
            # Read trajectory file to analyze ligand detachment
            traj_file = f"{d}/run{run_i}/steer_no_NVE/plumed.0.nvt.lammpstrj"
            
            try:
                with open(traj_file, "r") as f:
                    # Read first frame header
                    f.readline()  # Skip "ITEM: TIMESTEP"
                    TS = float(f.readline())  # Read timestep value
                    
                    f.readline()  # Skip "ITEM: NUMBER OF ATOMS"
                    atom_num = int(f.readline())  # Read number of atoms
                    data = []
                    
                    # Skip box information (4 lines)
                    for i in range(4):
                        f.readline()
                        
                    f.readline()  # Skip "ITEM: ATOMS" line
                    
                    # Process trajectory frames
                    for l in f:
                        # Read atom data for current frame
                        if len(data) < atom_num:
                            data.append(l.split())
                            
                        # When we have read all atoms for this frame
                        elif len(data) == atom_num:
                            # Convert to numpy array for easier manipulation
                            data = np.array(data).astype(float)
                            
                            # Extract ligand geometry (exclude last atom which is counter ion)
                            ligand_geo = data[atom_index[:-1]]
                            
                            # Extract other atoms (excluding ligand atoms)
                            other_geo = np.delete(data, atom_index, axis=0)

                            # Remove inorganic layer atoms from consideration
                            other_geo = np.array([i for i in other_geo if i[1] != 6 and i[1] != 17])

                            # Calculate separation criterion
                            min_ligand_geo_y = min(ligand_geo[:, 3])
                            max_other_geo_y = max(other_geo[:, 3])

                            # Check if ligand is completely detached
                            # Detachment occurs when minimum ligand y-coordinate exceeds
                            # maximum y-coordinate of all other atoms
                            if min_ligand_geo_y - max_other_geo_y > 0:
                                print(f"Detachment detected: {d}, run{run_i}, timestep {TS}")
                                disx_record[d][run_i].append(TS)
                                break  # Stop analyzing this run once detachment is found
                            else:
                                # Read next frame header
                                TS, atom_num, data = read_starting_line(f)
                                data = []  # Reset data for next frame
                                
            except FileNotFoundError:
                print(f"Warning: Could not find trajectory file: {traj_file}")
                continue
                                
    # Write results to output file
    current_dir = os.getcwd().split("/")[-1]
    output_file = f"{current_dir}_disx_record_noNVE_1_10.txt"
    
    with open(output_file, "w") as o:
        # Write detachment records for each temperature and run
        for temp in disx_record:
            for run_i in disx_record[temp]:
                o.write(f"{temp} {run_i} {disx_record[temp][run_i]}\n")
    
    print(f"Analysis complete. Results written to: {output_file}")


if __name__ == "__main__":
    """
    Script entry point when run directly from command line.
    
    Passes command line arguments (excluding script name) to main function.
    """
    main(sys.argv[1:])