#!/usr/bin/env python3
"""
LAMMPS Trajectory Analysis for Ligand Dissociation Detection
===========================================================

This script analyzes molecular dynamics simulation trajectories from LAMMPS to detect
when ligands completely dissociate from perovskite structures in 2D materials.


Author: Zhichen Nian

Usage:
------
python 2P_read_traj_def_remove.py

Example Output File:
-------------------
300K 1 [45000.0]
...
"""

import os
import sys
import shutil
import subprocess
import math
import numpy as np
import random
import re


def read_starting_line(f):
    """
    Parse LAMMPS trajectory frame header information.
    
    This function reads and parses the standard header of a LAMMPS dump file frame,
    which follows a specific format defined by the LAMMPS dump command. The header
    contains essential metadata about each trajectory frame.
    
    LAMMPS Dump File Format:
    -----------------------
    ITEM: TIMESTEP
    [timestep_value]
    ITEM: NUMBER OF ATOMS
    [number_of_atoms]
    ITEM: BOX BOUNDS [boundary_conditions]
    [xlo] [xhi]
    [ylo] [yhi] 
    [zlo] [zhi]
    ITEM: ATOMS [atom_attributes]
    [atom_data_lines...]
    
    Parameters:
    ----------
    f : file object
        Open file handle positioned at the beginning of a trajectory frame.
        The file pointer should be at the line immediately after "ITEM: TIMESTEP".
        
    Returns:
    -------
    tuple
        A 3-element tuple containing:
        - TS (float): Timestep value from the simulation
        - atom_num (int): Total number of atoms in the system
        - data (list): Empty list initialized for storing atomic coordinate data
    """
    # Read timestep value from the line immediately following "ITEM: TIMESTEP"
    # Convert to float to handle both integer and decimal timesteps
    TS = float(f.readline().strip())
    
    # Skip the "ITEM: NUMBER OF ATOMS" header line
    f.readline()
    
    # Read the actual number of atoms in the system
    # This is essential for knowing how many atom lines to read
    atom_num = int(f.readline().strip())
    
    # Initialize empty container for atomic coordinate data
    # This will be populated by the calling function
    data = []
    
    # Skip the next 4 lines containing box dimension information:
    # 1. "ITEM: BOX BOUNDS [pp pp pp]" or similar
    # 2. x-dimension bounds: xlo xhi
    # 3. y-dimension bounds: ylo yhi  
    # 4. z-dimension bounds: zlo zhi
    for i in range(4):
        f.readline()
        
    # Skip the "ITEM: ATOMS" header line which precedes the actual atom data
    # This line typically contains format specification like "id type x y z"
    f.readline()
    
    return TS, atom_num, data


def main(*args):
    """
    Main function to analyze ligand dissociation in 2D perovskite simulations.
    
    This function orchestrates the complete analysis workflow:
    1. Identifies temperature directories in the current working directory
    2. Processes multiple simulation runs for each temperature
    3. Extracts ligand atom information from PLUMED files
    4. Analyzes trajectory data to detect ligand removal
    5. Records and outputs the results
    
    The analysis focuses on detecting the moment when organic ligands completely
    separate from the inorganic perovskite framework based on spatial coordinates.
    
    Directory Structure Expected:
    ----------------------------
    Current directory should contain:
    - Temperature folders named like "270K", "300K", "350K", etc.
    - Each temperature folder contains run1/ through run10/
    - Each run folder contains steer_no_NVE/ subdirectory
    - Required files in steer_no_NVE/:
        * plumed.dat: Contains ligand atom definitions
        * plumed.0.nvt.lammpstrj: Trajectory coordinates
        
    Output:
    ------
    Creates a text file named "{current_directory}_disx_record_noNVE_1_10.txt"
    containing timesteps when ligand removal occurs for each temperature/run combination.
    
    File Format:
    -----------
    Each line: "[temperature] [run_number] [timestep_list]"
    Example: "300K 1 [50000.0]" means ligand removed at timestep 50000 in run 1 at 300K
    """
    
    # Get list of all items in current directory
    dir_list = os.listdir(".")
    
    # Filter to identify temperature directories
    # Pattern matches: one or more digits followed by 'K' (e.g., "270K", "300K", "1000K")
    # Only directories (not files) that match this pattern are included
    temp_dir = [d for d in dir_list if os.path.isdir(d) and re.match(r'^\d+K$', d)]
    print("Temperature directories found: ", temp_dir)
    
    # Initialize nested dictionary to store dissociation timesteps
    # Structure: {temperature: {run_number: [timestep_list]}}
    disx_record = {}
    
    # Process each temperature directory
    for d in temp_dir:
        disx_record[d] = {}
        print(f"Processing temperature directory: {d}")
        
        # Analyze runs 1 through 10 for each temperature
        # Range(1, 11, 1) generates [1, 2, 3, ..., 10]
        for run_i in range(1, 11, 1):
            # Initialize storage for current run
            disx_record[d][run_i] = []
            atom_index = []  # Will store ligand atom indices
            
            # Extract ligand atom information from PLUMED configuration
            plumed_file = f"{d}/run{run_i}/steer_no_NVE/plumed.dat"
            try:
                with open(plumed_file, "r") as f:
                    print(f"Reading PLUMED file: {plumed_file}")
                    
                    # Parse PLUMED file for ligand definitions
                    for line in f:
                        # Look for lines starting with "LC:" which define ligand centers
                        # PLUMED syntax typically: LC: GROUP ATOMS=1,2,3,4...
                        if line.split()[0] == "LC:":
                            # Split by both comma and equals sign to extract atom numbers
                            # Example: "LC: GROUP ATOMS=1,2,3" -> ["LC", " GROUP ATOMS", "1", "2", "3"]
                            atom_in_ligand = re.split(',|=', line)
                            
                            # Extract only numeric values and convert to indices
                            # Subtract 1 because LAMMPS uses 1-based indexing but Python uses 0-based
                            atom_index = np.array([int(item) for item in atom_in_ligand 
                                                 if item.isdigit()]) - 1
                            break  # Assuming only one LC definition per file
                            
            except FileNotFoundError:
                print(f"Warning: PLUMED file not found: {plumed_file}")
                continue
            except Exception as e:
                print(f"Error reading PLUMED file {plumed_file}: {e}")
                continue

            # Analyze trajectory file for ligand dissociation
            trajectory_file = f"{d}/run{run_i}/steer_no_NVE/plumed.0.nvt.lammpstrj"
            try:
                with open(trajectory_file, "r") as f:
                    print(f"Analyzing trajectory: {trajectory_file}")
                    
                    # Read initial frame header
                    f.readline()  # Skip "ITEM: TIMESTEP"
                    TS = float(f.readline().strip())  # Current timestep
                    
                    f.readline()  # Skip "ITEM: NUMBER OF ATOMS"
                    atom_num = int(f.readline().strip())  # Total atoms
                    
                    data = []  # Storage for current frame atom data
                    
                    # Skip box dimension information (4 lines)
                    for i in range(4):
                        f.readline()
                        
                    f.readline()  # Skip "ITEM: ATOMS" header
                    
                    # Process trajectory frame by frame
                    for line in f:
                        # Collect atom data for current frame
                        if len(data) < atom_num:
                            # Split line into columns and store
                            # Typical format: atom_id atom_type x y z [other_properties]
                            data.append(line.split())
                            
                        elif len(data) == atom_num:
                            # Complete frame collected, analyze for ligand separation
                            
                            # Convert string data to float for coordinate calculations
                            data = np.array(data).astype(float)
                            
                            # Extract coordinates of ligand atoms and other atoms
                            # Assuming coordinates are in columns 2, 3, 4 (x, y, z)
                            ligand_geo = data[atom_index]  # Ligand atom coordinates
                            other_geo = np.delete(data, atom_index, axis=0)  # Non-ligand coordinates
                            
                            # Calculate spatial separation metrics
                            # Using x-coordinate (column index 2) for separation analysis
                            min_ligand_x = np.min(ligand_geo[:, 2])  # Leftmost ligand atom
                            max_other_x = np.max(other_geo[:, 2])    # Rightmost non-ligand atom
                            
                            # Check for complete ligand separation
                            # If the closest ligand atom is further right than the furthest non-ligand atom,
                            # the ligand has completely separated from the perovskite structure
                            separation_distance = min_ligand_x - max_other_x
                            
                            if separation_distance > 0:
                                # Ligand removal detected!
                                print(f"Ligand removal detected: {d}, run{run_i}, timestep {TS}")
                                print(f"Separation distance: {separation_distance:.3f}")
                                disx_record[d][run_i].append(TS)
                                break  # Stop analyzing this run
                            else:
                                # No separation yet, read next frame
                                try:
                                    TS, atom_num, data = read_starting_line(f)
                                except:
                                    # End of file or read error
                                    break
                            
                            # Reset data container for next frame
                            data = []
                            
            except FileNotFoundError:
                print(f"Warning: Trajectory file not found: {trajectory_file}")
                continue
            except Exception as e:
                print(f"Error analyzing trajectory {trajectory_file}: {e}")
                continue
    
    # Generate output file with results
    current_dir = os.getcwd().split("/")[-1]  # Get current directory name
    output_filename = f"{current_dir}_disx_record_noNVE_1_10.txt"
    
    print(f"\nWriting results to: {output_filename}")
    
    try:
        with open(output_filename, "w") as output_file:
            # Write data for each temperature and run
            for temp in sorted(disx_record.keys()):  # Sort temperatures for consistent output
                for run_i in sorted(disx_record[temp].keys()):  # Sort run numbers
                    timesteps = disx_record[temp][run_i]
                    output_file.write(f"{temp} {run_i} {timesteps}\n")
                    
        print(f"Analysis complete. Results saved to {output_filename}")
        
        # Print summary statistics
        total_runs = sum(len(disx_record[temp]) for temp in disx_record)
        successful_removals = sum(len(disx_record[temp][run]) > 0 
                                for temp in disx_record 
                                for run in disx_record[temp])
        print(f"Summary: {successful_removals}/{total_runs} runs showed ligand removal")
        
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    """
    Script entry point with command line argument support.
    
    This script can be run from the command line with optional arguments,
    though currently no arguments are processed by the main function.
    
    Usage Examples:
    --------------
    python 2P_read_traj_def_remove.py
    
    """    
    main(sys.argv[1:])
