import os
import re # Import regular expressions
import pandas as pd

# --- Configuration ---

# 1. Set the main directory where all your job folders are
base_dir = "orca_calculations"

# 2. Set your validation criteria
success_string = "****ORCA TERMINATED NORMALLY****"
scf_fail_string = "SCF NOT CONVERGED AFTER"
expected_data_points = 75

# 3. Define the regex pattern to find *all* final energies
#    This looks for "FINAL SINGLE POINT ENERGY" followed by spaces
#    and captures the floating-point number (e.g., -1234.5678)
energy_pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

# 4. Define output file name
output_csv_file = "dft_scan_energies.csv"

# --- Main Script ---

# This list will hold pandas.Series objects (i.e., columns of energy data)
all_energy_data = []

# This will store the 'bond_distance' column from the first successful job
master_bond_distances = None

# Keep track of job statuses
passed_jobs = 0
salvaged_jobs = 0
failed_jobs = 0

print(f"Starting analysis in: {os.path.abspath(base_dir)}\n")

# Walk through all items in the base directory
for job_name in os.listdir(base_dir):
    job_dir = os.path.join(base_dir, job_name)
    
    # Only process directories
    if os.path.isdir(job_dir):
        print(f"--- Processing: {job_name} ---")
        
        out_file = os.path.join(job_dir, "orca.out")
        trj_file = os.path.join(job_dir, "orca.trjact.dat")

        check1_terminated_normally = False
        out_content = "" # To store the .out file content

        # === Pre-Check: Read the .out file ===
        try:
            with open(out_file, 'r') as f:
                out_content = f.read() # Read content for later checks
            if success_string in out_content:
                check1_terminated_normally = True
        except FileNotFoundError:
            print(f"  CHECK 1: {out_file} not found. (FAILED)")
        except Exception as e:
            print(f"  CHECK 1: Error reading {out_file}: {e} (FAILED)")

        # === Path A: Perfect Job (Terminated Normally) ===
        if check1_terminated_normally:
            print("  CHECK 1: Termination normal. (PASSED)")
            job_energies = []
            job_distances = []
            
            try:
                # This job says it's perfect, let's parse its trj_file
                with open(trj_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 2: # A data line
                        try:
                            job_distances.append(float(parts[0]))
                            job_energies.append(float(parts[1]))
                        except ValueError:
                            pass # Skip malformed lines
                
                # Now, validate the data we just parsed
                if len(job_energies) == expected_data_points:
                    print(f"  CHECK 2: Trajectory file is complete (75 points). (PASSED)")
                    print("  STATUS: Perfect job. Collecting data. 👍")
                    
                    # Create a pandas Series (a column)
                    energy_series = pd.Series(job_energies, name=job_name)
                    all_energy_data.append(energy_series)
                    
                    # NEW: If this is the first good job, save its distances
                    if master_bond_distances is None:
                        master_bond_distances = job_distances
                        print("  INFO: Stored master bond distances from this job.")
                        
                    passed_jobs += 1
                else:
                    # Terminated normally, but trj_file is corrupt/incomplete
                    print(f"  CHECK 2: Trajectory file is incomplete ({len(job_energies)} points). (FAILED)")
                    print("  STATUS: Inconsistent state. Skipping. ❌")
                    failed_jobs += 1

            except FileNotFoundError:
                print(f"  CHECK 2: {trj_file} not found. (FAILED)")
                print("  STATUS: Inconsistent state. Skipping. ❌")
                failed_jobs += 1
            except Exception as e:
                print(f"  CHECK 2: Error reading {trj_file}: {e} (FAILED)")
                failed_jobs += 1

        # === Path B: Failed Job (Check for SCF failure) ===
        elif scf_fail_string in out_content:
            print("  CHECK 1: Termination abnormal. (FAILED)")
            print("  INFO: 'SCF NOT CONVERGED' detected. Attempting to salvage...")
            
            # Use re.findall to get *ALL* energies
            matches = energy_pattern.findall(out_content)
            
            if matches:
                # Convert all found energy strings to floats
                salvaged_energies = [float(e) for e in matches]
                num_salvaged = len(salvaged_energies)
                print(f"  SALVAGED: Found {num_salvaged} energy points. ⚡")
                
                # Create a full-length list, padding with None
                # This ensures it has 75 elements, matching the perfect jobs
                padded_energies = salvaged_energies + [None] * (expected_data_points - num_salvaged)
                
                # Create a pandas Series, which will have 75 items
                energy_series = pd.Series(padded_energies, name=job_name)
                all_energy_data.append(energy_series)
                salvaged_jobs += 1
            else:
                print("  SALVAGE FAILED: 'SCF NOT CONVERGED' found, but no 'FINAL...ENERGY' lines.")
                failed_jobs += 1
        
        # === Path C: Failed Job (Other Reason) ===
        else:
            if out_content: # We only know if out_content was read
                print("  CHECK 1: Termination abnormal. (FAILED)")
                print("  INFO: Job failed (not an SCF error). Skipping. ❌")
            failed_jobs += 1
            
        print("-" * (len(job_name) + 20)) # Separator

# --- COMPILE FINAL CSV ---
print("\n--- Processing Complete ---")
print(f"Total jobs passed (perfect): {passed_jobs}")
print(f"Total jobs salvaged (partial): {salvaged_jobs}")
print(f"Total jobs failed/skipped: {failed_jobs}")

if not all_energy_data:
    print(f"\nNo data was collected. No '{output_csv_file}' will be written.")
else:
    # Combine all Series (columns) into one DataFrame
    # pandas will automatically align them, filling missing spots with NaN
    final_dataframe = pd.concat(all_energy_data, axis=1)
    
    # NEW: Add the bond_distance column
    if master_bond_distances is None:
        print("\nWARNING: No perfectly-terminated job was found.")
        print("         The 'bond_distance' column will be missing.")
    else:
        # insert() adds it at the beginning (location 0)
        final_dataframe.insert(0, 'bond_distance', master_bond_distances)
    
    # Add a 'Step' index (from 1 to 75)
    final_dataframe.index = range(1, len(final_dataframe) + 1)
    final_dataframe.index.name = "Step"
    
    output_path = os.path.join(base_dir, output_csv_file)
    final_dataframe.to_csv(output_path)
    
    print(f"\nSuccessfully compiled all data into:")
    print(f"{os.path.abspath(output_path)}")