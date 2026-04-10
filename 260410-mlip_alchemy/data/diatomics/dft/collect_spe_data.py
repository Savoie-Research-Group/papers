import os
import re
import pandas as pd

# --- Configuration ---

# 1. Set the main directory where all your job folders are
#    (Change this to the correct path for your new calculations)
base_dir = "orca_calculations" 

# 2. Set your validation criteria
success_string = "****ORCA TERMINATED NORMALLY****"

# 3. Define the regex pattern to find the final energy
#    This looks for "FINAL SINGLE POINT ENERGY", whitespace,
#    and captures the (possibly negative) floating-point number
energy_pattern = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+\.\d+)")

# 4. Define output file name
output_csv_file = "atom_reference_spe.csv"

# --- Main Script ---

# This list will store dictionaries of our results
all_results = []

# Keep track of job statuses
passed_jobs = 0
failed_jobs = 0

print(f"Starting analysis in: {os.path.abspath(base_dir)}\n")

# Walk through all items in the base directory
for job_name in os.listdir(base_dir):
    job_dir = os.path.join(base_dir, job_name)
    
    # Only process directories
    if os.path.isdir(job_dir):
        print(f"--- Processing: {job_name} ---")
        
        out_file = os.path.join(job_dir, "orca.out")
        
        # Initialize check and content variables
        check1_terminated_normally = False
        out_content = ""

        # --- Read the .out file ---
        try:
            with open(out_file, 'r') as f:
                out_content = f.read()
            
            # --- CHECK 1: Normal Termination ---
            if success_string in out_content:
                check1_terminated_normally = True
                print("  CHECK: Termination normal. (PASSED)")
            else:
                print("  CHECK: Termination string not found. (FAILED)")
                failed_jobs += 1
                
        except FileNotFoundError:
            print(f"  ERROR: {out_file} not found. (FAILED)")
            failed_jobs += 1
            print("-" * (len(job_name) + 20)) # Separator
            continue # Skip to the next folder
        except Exception as e:
            print(f"  ERROR: Could not read {out_file}: {e} (FAILED)")
            failed_jobs += 1
            print("-" * (len(job_name) + 20)) # Separator
            continue # Skip to the next folder

        # --- If Check 1 passed, extract energy ---
        if check1_terminated_normally:
            
            # Find *all* matches for the energy pattern
            matches = energy_pattern.findall(out_content)
            
            if matches:
                # Get the *last* match, which is the final one
                final_energy_str = matches[-1]
                final_energy = float(final_energy_str)
                
                print(f"  ENERGY: Found {final_energy} Hartree.")
                
                # Append the result as a dictionary
                all_results.append({
                    'system': job_name,
                    'energy_hartree': final_energy
                })
                passed_jobs += 1
                
            else:
                # This is an edge case: job terminated normally
                # but we couldn't find the energy string.
                print("  ERROR: Terminated normally, but 'FINAL...ENERGY' string not found. (FAILED)")
                failed_jobs += 1
        
        print("-" * (len(job_name) + 20)) # Separator

# --- COMPILE FINAL CSV ---
print("\n--- Processing Complete ---")
print(f"Jobs passed: {passed_jobs}")
print(f"Jobs failed/skipped: {failed_jobs}")

if not all_results:
    print(f"\nNo data was collected. No '{output_csv_file}' will be written.")
else:
    # Convert the list of dictionaries directly into a DataFrame
    final_dataframe = pd.DataFrame(all_results)
    
    # Re-order columns just to be sure
    final_dataframe = final_dataframe[['system', 'energy_hartree']]
    
    # Define the final output path (inside the base_dir)
    output_path = os.path.join(base_dir, output_csv_file)
    
    # Save to CSV, without the pandas index
    final_dataframe.to_csv(output_path, index=False)
    
    print(f"\nSuccessfully compiled all data into:")
    print(f"{os.path.abspath(output_path)}")