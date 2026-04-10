import os
import itertools

# 1. Define your ORCA input template as a string.
#    Placeholders are marked with <key>
template_string = """
! <KS> <functional> <basis>

%paras
    R = 8.0, 0.5, 75 # start, end, n_steps
end

* xyz <charge> <mult>
<atom> 0 0 0
<atom> {R} 0 0
*

# Never forget your bonus lines!!!
"""

# 2. Define all possible options for each placeholder.
#    This is the main section you will edit.
options = {
    'atom': ['F', 'Cl', 'Br', 'I'],
    'charge': ['0', '-2'],
    'mult': ['1'],
    'KS': ['RKS', 'UKS'],
    'functional': ['wB97M-V'],
    'basis': ['def2-TZVPD']
}

# 3. Define a base directory to hold all the jobs
base_dir = "orca_calculations"
os.makedirs(base_dir, exist_ok=True)

# --- Main Logic ---

# Get the keys and value-lists from your options dictionary
# We do this to ensure the order is consistent
option_keys = list(options.keys())
option_values = list(options.values())

# 4. Generate all unique combinations (Cartesian product)
total_jobs = 0
for combination in itertools.product(*option_values):
    
    # Create a dictionary for this specific combination, e.g.:
    # {'KS': 'RKS', 'functional': 'B3LYP', ...}
    job_params = dict(zip(option_keys, combination))
    
    # Create a unique, descriptive directory name
    # e.g., "RKS_B3LYP_def2-SVP_0_1_O_H"
    job_name = "_".join(str(v) for v in combination)
    job_dir = os.path.join(base_dir, job_name)
    
    # Create the directory
    os.makedirs(job_dir, exist_ok=True)
    
    # 5. Replace placeholders in the template
    input_content = template_string
    for key, value in job_params.items():
        placeholder = f"<{key}>"
        input_content = input_content.replace(placeholder, str(value))
        
    # 6. Write the new input file
    output_filename = os.path.join(job_dir, "orca.inp")
    with open(output_filename, 'w') as f:
        f.write(input_content)
        
    total_jobs += 1

print(f"\nSuccessfully generated {total_jobs} unique ORCA input files.")

