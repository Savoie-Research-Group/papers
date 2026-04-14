import os
import itertools

# 1. Define your ORCA input template as a string.
#    Placeholders are marked with <key>
template_string = """
! <KS> <functional> <basis>

%paras
    R [8.0 7.9 7.8 7.7 7.6 7.5 7.4 7.3 7.2 7.1
       7.0 6.9 6.8 6.7 6.6 6.5 6.4 6.3 6.2 6.1
       6.0 5.9 5.8 5.7 5.6 5.5 5.4 5.3 5.2 5.1
       5.0 4.9 4.8 4.7 4.6 4.5 4.4 4.3 4.2 4.1
       4.0 3.9 4.8 3.7 3.6 3.5 3.4 3.3 3.2 3.1
       3.0 2.9 2.8 2.7 2.6 2.5 2.4 2.3 2.2 2.1
       2.0 1.9 1.8 1.7 1.6 1.5 1.4 1.3 1.2 1.1
       1.0 0.9 0.9 0.7 0.6 0.5]
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

