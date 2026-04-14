#!/bin/bash

template_submit="/temp180/bsavoie2/emille35/mlip_alchemy/diatomic_reference/job.submit"
base_dir="/temp180/bsavoie2/emille35/mlip_alchemy/diatomic_reference/orca_calculations"

# Loop through every job directory created by the Python script
for job_dir in "$base_dir"/*/; do
    
    echo "Submitting job in: $job_dir"
    
    # Navigate into the job directory
    cd "$job_dir"

    # Copy template submission script
    cp "$template_submit" .
    echo here I am $PWD

    # Submit to queue
    qsub job.submit
        
    # Go back up to the parent directory to continue the loop
    cd "$base_dir"
    
done

echo "All jobs submitted."