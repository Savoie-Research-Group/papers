#!/bin/bash
# =============================================================================
# File: run_bond_scan_gpu.sh
# Author: Vignesh Sathyaseelan
# Email: vsathyas@purdue.edu
# Description:
#     SLURM batch script to evaluate MLIP (machine-learned interatomic potential)
#     energies for all XYZ structures in each subdirectory. Runs multiple MLIPs
#     (MACE, ORB, FAIR-Chem, AIMNet2) with appropriate Conda environments and
#     organizes results into per-folder output directories.
# =============================================================================

#SBATCH --job-name=bond-scan
#SBATCH --output=bond_scan.out
#SBATCH --error=bond_scan.err
#SBATCH -A bsavoie
#SBATCH -p a10
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --time=1:00:00

set -euo pipefail

# Initialize Conda
source /apps/external/conda/2024.09/etc/profile.d/conda.sh

# MLIPs to run (non-AIMNet)
MLIPS=("mace-omol" "orb-omol" "fairchem-omol")

# Charge/spin combinations
CHARGES=(0)
SPINS=(1)

# Folders containing xyz files
FOLDERS=(*/)

run_section() {
    local FOLDER="$1"
    local CHARGE="$2"
    local SPIN="$3"

    local TAG="${FOLDER%/}"
    local OUTPUT_DIR="results_${TAG}_charge${CHARGE}_spin${SPIN}"

    echo "=== Running folder: $FOLDER | charge=$CHARGE | spin=$SPIN ==="
    mkdir -p "$OUTPUT_DIR"

    # -----------------------
    # Non-AIMNet MLIPs
    # -----------------------
    conda activate /depot/bsavoie/apps/anaconda3/envs/matai

    for MLIP in "${MLIPS[@]}"; do
        echo "   → Running MLIP: $MLIP"
        python gen_bond_scan.py \
            --mlip "$MLIP" \
            --folder "$FOLDER" \
            --charge "$CHARGE" \
            --spin "$SPIN"
    done

    # -----------------------
    # AIMNet2
    # -----------------------
    conda activate /depot/bsavoie/apps/anaconda3/envs/AIMNET

    echo "   → Running MLIP: aimnet2"
    python gen_bond_scan.py \
        --mlip aimnet2 \
        --folder "$FOLDER" \
        --charge "$CHARGE" \
        --spin "$SPIN"

    # -----------------------
    # MOVE RESULTS
    # -----------------------
    echo "   → Moving results → $OUTPUT_DIR"
    mv energies_*.txt "$OUTPUT_DIR"
    echo "=== Completed: $FOLDER ==="
}

# Main loop
for FOLDER in "${FOLDERS[@]}"; do
    for i in "${!CHARGES[@]}"; do
        run_section "$FOLDER" "${CHARGES[$i]}" "${SPINS[$i]}"
    done
done
