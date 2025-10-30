## Code and Examples for the "Design of Hybrid Quasi-2D Perovskite Ligands to Improve Stability"

This directory contains the code and examples used in the research article titled "Design of Hybrid Quasi-2D Perovskite Ligands to Improve Stability".

The force field dataset for all ligands used in this work can be found in zenodo at: [10.5281/zenodo.17479019](https://doi.org/10.5281/zenodo.17479019)

## How to Use This Code
1. Clone the repository to your local machine:
   ```
   git clone <repository_url>
   ```
2. If start from scratch, the "generate_xxx_ligands.py" scripts can be used to generate the ligand structures. Modify the parameters in the scripts as needed. The step0.prepare_ligand_batch.py script can be use to generate ligand classical MD FF parameters.

3. To run the MD simulations, 
   - Make sure the your_ligand_name.xyz and your_ligand_name.db files are in the same directory as the MD input scripts, as well as the lib & util.
   - Use the ```step1.build_bulk_sims_quasi2D.py``` to generate the MD simulation input files.
   - Use the  ```step2.evaluate_octahedra_quasi2D.py``` to analyze the MD simulation results.
   - Use the ```step2.5.examine_stability.py``` to save the structural stability summaries.

4. Modify the parameters in the scripts as needed to fit your specific ligands and simulation conditions.
   