# Intralayer-Bidentate-Diammoniums-for-Stable-Two-Dimensional-Perovskites

## Project Overview
This repository contains computational data and analysis code for the study of intralayer bidentate diammonium organic cations in two-dimensional (2D) perovskites. 

## Directory Structure

### Ligand Systems
- **`2P/`** - Simulations for 2-phenylethylammonium (2P) ligand system
- **`4PmDA/`** - Simulations for 4PmDA ligand system  
- **`MeX/`** - Simulations for MeX ligand system

### Data and Analysis
Each ligand directory contains:
- **`300K/`** - Simulation input files and results at 300K (trajectories not included)
- **`*_read_traj_def_remove.py`** - Trajectory analysis scripts for ligand detachment detection
- **`extract_fe.py`** or **`extract_energy.py`** - Free energy extraction and analysis scripts

## Methodology

### Steered Molecular Dynamics (SMD)
- Simulations performed at 300K using steered MD to calculate binding free energies
- Ligand detachment trajectories analyzed to understand dissociation mechanisms
- Force-extension profiles used to determine thermodynamic properties

## Key Results
- Binding free energy calculations for different diammonium ligands
- Comparison of ligand stability in 2D perovskite frameworks
- Validation of computational methods against experimental data

## Notes
- Trajectory files are not included in this repository due to large file sizes
- Contact authors for access to full trajectory data if needed
- All simulations were performed on high-performance computing clusters

## Citation
[Paper citation to be added upon publication]

## Contact
Zhichen Nian (znian@nd.edu)
PI: Professor Brett Savoie (bsavoie2@nd.edu)

---

*Repository maintained by the Savoie Research Group*