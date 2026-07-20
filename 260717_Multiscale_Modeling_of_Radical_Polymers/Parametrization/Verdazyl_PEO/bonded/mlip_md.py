import os
import argparse
import torch
import torch._dynamo

from ase import units
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md import MDLogger
from ase.io import Trajectory #BY
from ase.build import molecule #BY

from fairchem.core import pretrained_mlip, FAIRChemCalculator

#from fairchem.models import load_model_checkpoint #BY

import logging
import numpy as np

torch._dynamo.config.suppress_errors = True

def infer_cell(atoms, padding=2.0):
    if atoms.get_cell().volume < 1e-3:
        pos = atoms.get_positions()
        mins = pos.min(axis=0)
        maxs = pos.max(axis=0)
        cell = maxs - mins + 2 * padding
        atoms.translate(-mins + padding)
        atoms.set_cell(cell)
        atoms.set_pbc([True, True, True])
    return atoms

def setup_calc(model, device):#BY
    if model == "fair":
        predictor=pretrained_mlip.get_predict_unit("uma-s-1", device=device.type)
        return FAIRChemCalculator(predictor, task_name="omol")
    else:
        raise ValueError(f"Unsupported model: {model}")
    
def write_lammpstrj(atoms, step, filename="trajectory.lammpstrj"):#BY
    unique_elements = sorted(set(atoms.get_chemical_symbols()))
    type_map = {element: i+1 for i, element in enumerate(unique_elements)}

    with open(filename, "a") as f:
        f.write(f"ITEM: TIMESTEP\n{step}\n")
        f.write(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")
        cell = atoms.get_cell()
        for i in range(3):
            f.write(f"0.0 {cell[i, i]:.6f}\n")
        f.write("ITEM: ATOMS id type x y z\n")
        for i, (symbol, pos) in enumerate(zip(atoms.get_chemical_symbols(), atoms.get_positions())):
            atom_type = type_map[symbol]
            f.write(f"{i+1} {atom_type} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyz_file")
    parser.add_argument("--model", choices=["orb", "fair", "mace"], default="fair")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--temp", type=float, default=298)
    parser.add_argument("--timestep_fs", type=float, default=1.0)
    parser.add_argument("--output", default="md_out.xyz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atoms = read(args.xyz_file)
    atoms = infer_cell(atoms)

    atoms.calc = setup_calc(args.model, device)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temp)

    dyn = Langevin(atoms, args.timestep_fs * units.fs, temperature_K=args.temp, friction=0.01 / units.fs)
    #dyn.attach(lambda: write(args.output, atoms.copy(), append=True), interval=10) (to get multiple frame xyz)
    dyn.attach(lambda: write_lammpstrj(atoms, dyn.nsteps), interval=1)#BY to get lammpstrj file too.
    dyn.attach(MDLogger(dyn, atoms, "md.log"), interval=1) #interval=10 means save every 10 steps (every 10fs) for energies, T ..
    dyn.run(args.steps)
    trajectory = Trajectory("my_md.traj", "w", atoms) #BY
    dyn.attach(trajectory.write, interval=1) #BY
    write(args.output, atoms.copy())  # This writes only the final frame to XYZ (to get final frame xyz)

if __name__ == "__main__":
    main()

