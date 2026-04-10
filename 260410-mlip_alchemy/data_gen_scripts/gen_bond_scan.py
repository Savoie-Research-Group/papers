#!/usr/bin/env python3
# =============================================================================
# File: gen_bond_scan.py
# Author: Vignesh Sathyaseelan
# Email: vsathyas@purdue.edu
# Description:
#     Evaluate MLIP (machine-learned interatomic potential) energies for all
#     XYZ structures inside a specified folder. Supports AIMNet2, MACE,
#     ORB, and FAIR-Chem models through ASE-compatible calculators.
#
# =============================================================================

import argparse
import os
import glob
import numpy as np
from ase.io import read


def get_mlip(name, device="cpu", charge=0, spin=None):
    """
    Return an ASE-compatible calculator for the requested MLIP model.
    """
    name = name.lower()

    # -------------------
    # AIMNet2
    # -------------------
    if name == "aimnet2":
        from aimnet2calc import AIMNet2ASE
        return AIMNet2ASE("aimnet2", charge=charge, mult=spin)

    # -------------------
    # MACE family
    # -------------------
    from mace.calculators import mace_mp, mace_omol

    mace_models = {
        "mace-large": "large",
        "mace-medium": "medium",
        "mace-matpes-r2scan-0": "mace-matpes-r2scan-0",
    }

    if name in mace_models:
        return mace_mp(
            model=mace_models[name],
            dispersion=True,
            default_dtype="float64",
            device=device,
        )

    if name == "mace-omol":
        return mace_omol(model="extra_large", device=device)

    # -------------------
    # ORB
    # -------------------
    if name == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device, precision="float32-high"
        )
        return ORBCalculator(orbff, device=device)

    if name == "orb-omol":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        orbff = pretrained.orb_v3_conservative_omol(
            device=device, precision="float64"
        )
        return ORBCalculator(orbff, device=device)

    # -------------------
    # FAIRChem
    # -------------------
    if name in ("fairchem-omat", "fairchem-omol"):
        from fairchem.core import FAIRChemCalculator, pretrained_mlip
        predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
        task = "omat" if name == "fairchem-omat" else "omol"
        return FAIRChemCalculator(predictor, task_name=task)

    raise ValueError(f"Unsupported MLIP model: {name}")


def evaluate_folder(folder, mlip, charge=0, spin=None, mode=None):
    """
    Read all .xyz files, compute energies, return list of (filename, energy).
    """
    xyz_files = sorted(glob.glob(os.path.join(folder, "*.xyz")))
    results = []

    for xyz_file in xyz_files:
        atoms = read(xyz_file)

        if charge != 0 or spin is not None:
            atoms.info.update({"charge": charge, "spin": spin})

        atoms.calc = mlip

        if mode == "aiment2":
            atoms.calc.set_charge(charge)
            atoms.calc.set_mult(spin)

        energy = atoms.get_potential_energy()

        if isinstance(energy, np.ndarray):
            energy = float(np.sum(energy))

        results.append((os.path.basename(xyz_file), energy))

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute MLIP energies for all XYZ files in a folder."
    )
    parser.add_argument(
        "--mlip",
        type=str,
        required=True,
        help=(
            "MLIP model name: aimnet2, mace-large, mace-medium, "
            "mace-matpes-r2scan-0, mace-omol, orb, orb-omol, "
            "fairchem-omat, fairchem-omol"
        ),
    )
    parser.add_argument("--folder", type=str, required=True, help="Folder containing .xyz files")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--charge", type=int, default=0, help="Total molecular charge")
    parser.add_argument("--spin", type=int, default=1, help="Spin multiplicity (2S+1)")
    args = parser.parse_args()

    mlip = get_mlip(
        args.mlip,
        device=args.device,
        charge=args.charge,
        spin=args.spin,
    )

    is_aimnet2 = args.mlip.lower() == "aimnet2"
    results = evaluate_folder(
        args.folder,
        mlip,
        charge=args.charge,
        spin=args.spin,
        mode="aiment2" if is_aimnet2 else None,
    )

    output_file = f"energies_{args.mlip}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# File\tEnergy (eV)\n")
        for fname, energy in results:
            f.write(f"{fname}\t{energy:.10f}\n")

    print(f"Saved energies to {output_file}")


if __name__ == "__main__":
    main()
