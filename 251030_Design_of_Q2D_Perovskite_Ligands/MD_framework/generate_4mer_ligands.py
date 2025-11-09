import os, sys, shutil, math, argparse, subprocess
import openbabel
from openbabel import pybel
from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import Draw

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

def main(*args):
    print("python " + "  ".join(sys.argv))
    print("-" * 80 + '\n')
    sys.stdout = Tee(sys.stdout, open('generate_all_ligands.log', 'w'))
    
    # 16 functional groups, rearranged to match the ordering from previous data
    fg_candidates = [
       "[Na]C1=CC=C([Cs])C=C1[K]",
       "[Na]C1=C([K])C=C([Cs])O1",
       "[Na]C1=C([K])C=C([Cs])S1",
       "[Na]C1=NC([K])=C([Cs])S1",
       "[Na]C1=NC([K])=C([Cs])N1",
       "[Cs]C1=NC=C([Na])C([K])=C1",
       "[Cs]C1=C([K])N=C([Na])O1",
       "[Na]C1=C([K])C(SC([Cs])=C2)=C2S1",
       "[Na]C1=C([K])C(C=C([Cs])S2)=C2S1",
       "[Na]C1=C([K])C=C([Cs])C2=NSN=C21",
       "[Cs]C1=CC(C([K])C2=C3C=CC([Na])=C2)=C3C=C1",
       "[Na]C1=CC(C=C(OC([Cs])=C2)C2=C3[K])=C3O1",
       "[Na]C1=CC(C=C(SC([Cs])=C2)C2=C3[K])=C3S1",
       "[Cs]C1=CC2=C(C(C=C(C([K])C3=C4SC([Na])=C3)C4=C5)=C5C2)S1",
       "O=C(NC([Na])=C12)C1=C([Cs])NC2=O",
       "O=C(N1)C2=C([Na])SC([Cs])=C2C1=O"
    ]

    # Add K atom to the last two functional groups
    fg_candidates[-2] = "O=C(NC([Na])=C12)C1=C([Cs])N([K])C2=O" 
    fg_candidates[-1] = "O=C(N1([K]))C2=C([Na])SC([Cs])=C2C1=O"
    
    # Reorder functional groups
    mask_fg = [0, 1, 2, 7, 8, 15, 14, 9, 11, 12, 13, 10, 5, 6, 3, 4]
    fg_candidates = [fg_candidates[i] for i in mask_fg]
    
    print("Functional groups (16 total):")
    for i, fg in enumerate(fg_candidates):
        print(f"  {i+1}: {fg}")
    print()
    
    fg_common_name = [1, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18]
    
    # Anchor groups
    anchor_common_name = ["1C", "2C", "2C-O", "3C-O", "3C-conj", "3C"]
    anchor_candidates = ["[NH3+]C[*]", "[NH3+]CC[*]", "[NH3+]CCC[*]", 
                        "[NH3+]C/C=C/[*]", "[NH3+]CCO[*]", "[NH3+]CCCO[*]"]
    
    # Reorder anchors
    mask_anchor = [0, 1, 4, 5, 3, 2]
    anchor_candidates = [anchor_candidates[i] for i in mask_anchor]
    
    print("Anchors (6 total):")
    for i, (name, smiles) in enumerate(zip(anchor_common_name, anchor_candidates)):
        print(f"  {name}: {smiles}")
    print()
    
    # Side chains
    side_chain_candidates = ["*[H]", "*C", "*CC", "*F", "*OC", "*C#N"]
    sc_common_name = ["H", "Me", "Et", "F", "OMe", "CN"]
    
    print("Side chains (6 total):")
    for i, (name, smiles) in enumerate(zip(sc_common_name, side_chain_candidates)):
        print(f"  {name}: {smiles}")
    print()
    
    # Create output directory
    output_dir = "all_ligands_generated"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
    print()
    
    # Calculate total number of ligands
    total_ligands = len(anchor_candidates) * len(fg_candidates) * len(fg_candidates) * len(side_chain_candidates)
    print(f"Total ligands to generate: {total_ligands}")
    print(f"  = {len(anchor_candidates)} anchors × {len(fg_candidates)} FG_A × {len(fg_candidates)} FG_B × {len(side_chain_candidates)} side chains")
    print("-" * 80 + '\n')
    
    # Counter for generated ligands
    count = 0
    failed = 0
    
    # Iterate through all combinations
    for anchor_idx, (anchor_name, anchor_smiles) in enumerate(zip(anchor_common_name, anchor_candidates)):
        print(f"\n{'='*80}")
        print(f"Processing anchor: {anchor_name} ({anchor_idx+1}/{len(anchor_candidates)})")
        print(f"{'='*80}\n")
        
        # Create subdirectory for this anchor
        anchor_dir = os.path.join(output_dir, f"{anchor_name}_ready")
        if not os.path.exists(anchor_dir):
            os.makedirs(anchor_dir)
        
        for fg_A_idx, fg_A_name in enumerate(fg_common_name):
            fg_A_smiles = fg_candidates[fg_A_idx]
            
            for fg_B_idx, fg_B_name in enumerate(fg_common_name):
                fg_B_smiles = fg_candidates[fg_B_idx]
                
                for sc_idx, sc_name in enumerate(sc_common_name):
                    sc_smiles = side_chain_candidates[sc_idx]
                    
                    # Generate ligand name
                    ligand_name = f"NH3_{anchor_name}_Angew-{fg_A_name}+H-Angew-{fg_B_name}+{sc_name}_minimized.xyz"
                    ligand_path = os.path.join(anchor_dir, ligand_name)
                    
                    # Skip if already exists
                    if os.path.exists(ligand_path):
                        count += 1
                        if count % 100 == 0:
                            print(f"Progress: {count}/{total_ligands} ({100*count/total_ligands:.1f}%) - Skipped (already exists): {ligand_name}")
                        continue
                    
                    try:
                        # Combine molecules
                        temp_mol, temp_mol_smi = _combine_mol(
                            Chem.MolFromSmiles(anchor_smiles),
                            Chem.MolFromSmiles(fg_A_smiles),
                            Chem.MolFromSmiles(fg_B_smiles),
                            Chem.MolFromSmiles(sc_smiles)
                        )
                        
                        # Convert to XYZ with optimization
                        smi2xyz(temp_mol_smi, ligand_path, opt_option=True)
                        
                        count += 1
                        if count % 100 == 0:
                            print(f"Progress: {count}/{total_ligands} ({100*count/total_ligands:.1f}%) - Generated: {ligand_name}")
                    
                    except Exception as e:
                        failed += 1
                        print(f"ERROR: Failed to generate {ligand_name}")
                        print(f"       {str(e)}")
                        continue
    
    print("\n" + "="*80)
    print("GENERATION COMPLETE")
    print("="*80)
    print(f"Total ligands attempted: {total_ligands}")
    print(f"Successfully generated: {count - failed}")
    print(f"Skipped (already exist): {count - (count - failed) - failed}")
    print(f"Failed: {failed}")
    print("="*80)
    
    return 


def _combine_mol(anchor, ligand_A, ligand_B, side_chain):
    """
    Combine anchor, two functional groups (ligand_A, ligand_B), and side chain to form a 4mer ligand.
    
    Uses marker atoms:
    - Na (11): connection points on functional groups
    - K (19): side chain attachment points  
    - Cs (55): anchor attachment points
    - * (0): wildcard attachment points on anchor and side chains
    """
    
    # Step 1: Remove side chain tag atom (K) on ligand A
    clean_ligand_A = Chem.RWMol(ligand_A)
    for at in ligand_A.GetAtoms():
        if at.GetAtomicNum() == 19:  # K
            clean_ligand_A.RemoveAtom(at.GetIdx())
            break
    
    product = Chem.RWMol(clean_ligand_A)
    
    # Step 2: Add anchor to the product
    # Find attachment point on anchor (*)
    attach_idx = None
    rgroup = Chem.RWMol(anchor)
    
    for rg_at in rgroup.GetAtoms():
        if rg_at.GetAtomicNum() == 0:  # *
            attach_idx = rg_at.GetNeighbors()[0].GetIdx()
            if rg_at.GetIdx() < attach_idx:
                attach_idx -= 1
            rgroup.RemoveAtom(rg_at.GetIdx())
            break
    
    if attach_idx is None:
        raise ValueError("Invalid anchor: no attachment point found")
    
    # Connect anchor to ligand A (via Cs)
    prev_atom_count = product.GetNumAtoms()
    product = Chem.RWMol(Chem.CombineMols(product, rgroup))
    
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 55:  # Cs
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
            break
    
    # Step 3: Attach ligand B (first occurrence)
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 11:  # Na
            # Find attachment point on ligand B (Cs)
            attach_idx = None
            rgroup = Chem.RWMol(ligand_B)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 55:  # Cs
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            
            if attach_idx is None:
                raise ValueError("Invalid ligand_B: no attachment point found")
            
            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
            break
    
    # Step 4: Attach side chain (first occurrence)
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 19:  # K
            attach_idx = None
            rgroup = Chem.RWMol(side_chain)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:  # *
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            
            if attach_idx is None:
                raise ValueError("Invalid side chain: no attachment point found")
            
            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
            break
    
    # Step 5: Attach ligand A again (second occurrence)
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 11:  # Na
            attach_idx = None
            rgroup = Chem.RWMol(clean_ligand_A)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 55:  # Cs
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            
            if attach_idx is None:
                raise ValueError("Invalid clean_ligand_A: no attachment point found")
            
            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
            break
    
    # Step 6: Attach ligand B again (second occurrence)
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 11:  # Na
            attach_idx = None
            rgroup = Chem.RWMol(ligand_B)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 11:  # Na
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            
            if attach_idx is None:
                raise ValueError("Invalid ligand_B (second): no attachment point found")
            
            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
            break
    
    # Step 7: Attach side chain again (second occurrence)
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 19:  # K
            attach_idx = None
            rgroup = Chem.RWMol(side_chain)
            
            for rg_at in rgroup.GetAtoms():
                if rg_at.GetAtomicNum() == 0:  # *
                    attach_idx = rg_at.GetNeighbors()[0].GetIdx()
                    if rg_at.GetIdx() < attach_idx:
                        attach_idx -= 1
                    rgroup.RemoveAtom(rg_at.GetIdx())
                    break
            
            if attach_idx is None:
                raise ValueError("Invalid side chain (second): no attachment point found")
            
            prev_atom_count = product.GetNumAtoms()
            product = Chem.RWMol(Chem.CombineMols(product, rgroup))
            product.AddBond(at.GetNeighbors()[0].GetIdx(), attach_idx + prev_atom_count)
            product.RemoveAtom(at.GetIdx())
            break
    
    # Step 8: Remove remaining Cs marker on second ligand B
    product_ori = deepcopy(product)
    for at in product_ori.GetAtoms():
        if at.GetAtomicNum() == 55:  # Cs
            product.RemoveAtom(at.GetIdx())
            break
    
    product_smi = Chem.MolToSmiles(product)
    product_smi = product_smi.replace('~', '-')
    
    return product, product_smi


def smi2xyz(smiles, xyzname, opt_option=True):
    """
    Convert SMILES string to XYZ file.
    
    Args:
        smiles: SMILES string representation of molecule
        xyzname: output XYZ filename
        opt_option: if True, perform geometry optimization
    """
    if opt_option:
        # Read SMILES, add hydrogens, and minimize structure
        mol = pybel.readstring("smi", smiles)
        pybel._builder.Build(mol.OBMol)
        mol.addh()
        globalopt(mol)
        
        # Write XYZ file
        xyz = pybel.Outputfile("xyz", xyzname, overwrite=True)
        xyz.write(mol)
        xyz.close()
    else:
        # Quick conversion without optimization
        subprocess.call(f'obabel -:"{smiles}" -oxyz -O {xyzname} --gen3d', shell=True)
    
    return


def globalopt(mol, debug=False, fast=False):
    """
    Perform UFF/MMFF94-based geometry optimization using OpenBabel.
    
    Args:
        mol: pybel molecule object
        debug: if True, output intermediate structures
        fast: if True, use faster but less thorough optimization
    """
    # Initialize forcefield (prefer MMFF94, fallback to UFF)
    ff = pybel._forcefields["mmff94"]
    success = ff.Setup(mol.OBMol)
    if not success:
        ff = pybel._forcefields["uff"]
        success = ff.Setup(mol.OBMol)
        if not success:
            raise RuntimeError("Cannot set up forcefield")
    
    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "1.sdf", overwrite=True)
    
    # Initial structure minimization
    if fast:
        ff.SteepestDescent(50, 1.0e-3)
    else:
        ff.SteepestDescent(500, 1.0e-4)
    
    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "2.sdf", overwrite=True)
    
    # Find lowest-energy conformer
    if fast:
        ff.WeightedRotorSearch(20, 5)
    else:
        ff.WeightedRotorSearch(100, 20)
    
    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "3.sdf", overwrite=True)
    
    # Final minimization
    if fast:
        ff.ConjugateGradients(50, 1.0e-4)
    else:
        ff.ConjugateGradients(500, 1.0e-6)
    
    if debug:
        ff.GetCoordinates(mol.OBMol)
        mol.write("sdf", "4.sdf", overwrite=True)
    
    return


if __name__ == "__main__":
    main(*sys.argv[1:])
