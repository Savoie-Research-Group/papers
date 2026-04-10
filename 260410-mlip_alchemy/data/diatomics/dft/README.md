# Laying out diatomic calculation requirements

Systems:
- Cations: H, Li, Na, K
- Anions: F, Cl, Br, I
- Cations_2 at neutral and +2 charge (`generate_dication.py`)
    - H2 at +2 charge failed entirely, as ORCA refuses to do anything without electrons
- Anions_2 at neutral and -2 charge (`generate_dianion.py`)
- Cation_Anion at neutral, +2 charge, and -2 charge (`generate_cross.py`)
- 24 neutral systems
- 40 charged systems
- **All systems are computed with singlet spin multiplicity!**
- Start at 8.0 angstroms, scan until 0.5 angstroms, interval of 0.1 angstroms
- Level of theory: wB97M-V/def2-TZVPD (to match OMol25)
- Unrestricted and restricted DFT
- Total number of calculations: 2 x (24 + 40) = 128
- Total number of scans collected: 126 (missing H2 at +2 results for RKS and UKS)
    - Data collected in `dft_scan_energies.csv`

Dissociation plot reference energy (0 eV):
- Neutral radicals (doublet) for all 8 elements
- Computed at multiple levels of theory (`generate_atoms.py`)
    - wB97M-V/def2-TZVPD
    - wB97M-D3BJ/def2-TZVPP
- Unrestricted DFT reference is required for doublets
    - I didn't specify it in the input file, because ORCA is smart enough to switch to UKS in this instance
- 16 single point energies collected
    - Data collected in `atom_reference_spe.csv`