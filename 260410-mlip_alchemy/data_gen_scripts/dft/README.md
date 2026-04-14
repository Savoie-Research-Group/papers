# Laying out diatomic calculation requirements

Systems:
- Cations: H, Li, Na, K
- Anions: F, Cl, Br, I
- Cations_2 at neutral and +2 charge
- Anions_2 at neutral and -2 charge
- Cation_Anion at neutral, +2 charge, and -2 charge
- 24 neutral systems
- 40 charged systems
- **All systems should have singlet spin multiplicity!**

DFT scans:
- Start at 8.0 angstroms, scan until 0.5 angstroms, interval of 0.1 angstroms
- Level of theory: wB97M-V/def2-TZVPD (to match OMol25)
- Unrestricted and restricted DFT

Dissociation plot reference energy (0 eV):
- Neutral radicals (doublet) for all 8 elements
- Computed at multiple levels of theory
    - wB97M-V/def2-TZVPD
    - wB97M-D3BJ/def2-TZVPP
- Unrestricted DFT reference is required

CASPT2 spot-checks
- Done for neutral F2 and NaCl, so we can show uDFT/rDFT vs CASPT2
    - Homolytic cleavage: F2
    - Heterolytic cleavage: NaCl
- Used def2-TZVPD basis set, so that we can compare against wB97M-V/def2-TZVPD
- Start at 12.0 angstroms, scan until 0.8 angstroms, interval of 0.1 angstroms
- Use 12.0 angstrom energy as the reference "isolated species" energy
