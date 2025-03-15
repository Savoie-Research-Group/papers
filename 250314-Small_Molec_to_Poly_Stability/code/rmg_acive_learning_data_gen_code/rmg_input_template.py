# Data sources
database(
    thermoLibraries = ['primaryThermoLibrary'],
    reactionLibraries = [],
    seedMechanisms = [],
    transportLibraries=[],
    kineticsDepositories = ['training'],
    kineticsFamilies = ['1+2_Cycloaddition',
                        '1,2-Birad_to_alkene',
                        '1,2_Insertion_CO',
                        '1,2_Insertion_carbene',
                        '1,3_Insertion_CO2',
                        '1,3_Insertion_ROR',
                        '1,4_Cyclic_birad_scission',
                        '1,4_Linear_birad_scission',
                        '2+2_cycloaddition',
                        'Birad_recombination',
                        'CO_Disproportionation',
                        'Birad_R_Recombination',
                        'Cyclic_Ether_Formation',
                        'Diels_alder_addition',
                        'Diels_alder_addition_Aromatic',
                        'Disproportionation',
                        'HO2_Elimination_from_PeroxyRadical',
                        'H_Abstraction',
                        'Intra_Retro_Diels_alder_bicyclic',
                        'Intra_Disproportionation',
                        'Intra_R_Add_Endocyclic',
                        'Intra_R_Add_Exocyclic',
                        'R_Addition_COm',
                        'R_Addition_MultipleBond',
                        'R_Recombination',
                        'intra_H_migration',
                        'intra_NO2_ONO_conversion',
                        'intra_OH_migration',
                        '1,3_sigmatropic_rearrangement',
                        'Singlet_Carbene_Intra_Disproportionation',
                        'Singlet_Val6_to_triplet',
                        'Intra_5_membered_conjugated_C=C_C=C_addition',
                        'Intra_Diels_alder_monocyclic',
                        'Concerted_Intra_Diels_alder_monocyclic_1,2_shiftH',
                        'Intra_2+2_cycloaddition_Cd',
                        'Intra_ene_reaction',
                        'Cyclopentadiene_scission',
                        '6_membered_central_C-C_shift',
                        'Intra_R_Add_Exo_scission',
                        '1,2_shiftC',
                        '1,2_NH3_elimination',
                        '1,3_NH3_elimination',
                        'Retroene',
                        'Ketoenol',
                        'Cl_Abstraction',
                        'F_Abstraction',
                        'Disproportionation-Y',
                        'XY_Addition_MultipleBond',
                        '1,2_XY_interchange',
                        'halocarbene_recombination',
                        'halocarbene_recombination_double',
                        'XY_elimination_hydroxyl',
                        'intra_halogen_migration'],
    kineticsEstimator = 'rate rules',
)

# List of species
species(
    label='<smi>',
    reactive=True,
    structure=SMILES('<smi>'),
)

# Reaction systems
simpleReactor(
    temperature=(1350,'K'),
    pressure=(1.0,'bar'),
    initialMoleFractions={
        '<smi>': 1.0,
    },
    terminationConversion={
        '<smi>': 0.5,
    },
    terminationTime=(1e8,'s'),
)

simulator(
    atol=1e-16,
    rtol=1e-8,
)

model(
    toleranceKeepInEdge=0.005,
    toleranceMoveToCore=0.05,
    toleranceInterruptSimulation=1e8,
    maximumEdgeSpecies=100000,
    minCoreSizeForPrune=50,  # default 50
    minSpeciesExistIterationsForPrune=2,  # default 2
    filterReactions=True,
    filterThreshold=1e8, # default 1e8
)

generatedSpeciesConstraints(
    allowed=['input species', 'seed mechanisms', 'reaction libraries'],
    maximumCarbonAtoms='<smi>'.lower().count('c') + 2  # heuristic for hydrocarbons
)

options(
    units='si',
    generateOutputHTML=False,
    generatePlots=False,
    saveEdgeSpecies=False,
    saveSimulationProfiles=False,
)
