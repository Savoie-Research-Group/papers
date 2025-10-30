#!/bin/env python
# author: Stephen Shiring
import argparse
import math
import sys
import os
import numpy
import random
import datetime
from copy import deepcopy
from scipy.spatial.distance import cdist

# Add TAFFI Lib to path
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2]) + '/lib')
sys.path.append('/Users/nianz/Desktop/Research/high-throughput-2D-Perovskite-ligand-screening-tool/MD/lib')
import adjacency

import id_types


def main(argv):
    parser = argparse.ArgumentParser(description='Generate a perovskite crystal structure.')

    # positional arguments
    parser.add_argument('cation', type=str,
                        help='Identity of element A / cation. Specify a single atom or an *.xyz file for multiple atoms / small molecule.')

    parser.add_argument('metal', type=str,
                        help='Identity of element B / metal cation.')

    parser.add_argument('anion', type=str,
                        help='Identity of element X / anion. Also accepts a space-delimited string of 2 atoms to generate a heterojunction.')

    parser.add_argument('bond_length', type=float,
                        help='"Bond length" between metal-halide. Metal-metal distance is twice this (so unit cell parameter is inputed as half).')

    # optional arguments
    parser.add_argument('-o', dest='output', type=str, default='perovskite',
                        help='Name of output files. Default = \'perovskite\'')

    parser.add_argument('-dims', dest='dims', default="0 0 0",
                        help='This is supplied as a space-delimited string of integers. (default: "0 0 0")')

    parser.add_argument('-q', dest='charges', default="1.00 2.00 -1.00",
                        help='Space-delimited string specifying the charge for the cation, metal, and anion. Cation will also be applied to surface cation; if specified, can override charge normalization to 1 for surface cation (values must match exactly). Default: "1.00 2.00 -1.00" ')

    parser.add_argument('-surface', dest='surface_cation', type=str, default=None,
                        help='Identity of single atom for surface cation (placed +/- z). Also accepts a *.xyz file for multiple atoms / small molecule. First atom must be head group, second atom is tail atom to define long axis.')

    parser.add_argument('-SO', dest='surface_cation_orientation', type=str, default='0.0 45.0 0.0',
                        help='Space-delimited string specifying the tilt, sweep, and orientation angles for surface cation insertion. All three must be specified. Default: "0.0 45.0 0.0"')

    parser.add_argument('-FF', dest='FF_db', type=str, default='/home/sshiring/bin/taffi/Data/TAFFI.db',
                        help='Specify path to force field database. Default: /home/sshiring/bin/taffi/Data/TAFFI.db')

    parser.add_argument('-surface_FF', dest='surface_FF_db', type=str, default='/home/sshiring/bin/taffi/Data/TAFFI.db',
                        help='Specify path to surface cation force field database. Default: /home/sshiring/bin/taffi/Data/TAFFI.db')

    parser.add_argument('--hydrate', dest='hydrate', action='store_const', const=True, default=False,
                        help='When invoked, hydrates the perovskite. Default: Off. When off, head atoms of surface cations should be tethered to their initial position. When off, an output containing the LAMMPS fix with proper atom IDs will be written.')

    parser.add_argument('-headspace', dest='headspace', type=float, default=10.0,
                        help='Length of y axis to hydrate. Total volume will be x and z from box volume. Both sides of box will be hydrated. Default: 10.0 A')

    parser.add_argument('-s', dest='spacer', type=float, default=1.5,
                        help='Set the spacer distance between the water surfaces and perovksite surfaces after box is rescaled for proper density. If set to -1, then the water step size is used. Default: 1.5 A.')

    # For the edge of perovskite cristal?
    parser.add_argument('--bottom', dest='bottom', action='store_const', const=True, default=False,
                        help='When invoked, close off bottom layer of the perovskite. Default: Off.')

    parser.add_argument('--monolayer', dest='monolayer', action='store_const', const=True, default=False,
                        help='When invoked, the geometry will be that of a monolayer (i.e y will be set to 0, regardless of value specified in dimensions and cation will be ignored. A surface cation must be specified.). Default: Off.')

    parser.add_argument('--quasi_2D', dest='quasi_2D', action='store_const', const=True, default=False, 
                        help='When invoked, the geometry will be that of a quasi-2D perovskite(i.e there must be surface cation and surface FF provided as well')

    parser.add_argument('--water_pairs', dest='water_pairs', action='store_const', const=True, default=False,
                        help='When invoked, write out pairs for addition of a single water molecule. Default: Off.')

    parser.add_argument('-metal_vacancy', dest='metal_vacancy', type=int, default=0,
                        help='Number of metal vacancies to introduce. Default: 0.')

    parser.add_argument('-anion_vacancy', dest='anion_vacancy', type=int, default=0,
                        help='Number of anion vacancies to introduce. Default: 0.')

    parser.add_argument('-SC_vacancy', dest='surface_cation_vacancy', type=int, default=0,
                        help='Number of surface cation vacancies to introduce when building a monolayer. Default: 0.')

    parser.add_argument('-mixing_rule', dest='mixing_rule', type=str, default='none',
                        help='Define the mixing rule to be used for misisng LJ parameters. Waldman-Hagler (wh) and Lorentz-Berthelot (lb) and "none" are valid options. When set to "none", will only read from force field database and exit if there are any missing parameters. default: none')

    parser.add_argument('--UFF_supplement', dest='UFF_supplement', action='store_const', const=True, default=False,
                        help='When invoked, supplement any missing LJ parameters with UFF parameters. Default: Off.')

    parser.add_argument('--print_lj', dest='print_lj', action='store_const', const=True, default=False,
                        help='When invoked, print out the origin of LJ parameters. Default: Off.')

    parser.add_argument('--anywhere', dest='vacancies_anywhere', action='store_const', const=True, default=False,
                        help='When invoked, if placing vacancies in a monolayer, place them anywhere in the perovskite instead of only along the interface. Default: Off (places them only along the interface).')

    parser.add_argument('-y_pad', dest='y_padding', type=float, default=0.0,
                        help='Add this amount to final +/- y dimensions to create a headspace volume in y direction. Default: 0.0 A')

    parser.add_argument('--debug', dest='debug', action='store_const', const=True, default=False,
                        help='When invoked, will output a debug file to assist in identifying surface cation atom types and charges when attempting to match to other force fields. Default: Off.')

    args = parser.parse_args()

    #Process if a folder has been specified
    if os.path.isdir(args.output):
        pass
        # print(
        #     'ERROR: A folder matching the specified output argument already exists. Aborting to avoid overwritten any existing data...')
        # exit()
    else:
        os.mkdir(args.output)

    sys.stdout = Logger(args.output + '/' + args.output)

    print("{}\n\nPROGRAM CALL: python {}\n".format('-' * 150, ' '.join([i for i in sys.argv])))

    # Dictionary to hold options
    options = {}

    # Ensure that three dimensions have been specified; if not, repeat last value intil 3 is reached.
    args.dims = [int(_) for _ in args.dims.split()]
    while len(args.dims) < 3: args.dims += [args.dims[-1]]

    if args.monolayer:
        args.dims[1] = 0

        if args.surface_cation == None:
            print('ERROR: When requesting a monolayer, the surface cation must be specified. Aborting...')
            exit()

        if args.bottom:
            print('ERROR: When requesting a monolayer, the bottom flag must not be invoked. Aborting...')
            exit()

    n_cations = 0 if args.monolayer else 1

    args.charges = [float(_) for _ in args.charges.split()]
    if len(args.charges) < 3:
        print('ERROR: Charges must be specified for the cation, metal, and anion.')
        exit()

    # Why we need to specify the charges for ions? they are treated as atoms but not ions if we don't specify that?
    cation_charge = args.charges[0]
    metal_charge = args.charges[1]
    anion_charge = args.charges[2]
    surface_cation_charge = args.charges[0]

    args.mixing_rule = args.mixing_rule.lower()
    if args.mixing_rule not in ['wh', 'lb', 'none']:
        print(
            'ERROR: Supplied -mixing_rule ({}) not accepted. Only "lb", "wh", or "none" are accepted. Exiting...'.format(
                args.mixing_rule))
        exit()
    if args.mixing_rule == 'none':
        print('WARNING: A mixing rule of "none" has been specified...')

    anion_vacancy_flag = False
    surface_cation_vacancy_flag = False
    metal_vacancy_flag = False
    if args.anion_vacancy > 0: anion_vacancy_flag = True
    if args.surface_cation_vacancy > 0: surface_cation_vacancy_flag = True
    if args.metal_vacancy > 0: metal_vacancy_flag = True

    # Why we need to specify the charges for ions? they are treated as atoms but not ions if we don't specify that?
    #    cation_charge = 1.0 #1.360
    #    metal_charge = 2.00 #2.0909 # integer charge: 2.00 # MAPbI: 2.03
    #    anion_charge = -1.00 #-1.1639 # integer charge: -1.00 #   MAPbI: -1.13
    #    surface_cation_charge = -1.0

    # Process the anion
    build_heterojunction = False
    anions = args.anion.split()

    if len(anions) == 2:
        print('Two anions specified, building heterojunction.')
        build_heterojunction = True
        # args.bottom = True

        if anions[0] not in ['I', 'Br'] and anions[1] not in ['I', 'Br']:
            print(
                'ERROR: Currently only building heterojunctions composed of I and Br anions are supported. Aborting...')
            exit()

        Anion = parse_ion(anions[0], anion_charge, args.FF_db, 'anion')
        Anion_2 = parse_ion(anions[1], anion_charge, args.FF_db, 'anion')

    elif len(anions) == 1:
        if args.anion.lower().endswith('.xyz'):
            print('ERROR: Specifying an *.xyz file for the anion is currently not supported. Exiting...')
            exit()
        Anion = parse_ion(args.anion, anion_charge, args.FF_db, 'anion')

    else:
        print("ERROR: More than 2 anions specified. Exiting....")
        exit()

    # Process the metal
    # If building a heterojunction, because the Pb-Pb parameters are different between Br and I MYP
    # models to account for lattice spacing, need to define 2 different atom types for Pb to account for them.
    if build_heterojunction:
        # Pb_I
        Metal = parse_metal(args.metal, metal_charge)
        Metal["atom_types"] = ['[82[53]]']
        Metal["masses"] = {Metal["atom_types"][0]: get_masses(Metal["elements"])[0]}

        # Pb_Br
        Metal_2 = parse_metal(args.metal, metal_charge)
        Metal_2["atom_types"] = ['[82[35]]']
        Metal_2["masses"] = {Metal_2["atom_types"][0]: get_masses(Metal_2["elements"])[0]}

    else:
        Metal = parse_metal(args.metal, metal_charge)


        if args.anion == 'I':
            Metal["atom_types"] = ['[82[53]]']
            Metal["masses"] = {Metal["atom_types"][0]: get_masses(Metal["elements"])[0]}
        elif args.anion == 'Br':
            Metal["atom_types"] = ['[82[35]]']
            Metal["masses"] = {Metal["atom_types"][0]: get_masses(Metal["elements"])[0]}
        else:
            print('ERROR: Unsupported halide/anion specified. Currently only Br and I are supported. Exiting...')
            exit()

    # Process the cation
    Cation = parse_ion(args.cation, cation_charge, args.FF_db, 'cation')

    # Process the surface cation
    surface_cation = parse_surface_cation(args.surface_cation, surface_cation_charge, args.surface_FF_db, args.debug,
                                          args.output)
    if args.surface_cation != None:
        args.surface_cation_orientation = [float(_) for _ in args.surface_cation_orientation.split()]
        if len(args.surface_cation_orientation) != 3:
            print(
                'ERROR: Surface cation specified but not all surface cation orientation angles specified. Please ensure all three are specified. Exiting...')
            exit()

    # Process water molecule if hydration is requested
    options['TIP4P_flag'] = False
    if args.hydrate:
        print('\nHydration requested, processing water molecule...')

        if not os.path.isfile(args.FF_db):
            print('ERROR: Specified force field file ({}) does not exist. Aborting...'.format(args.FF_db))
            exit()

        # Build water molecule (coordinates taken from: http://www.nyu.edu/classes/tuckerman/adv.chem/lectures/lecture_12/node1.html)
        Water = {}

        Water["count"] = 3
        # Water["geometry"] = numpy.array([[0.0, 0.0, 0.0], [0.7907, 0.6122, 0], [-0.7907, 0.6122, 0]])                      # original coordinates from above
        Water["geometry"] = numpy.array([[0.0, -0.40813333, 0.0], [0.7907, 0.20406667, 0], [-0.7907, 0.20406667,
                                                                                            0]])  # coordinates that have been centered about the origin
        Water["elements"] = ['O', 'H', 'H']
        Water["adj_mat"] = adjacency.Table_generator(Water["elements"], Water["geometry"])
        Water["atom_types"] = id_types.id_types(Water["elements"], Water["adj_mat"])

        Water["bonds"], Water["bond_types"], Water["bond_params"], Water["angles"], Water["angle_types"], Water[
            "angle_params"], Water["dihedrals"], Water["dihedral_types"], Water["dihedral_params"], Water["impropers"], \
        Water["improper_types"], Water["improper_params"], Water["charges"], Water["masses"], Water["VDW_params"], \
        Water["VDW_comments"] = Find_parameters(Water["adj_mat"], Water["geometry"], Water["atom_types"], args.FF_db,
                                                Improper_flag=False)
        Water["mol_box"] = (min(Water["geometry"][:, 0]), max(Water["geometry"][:, 0]), min(Water["geometry"][:, 1]),
                            max(Water["geometry"][:, 1]), -1.0, 1.0)  # Define box circumscribing the water molecule
        Water[
            "step_size"] = 4.69576378072  # Define the step size (the geometric norm of the circumscribing cube) for tiling
        Water["volume"] = (Water["mol_box"][1] - Water["mol_box"][0]) * (Water["mol_box"][3] - Water["mol_box"][2]) * 1

        for key in Water["VDW_params"]:
            if Water["VDW_params"][key][0].lower() == 'tip4p':
                options['TIP4P_flag'] = True
                break

    # Build the unit cell depending on whether or not monolayer is specified.
    # Unit cell is contains atoms identical to the unit cell specifications: A_2 B X_4 for monolayer or A B X_3 for multilayer (where A is cation, B is metal, X is anion)
    print('\nBuilding unit cell...')
    if args.monolayer:
        print('\tConstructing monolayer unit cell (A2BX4)...')
    elif args.quasi_2D:
        print('\tConstructing edgelayer unit cell (BX4)...')
        print('\tConstructing multilayer unit cell (ABX3)...')
    else:
        print('\tConstructing multilayer unit cell (ABX3)...')

    unit_cell = build_unit_cell(Metal, Anion, Cation, args.bond_length, args.monolayer, False, False)
    # if args.quasi_2D:
    unit_cell_no_cation = build_bottom_quasi2D(Metal, Anion, Cation, args.bond_length, False)
        # print(unit_cell_no_cation)
    unit_cell_plusAnion = build_top_quasi2D(Metal, Anion, Cation, args.bond_length, args.quasi_2D, False, False)
        # print(unit_cell_plusAnion)
    if build_heterojunction: unit_cell_2 = build_unit_cell(Metal_2, Anion_2, Cation, args.bond_length, args.monolayer,
                                                           False, False)

    # !!
    # Not completly implemented yet for a heterojunction, most likely to be unneeded.
    # To implement: will need to actually use the bottom_2 unit cell, as of right now it will just place
    # the first unit cell across the entire length of the crystal. So, e.g., I will be underneath Br,
    # instead of Br being underneath Br.
    # !!
    bottom = build_bottom(Metal, Anion, Cation, args.bond_length, build_heterojunction)


    # if build_heterojunction:
    #    bottom_2 = build_bottom(Metal_2, Anion_2, Cation, args.bond_length, build_heterojunction)

    expected_unit_cell_count = (args.dims[0] + 1) * (args.dims[1] + 1) * (args.dims[2] + 1)
    print('\tNumber of unit cells expected: {}'.format(expected_unit_cell_count))
    print('\tNumber of atoms in unit cell ({}): {}'.format('A2BX4' if args.monolayer else 'ABX3', (
                len(unit_cell["elements"]) + 2 * len(surface_cation["elements"])) if args.monolayer else len(
        unit_cell["elements"])))

    print('\nBuilding perovskite...')

    # Calculate the total number of atoms needed...

    # Number of interior atoms
    if args.quasi_2D:
        N_bot = (args.dims[0] + 1) * (args.dims[2] + 1) * len(unit_cell_no_cation['elements'])
        N_top = (args.dims[0] + 1) * (args.dims[2] + 1) * len(unit_cell_plusAnion['elements'])
        N_interior = (args.dims[0] + 1) * (args.dims[1] - 1) * (args.dims[2] + 1) * len(unit_cell["elements"])
    else:
        N_bot = N_top = 0
        N_interior = (args.dims[0] + 1) * (args.dims[1] + 1) * (args.dims[2] + 1) * len(unit_cell["elements"])

    # Number of surface atoms (surface is along the xz planes)
    if args.surface_cation != None:
        N_surface = 2 * ((args.dims[0] + 1) * (args.dims[2] + 1) * len(surface_cation["elements"]))
    else:
        N_surface = 0

    # Number of atoms to place along the bottom to cap the dangling interior unit cell atoms
    if args.bottom:
        N_bottom = (args.dims[0] + 1) * (args.dims[2] + 1) * len(bottom["elements"])
    else:
        N_bottom = 0

    # Total up the number of atoms in the simulation
    N_atoms = N_interior + N_surface + N_bot + N_top + N_bottom - args.anion_vacancy * len(
        Anion["elements"]) - args.surface_cation_vacancy * len(surface_cation["elements"]) - args.metal_vacancy
    N_vacancies = args.anion_vacancy + args.metal_vacancy

    # Draw some random unit cell numbers to place the vacancies in. If we are building a heterojunction, first find the IDs of the unit cells that are along the
    # the interface and draw from this list. Else, just draw a random number from the entire set of unit cell IDs. The indexing starts at 1 since we are counting.
    # Only restriction: a unit cell cannot have both a metal and halide vacancy, only one or the other. If the combined total of vacancies exceeds the available sites,
    # throw an error and exit.
    anion_vacancies_list = []
    anion_vacancies_coords = []
    surface_cation_vacancies_list = []
    metal_vacancies_list = []

    if args.surface_cation_vacancy > (expected_unit_cell_count * 2):
        print(
            '\nERROR: The number of specified surface cation vacancies exceeds the number of surface cations (2 per unit cell). Please reduce the specified number. Exiting...')
        exit()

    if args.anion_vacancy + args.surface_cation_vacancy + args.metal_vacancy > 0:
        if build_heterojunction and not args.vacancies_anywhere:
            interface_ids = []
            i = 1
            x_2 = float((args.dims[0]) / 2)
            for x in range(args.dims[0] + 1):
                for y in range(args.dims[1] + 1):
                    for z in range(args.dims[2] + 1):
                        if x == x_2:
                            interface_ids.append(i)
                        elif x == x_2 + 1:
                            interface_ids.append(i)
                        i += 1

            if N_vacancies > len(interface_ids):
                print(
                    '\nERROR: The combined number of specified metal and halide vacancies exceeds the number of interface sites. Please reduce the specified amount. Aborting...')
                print('NOTE: Each unit cell can only receive one type of vacancy. ')
                exit()

            while len(anion_vacancies_list) < args.anion_vacancy:
                indx = random.choice(interface_ids)
                if indx not in anion_vacancies_list:
                    anion_vacancies_list.append(indx)

            while len(surface_cation_vacancies_list) < args.surface_cation_vacancy:
                indx = random.choice(interface_ids)
                if indx not in surface_cation_vacancies_list:
                    surface_cation_vacancies_list.append(indx)

            while len(metal_vacancies_list) < args.metal_vacancy:
                indx = random.choice(interface_ids)
                if indx not in metal_vacancies_list and indx not in anion_vacancies_list:
                    metal_vacancies_list.append(indx)

        else:
            if N_vacancies > expected_unit_cell_count:
                print(
                    '\nERROR: The combined number of specified metal and halide vacancies exceeds the number of unit cell sites. Please reduce the specified amount. Aborting...')
                print('NOTE: Each unit cell can only receive one type of vacancy. ')
                exit()

            while len(anion_vacancies_list) < args.anion_vacancy:
                indx = random.randint(1, expected_unit_cell_count)
                if indx not in anion_vacancies_list:
                    anion_vacancies_list.append(indx)

            while len(surface_cation_vacancies_list) < args.surface_cation_vacancy:
                indx = random.randint(1, expected_unit_cell_count)
                if indx not in surface_cation_vacancies_list:
                    surface_cation_vacancies_list.append(indx)

            while len(metal_vacancies_list) < args.metal_vacancy:
                indx = random.randint(1, expected_unit_cell_count)
                if indx not in metal_vacancies_list and indx not in anion_vacancies_list:
                    metal_vacancies_list.append(indx)

    if args.anion_vacancy != 0:
        print('\tNumber of anion vacancies specified: {}'.format(args.anion_vacancy))
        print('\tUnit cell IDs selected for anion vacany {}'.format(', '.join([str(_) for _ in anion_vacancies_list])))
        if build_heterojunction and args.vacancies_anywhere:
            print('    \t--anywhere flag set, not restricting vacancies to only along the junction')

    if args.surface_cation_vacancy != 0:
        print('\tNumber of surface cation vacancies specified: {}'.format(args.surface_cation_vacancy))
        print('\tUnit cell IDs selected for surface cation vacany {}'.format(
            ', '.join([str(_) for _ in surface_cation_vacancies_list])))
        if build_heterojunction and args.vacancies_anywhere:
            print('    \t--anywhere flag set, not restricting vacancies to only along the junction')

    if args.metal_vacancy != 0:
        print('\tNumber of metal vacancies specified: {}'.format(args.metal_vacancy))
        print('\tUnit cell IDs selected for metal vacany {}'.format(', '.join([str(_) for _ in metal_vacancies_list])))
        if build_heterojunction and args.vacancies_anywhere:
            print('    \t--anywhere flag set, not restricting vacancies to only along the junction')

    # Empty array for simulation box coordinates, list of elements, and list of surface indices
    # (corresponding to indices in sim_box; to match an *.xyz file,
    # need to be incremented by 1 since the *.xyz file starts counting at 1)
    sim_box = numpy.zeros([N_atoms, 3])
    surface_indices = []
    surface_head_indices = []  # records list of the indices of the head atom (defined as being the first atom in the geometry) in the surface cation (should be N in the example of Letian's molecules)

    sim_data = {}
    sim_data["elements"] = []
    sim_data["atom_types"] = []
    sim_data["adj_mat"] = numpy.zeros([N_atoms, N_atoms])
    sim_data["masses"] = []
    sim_data["charges"] = []

    sim_data["atom_types"] = []
    sim_data["bonds"] = []
    sim_data["bond_types"] = []
    sim_data["angles"] = []
    sim_data["angle_types"] = []
    sim_data["dihedrals"] = []
    sim_data["dihedral_types"] = []
    sim_data["impropers"] = []
    sim_data["improper_types"] = []
    sim_data["molecules"] = []

    sim_data["bond_params"] = {}
    sim_data["bond_params"].update(Anion["bond_params"])
    sim_data["bond_params"].update(Cation["bond_params"])
    sim_data["bond_params"].update(surface_cation["bond_params"])
    if build_heterojunction: sim_data["bond_params"].update(Anion_2["bond_params"])
    if args.hydrate: sim_data["bond_params"].update(Water["bond_params"])

    sim_data["angle_params"] = {}
    sim_data["angle_params"].update(Anion["angle_params"])
    sim_data["angle_params"].update(Cation["angle_params"])
    sim_data["angle_params"].update(surface_cation["angle_params"])
    if build_heterojunction: sim_data["angle_params"].update(Anion_2["angle_params"])
    if args.hydrate: sim_data["angle_params"].update(Water["angle_params"])

    sim_data["dihedral_params"] = {}
    sim_data["dihedral_params"].update(Anion["dihedral_params"])
    sim_data["dihedral_params"].update(Cation["dihedral_params"])
    sim_data["dihedral_params"].update(surface_cation["dihedral_params"])
    if build_heterojunction: sim_data["dihedral_params"].update(Anion_2["dihedral_params"])
    if args.hydrate: sim_data["dihedral_params"].update(Water["dihedral_params"])

    sim_data["improper_params"] = {}
    sim_data["improper_params"].update(Anion["improper_params"])
    sim_data["improper_params"].update(Cation["improper_params"])
    sim_data["improper_params"].update(surface_cation["improper_params"])
    if build_heterojunction: sim_data["improper_params"].update(Anion_2["improper_params"])
    if args.hydrate: sim_data["improper_params"].update(Water["improper_params"])

    sim_data["VDW_params"] = {}
    sim_data["VDW_params"].update(Anion["VDW_params"])
    sim_data["VDW_params"].update(Cation["VDW_params"])
    sim_data["VDW_params"].update(surface_cation["VDW_params"])
    if build_heterojunction: sim_data["VDW_params"].update(Anion_2["VDW_params"])
    if args.hydrate: sim_data["VDW_params"].update(Water["VDW_params"])

    sim_data["VDW_comments"] = {}
    sim_data["VDW_comments"].update(Anion["VDW_comments"])
    sim_data["VDW_comments"].update(Cation["VDW_comments"])
    sim_data["VDW_comments"].update(surface_cation["VDW_comments"])
    if build_heterojunction: sim_data["VDW_comments"].update(Anion_2["VDW_comments"])
    if args.hydrate: sim_data["VDW_comments"].update(Water["VDW_comments"])

    sim_data["all_masses"] = {}
    sim_data["all_masses"].update(Metal["masses"])
    if build_heterojunction: sim_data["all_masses"].update(Metal_2["masses"])
    sim_data["all_masses"].update(Anion["masses"])
    sim_data["all_masses"].update(Cation["masses"])
    sim_data["all_masses"].update(surface_cation["masses"])
    if build_heterojunction: sim_data["all_masses"].update(Anion_2["masses"])
    if args.hydrate: sim_data["all_masses"].update(Water["masses"])

    # Internal tracking
    atoms_placed = 0
    unit_cells_placed = 0
    surface_atoms_placed = 0
    mols_placed = 0
    place_anion_vacancy = False
    place_surface_vacancy = False
    centroids_list = []

    print('\tNumber of atoms expected: {}\n'.format(N_atoms))

    # Begin the loop to start placing unit cells, surface anions, and capping atoms
    for x in range(args.dims[0] + 1):
        for y in range(args.dims[1] + 1):
            for z in range(args.dims[2] + 1):
                n_cations = 1
                if args.monolayer:
                    n_cations = 0
                elif args.quasi_2D:
                    if y == 0:
                        n_cations = 0

                # For values of x > 1/2 dims[0], place second unit cell if building a heterojunction
                place_heterojunction = False
                if build_heterojunction:
                    if x > float((args.dims[0]) / 2):
                        place_heterojunction = True

                # Determine when to place the vacancy
                place_anion_vacancy = False
                place_surface_vacancy = False
                place_metal_vacancy = False
                if unit_cells_placed + 1 in anion_vacancies_list:
                    if anion_vacancy_flag == True:
                        place_anion_vacancy = True
                        print('\tPlacing anion vacancy at unit cell position {} {} {}'.format(x, y, z))
                        vacancy_unit_cell = build_unit_cell(Metal, Anion, Cation, args.bond_length, args.monolayer,
                                                            True, False)
                        if build_heterojunction: vacancy_unit_cell_2 = build_unit_cell(Metal_2, Anion_2, Cation,
                                                                                       args.bond_length, args.monolayer,
                                                                                       True, False)

                if unit_cells_placed + 1 in surface_cation_vacancies_list:
                    if surface_cation_vacancy_flag == True:
                        place_surface_vacancy = True
                        # Randomly pick 0 or 1 to determine to place vacancy below (0) or above (1) metal plane
                        surface_cation_vacancy_location = random.choice([0, 1])

                if unit_cells_placed + 1 in metal_vacancies_list:
                    if metal_vacancy_flag == True:
                        place_metal_vacancy = True
                        print('\tPlacing metal vacancy at unit cell position {} {} {}'.format(x, y, z))
                        vacancy_unit_cell = build_unit_cell(Metal, Anion, Cation, args.bond_length, args.monolayer,
                                                            False, True)
                        if build_heterojunction: vacancy_unit_cell_2 = build_unit_cell(Metal_2, Anion_2, Cation,
                                                                                       args.bond_length, args.monolayer,
                                                                                       False, True)

                # For each cation site, randomize the cation orientation then place it inside the unit cell




                for i in range(n_cations):

                    # do cation site #1
                    geom = deepcopy(Cation["geometry"])

                    # Randomize the cation orientation
                    # perform x rotations
                    angle = random.random() * 360
                    for count_j, j in enumerate(geom):
                        geom[count_j, :] = axis_rot(j, numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]),
                                                    angle, mode='angle')

                    # perform y rotations
                    angle = random.random() * 360
                    for count_j, j in enumerate(geom):
                        geom[count_j, :] = axis_rot(j, numpy.array([0.0, 1.0, 0.0]), numpy.array([0.0, 0.0, 0.0]),
                                                    angle, mode='angle')

                    # perform z rotations
                    angle = random.random() * 360
                    for count_j, j in enumerate(geom):
                        geom[count_j, :] = axis_rot(j, numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 0.0]),
                                                    angle, mode='angle')

                    # set initial cation position
                    geom += numpy.array([args.bond_length, -args.bond_length, args.bond_length])

                    # Update the unit cell with the randomized cation coordinates (Cation always starts at 4, since there are 4 preceeding atoms already in the unit cell)
                    if place_anion_vacancy or place_metal_vacancy:
                        if place_heterojunction:
                            vacancy_unit_cell_2["geometry"][
                            3 + i * len(Cation["elements"]): 3 + (i + 1) * len(Cation["elements"])] = geom
                        else:
                            vacancy_unit_cell["geometry"][
                            3 + i * len(Cation["elements"]): 3 + (i + 1) * len(Cation["elements"])] = geom
                    else:
                        if place_heterojunction:
                            unit_cell_2["geometry"][
                            4 + i * len(Cation["elements"]): 4 + (i + 1) * len(Cation["elements"])] = geom
                        else:
                            unit_cell["geometry"][
                                    4 + i * len(Cation["elements"]): 4 + (i + 1) * len(Cation["elements"])] = geom
                            # if args.quasi_2D:
                            if y == args.dims[1]:
                                    unit_cell_plusAnion["geometry"][5 + i * len(Cation["elements"]): 5 + (i + 1) * len(Cation["elements"])] = geom

                # Translate the unit cell to the current position
                if place_anion_vacancy or place_metal_vacancy:
                    if place_heterojunction:
                        geom = vacancy_unit_cell_2["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), y * (args.bond_length * 2), z * (args.bond_length * 2)])

                        # We need this reference geometry to properly place the surface cation, since that should be placed where the now-missing anion is
                        geom_reference = unit_cell["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), y * (args.bond_length * 2), z * (args.bond_length * 2)])

                        # Place unit cell in simulation box. Update lists
                        sim_box[atoms_placed:atoms_placed + len(vacancy_unit_cell_2["elements"])] = geom
                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(vacancy_unit_cell_2["elements"])),
                        atoms_placed:(atoms_placed + len(vacancy_unit_cell_2["elements"]))] = vacancy_unit_cell_2[
                            "adj_mat"]
                        sim_data["elements"] = sim_data["elements"] + vacancy_unit_cell_2["elements"]
                        sim_data["masses"] = sim_data["masses"] + vacancy_unit_cell_2["masses"]
                        sim_data["charges"] = sim_data["charges"] + vacancy_unit_cell_2["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + vacancy_unit_cell_2["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 vacancy_unit_cell_2["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + vacancy_unit_cell_2["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                            vacancy_unit_cell_2["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + vacancy_unit_cell_2["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in vacancy_unit_cell_2["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + vacancy_unit_cell_2["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in vacancy_unit_cell_2["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + vacancy_unit_cell_2["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(
                            vacancy_unit_cell_2["elements"])
                        mols_placed += 1

                    else:
                        geom = vacancy_unit_cell["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), y * (args.bond_length * 2), z * (args.bond_length * 2)])

                        # We need this reference geometry to properly place the surface cation, since that should be placed where the now-missing anion is
                        geom_reference = unit_cell["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), y * (args.bond_length * 2), z * (args.bond_length * 2)])

                        # Place unit cell in simulation box. Update lists
                        sim_box[atoms_placed:atoms_placed + len(vacancy_unit_cell["elements"])] = geom
                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"])),
                        atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"]))] = vacancy_unit_cell["adj_mat"]
                        sim_data["elements"] = sim_data["elements"] + vacancy_unit_cell["elements"]
                        sim_data["masses"] = sim_data["masses"] + vacancy_unit_cell["masses"]
                        sim_data["charges"] = sim_data["charges"] + vacancy_unit_cell["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + vacancy_unit_cell["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 vacancy_unit_cell["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + vacancy_unit_cell["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                            vacancy_unit_cell["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + vacancy_unit_cell["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in vacancy_unit_cell["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + vacancy_unit_cell["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in vacancy_unit_cell["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + vacancy_unit_cell["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(
                            vacancy_unit_cell["elements"])
                        mols_placed += 1

                else:
                    if place_heterojunction:
                        # Place unit cell in simulation box. Update lists
                        geom = unit_cell_2["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), y * (args.bond_length * 2), z * (args.bond_length * 2)])
                        sim_box[atoms_placed:atoms_placed + len(unit_cell_2["elements"])] = geom
                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell_2["elements"])),
                        atoms_placed:(atoms_placed + len(unit_cell_2["elements"]))] = unit_cell_2["adj_mat"]
                        sim_data["elements"] = sim_data["elements"] + unit_cell_2["elements"]
                        sim_data["masses"] = sim_data["masses"] + unit_cell_2["masses"]
                        sim_data["charges"] = sim_data["charges"] + unit_cell_2["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + unit_cell_2["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 unit_cell_2["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + unit_cell_2["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                            unit_cell_2["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + unit_cell_2["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in unit_cell_2["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + unit_cell_2["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in unit_cell_2["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + unit_cell_2["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(unit_cell_2["elements"])
                        mols_placed += 1

                    else:
                        # Place unit cell in simulation box. Update lists
                        if args.quasi_2D:
                            if y == 0:
                                update_simbox(x, y, z, args.bone_length, sim_box, sim_data, atoms_placed, unit_cell_no_cation)

                        if y == 0:
                            print(unit_cell_no_cation)

                        elif y == args.dims[1]:
                            print(unit_cell_plusAnion)
                        
                        geom = unit_cell["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), y * (args.bond_length * 2), z * (args.bond_length * 2)])
                        sim_box[atoms_placed:atoms_placed + len(unit_cell["elements"])] = geom
                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell["elements"])),
                        atoms_placed:(atoms_placed + len(unit_cell["elements"]))] = unit_cell["adj_mat"]
                        sim_data["elements"] = sim_data["elements"] + unit_cell["elements"]
                        sim_data["masses"] = sim_data["masses"] + unit_cell["masses"]
                        sim_data["charges"] = sim_data["charges"] + unit_cell["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + unit_cell["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 unit_cell["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + unit_cell["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                            unit_cell["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + unit_cell["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in unit_cell["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + unit_cell["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in unit_cell["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + unit_cell["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(unit_cell["elements"])
                        mols_placed += 1

                centroid = geom.mean(axis=0)

                for j in range(len(centroids_list)):
                    distance = numpy.around(numpy.linalg.norm(centroid - centroids_list[j]), decimals=5)
                    if distance == 0:
                        print('ERROR: OVERLAPPING CENTROIDS')

                centroids_list.append(centroid)

                # Increment lists
                if place_anion_vacancy or place_metal_vacancy:
                    if place_heterojunction:
                        atoms_placed += len(vacancy_unit_cell_2["elements"])
                    else:
                        atoms_placed += len(vacancy_unit_cell["elements"])
                else:
                    if place_heterojunction:
                        atoms_placed += len(unit_cell_2["elements"])
                    else:
                        atoms_placed += len(unit_cell["elements"])
                unit_cells_placed += 1

                # Handle the surface cations slightly differently depending on whether or not we are building a monolayer. For a monolayer, place both top/bottom at the same time
                # For a multilayer, check if we are on an end and place a surface cation accordingly (on the xz plane)
                # while we are checking, if we are on the bottom, also place the capping atoms to eliminate any dangling atoms from the unit cell
                if args.monolayer:

                    # Place the bottom capping atoms, if requested
                    if args.bottom:
                        # Translate to current position
                        bottom_geom_place = bottom["geometry"] + numpy.array(
                            [x * (args.bond_length * 2), -(args.bond_length * 2), z * (args.bond_length * 2)])

                        v_1 = bottom_geom_place[1] - bottom_geom_place[0]
                        v_2 = bottom_geom_place[2] - bottom_geom_place[0]
                        v_n = numpy.cross(v_1, v_2)
                        v_n = v_n / numpy.linalg.norm(v_n)

                        # Place inside box
                        sim_box[atoms_placed:atoms_placed + len(bottom["elements"])] = bottom_geom_place

                        # Update lists
                        atoms_placed += len(bottom["elements"])
                        sim_data["elements"] = sim_data["elements"] + bottom["elements"]
                        if place_anion_vacancy or place_metal_vacancy:
                            if place_heterojunction:
                                sim_data["adj_mat"][atoms_placed:(atoms_placed + len(vacancy_unit_cell_2["elements"])),
                                atoms_placed:(atoms_placed + len(vacancy_unit_cell_2["elements"]))] = bottom["adj_mat"]
                            else:
                                sim_data["adj_mat"][atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"])),
                                atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"]))] = bottom["adj_mat"]
                        else:
                            if place_heterojunction:
                                sim_data["adj_mat"][atoms_placed:(atoms_placed + len(bottom["elements"])),
                                atoms_placed:(atoms_placed + len(bottom["elements"]))] = bottom["adj_mat"]
                            else:
                                sim_data["adj_mat"][atoms_placed:(atoms_placed + len(bottom["elements"])),
                                atoms_placed:(atoms_placed + len(bottom["elements"]))] = bottom["adj_mat"]

                        sim_data["masses"] = sim_data["masses"] + bottom["masses"]
                        sim_data["charges"] = sim_data["charges"] + bottom["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + bottom["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 bottom["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + bottom["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in bottom["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + bottom["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in bottom["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + bottom["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in bottom["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + bottom["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(bottom["elements"])
                        mols_placed += 1

                    else:
                        # Did not set any bottom capping atoms, so set the vectors to the appropriate metal atoms

                        if place_anion_vacancy or place_metal_vacancy:
                            v_1 = geom_reference[1] - geom_reference[0]
                            v_2 = geom_reference[3] - geom_reference[0]
                        else:
                            v_1 = geom[1] - geom[0]
                            v_2 = geom[3] - geom[0]
                        v_n = numpy.cross(v_1, v_2)
                        v_n = v_n / numpy.linalg.norm(v_n)

                    #############################
                    #############################
                    ## Placing surface cations ##
                    #############################
                    #############################

                    # Place two copies of the surface cation, one on +y and the other -y
                    # If we are placing a vacancy, set appropriate location to place it
                    if place_surface_vacancy:
                        if surface_cation_vacancy_location == 0:
                            place_bottom_SC = False
                            place_top_SC = True
                            print('\tPlacing surface vacancy on bottom at unit cell position {} {} {}'.format(x, y, z))
                        else:
                            place_bottom_SC = True
                            place_top_SC = False
                            print('\tPlacing surface vacancy on top at unit cell position {} {} {}'.format(x, y, z))
                    else:
                        place_bottom_SC = True
                        place_top_SC = True

                    ################################################
                    # Place first copy of surface cation on bottom #
                    ################################################

                    if place_bottom_SC:
                        # Get copy #1 of surface cation geometry
                        surface_geom = deepcopy(surface_cation["geometry"])

                        # Position surface cation #1 geometry relative to the unit cell
                        surface_geom += numpy.array([args.bond_length, -args.bond_length, args.bond_length])

                        # Translate geometry to correct location, depending on if we are at the bottom (y = 0) or top (y = args.dims[1]) of the simulation cell.
                        if args.bottom:
                            # place = surface_geom + numpy.array([x*(args.bond_length*2), -(args.bond_length*2), z*(args.bond_length*2)])
                            place = surface_geom + numpy.array(
                                [x * (args.bond_length * 2), -(args.bond_length), z * (args.bond_length * 2)])

                        else:
                            # place = surface_geom + numpy.array([x*(args.bond_length*2), -(args.bond_length), z*(args.bond_length*2)])
                            place = surface_geom + numpy.array(
                                [x * (args.bond_length * 2), 0.0, z * (args.bond_length * 2)])

                        place_origin = place[0]

                        # Define long-axis vector and normalize
                        # v_la = place[surface_cation["tail_atom"]] - place[0]
                        v_la = place[1] - place[0]
                        v_la = v_la / numpy.linalg.norm(v_la)

                        # Angle to rotate by, convert to degrees
                        angle = (math.acos(numpy.dot(v_n, v_la) / (
                                    numpy.linalg.norm(v_la) * numpy.linalg.norm(v_n))) * 180.0 / math.pi)

                        v_n_la = numpy.cross(v_n, v_la)

                        # rotate surface anion to be normal to the top/bottom
                        for count_j, j in enumerate(place):
                            place[count_j, :] = axis_rot(j, v_n_la, place_origin, -angle, mode='angle')

                        #
                        # Procedure is to apply tilt, then sweep, then rotation operations, in that order.
                        #

                        # Tilt relative to x axis
                        for count_k, k in enumerate(place):
                            place[count_k, :] = axis_rot(k, numpy.array([1.0, 0.0, 0.0]), place[0],
                                                         args.surface_cation_orientation[0], mode='angle')

                        # Sweep relative to the y axis (sweeps around the unit cell based on the tilt angle)
                        for count_k, k in enumerate(place):
                            place[count_k, :] = axis_rot(k, numpy.array([0.0, 1.0, 0.0]), place[0],
                                                         args.surface_cation_orientation[1], mode='angle')

                        # Rotate about molecular long axis (spins the molecule)
                        for count_k, k in enumerate(place):
                            place[count_k, :] = axis_rot(k, v_la, place[0], args.surface_cation_orientation[2],
                                                         mode='angle')

                        # Insert the surface anions into the simulation box
                        sim_box[atoms_placed:atoms_placed + len(surface_cation["elements"])] = place

                        # Record the indices of the surface anions. These are the numpy array indices, which start numbering at 0;
                        # most molecular visualization programs start counting at 1, so when writing these out remember to add 1 to each.
                        for i in range(atoms_placed, atoms_placed + len(surface_cation["elements"])):
                            surface_indices.append(i)

                        # Record index of head atom in surface cation
                        surface_head_indices.append(atoms_placed)

                        # Update lists
                        sim_data["elements"] = sim_data["elements"] + surface_cation["elements"]
                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(surface_cation["elements"])),
                        atoms_placed:(atoms_placed + len(surface_cation["elements"]))] = surface_cation["adj_mat"]
                        sim_data["masses"] = sim_data["masses"] + [surface_cation["masses"][j] for j in
                                                                   surface_cation["atom_types"]]
                        sim_data["charges"] = sim_data["charges"] + surface_cation["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + surface_cation["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 surface_cation["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + surface_cation["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                            surface_cation["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + surface_cation["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in surface_cation["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + surface_cation["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in surface_cation["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + surface_cation["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(surface_cation["elements"])
                        atoms_placed += len(surface_cation["elements"])
                        mols_placed += 1
                        surface_atoms_placed += 1

                    ###################################################
                    # Place the second surface cation on the top side #
                    ###################################################

                    if place_top_SC:
                        if place_anion_vacancy or place_metal_vacancy:
                            v_1 = geom_reference[1] - geom_reference[0]
                            v_2 = geom_reference[3] - geom_reference[0]
                        else:
                            v_1 = geom[1] - geom[0]
                            v_2 = geom[3] - geom[0]
                        v_n = numpy.cross(v_1, v_2)
                        v_n = v_n / numpy.linalg.norm(v_n)

                        # Get copy 2 of surface cation geometry
                        surface_geom = deepcopy(surface_cation["geometry"])

                        # Position the geometry relative to the unit cell
                        surface_geom += numpy.array([args.bond_length, args.bond_length, args.bond_length])

                        place = surface_geom + numpy.array(
                            [x * (args.bond_length * 2), 0.0, z * (args.bond_length * 2)])
                        v_n = -1 * v_n

                        place_origin = place[0]

                        # Define long-axis vector and normalize
                        v_la = place[surface_cation["tail_atom"]] - place[0]
                        # v_la = place[1] - place[0]
                        v_la = v_la / numpy.linalg.norm(v_la)

                        # Angle to rotate by, convert to degrees
                        angle = (math.acos(numpy.dot(v_n, v_la) / (
                                    numpy.linalg.norm(v_la) * numpy.linalg.norm(v_n))) * 180.0 / math.pi)

                        v_n_la = numpy.cross(v_n, v_la)

                        #                        # rotate surface anion to be normal to the top/bottom
                        for count_j, j in enumerate(place):
                            place[count_j, :] = axis_rot(j, v_n_la, place_origin, -angle, mode='angle')

                        #
                        # Procedure is to apply tilt, then sweep, then rotation operations, in that order.
                        #

                        # Tilt relative to x axis
                        for count_k, k in enumerate(place):
                            place[count_k, :] = axis_rot(k, numpy.array([1.0, 0.0, 0.0]), place[0],
                                                         args.surface_cation_orientation[0], mode='angle')

                        # Sweep relative to the y axis (sweeps around the unit cell based on the tilt angle)
                        for count_k, k in enumerate(place):
                            place[count_k, :] = axis_rot(k, numpy.array([0.0, 1.0, 0.0]), place[0],
                                                         args.surface_cation_orientation[1], mode='angle')

                        # Rotate about molecular long axis (spins the molecule)
                        for count_k, k in enumerate(place):
                            place[count_k, :] = axis_rot(k, v_la, place[0], args.surface_cation_orientation[2],
                                                         mode='angle')

                        # Insert the surface anions into the simulation box
                        sim_box[atoms_placed:atoms_placed + len(surface_cation["elements"])] = place

                        # Record the indices of the surface anions. These are the numpy array indices, which start numbering at 0;
                        # most molecular visualization programs start counting at 1, so when writing these out remember to add 1 to each.
                        for i in range(atoms_placed, atoms_placed + len(surface_cation["elements"])):
                            surface_indices.append(i)

                        # Record index of head atom in surface cation
                        surface_head_indices.append(atoms_placed)

                        # Update lists
                        sim_data["elements"] = sim_data["elements"] + surface_cation["elements"]
                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(surface_cation["elements"])),
                        atoms_placed:(atoms_placed + len(surface_cation["elements"]))] = surface_cation["adj_mat"]
                        sim_data["masses"] = sim_data["masses"] + [surface_cation["masses"][j] for j in
                                                                   surface_cation["atom_types"]]
                        sim_data["charges"] = sim_data["charges"] + surface_cation["charges"]
                        sim_data["atom_types"] = sim_data["atom_types"] + surface_cation["atom_types"]
                        sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                 surface_cation["bonds"]]
                        sim_data["bond_types"] = sim_data["bond_types"] + surface_cation["bond_types"]
                        sim_data["angles"] = sim_data["angles"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                            surface_cation["angles"]]
                        sim_data["angle_types"] = sim_data["angle_types"] + surface_cation["angle_types"]
                        sim_data["dihedrals"] = sim_data["dihedrals"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in surface_cation["dihedrals"]]
                        sim_data["dihedral_types"] = sim_data["dihedral_types"] + surface_cation["dihedral_types"]
                        sim_data["impropers"] = sim_data["impropers"] + [
                            (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
                            in surface_cation["impropers"]]
                        sim_data["improper_types"] = sim_data["improper_types"] + surface_cation["improper_types"]
                        sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(surface_cation["elements"])
                        atoms_placed += len(surface_cation["elements"])
                        mols_placed += 1
                        surface_atoms_placed += 1

                else:
                    if y == 0 or y == args.dims[1]:

                        if y == 0:
                            # Place a set of capping atoms if needed, update lists

                            if args.bottom:
                                # Translate to current position
                                bottom_geom_place = bottom["geometry"] + numpy.array(
                                    [x * (args.bond_length * 2), -(args.bond_length * 2), z * (args.bond_length * 2)])

                                v_1 = bottom_geom_place[1] - bottom_geom_place[0]
                                v_2 = bottom_geom_place[2] - bottom_geom_place[0]
                                v_n = numpy.cross(v_1, v_2)
                                v_n = v_n / numpy.linalg.norm(v_n)

                                # Place inside box
                                sim_box[atoms_placed:atoms_placed + len(bottom["elements"])] = bottom_geom_place

                                # Update lists
                                atoms_placed += len(bottom["elements"])
                                sim_data["elements"] = sim_data["elements"] + bottom["elements"]
                                if place_anion_vacancy or place_metal_vacancy:
                                    sim_data["adj_mat"][
                                    atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"])),
                                    atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"]))] = \
                                    vacancy_unit_cell["adj_mat"]
                                else:
                                    if place_heterojunction:
                                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell_2["elements"])),
                                        atoms_placed:(atoms_placed + len(unit_cell_2["elements"]))] = unit_cell_2[
                                            "adj_mat"]
                                    else:
                                        sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell["elements"])),
                                        atoms_placed:(atoms_placed + len(unit_cell["elements"]))] = unit_cell["adj_mat"]

                                sim_data["masses"] = sim_data["masses"] + bottom["masses"]
                                sim_data["charges"] = sim_data["charges"] + bottom["charges"]
                                sim_data["atom_types"] = sim_data["atom_types"] + bottom["atom_types"]
                                sim_data["bonds"] = sim_data["bonds"] + [
                                    (j[0] + atoms_placed - 4, j[1] + atoms_placed - 4) for j in bottom["bonds"]]
                                sim_data["bond_types"] = sim_data["bond_types"] + bottom["bond_types"]
                                sim_data["angles"] = sim_data["angles"] + [
                                    (j[0] + atoms_placed - 4, j[1] + atoms_placed - 4, j[2] + atoms_placed - 4) for j in
                                    bottom["angles"]]
                                sim_data["angle_types"] = sim_data["angle_types"] + bottom["angle_types"]
                                sim_data["dihedrals"] = sim_data["dihedrals"] + [(j[0] + atoms_placed - 4,
                                                                                  j[1] + atoms_placed - 4,
                                                                                  j[2] + atoms_placed - 4,
                                                                                  j[3] + atoms_placed - 4) for j in
                                                                                 bottom["dihedrals"]]
                                sim_data["dihedral_types"] = sim_data["dihedral_types"] + bottom["dihedral_types"]
                                sim_data["impropers"] = sim_data["impropers"] + [(j[0] + atoms_placed - 4,
                                                                                  j[1] + atoms_placed - 4,
                                                                                  j[2] + atoms_placed - 4,
                                                                                  j[3] + atoms_placed - 4) for j in
                                                                                 bottom["impropers"]]
                                sim_data["improper_types"] = sim_data["improper_types"] + bottom["improper_types"]
                                sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(bottom["elements"])
                                mols_placed += 1

                            else:
                                v_1 = geom[1] - geom[0]
                                v_2 = geom[3] - geom[0]
                                v_n = numpy.cross(v_1, v_2)
                                v_n = v_n / numpy.linalg.norm(v_n)

                        elif y == args.dims[1]:
                            v_1 = geom[1] - geom[0]
                            v_2 = geom[3] - geom[0]

                            # print(v_1, v_2)
                            v_n = numpy.cross(v_1, v_2)
                            v_n = v_n / numpy.linalg.norm(v_n)
                            

                        # Place surface anions only if they've been specified
                        if args.surface_cation != None:

                            # Get copy of surface anion geometry
                            surface_geom = deepcopy(surface_cation["geometry"])

                            # DEFINE UNIT CELL
                            # Position the geometry relative to the unit cell
                            surface_geom += numpy.array([args.bond_length, -args.bond_length, args.bond_length])

                            # Translate geometry to correct location, depending on if we are at the bottom (y = 0) or top (y = args.dims[1]) of the simulation cell.
                            if y == 0:
                                if args.bottom:
                                    place = surface_geom + numpy.array(
                                        [x * (args.bond_length * 2), -(args.bond_length * 2),
                                         z * (args.bond_length * 2)])
                                else:
                                    place = surface_geom + numpy.array(
                                        [x * (args.bond_length * 2), -(args.bond_length), z * (args.bond_length * 2)])

                            elif y == args.dims[1]:
                                place = surface_geom + numpy.array(
                                    [x * (args.bond_length * 2), y * (args.bond_length * 4),
                                     z * (args.bond_length * 2)])
                                v_n = 1 * v_n
                            
                            #print(v_n)
                            place_origin = place[0]

                            # Define long-axis vector and normalize
                            v_la = place[surface_cation["tail_atom"]] - place[0]
                            # v_la = place[1] - place[0]
                            v_la = v_la / numpy.linalg.norm(v_la)

                            # Angle to rotate by, convert to degrees
                            angle = (math.acos(numpy.dot(v_n, v_la) / (
                                        numpy.linalg.norm(v_la) * numpy.linalg.norm(v_n))) * 180.0 / math.pi)

                            v_n_la = numpy.cross(v_n, v_la)

                            #
                            # Procedure is to apply tilt, then sweep, then rotation operations, in that order.
                            #

                            # Tilt relative to x axis
                            for count_k, k in enumerate(place):
                                place[count_k, :] = axis_rot(k, numpy.array([1.0, 0.0, 0.0]), place[0],
                                                             args.surface_cation_orientation[0], mode='angle')

                            # Sweep relative to the y axis (sweeps around the unit cell based on the tilt angle)
                            for count_k, k in enumerate(place):
                                place[count_k, :] = axis_rot(k, numpy.array([0.0, 1.0, 0.0]), place[0],
                                                             args.surface_cation_orientation[1], mode='angle')

                            # Rotate about molecular long axis (spins the molecule)
                            for count_k, k in enumerate(place):
                                place[count_k, :] = axis_rot(k, v_la, place[0], args.surface_cation_orientation[2],
                                                             mode='angle')

                            # print(place[0])

                            # Insert the surface anions into the simulation box
                            sim_box[atoms_placed:atoms_placed + len(surface_cation["elements"])] = place
                            
                            # Record the indices of the surface anions. These are the numpy array indices, which start numbering at 0;
                            # most molecular visualization programs start counting at 1, so when writing these out remember to add 1 to each.
                            for i in range(atoms_placed, atoms_placed + len(surface_cation["elements"])):
                                surface_indices.append(i)

                            # Record index of head atom in surface cation
                            surface_head_indices.append(atoms_placed)

                            # Update lists
                            sim_data["elements"] = sim_data["elements"] + surface_cation["elements"]
                            if place_anion_vacancy or place_metal_vacancy:
                                sim_data["adj_mat"][atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"])),
                                atoms_placed:(atoms_placed + len(vacancy_unit_cell["elements"]))] = vacancy_unit_cell[
                                    "adj_mat"]
                            else:
                                if place_heterojunction:
                                    sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell_2["elements"])),
                                    atoms_placed:(atoms_placed + len(unit_cell_2["elements"]))] = unit_cell_2["adj_mat"]
                                else:
                                    sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell["elements"])),
                                    atoms_placed:(atoms_placed + len(unit_cell["elements"]))] = unit_cell["adj_mat"]

                            sim_data["masses"] = sim_data["masses"] + [surface_cation["masses"][j] for j in
                                                                       surface_cation["atom_types"]]
                            sim_data["charges"] = sim_data["charges"] + surface_cation["charges"]
                            sim_data["atom_types"] = sim_data["atom_types"] + surface_cation["atom_types"]
                            sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                                     surface_cation["bonds"]]
                            sim_data["bond_types"] = sim_data["bond_types"] + surface_cation["bond_types"]
                            sim_data["angles"] = sim_data["angles"] + [
                                (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
                                surface_cation["angles"]]
                            sim_data["angle_types"] = sim_data["angle_types"] + surface_cation["angle_types"]
                            sim_data["dihedrals"] = sim_data["dihedrals"] + [
                                (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for
                                j in surface_cation["dihedrals"]]
                            sim_data["dihedral_types"] = sim_data["dihedral_types"] + surface_cation["dihedral_types"]
                            sim_data["impropers"] = sim_data["impropers"] + [
                                (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for
                                j in surface_cation["impropers"]]
                            sim_data["improper_types"] = sim_data["improper_types"] + surface_cation["improper_types"]
                            sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(
                                surface_cation["elements"])
                            atoms_placed += len(surface_cation["elements"])
                            mols_placed += 1
                            surface_atoms_placed += 1

    perovskite_atoms_placed = atoms_placed
    # Hydrate, if requested

    if args.hydrate:
        # Add water molecules in above and below xz plane, in a volume defined by headspace (y distance) for each side.
        print('\nHydrating...')
        N_A = 6.0221413e23
        Density = 1.0

        # Find current sim box dimensions to use the x and z lengths to define the headspace volume
        # sim_box_dims = ( -args.bond_length, ((2*args.bond_length)*args.dims[0]) + args.bond_length, min(sim_box[:,1])-args.bond_length, max(sim_box[:,1]), -args.bond_length, ((2*args.bond_length)*args.dims[2]) + args.bond_length )
        # reduce max width by bond_length so water molecules aren't tiled right along the edge of x and z, may have issues with LAMMPS wrapping the molecules on the edge over to the other side,
        # resulitng in overlapping coordinates
        sim_box_dims = (
        -args.bond_length, ((2 * args.bond_length) * args.dims[0]), min(sim_box[:, 1]) - args.bond_length,
        max(sim_box[:, 1]), -args.bond_length, ((2 * args.bond_length) * args.dims[2]))

        # convert everything from Ang to cm, since that is what LAMMPS uses in its density calc g(cm^3)
        length_x = ((sim_box_dims[1] - sim_box_dims[0]) * 10.0 ** (-8))  # cm
        length_y = ((args.headspace) * 10.0 ** (-8))  # cm
        length_z = ((sim_box_dims[5] - sim_box_dims[4]) * 10.0 ** (-8))  # cm

        headspace_volume = length_x * length_y * length_z  # cm^3
        target_mass = headspace_volume

        # Convert mass of water to grams
        Water["total_mass"] = sum([Water["masses"][i] for i in Water["atom_types"]]) / N_A

        # Find an integer number of water molecules needed to reach the target mass
        target_N_water = int(math.ceil(target_mass / Water["total_mass"]))

        # Find the largest integer needed to build a cube of water molecules with the required mass
        dimensionality = int(math.ceil(target_N_water ** (1. / 3.)))

        print('length x: {} cm'.format(length_x))
        print('length y: {} cm'.format(length_y))
        print('length z: {} cm'.format(length_z))
        print('headspace_volume: {} cm^3'.format(headspace_volume))
        print('target_mass: {} g'.format(target_mass))
        print('target density: {} g/cm^3'.format(target_mass / headspace_volume))
        print('water total mass: {} g/mol'.format(sum([Water["masses"][i] for i in Water["atom_types"]])))
        print('water total mass: {} g'.format(Water["total_mass"]))
        print('target N waters: {}'.format(target_N_water))
        print('cube dimensionality: {}'.format(dimensionality))

        # Find molecular centers
        Centers = numpy.zeros([dimensionality ** 3, 3])
        Centers_bottom = numpy.zeros([dimensionality ** 3, 3])
        count = 0

        # Loop over y, x, z. Want to build y layer-by-layer, so i corresponds to y. That way the perovskite layer should be completely covered by water molecules, and not an unfinished layer will be towards the end
        for i in range(dimensionality):
            for j in range(dimensionality):
                for k in range(dimensionality):

                    # Account for volume excluded by the surface anions
                    if args.surface_cation:
                        Centers[count] = numpy.array([((Water["step_size"]) * j),
                                                      ((Water["step_size"]) * i) + sim_box_dims[3] + Water["step_size"],
                                                      ((Water["step_size"]) * k)])
                        Centers_bottom[count] = numpy.array([((Water["step_size"]) * j),
                                                             sim_box_dims[2] - (Water["step_size"] * i) - Water[
                                                                 "step_size"], ((Water["step_size"]) * k)])

                    else:
                        Centers[count] = numpy.array(
                            [((Water["step_size"]) * j), ((Water["step_size"]) * i) + sim_box_dims[3],
                             ((Water["step_size"]) * k)])
                        Centers_bottom[count] = numpy.array(
                            [((Water["step_size"]) * j), sim_box_dims[2] - (Water["step_size"] * i),
                             ((Water["step_size"]) * k)])
                    count = count + 1

                    # Start placing waters
        print('\n PLACING WATERS...\n')
        hydrate_data = {}
        hydrate_data["elements"] = []
        hydrate_data["adj_mat"] = numpy.zeros(
            [target_N_water * len(Water["elements"]), target_N_water * len(Water["elements"])])
        hydrate_data["masses"] = []
        hydrate_data["charges"] = []
        hydrate_data["atom_types"] = []
        hydrate_data["bonds"] = []
        hydrate_data["bond_types"] = []
        hydrate_data["angles"] = []
        hydrate_data["angle_types"] = []
        hydrate_data["dihedrals"] = []
        hydrate_data["dihedral_types"] = []
        hydrate_data["impropers"] = []
        hydrate_data["improper_types"] = []
        hydrate_data["molecules"] = []

        hydrate_bottom_data = {}
        hydrate_bottom_data["elements"] = []
        hydrate_bottom_data["adj_mat"] = numpy.zeros(
            [target_N_water * len(Water["elements"]), target_N_water * len(Water["elements"])])
        hydrate_bottom_data["masses"] = []
        hydrate_bottom_data["charges"] = []
        hydrate_bottom_data["atom_types"] = []
        hydrate_bottom_data["bonds"] = []
        hydrate_bottom_data["bond_types"] = []
        hydrate_bottom_data["angles"] = []
        hydrate_bottom_data["angle_types"] = []
        hydrate_bottom_data["dihedrals"] = []
        hydrate_bottom_data["dihedral_types"] = []
        hydrate_bottom_data["impropers"] = []
        hydrate_bottom_data["improper_types"] = []
        hydrate_bottom_data["molecules"] = []

        water_box = numpy.zeros([target_N_water * len(Water["elements"]), 3])
        water_box_bottom = numpy.zeros([target_N_water * len(Water["elements"]), 3])

        waters_placed = 0
        mol_index = 0
        atom_index = 0

        while waters_placed < target_N_water:

            # First do the top headspace
            # Copy water geometry and randomize its orientations
            water_geom = deepcopy(Water["geometry"])

            # perform x rotations
            angle = random.random() * 360
            for count_j, j in enumerate(water_geom):
                water_geom[count_j, :] = axis_rot(j, numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]), angle,
                                                  mode='angle')

            # perform y rotations
            angle = random.random() * 360
            for count_j, j in enumerate(water_geom):
                water_geom[count_j, :] = axis_rot(j, numpy.array([0.0, 1.0, 0.0]), numpy.array([0.0, 0.0, 0.0]), angle,
                                                  mode='angle')

            # perform z rotations
            angle = random.random() * 360
            for count_j, j in enumerate(water_geom):
                water_geom[count_j, :] = axis_rot(j, numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 0.0]), angle,
                                                  mode='angle')

            # Move the current molecule to its box and append to sim_box
            water_geom += Centers[mol_index]
            water_box[atom_index:(atom_index + len(Water["geometry"])), :] = water_geom

            # Now do bottom headspace (tailspace?)
            # Copy water geometry and randomize its orientations
            water_geom = deepcopy(Water["geometry"])

            # perform x rotations
            angle = random.random() * 360
            for count_j, j in enumerate(water_geom):
                water_geom[count_j, :] = axis_rot(j, numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]), angle,
                                                  mode='angle')

            # perform y rotations
            angle = random.random() * 360
            for count_j, j in enumerate(water_geom):
                water_geom[count_j, :] = axis_rot(j, numpy.array([0.0, 1.0, 0.0]), numpy.array([0.0, 0.0, 0.0]), angle,
                                                  mode='angle')

            # perform z rotations
            angle = random.random() * 360
            for count_j, j in enumerate(water_geom):
                water_geom[count_j, :] = axis_rot(j, numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 0.0]), angle,
                                                  mode='angle')

            # Move current molecule to box
            water_geom += Centers_bottom[mol_index]
            water_box_bottom[atom_index:(atom_index + len(Water["geometry"])), :] = water_geom

            # Extend lists for top and bottom waters
            hydrate_data["elements"] = hydrate_data["elements"] + Water["elements"]
            hydrate_data["adj_mat"][atom_index:(atom_index + len(Water["elements"])),
            atom_index:(atom_index + len(Water["elements"]))] = Water["adj_mat"]
            hydrate_data["masses"] = hydrate_data["masses"] + [Water["masses"][j] for j in Water["atom_types"]]
            hydrate_data["charges"] = hydrate_data["charges"] + Water["charges"]
            hydrate_data["atom_types"] = hydrate_data["atom_types"] + Water["atom_types"]
            hydrate_data["bonds"] = hydrate_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                             Water["bonds"]]
            hydrate_data["bond_types"] = hydrate_data["bond_types"] + Water["bond_types"]
            hydrate_data["angles"] = hydrate_data["angles"] + [
                (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in Water["angles"]]
            hydrate_data["angle_types"] = hydrate_data["angle_types"] + Water["angle_types"]
            hydrate_data["dihedrals"] = hydrate_data["dihedrals"] + [
                (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j in
                Water["dihedrals"]]
            hydrate_data["dihedral_types"] = hydrate_data["dihedral_types"] + Water["dihedral_types"]
            hydrate_data["impropers"] = hydrate_data["impropers"] + [
                (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j in
                Water["impropers"]]
            hydrate_data["improper_types"] = hydrate_data["improper_types"] + Water["improper_types"]

            mols_placed += 1
            hydrate_data["molecules"] = hydrate_data["molecules"] + [mols_placed] * len(Water["elements"])
            hydrate_bottom_data["elements"] = hydrate_bottom_data["elements"] + Water["elements"]
            hydrate_bottom_data["adj_mat"][atom_index:(atom_index + len(Water["elements"])),
            atom_index:(atom_index + len(Water["elements"]))] = Water["adj_mat"]
            hydrate_bottom_data["masses"] = hydrate_bottom_data["masses"] + [Water["masses"][j] for j in
                                                                             Water["atom_types"]]
            hydrate_bottom_data["charges"] = hydrate_bottom_data["charges"] + Water["charges"]
            hydrate_bottom_data["atom_types"] = hydrate_bottom_data["atom_types"] + Water["atom_types"]
            hydrate_bottom_data["bonds"] = hydrate_bottom_data["bonds"] + [
                (j[0] + atom_index + len(sim_box) + len(water_box), j[1] + atom_index + len(sim_box) + len(water_box))
                for j in Water["bonds"]]
            hydrate_bottom_data["bond_types"] = hydrate_bottom_data["bond_types"] + Water["bond_types"]
            hydrate_bottom_data["angles"] = hydrate_bottom_data["angles"] + [(j[0] + atom_index + len(sim_box) + len(
                water_box), j[1] + atom_index + len(sim_box) + len(water_box), j[2] + atom_index + len(sim_box) + len(
                water_box)) for j in Water["angles"]]
            hydrate_bottom_data["angle_types"] = hydrate_bottom_data["angle_types"] + Water["angle_types"]
            hydrate_bottom_data["dihedrals"] = hydrate_bottom_data["dihedrals"] + [(j[0] + atom_index + len(
                sim_box) + len(water_box), j[1] + atom_index + len(sim_box) + len(water_box), j[2] + atom_index + len(
                sim_box) + len(water_box), j[3] + atom_index + len(sim_box) + len(water_box)) for j in
                                                                                   Water["dihedrals"]]
            hydrate_bottom_data["dihedral_types"] = hydrate_bottom_data["dihedral_types"] + Water["dihedral_types"]
            hydrate_bottom_data["impropers"] = hydrate_bottom_data["impropers"] + [(j[0] + atom_index + len(
                sim_box) + len(water_box), j[1] + atom_index + len(sim_box) + len(water_box), j[2] + atom_index + len(
                sim_box) + len(water_box), j[3] + atom_index + len(sim_box) + len(water_box)) for j in
                                                                                   Water["impropers"]]
            hydrate_bottom_data["improper_types"] = hydrate_bottom_data["improper_types"] + Water["improper_types"]

            waters_placed += 1
            mol_index += 1
            atoms_placed += len(Water["geometry"])
            atom_index += len(Water["geometry"])
            mols_placed += 1
            hydrate_data["molecules"] = hydrate_data["molecules"] + [mols_placed] * len(Water["elements"])

        water_box_dims = numpy.array(
            [min(water_box[:, 0]), max(water_box[:, 0]), min(water_box[:, 1]), max(water_box[:, 1]),
             min(water_box[:, 2]), max(water_box[:, 2])])

        print("\n\tRescaling water headspace box size and coordinates to match density of water (1 g/cm^3):\n".format(
            Density))

        mass_in_box = sum(hydrate_data["masses"]) / N_A  # sum atomic masses then convert to g
        print('Total mass in box: {}'.format(mass_in_box))

        # box_vol = ((water_box_dims[1]-water_box_dims[0])*10.0**(-8))**3   # Standard density units are per cm^3 in lammps
        b1 = ((water_box_dims[1] - water_box_dims[0]) * 10.0 ** (-8))  # cm   length*(10^(-8))
        b2 = ((water_box_dims[3] - water_box_dims[2]) * 10.0 ** (-8))  # cm
        b3 = ((water_box_dims[5] - water_box_dims[4]) * 10.0 ** (-8))  # cm
        box_vol = b1 * b2 * b3  # in cm^3

        current_density = mass_in_box / box_vol  # save current density to variable
        rescale_factor = (current_density / Density) ** (
                    1. / 3.)  # calculate rescale_factor based on the density ratio. cubed root is owing to the 3-dimensionality of the sim box.
        print("\t\tmass_in_box:     {:< 12.6f} g".format(mass_in_box))
        print("\t\tbox_vol:         {:< 12.6f} cm^3".format(box_vol))
        print("\t\tcurrent_density: {:< 12.6f} g/cm^3".format(current_density))
        print("\t\trescale_factor:  {:< 12.6f} ".format(rescale_factor))

        # If the requested density is less than the current density, then only the box is expanded
        if rescale_factor > 1:
            print('expanding box')
            water_box_dims *= rescale_factor
        # For a contraction, rescale coordinates and box to ensure there are no clashes.
        else:
            print("\n\tFixing the mass density by rescaling/contracting the coordinates by {}\n".format(rescale_factor))
            water_box_dims *= rescale_factor
            water_box *= rescale_factor

            water_box_bottom *= rescale_factor

            # Want to make sure that there is an appropriate offset between the top of the perovskite layer and the bottom of headspace water, and an appropriate offest between the bottom of the perovskite layer and the top of the tailsapce water
            # find dimensions of the headspace water box
            wb_dims = (min(water_box[:, 0]), max(water_box[:, 0]), min(water_box[:, 1]), max(water_box[:, 1]),
                       min(water_box[:, 2]), max(water_box[:, 2]))

            # find dimensions of the tailspace water box
            wbb_dims = (min(water_box_bottom[:, 0]), max(water_box_bottom[:, 0]), min(water_box_bottom[:, 1]),
                        max(water_box_bottom[:, 1]), min(water_box_bottom[:, 2]), max(water_box_bottom[:, 2]))

            # find difference between the top of the perovskite layer and bottom of headspace (y axis)
            headspace_offset = wb_dims[2] - sim_box_dims[3]

            # find different between the bottom of the perovskite layer and top of tailspace (y axis)
            tailspace_offset = (sim_box_dims[2] + args.bond_length) - wbb_dims[3]

            # First subtract / add to headspace / tailspace to align with last row, then add / subtract the water molecule spacer distance (set to 2A or water step size). For water, make sure to add in the
            # bond distance for the bottom of the perovskite dimension
            water_box -= numpy.array([0.0, headspace_offset, 0.0])
            water_box_bottom += numpy.array([0.0, tailspace_offset, 0.0])

            if args.spacer == -1:
                args.spacer = Water["step_size"]

            print('Water spacer value: {}'.format(args.spacer))

            water_box += numpy.array([0.0, args.spacer, 0.0])
            water_box_bottom -= numpy.array([0.0, args.spacer, 0.0])

        b1 = ((water_box_dims[1] - water_box_dims[0]) * 10.0 ** (-8))  # cm
        b2 = ((water_box_dims[3] - water_box_dims[2]) * 10.0 ** (-8))  # cm
        b3 = ((water_box_dims[5] - water_box_dims[4]) * 10.0 ** (-8))  # cm
        box_vol = b1 * b2 * b3  # in cm^3

        current_density = mass_in_box / box_vol
        print('\t\tNew density: {:< 12.6f} g/cm^3'.format(current_density))

        print('new dimensions:')
        print(water_box_dims)
        print('\t\tNew volume: {}'.format(box_vol))

        water_box_dims_expected = numpy.array([sim_box_dims[0], sim_box_dims[1], sim_box_dims[3] + Water["step_size"],
                                               sim_box_dims[3] + Water["step_size"] + args.headspace, sim_box_dims[4],
                                               sim_box_dims[5]])
        b1 = ((water_box_dims[1] - water_box_dims[0]) * 10.0 ** (-8))  # cm
        b2 = ((water_box_dims[3] - water_box_dims[2]) * 10.0 ** (-8))  # cm
        b3 = ((water_box_dims[5] - water_box_dims[4]) * 10.0 ** (-8))  # cm
        box_vol = b1 * b2 * b3  # in cm^3
        current_density = mass_in_box / box_vol
        print('density check: {}'.format(current_density))

        # Because we now have water above/below the perovskite, the perovskite itself is now periodic only in x and z directions, so pad those minimum dimensions by one bond length. y has a head/tail space filled with water.
        sim_box_dims = (-args.bond_length, ((2 * args.bond_length) * args.dims[0]) + args.bond_length,
                        min(sim_box[:, 1]) - args.bond_length - args.y_padding, max(sim_box[:, 1]) + args.y_padding,
                        -args.bond_length, ((2 * args.bond_length) * args.dims[2]) + args.bond_length)

    else:
        if args.monolayer:
            # Periodic in x and z, but y is variable because of the cation. Add 50% of width for some spacing, otherwise simulation (pressure) may not be stable due to the box being too small.
            sim_box_dims = (-args.bond_length, ((2 * args.bond_length) * args.dims[0]) + args.bond_length,
                            min(sim_box[:, 1]) + (min(sim_box[:, 1]) * 0.5) - args.y_padding,
                            max(sim_box[:, 1]) + (max(sim_box[:, 1]) * 0.5) + args.y_padding, -args.bond_length,
                            ((2 * args.bond_length) * args.dims[2]) + args.bond_length)
        else:
            # No water was added, so the box is periodic in x, y, z. Need to pad the dimensions by a bond length.
            sim_box_dims = (-args.bond_length, ((2 * args.bond_length) * args.dims[0]) + args.bond_length,
                            -args.bond_length - args.y_padding,
                            ((2 * args.bond_length) * args.dims[1]) + args.bond_length + args.y_padding,
                            -args.bond_length, ((2 * args.bond_length) * args.dims[2]) + args.bond_length)

    # Get total charge
    sim_data["total_charge"] = sum(sim_data["charges"])

    # Statistics
    print('\nStatistics:')
    print('\tNumber of atoms placed: {}'.format(perovskite_atoms_placed))
    print('\tNumber of unit cells placed: {}'.format(unit_cells_placed))
    print('\tBox dimensions: \n\t\t{:< 10.3f} {:< 10.3f}\n\t\t{:< 10.3f} {:< 10.3f}\n\t\t{:< 10.3f} {:< 10.3f}'.format(
        sim_box_dims[0], sim_box_dims[1], sim_box_dims[2], sim_box_dims[3], sim_box_dims[4], sim_box_dims[5]))
    # print '\tSurface indices: {}'.format(', '.join(str(_+1) for _ in surface_indices))
    print('\tIndices of head atoms in surface cations (LAMMPS style): {}'.format(
        ' '.join(str(_ + 1) for _ in surface_head_indices)))
    print('\tNumber of Surface molecules placed: {}'.format(surface_atoms_placed))
    print('\tTotal charge: {:< 12.6f} |e|'.format(sim_data["total_charge"]))
    print('Elements number: {}'.format(len(sim_data['elements'])))

    if args.hydrate:
        print('\tTotal water molecules placed: {}'.format(waters_placed * 2))
        print('\tTotal water atoms placed: {}'.format(len(water_box) + len(water_box_bottom)))

        # Combine the sim_box, water headspace, water bottom headspace arrays into a single array
        count = len(sim_box) + len(water_box) + len(water_box_bottom)
        total_box = numpy.zeros([count, 3])

        print('TOTAL ATOMS COUNT: {}'.format(count))

        total_box[0:len(sim_box), :] = sim_box
        total_box[len(sim_box):(len(sim_box) + len(water_box)), :] = water_box
        total_box[(len(sim_box) + len(water_box)):(len(sim_box) + len(water_box) + (len(water_box_bottom))),
        :] = water_box_bottom

        # Combine adjaceny matrices
        adj_count = len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"]) + len(hydrate_bottom_data["adj_mat"])

        total_adj_mat = numpy.zeros([adj_count, adj_count])
        total_adj_mat[0:len(sim_data["adj_mat"]), 0:len(sim_data["adj_mat"])] = sim_data["adj_mat"]
        total_adj_mat[len(sim_data["adj_mat"]):(len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"])),
        len(sim_data["adj_mat"]):(len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"]))] = hydrate_data["adj_mat"]
        total_adj_mat[(len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"])):(
                    len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"]) + len(hydrate_bottom_data["adj_mat"])),
        (len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"])):(
                    len(sim_data["adj_mat"]) + len(hydrate_data["adj_mat"]) + len(hydrate_bottom_data["adj_mat"]))] = \
        hydrate_bottom_data["adj_mat"]

        sim_data["elements"] = sim_data["elements"] + hydrate_data["elements"] + hydrate_bottom_data["elements"]
        sim_data["masses"] = sim_data["masses"] + hydrate_data["masses"] + hydrate_bottom_data["masses"]
        sim_data["charges"] = sim_data["charges"] + hydrate_data["charges"] + hydrate_bottom_data["charges"]

        sim_data["atom_types"] = sim_data["atom_types"] + hydrate_data["atom_types"] + hydrate_bottom_data["atom_types"]
        sim_data["bonds"] = sim_data["bonds"] + hydrate_data["bonds"] + hydrate_bottom_data["bonds"]
        sim_data["bond_types"] = sim_data["bond_types"] + hydrate_data["bond_types"] + hydrate_bottom_data["bond_types"]
        sim_data["angles"] = sim_data["angles"] + hydrate_data["angles"] + hydrate_bottom_data["angles"]
        sim_data["angle_types"] = sim_data["angle_types"] + hydrate_data["angle_types"] + hydrate_bottom_data[
            "angle_types"]
        sim_data["dihedrals"] = sim_data["dihedrals"] + hydrate_data["dihedrals"] + hydrate_bottom_data["dihedrals"]
        sim_data["dihedral_types"] = sim_data["dihedral_types"] + hydrate_data["dihedral_types"] + hydrate_bottom_data[
            "dihedral_types"]
        sim_data["impropers"] = sim_data["impropers"] + hydrate_data["impropers"] + hydrate_bottom_data["impropers"]
        sim_data["improper_types"] = sim_data["improper_types"] + hydrate_data["improper_types"] + hydrate_bottom_data[
            "improper_types"]

        sim_data["molecules"] = sim_data["molecules"] + hydrate_data["molecules"] + hydrate_data["molecules"]

    else:
        total_box = sim_box
        total_adj_mat = sim_data["adj_mat"]

    if args.hydrate:
        sim_box_dims = (-args.bond_length, ((2 * args.bond_length) * args.dims[0]) + args.bond_length,
                        min(total_box[:, 1]) - args.bond_length - args.y_padding, max(total_box[:, 1]) + args.y_padding,
                        -args.bond_length, ((2 * args.bond_length) * args.dims[2]) + args.bond_length)
    else:
        if args.monolayer:
            # Need some headspace above the surface cations in a monolayer, otherwise the simulation box will be unstable. Arbitrarily add half the length of half the box to the y dimensions; an npt sim will get the "correct" box dims.
            # otherwise, if studying a vacancy, will need to like quadruple this so the surface cation has space to move out of plane.
            sim_box_dims = (-args.bond_length, ((2 * args.bond_length) * args.dims[0]) + args.bond_length,
                            min(sim_box[:, 1]) + (min(sim_box[:, 1]) * 0.5) - args.y_padding,
                            max(sim_box[:, 1]) + (max(sim_box[:, 1]) * 0.5) + args.y_padding, -args.bond_length,
                            ((2 * args.bond_length) * args.dims[2]) + args.bond_length)
        else:
            sim_box_dims = (-args.bond_length, ((2 * args.bond_length) * args.dims[0]) + args.bond_length,
                            -args.bond_length - args.y_padding,
                            ((2 * args.bond_length) * args.dims[1]) + args.bond_length + args.y_padding,
                            -args.bond_length, ((2 * args.bond_length) * args.dims[2]) + args.bond_length)

    # Write out the *.xyz output file
    print('\nWriting "{}.xyz" output'.format(args.output))
    with open(args.output + '/' + args.output + '.xyz', 'w') as f:
        f.write('{}\n'.format(len(sim_data["elements"])))

        if build_heterojunction:
            f.write('perovskite_{}{}{}\n'.format(anions[0] + '-' + anions[1], args.metal, args.cation))
        else:
            f.write('perovskite_{}{}{}\n'.format(args.anion, args.metal, args.cation))

        for i in range(len(sim_data["elements"])):
            f.write('{:<40s} {:<20.6f} {:<20.6f} {:<20.6f}\n'.format(sim_data["elements"][i], total_box[i][0],
                                                                     total_box[i][1], total_box[i][2]))

    print('\nWriting LAMMPS data file "{}.data"'.format(args.output))

    # Generate VDW parameters
    VDW_params = initialize_VDW(sorted(set(sim_data["atom_types"])), 1.0, 1.0, sim_data["VDW_params"], 0,
                                args.mixing_rule, args.UFF_supplement, args.print_lj)

    # Generate Simulation Dictionaries
    # The bond, angle, and diehdral parameters for each molecule are combined into one dictionary
    Bond_params = {};
    Angle_params = {};
    Dihedral_params = {};
    Improper_params = {};
    Masses = {}
    for j in list(sim_data["bond_params"].keys()): Bond_params[j] = sim_data["bond_params"][j]
    for j in list(sim_data["angle_params"].keys()): Angle_params[j] = sim_data["angle_params"][j]
    for j in list(sim_data["dihedral_params"].keys()): Dihedral_params[j] = sim_data["dihedral_params"][j]
    for j in list(sim_data["improper_params"].keys()): Improper_params[j] = sim_data["improper_params"][j]
    for j in list(sim_data["all_masses"].keys()): Masses[j] = sim_data["all_masses"][j]

    Atom_type_dict, Bond_type_dict, Angle_type_dict, fixed_modes = Write_data(args.output + '/' + args.output,
                                                                              sim_data["atom_types"], sim_box_dims,
                                                                              sim_data["elements"], total_box,
                                                                              sim_data["bonds"], sim_data["bond_types"],
                                                                              Bond_params, sim_data["angles"],
                                                                              sim_data["angle_types"], Angle_params,
                                                                              sim_data["dihedrals"],
                                                                              sim_data["dihedral_types"],
                                                                              Dihedral_params, sim_data["impropers"],
                                                                              sim_data["improper_types"],
                                                                              Improper_params, sim_data["charges"],
                                                                              VDW_params, Masses, sim_data["molecules"],
                                                                              sim_data['VDW_comments'], False)

    print('\nLead atom types:')
    if '[82[35]]' in list(Atom_type_dict.keys()):
        print('    Pb-Br: {}'.format(Atom_type_dict['[82[35]]']))
    if '[82[53]]' in list(Atom_type_dict.keys()):
        print('    Pb-I: {}'.format(Atom_type_dict['[82[53]]']))

    if args.hydrate:
        print('\nWater bond LAMMPS ID: {}'.format(Bond_type_dict[('[1[8[1]]]', '[8[1][1]]')]))
        print('Water angle LAMMPS ID: {}'.format(Angle_type_dict[('[1[8[1]]]', '[8[1][1]]', '[1[8[1]]]')]))
        print(
            'LAMMPS fix shake command to hold water bond and angle rigid:\n         fix    water_spce all shake 0.0001 20 0 b {} a {}'.format(
                Bond_type_dict[('[1[8[1]]]', '[8[1][1]]')], Angle_type_dict[('[1[8[1]]]', '[8[1][1]]', '[1[8[1]]]')]))
        if options['TIP4P_flag']:
            print('LAMMPS pair_style for TIP4P:\n         lj/cut/tip4p/long {} {} {} {} 0.1250 15.0 15.0'.format(
                Atom_type_dict['[8[1][1]]'], Atom_type_dict['[1[8[1]]]'], Bond_type_dict[('[1[8[1]]]', '[8[1][1]]')],
                Angle_type_dict[('[1[8[1]]]', '[8[1][1]]', '[1[8[1]]]')]))

    print('\nWriting atom type correspondence file "{}_correspondence.map"'.format(args.output))
    with open(args.output + '/' + args.output + '_correspondence.map', 'w') as f:
        f.write('{}\n'.format(len(sim_data["elements"])))

        if build_heterojunction:
            f.write('perovskite_{}{}{}\n\n'.format(anions[0] + '-' + anions[1], args.metal, args.cation))
        else:
            f.write('perovskite_{}{}{}\n\n'.format(args.anion, args.metal, args.cation))

        for key in Atom_type_dict:
            f.write('{0:<50} {1:<3} \n'.format(key, Atom_type_dict[key]))

        f.write('\n\n')
        for i in range(len(sim_data["elements"])):
            f.write('{}\t {}\t {}\t {}\n'.format(i, sim_data["elements"][i], sim_data["atom_types"][i],
                                                 Atom_type_dict[sim_data["atom_types"][i]]))

    # Write map file to more easily post-process the trajectory
    print("Writing mapfile ({})...".format(args.output + '_map.map'))
    write_map(args.output + '/' + args.output, sim_data["elements"], sim_data["atom_types"], sim_data["charges"],
              sim_data["masses"], total_adj_mat, numpy.zeros([len(sim_data["elements"])]), mols_placed)

    if not args.hydrate:
        print('Writing LAMMPS tether fix to "{}"...'.format(args.output + '_tether.fix'))
        with open(args.output + '/' + args.output + '_tether.fix', 'w') as f:
            f.write('group surface-heads id {}\n'.format(' '.join(str(_ + 1) for _ in surface_head_indices)))
            f.write('fix tether surface-heads spring/self 10.0\n\nunfix		tether')

    if args.water_pairs:
        print('depricated...')

    print('\nFinished!\n')
    return


# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2.
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical
# y-axis results in 1,0,0.
#
# Point: 1x3 array, coordinates to be rotated
# v1: 1x3 array, rotation direction vector
# v2: 1x3 array, point the rotation passes through (default is the origin)
# theta: scalar, magnitude of the rotation (defined by default in degrees) (default performs no rotation)

# from TAFFI/Lib/transify.py. Original author: Brett Savoie
def axis_rot(Point, v1, v2=[0.0, 0.0, 0.0], theta=0.0, mode='angle'):
    # Temporary variable for performing the transformation
    rotated = numpy.array([Point[0], Point[1], Point[2]])

    # If mode is set to 'angle' then theta needs to be converted to radians to be compatible with the
    # definition of the rotation vectors
    if mode == 'angle':
        theta = theta * numpy.pi / 180.0

    # Rotation carried out using formulae defined here (11/22/13) http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/)
    # Adapted for the assumption that v1 is the direction vector and v2 is a point that v1 passes through
    a = v2[0]
    b = v2[1]
    c = v2[2]
    u = v1[0]
    v = v1[1]
    w = v1[2]
    L = u ** 2 + v ** 2 + w ** 2

    # Rotate Point
    x = rotated[0]
    y = rotated[1]
    z = rotated[2]

    # x-transformation
    rotated[0] = (a * (v ** 2 + w ** 2) - u * (b * v + c * w - u * x - v * y - w * z)) \
                 * (1.0 - math.cos(theta)) + L * x * math.cos(theta) + L ** (0.5) * (
                             -c * v + b * w - w * y + v * z) * math.sin(theta)

    # y-transformation
    rotated[1] = (b * (u ** 2 + w ** 2) - v * (a * u + c * w - u * x - v * y - w * z)) \
                 * (1.0 - math.cos(theta)) + L * y * math.cos(theta) + L ** (0.5) * (
                             c * u - a * w + w * x - u * z) * math.sin(theta)

    # z-transformation
    rotated[2] = (c * (u ** 2 + v ** 2) - w * (a * u + b * v - u * x - v * y - w * z)) \
                 * (1.0 - math.cos(theta)) + L * z * math.cos(theta) + L ** (0.5) * (
                             -b * u + a * v - v * x + u * y) * math.sin(theta)

    rotated = rotated / L
    return rotated


def read_xyz(xyz_file):
    # Reads an *.xyz file and returns the origin-centered geometry and elements, plus the centroid (which should be (0,0,0))

    # Check to make sure the data file exists.
    if not os.path.isfile(xyz_file):
        print('\nERROR: Specified .yxz file "{}" not found. Aborting....\n'.format(xyz_file))
        exit()

    elements = []
    with open(xyz_file, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                atom_count = int(line)
                geometry = numpy.zeros([atom_count, 3])

            elif count == 1:
                # skip the blank line / comment line
                continue

            else:
                fields = line.split()
                elements.append(str(fields[0]))
                geometry[count - 2] = float(fields[1]), float(fields[2]), float(fields[3])

    # Center the molecule at the origin
    geometry -= (numpy.mean(geometry[:, 0]), numpy.mean(geometry[:, 1]), numpy.mean(geometry[:, 2]))

    # Compute centroid
    centroid = geometry.mean(axis=0)

    return atom_count, elements, geometry, centroid


def get_masses(elements):
    # Returns the mass of an element
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H': 1.00794, 'He': 4.002602, 'Li': 6.941, 'Be': 9.012182, 'B': 10.811, 'C': 12.011, 'N': 14.00674,
                 'O': 15.9994, 'F': 18.9984032, 'Ne': 20.1797, \
                 'Na': 22.989768, 'Mg': 24.3050, 'Al': 26.981539, 'Si': 28.0855, 'P': 30.973762, 'S': 32.066,
                 'Cl': 35.4527, 'Ar': 39.948, \
                 'K': 39.0983, 'Ca': 40.078, 'Sc': 44.955910, 'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961,
                 'Mn': 54.938049, 'Fe': 55.845, 'Co': 58.933200, 'Ni': 58.6934, 'Cu': 63.546, 'Zn': 65.39, \
                 'Ga': 69.723, 'Ge': 72.61, 'As': 74.92159, 'Se': 78.96, 'Br': 79.904, 'Kr': 83.80, \
                 'Rb': 85.4678, 'Sr': 87.62, 'Y': 88.90585, 'Zr': 91.224, 'Nb': 92.90638, 'Mo': 95.94, 'Tc': 98.0,
                 'Ru': 101.07, 'Rh': 102.90550, 'Pd': 106.42, 'Ag': 107.8682, 'Cd': 112.411, \
                 'In': 114.818, 'Sn': 118.710, 'Sb': 121.760, 'Te': 127.60, 'I': 126.90447, 'Xe': 131.29, \
                 'Cs': 132.90545, 'Ba': 137.327, 'La': 138.9055, 'Hf': 178.49, 'Ta': 180.9479, 'W': 183.84,
                 'Re': 186.207, 'Os': 190.23, 'Ir': 192.217, 'Pt': 195.078, 'Au': 196.96655, 'Hg': 200.59, \
                 'Tl': 204.3833, 'Pb': 207.2, 'Bi': 208.98038, 'Po': 209.0, 'At': 210.0, 'Rn': 222.0}

    # masses = [ mass_dict[i] for i in elements ]
    masses = [mass_dict[i] for i in elements]
    # return mass_dict[elements]
    return masses


# A wrapper for the commands to parse the bonds, angles, and dihedrals from the adjacency matrix.
# Returns:   list of atomtypes, bond_types, bond instances, angle_types, angle instances, dihedral_types,
#            diehdral instances, charges, and VDW parameters.
def Find_parameters(Adj_mat, Geometry, Atom_types, FF_db="FF_file", Improper_flag=False):
    # List comprehension to determine bonds from a loop over the adjacency matrix. Iterates over rows (i) and individual elements
    # ( elements A[count_i,count_j] = j ) and stores the bond if the element is "1". The count_i < count_j condition avoids
    # redudant bonds (e.g., (i,j) vs (j,i) ). By convention only the i < j definition is stored.
    print("\t\tParsing bonds...")
    Bonds = [(count_i, count_j) for count_i, i in enumerate(Adj_mat) for count_j, j in enumerate(i) if j == 1]
    Bond_types = [(Atom_types[i[0]], Atom_types[i[1]]) for i in Bonds]

    # List comprehension to determine angles from a loop over the bonds. Note, since there are two bonds in every angle, there will be
    # redundant angles stored (e.g., (i,j,k) vs (k,j,i) ). By convention only the i < k definition is stored.
    print("\t\tParsing angles...")
    Angles = [(count_j, i[0], i[1]) for i in Bonds for count_j, j in enumerate(Adj_mat[i[0]]) if
              j == 1 and count_j not in i]
    Angle_types = [(Atom_types[i[0]], Atom_types[i[1]], Atom_types[i[2]]) for i in Angles]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("\t\tParsing dihedrals...")
    Dihedrals = [(count_j, i[0], i[1], i[2]) for i in Angles for count_j, j in enumerate(Adj_mat[i[0]]) if
                 j == 1 and count_j not in i]
    Dihedral_types = [(Atom_types[i[0]], Atom_types[i[1]], Atom_types[i[2]], Atom_types[i[3]]) for i in Dihedrals]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    if Improper_flag:    print("\t\tParsing impropers...")
    Impropers = [(i[1], i[0], i[2], count_j) for i in Angles for count_j, j in enumerate(Adj_mat[i[1]]) if
                 j == 1 and count_j not in i]
    Improper_types = [(Atom_types[i[0]], Atom_types[i[1]], Atom_types[i[2]], Atom_types[i[3]]) for i in Impropers]

    # Canonicalize the modes
    for i in range(len(Bonds)):
        Bond_types[i], Bonds[i] = adjacency.canon_bond(Bond_types[i], ind=Bonds[i])
    for i in range(len(Angles)):
        Angle_types[i], Angles[i] = adjacency.canon_angle(Angle_types[i], ind=Angles[i])
    for i in range(len(Dihedrals)):
        Dihedral_types[i], Dihedrals[i] = adjacency.canon_dihedral(Dihedral_types[i], ind=Dihedrals[i])
    for i in range(len(Impropers)):
        Improper_types[i], Impropers[i] = adjacency.canon_improper(Improper_types[i], ind=Impropers[i])

        # Remove redundancies
    if len(Bonds) > 0:
        Bonds, Bond_types = list(zip(*[(i, Bond_types[count_i]) for count_i, i in enumerate(Bonds) if count_i ==
                                       [count_j for count_j, j in enumerate(Bonds) if j == i or j[::-1] == i][0]]))

    if len(Angles) > 0:
        Angles, Angle_types = list(zip(*[(i, Angle_types[count_i]) for count_i, i in enumerate(Angles) if count_i ==
                                         [count_j for count_j, j in enumerate(Angles) if j == i or j[::-1] == i][0]]))

    if len(Dihedrals) > 0:
        Dihedrals, Dihedral_types = list(zip(*[(i, Dihedral_types[count_i]) for count_i, i in enumerate(Dihedrals) if
                                               count_i == [count_j for count_j, j in enumerate(Dihedrals) if
                                                           j == i or j[::-1] == i][0]]))

    if len(Impropers) > 0:
        Impropers, Improper_types = list(zip(*[(i, Improper_types[count_i]) for count_i, i in enumerate(Impropers) if
                                               count_i == [count_j for count_j, j in enumerate(Impropers) if
                                                           j[0] == i[0] and len(set(i[1:]).intersection(set(j[1:])))][
                                                   0]]))

    ##############################################################
    # Read in parameters: Here the stretching, bending, dihedral #
    # and non-bonding interaction parameters are read in from    #
    # parameters file. Mass and charge data is also included.    #
    # The program looks for a simple match for the first entry   #
    # in each line with one of the bond or angle types.          #
    # INPUT: param_file, BOND_TYPES_LIST, ANGLE_TYPES_LIST       #
    #        DIHEDRAL_TYPES_LIST, ELEMENTS                       #
    # OUTPUT: CHARGES, MASSES, BOND_PARAMS, ANGLE_PARAMS,        #
    #         DIHERAL_PARAMS, PW_PARAMS                          #
    ##############################################################

    # Initialize dictionaries

    # Read in masses and charges
    Masses = {}
    with open(FF_db, 'r') as f:
        content = f.readlines()

    for lines in content:
        fields = lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'atom':
            Masses[fields[1]] = float(fields[3])

            # Read in bond parameters
    Bond_params = {}
    with open(FF_db, 'r') as f:
        content = f.readlines()

    for lines in content:
        fields = lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0] == 'bond':
            if fields[3] == "harmonic":
                Bond_params[(fields[1], fields[2])] = [float(fields[4]), float(fields[5])]
            elif fields[3] == "fixed":
                Bond_params[(fields[1], fields[2])] = ["fixed", 0.0, float(fields[4])]
            else:
                print(
                    "ERROR: only harmonic and fixed bond definitions are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Read in angle parameters
    Angle_params = {}
    with open(FF_db, 'r') as f:
        content = f.readlines()

    for lines in content:
        fields = lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'angle':
            if fields[4] == "harmonic":
                Angle_params[(fields[1], fields[2], fields[3])] = [float(fields[5]), float(fields[6])]
            elif fields[4] == "fixed":
                Angle_params[(fields[1], fields[2], fields[3])] = ["fixed", 0.0, float(fields[5])]
            else:
                print(
                    "ERROR: only harmonic angle definitions are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Read in dihedral parameters
    Dihedral_params = {}
    with open(FF_db, 'r') as f:
        content = f.readlines()

    for lines in content:
        fields = lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['dihedral', 'torsion']:
            if fields[5] == "opls":
                Dihedral_params[(fields[1], fields[2], fields[3], fields[4])] = [fields[5]] + [float(i) for i in
                                                                                               fields[6:10]]
            elif fields[5] == "harmonic":
                Dihedral_params[(fields[1], fields[2], fields[3], fields[4])] = [fields[5]] + [float(fields[6]),
                                                                                               int(float(fields[7])),
                                                                                               int(float(fields[8]))]
            else:
                print(
                    "ERROR: Only opls and harmonic dihedral types are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Read in improper parameters
    Improper_params = {}
    with open(FF_db, 'r') as f:
        content = f.readlines()

    for lines in content:
        fields = lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['improper']:
            if fields[5] == "harmonic":
                Improper_params[(fields[1], fields[2], fields[3], fields[4])] = [fields[5]] + [float(fields[6]),
                                                                                               float(fields[7])]
            else:
                print(
                    "ERROR: Only opls type dihedral definitions are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Search for charges based on atom type
    with open(FF_db, 'r') as f:
        content = f.readlines()

    Charges = numpy.zeros(len(Atom_types))
    Charges.fill(-100.0)
    for i in range(len(Atom_types)):
        for lines in content:
            fields = lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue

            if fields[0].lower() in ['charge'] and Atom_types[i] == fields[1]:
                Charges[i] = float(fields[2])

    # Search for VDW parameters
    VDW_params = {}
    VDW_comments = {}
    with open(FF_db, 'r') as f:
        for lines in f:
            fields = lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue

            if fields[0].lower() in ['vdw']:

                # Only two parameters are required for lj types
                if fields[3].lower() == "lj":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1], fields[2])] = [fields[3], float(fields[4]), float(fields[5])]
                        if len(fields) > 6:
                            VDW_comments[(fields[1], fields[2])] = ' '.join(fields[6:])
                        else:
                            VDW_comments[(fields[1], fields[2])] = ''
                    else:
                        VDW_params[(fields[2], fields[1])] = [fields[3], float(fields[4]), float(fields[5])]
                        if len(fields) > 6:
                            VDW_comments[(fields[2], fields[1])] = ' '.join(fields[6:])
                        else:
                            VDW_comments[(fields[2], fields[1])] = ''

                elif fields[3].lower() == "tip4p":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1], fields[2])] = [fields[3], float(fields[4]), float(fields[5])]
                        if len(fields) > 6:
                            VDW_comments[(fields[1], fields[2])] = ' '.join(fields[6:])
                        else:
                            VDW_comments[(fields[1], fields[2])] = ''
                    else:
                        VDW_params[(fields[2], fields[1])] = [fields[3], float(fields[4]), float(fields[5])]
                        if len(fields) > 6:
                            VDW_comments[(fields[2], fields[1])] = ' '.join(fields[6:])
                        else:
                            VDW_comments[(fields[2], fields[1])] = ''

                elif fields[3].lower() == "buck":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1], fields[2])] = [fields[3], float(fields[4]), float(fields[5]),
                                                              float(fields[6])]
                        if len(fields) > 7:
                            VDW_comments[(fields[1], fields[2])] = ' '.join(fields[7:])
                        else:
                            VDW_comments[(fields[1], fields[2])] = ''
                    else:
                        VDW_params[(fields[2], fields[1])] = [fields[3], float(fields[4]), float(fields[5]),
                                                              float(fields[6])]
                        if len(fields) > 7:
                            VDW_comments[(fields[2], fields[1])] = ' '.join(fields[7:])
                        else:
                            VDW_comments[(fields[2], fields[1])] = ''

    # Check for missing parameters
    Missing_masses = [i for i in Atom_types if str(i) not in list(Masses.keys())]
    Missing_charges = [count_i for count_i, i in enumerate(Charges) if i == -100.0];
    Missing_charges = [Atom_types[i] for i in Missing_charges]
    Missing_bonds = [i for i in Bond_types if (i[0], i[1]) not in list(Bond_params.keys())]
    Missing_angles = [i for i in Angle_types if (i[0], i[1], i[2]) not in list(Angle_params.keys())]
    Missing_dihedrals = [i for i in Dihedral_types if (i[0], i[1], i[2], i[3]) not in list(Dihedral_params.keys())]
    Missing_impropers = []
    if Improper_flag is True: Missing_impropers = [i for i in Improper_types if
                                                   (i[0], i[1], i[2], i[3]) not in list(Improper_params.keys())]

    # Print diagnostics on missing parameters and quit if the prerequisites are missing.
    if (len(Missing_masses) + len(Missing_charges) + len(Missing_bonds) + len(Missing_angles) + len(
            Missing_dihedrals) + len(Missing_impropers)) > 0:
        print("\nUh Oh! There are missing FF parameters...\n")

        if Missing_masses:
            print("\nMissing masses for the following atom types:")
            for i in set(Missing_masses):
                print(i)

        if Missing_charges:
            print("\nMissing charges for the following atom types:")
            for i in set(Missing_charges):
                print(i)

        if Missing_bonds:
            print("\nMissing bond parameters for the following bond types:")
            for i in set(Missing_bonds):
                print('{}    {}'.format(i[0], i[1]))

        if Missing_angles:
            print("\nMissing angle parameters for the following angle types:")
            for i in set(Missing_angles):
                print('{}    {}    {}'.format(i[0], i[1], i[2]))

        if Missing_dihedrals:
            print("\nMissing dihedral parameters for the following dihedral types:")
            for i in set(Missing_dihedrals):
                print('{}    {}    {}    {}'.format(i[0], i[1], i[2], i[3]))

        if Improper_flag and Missing_impropers:
            print("\nMissing improper parameters for the following improper types:")
            for i in set(Missing_impropers):
                print('{}    {}    {}    {}'.format(i[0], i[1], i[2], i[3]))

        print("\nEnsure the specification of the missing parameters. Exiting...")
        quit()

    return list(Bonds), list(Bond_types), Bond_params, list(Angles), list(Angle_types), Angle_params, list(
        Dihedrals), list(Dihedral_types), Dihedral_params, list(Impropers), list(
        Improper_types), Improper_params, Charges.tolist(), Masses, VDW_params, VDW_comments


# Wrapper function for writing the lammps data file which is read by the .in.init file during run initialization
def Write_data(Filename, Atom_types, Sim_Box, Elements, Geometry, Bonds, Bond_types, Bond_params, Angles, Angle_types,
               Angle_params, Dihedrals, Dihedral_types, Dihedral_params, \
               Impropers, Improper_types, Improper_params, Charges, VDW_params, Masses, Molecule, VDW_comments,
               Improper_flag=False):
    # Write an xyz for easy viewing
    # with open(Filename+'/'+Filename.split('/')[-1]+'.xyz','w') as f:
    # with open(Filename.split('/')[-1]+'.xyz','w') as f:
    with open(Filename + '.xyz', 'w') as f:
        f.write('{}\n\n'.format(len(Geometry)))
        for count_i, i in enumerate(Geometry):
            f.write('{:20s} {:< 20.6f} {:< 20.6f} {:< 20.6f}\n'.format(Elements[count_i], i[0], i[1], i[2]))

    # Create type dictionaries (needed to convert each atom,bond,angle, and dihedral type to consecutive numbers as per LAMMPS convention)
    # Note: LAMMPS orders atomtypes, bonds, angles, dihedrals, etc as integer types. Each of the following dictionaries holds the mapping between
    #       the true type (held in the various types lists) and the lammps type_id, obtained by enumerated iteration over the respective set(types).
    Atom_type_dict = {}
    for count_i, i in enumerate(sorted(set(Atom_types))):
        for j in Atom_types:
            if i == j:
                Atom_type_dict[i] = count_i + 1
            if i in list(Atom_type_dict.keys()):
                break
    Bond_type_dict = {}
    for count_i, i in enumerate(sorted(set(Bond_types))):
        for j in Bond_types:
            if i == j:
                Bond_type_dict[i] = count_i + 1
            if i in list(Bond_type_dict.keys()):
                break
    Angle_type_dict = {}
    for count_i, i in enumerate(sorted(set(Angle_types))):
        for j in Angle_types:
            if i == j:
                Angle_type_dict[i] = count_i + 1
            if i in list(Angle_type_dict.keys()):
                break
    Dihedral_type_dict = {}
    for count_i, i in enumerate(sorted(set(Dihedral_types))):
        for j in Dihedral_types:
            if i == j:
                Dihedral_type_dict[i] = count_i + 1
            if i in list(Dihedral_type_dict.keys()):
                break
    Improper_type_dict = {}
    for count_i, i in enumerate(sorted(set(Improper_types))):
        for j in Improper_types:
            if i == j:
                Improper_type_dict[i] = count_i + 1
            if i in list(Improper_type_dict.keys()):
                break

    # Write the data file
    # with open(Filename+'/'+Filename.split('/')[-1]+'.data','w') as f:
    # with open(Filename.split('/')[-1]+'.data','w') as f:
    with open(Filename + '.data', 'w') as f:

        # Write system properties
        f.write("LAMMPS data file via perovskite_builder, on {}\n\n".format(datetime.datetime.now()))

        f.write("{} atoms\n".format(len(Elements)))
        f.write("{} atom types\n".format(len(set(Atom_types))))
        if len(Bonds) > 0:
            f.write("{} bonds\n".format(len(Bonds)))
            f.write("{} bond types\n".format(len(set(Bond_types))))
        if len(Angles) > 0:
            f.write("{} angles\n".format(len(Angles)))
            f.write("{} angle types\n".format(len(set(Angle_types))))
        if len(Dihedrals) > 0:
            f.write("{} dihedrals\n".format(len(Dihedrals)))
            f.write("{} dihedral types\n".format(len(set(Dihedral_types))))
        if Improper_flag and len(Impropers) > 0:
            f.write("{} impropers\n".format(len(Impropers)))
            f.write("{} improper types\n".format(len(set(Improper_types))))
        f.write("\n")

        # Write box dimensions
        f.write("{:< 20.16f} {:< 20.16f} xlo xhi\n".format(Sim_Box[0], Sim_Box[1]))
        f.write("{:< 20.16f} {:< 20.16f} ylo yhi\n".format(Sim_Box[2], Sim_Box[3]))
        f.write("{:< 20.16f} {:< 20.16f} zlo zhi\n".format(Sim_Box[4], Sim_Box[5]))

        # Write Masses
        f.write("\nMasses\n\n")
        for count_i, i in enumerate(sorted(set(Atom_types))):
            for j in set(Atom_types):
                if Atom_type_dict[j] == count_i + 1:
                    f.write("{} {:< 8.6f}\n".format(count_i + 1, Masses[str(j)]))  # count_i+1 bc of LAMMPS 1-indexing
        f.write("\n")

        # Write Atoms
        f.write("Atoms\n\n")
        for count_i, i in enumerate(Atom_types):
            f.write("{:<8d} {:< 4d} {:< 4d} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:d} {:d} {:d}\n" \
                    .format(count_i + 1, Molecule[count_i], Atom_type_dict[i], Charges[count_i], Geometry[count_i, 0],
                            Geometry[count_i, 1], Geometry[count_i, 2], 0, 0, 0))

        # Write Bonds
        if len(Bonds) > 0:
            f.write("\nBonds\n\n")
            for count_i, i in enumerate(Bonds):
                f.write(
                    "{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1, Bond_type_dict[Bond_types[count_i]], i[0] + 1,
                                                           i[1] + 1))
                # f.write("{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Bond_type_dict[Bond_types[count_i]],i[0]+5,i[1]+5))

        # Write Angles
        if len(Angles) > 0:
            f.write("\nAngles\n\n")
            for count_i, i in enumerate(Angles):
                f.write(
                    "{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1, Angle_type_dict[Angle_types[count_i]],
                                                                  i[0] + 1, i[1] + 1, i[2] + 1))
                # f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Angle_type_dict[Angle_types[count_i]],i[0]+5,i[1]+5,i[2]+5))

        # Write Dihedrals
        if len(Dihedrals) > 0:
            f.write("\nDihedrals\n\n")
            for count_i, i in enumerate(Dihedrals):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1, Dihedral_type_dict[
                    Dihedral_types[count_i]], i[0] + 1, i[1] + 1, i[2] + 1, i[3] + 1))  # original line
                # f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Dihedral_type_dict[Dihedral_types[count_i]],i[0]+5,i[1]+5,i[2]+5,i[3]+5))  #+1 already there. added 4 to get 5

        # Write Impropers
        if Improper_flag and len(Impropers) > 0:
            f.write("\nImpropers\n\n")
            for count_i, i in enumerate(Impropers):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i + 1, Improper_type_dict[
                    Improper_types[count_i]], i[0] + 1, i[1] + 1, i[2] + 1, i[3] + 1))
                # f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Improper_type_dict[Improper_types[count_i]],i[0]+5,i[1]+5,i[2]+5,i[3]+5))

    # Write the settings file
    fixed_modes = {'bonds': [], 'angles': []}
    # with open(Filename+'/'+Filename.split('/')[-1]+'.in.settings','w') as f:
    # with open(Filename.split('/')[-1]+'.in.settings','w') as f:
    with open(Filename + '.in.settings', 'w') as f:

        # Write non-bonded interactions (the complicated form of this loop is owed
        # to desire to form a nicely sorted list in terms of the lammps atom types
        # Note: Atom_type_dict was initialize according to sorted(set(Atom_types)
        #       so iterating over this list (twice) orders the pairs, too.
        f.write("     {}\n".format("# Non-bonded interactions (pair-wise)"))
        for count_i, i in enumerate(sorted(set(Atom_types))):
            for count_j, j in enumerate(sorted(set(Atom_types))):
                lj_note = ''

                # Skip duplicates
                if count_j < count_i:
                    continue

                # Conform to LAMMPS i <= j formatting
                if Atom_type_dict[i] <= Atom_type_dict[j]:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff", Atom_type_dict[i], Atom_type_dict[j]))
                else:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff", Atom_type_dict[j], Atom_type_dict[i]))

                # Determine key (ordered by initialize_VDW function such that i > j)
                if i > j:
                    key = (i, j)
                else:
                    key = (j, i)

                for k in VDW_params[key]:
                    if type(k) is str:
                        f.write("{:20s} ".format(k))
                    if type(k) is float:
                        f.write("{:< 20.6f} ".format(k))
                try:
                    lj_note = VDW_comments[key]
                except:
                    lj_note = ''

                f.write("        {}\n".format(lj_note))

        # Write stretching interactions
        # Note: Bond_type_dict was initialized by looping over sorted(set(Bond_types)), so
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Stretching interactions"))
        for i in sorted(set(Bond_types)):
            f.write("     {:20s} {:<10d} ".format("bond_coeff", Bond_type_dict[i]))
            for j in Bond_params[i]:
                if j == "fixed":
                    continue
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
            f.write("\n")

            # populate fixed_modes
            if Bond_params[i][0] == "fixed":
                fixed_modes["bonds"] += [Bond_type_dict[i]]

        # Write bending interactions
        # Note: Angle_type_dict was initialized by looping over sorted(set(Angle_types)), so
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Bending interactions"))
        for i in sorted(set(Angle_types)):
            f.write("     {:20s} {:<10d} ".format("angle_coeff", Angle_type_dict[i]))
            for j in Angle_params[i]:
                if j == "fixed":
                    continue
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
            f.write("\n")

            # populate fixed_modes
            if Angle_params[i][0] == "fixed":
                fixed_modes["angles"] += [Angle_type_dict[i]]

        # Write dihedral interactions
        # Note: Dihedral_type_dict was initialized by looping over sorted(set(Dihedral_types)), so
        #       iterating over this list resulted in ordered parameters in the in.settings file
        f.write("\n     {}\n".format("# Dihedral interactions"))
        for i in sorted(set(Dihedral_types)):
            f.write("     {:20s} {:<10d} ".format("dihedral_coeff", Dihedral_type_dict[i]))
            for j in Dihedral_params[i][
                     1:]:  # Starting from [1:] surpresses writing out the dihedral type, since they all should be OPLS for the perovskites
                if type(j) is str:
                    f.write("{:20s} ".format(j))
                if type(j) is float:
                    f.write("{:< 20.6f} ".format(j))
                if type(j) is int:
                    f.write("{:< 20d} ".format(j))
            f.write("\n")

        # Write improper interactions
        # Note: Improper_type_dict was initialized by looping over sorted(set(Improper_types)), so
        #       iterating over this list resulted in ordered parameters in the in.settings file
        if Improper_flag:
            f.write("\n     {}\n".format("# Improper interactions"))
            for i in sorted(set(Improper_types)):
                f.write("     {:20s} {:<10d} ".format("improper_coeff", Improper_type_dict[i]))
                for j in Improper_params[i][1:]:
                    if type(j) is str:
                        f.write("{:20s} ".format(j))
                    if type(j) is float:
                        f.write("{:< 20.6f} ".format(j))
                    if type(j) is int:
                        f.write("{:< 20d} ".format(j))
                f.write("\n")

    return Atom_type_dict, Bond_type_dict, Angle_type_dict, fixed_modes


# Description: Initialize VDW_dict based on UFF parameters for the initial guess of the fit.
def initialize_VDW(atomtypes, sigma_scale=1.0, eps_scale=1.0, VDW_FF={}, Force_UFF=0, mixing_rule='none',
                   UFF_supplement=False, print_LJ=False):
    # Initialize UFF parameters (tuple corresponds to eps,sigma pairs for each element)
    # Taken from UFF (Rappe et al. JACS 1992)
    # Note: LJ parameters in the table are specificed in the eps,r_min form rather than eps,sigma
    #       the conversion between r_min and sigma is sigma = r_min/2^(1/6)
    # Note: Units for sigma = angstroms and eps = kcal/mol
    UFF_dict = {1: (0.044, 2.5711337005530193), 2: (0.056, 2.1043027722474816), 3: (0.025, 2.183592758161972),
                4: (0.085, 2.4455169812952313), \
                5: (0.180, 3.6375394661670053), 6: (0.105, 3.4308509635584463), 7: (0.069, 3.260689308393642),
                8: (0.060, 3.1181455134911875), \
                9: (0.050, 2.996983287824101), 10: (0.042, 2.88918454292912), 11: (0.030, 2.657550876212632),
                12: (0.111, 2.6914050275019648), \
                13: (0.505, 4.008153332913386), 14: (0.402, 3.82640999441276), 15: (0.305, 3.694556984127987),
                16: (0.274, 3.594776327696269), \
                17: (0.227, 3.5163772404999194), 18: (0.185, 3.4459962417668324), 19: (0.035, 3.396105913550973),
                20: (0.238, 3.0281647429590133), \
                21: (0.019, 2.935511276272418), 22: (0.017, 2.828603430095577), 23: (0.016, 2.800985569833227),
                24: (0.015, 2.6931868249382456), \
                25: (0.013, 2.6379511044135446), 26: (0.013, 2.5942970672246677), 27: (0.014, 2.558661118499054),
                28: (0.015, 2.5248069672097215), \
                29: (0.005, 3.113691019900486), 30: (0.124, 2.4615531582217574), 31: (0.415, 3.904809081609107),
                32: (0.379, 3.813046513640652), \
                33: (0.309, 3.7685015777336357), 34: (0.291, 3.746229109780127), 35: (0.251, 3.731974730289881),
                36: (0.220, 3.689211591819145), \
                37: (0.040, 3.6651573264293558), 38: (0.235, 3.2437622327489755), 39: (0.072, 2.980056212179435),
                40: (0.069, 2.78316759547042), \
                41: (0.059, 2.819694442914174), 42: (0.056, 2.7190228877643157), 43: (0.048, 2.670914356984738),
                44: (0.056, 2.6397329018498255), \
                45: (0.053, 2.6094423454330538), 46: (0.048, 2.5827153838888437), 47: (0.036, 2.804549164705788),
                48: (0.228, 2.537279549263686), \
                49: (0.599, 3.976080979060334), 50: (0.567, 3.9128271700723705), 51: (0.449, 3.937772334180300),
                52: (0.398, 3.982317270087316), \
                53: (0.339, 4.009044231631527), 54: (0.332, 3.923517954690054), 55: (0.045, 4.024189509839913),
                56: (0.364, 3.2989979532736764), \
                72: (0.072, 2.798312873678806), 73: (0.081, 2.8241489365048755), 74: (0.067, 2.734168165972701),
                75: (0.066, 2.631714813386562), \
                76: (0.037, 2.7796040005978586), 77: (0.073, 2.5301523595185635), 78: (0.080, 2.453535069758495),
                79: (0.039, 2.9337294788361374), \
                80: (0.385, 2.4098810325696176), 81: (0.680, 3.872736727756055), 82: (0.663, 3.828191791849038),
                83: (0.518, 3.893227398273283), \
                84: (0.325, 4.195242063722858), 85: (0.284, 4.231768911166611), 86: (0.248, 4.245132391938716)}

    # NEW: Initialize VDW_dict first guess based on element types and Lorentz-Berthelot mixing rules
    # Order of operations: (1) If the parameters are in the supplied FF database then they are used as is
    # (2) If the self-terms are in the supplied FF datebase then they are used to generate the mixed
    # interactions (3) UFF parameters are used.
    VDW_dict = {}
    origin = {}
    VDW_styles = []
    missing_params = False
    missing_list = []
    missing_pairs = []

    for count_i, i in enumerate(atomtypes):
        for count_j, j in enumerate(atomtypes):
            if count_i < count_j:
                continue

            # Check for parameters in the database
            if (i, j) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(i, j)][0] == "lj":
                    VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(i, j)][0] == "buck":
                    VDW_type = "buck/coul/long"
                elif VDW_FF[(i, j)][0] == "tip4p":
                    VDW_type = "lj/cut/tip4p/long"
                else:
                    print("\n\nERROR in initialize_VDW: only lj and buck pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j:
                    VDW_dict[(i, j)] = [VDW_type] + VDW_FF[(i, j)][1:]
                else:
                    VDW_dict[(j, i)] = [VDW_type] + VDW_FF[(i, j)][1:]
                origin[(i, j)] = origin[(j, i)] = "read"

            # Check for reverse combination
            elif (j, i) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(j, i)][0] == "lj":
                    VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(j, i)][0] == "buck":
                    VDW_type = "buck/coul/long"
                elif VDW_FF[(i, j)][0] == "tip4p":
                    VDW_type = "lj/cut/tip4p/long"
                else:
                    print("\n\nERROR in initialize_VDW: only lj and buck pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j:
                    VDW_dict[(i, j)] = [VDW_type] + VDW_FF[(j, i)][1:]
                else:
                    VDW_dict[(j, i)] = [VDW_type] + VDW_FF[(j, i)][1:]
                origin[(i, j)] = origin[(j, i)] = "read"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i, i) in VDW_FF and (j, j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'lb':

                # Check compatibility with mixing rules
                if VDW_FF[(i, i)][0].lower() not in ["lj", "tip4p"] or VDW_FF[(j, j)][0].lower() not in ["lj", "tip4p"]:
                    print(
                        "\n\nERROR in initialize_VDW: only lj styles support mixing rules ({},{})x({},{}). Exiting...".format(
                            i, VDW_FF[(i, i)][0], j, VDW_FF[(j, j)][0]))
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                eps = (VDW_FF[(i, i)][1] * VDW_FF[(j, j)][1]) ** (0.5) * eps_scale
                sigma = (VDW_FF[(i, i)][2] + VDW_FF[(j, j)][2]) / 2.0 * sigma_scale
                if i > j:
                    VDW_dict[(i, j)] = [VDW_type, eps, sigma]
                else:
                    VDW_dict[(j, i)] = [VDW_type, eps, sigma]
                origin[(i, j)] = origin[(j, i)] = "lb"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i, i) in VDW_FF and (j, j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'wh':

                # Check compatibility with mixing rules
                if VDW_FF[(i, i)][0].lower() not in ["lj", "tip4p"] or VDW_FF[(j, j)][0].lower() not in ["lj", "tip4p"]:
                    print(
                        "\n\nERROR in initialize_VDW: only lj styles support mixing rules ({},{})x({},{}). Exiting...".format(
                            i, VDW_FF[(i, i)][0], j, VDW_FF[(j, j)][0]))
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                sigma = ((VDW_FF[(i, i)][2] ** (6.0) + VDW_FF[(j, j)][2] ** (6.0)) / 2.0) ** (1.0 / 6.0)
                eps = (VDW_FF[(i, i)][1] * VDW_FF[(i, i)][2] ** (6.0) * VDW_FF[(j, j)][1] * VDW_FF[(j, j)][2] ** (
                    6.0)) ** (0.5) / sigma ** (6.0)
                if i > j:
                    VDW_dict[(i, j)] = [VDW_type, eps, sigma]
                else:
                    VDW_dict[(j, i)] = [VDW_type, eps, sigma]
                origin[(i, j)] = origin[(j, i)] = "wh"

            else:
                if Force_UFF == 1 or UFF_supplement:
                    VDW_type = "lj/cut/coul/long"
                    type_1 = int(i.split('[')[1].split(']')[0])
                    type_2 = int(j.split('[')[1].split(']')[0])
                    eps = (UFF_dict[type_1][0] * UFF_dict[type_2][0]) ** (0.5) * eps_scale
                    sigma = (UFF_dict[type_1][1] + UFF_dict[type_2][1]) / 2.0 * sigma_scale
                    if i > j:
                        VDW_dict[(i, j)] = [VDW_type, eps, sigma]
                    else:
                        VDW_dict[(j, i)] = [VDW_type, eps, sigma]
                    origin[(i, j)] = origin[(j, i)] = "UFF"

                else:
                    missing_list.append("\n\ninitialize_VDW: Database missing vdw parameters for ({},{}).".format(i, j))
                    missing_pairs.append("( {} , {} )".format(i, j))
                    missing_params = True
                    VDW_type = "missing"

            # Collect a list of the LAMMPS styles used in the simulation
            VDW_styles += [VDW_type]

    if missing_params:
        print('ERROR in initialize_VDW: Missing the following parameters:')
        for m in missing_list:
            print(m)
        print('\nPairs list:')
        for m in missing_pairs:
            print(m)
        print('\nAborting...')
        exit()

    if print_LJ:
        # Print summary
        print("\n{}".format("*" * 177))
        print("* {:^173s} *".format(
            "Initializing VDW parameters for the simulation (those with * were read from the FF file(s))"))
        print("*{}*".format("-" * 175))
        print("* {:<50s} {:<50s} {:<20s}  {:<18s} {:<18s} {:<8s}    *".format("Type", "Type", "VDW_type",
                                                                              "eps (kcal/mol)", "sigma (angstroms)",
                                                                              "origin"))
        print("{}".format("*" * 177))
        for j in list(VDW_dict.keys()):
            print("  {:<50s} {:<50s} {:<20s} {:< 18.4f} {:< 18.4f}  {:<18s}".format(j[0], j[1], VDW_dict[j][0],
                                                                                    VDW_dict[j][1], VDW_dict[j][2],
                                                                                    origin[j]))
        print("")

    return VDW_dict


# Wrapper function for the write commands for creating the *.map file
def write_map(Filename, Elements, Atom_types, Charges, Masses, Adj_mat, Structure, N_mol):
    # Open file for writing and write header (first two lines of the map file are header)
    with open(Filename + '_map.map', 'w') as f:
        f.write(
            '{} {}\n {:<50} {:<10} {:<10} {:<14}  {:<13} {}\n'.format(len(Atom_types), N_mol, 'Atom_type', 'Element',
                                                                      'Structure', 'Mass', 'Charge', 'Adj_mat'))
        for count_i, i in enumerate(Atom_types):
            adj_mat_entry = (' ').join([str(count_j) for count_j, j in enumerate(Adj_mat[count_i, :]) if j == 1])
            f.write(
                ' {:<50} {:<10} {:< 9d} {:<14.6f} {:< 14.8f} {}\n'.format(i, Elements[count_i], int(Structure[count_i]),
                                                                          Masses[count_i], Charges[count_i],
                                                                          adj_mat_entry))
        f.close()


def parse_metal(element, charge):
    Metal = {}
    Metal["count"] = 1
    Metal["elements"] = [element]
    Metal["geometry"] = numpy.zeros([1, 3])
    Metal["centroid"] = numpy.zeros([1, 3])
    Metal["charges"] = [charge]
    Metal["adj_mat"] = adjacency.Table_generator(Metal["elements"], Metal["geometry"])
    Metal["atom_types"] = id_types.id_types(Metal["elements"], Metal["adj_mat"])
    Metal["masses"] = {Metal["atom_types"][0]: get_masses(Metal["elements"])[0]}
    Metal["bonds"] = Metal["bond_types"] = Metal["bond_params"] = Metal["angles"] = Metal["angle_types"] = Metal[
        "angle_params"] = Metal["dihedrals"] = Metal["dihedral_types"] = Metal["dihedral_params"] = Metal["impropers"] = \
    Metal["improper_types"] = Metal["improper_params"] = Metal["VDW_params"] = Metal["VDW_comments"] = []
    return Metal


def parse_ion(element, charge, FF_db, ion):
    Ion = {}
    # Was a *.xyz file specified?
    if element.lower().endswith('.xyz'):
        print('\n*.xyz file specified for {}. Reading file "{}"...'.format(ion, element))

        if not os.path.isfile(FF_db):
            print('ERROR: Specified force field file ({}) does not exist. Aborting...'.format(FF_db))
            exit()

        Ion["count"], Ion["elements"], Ion["geometry"], Ion["centroid"] = read_xyz(element)
        Ion["adj_mat"] = adjacency.Table_generator(Ion["elements"], Ion["geometry"])
        Ion["atom_types"] = id_types.id_types(Ion["elements"], Ion["adj_mat"])

        print('\tParsing force field information...')
        Ion["bonds"], Ion["bond_types"], Ion["bond_params"], Ion["angles"], Ion["angle_types"], Ion["angle_params"], \
        Ion["dihedrals"], Ion["dihedral_types"], Ion["dihedral_params"], Ion["impropers"], Ion["improper_types"], Ion[
            "improper_params"], Ion["charges"], Ion["masses"], Ion["VDW_params"], Ion["VDW_comments"] = Find_parameters(
            Ion["adj_mat"], Ion["geometry"], Ion["atom_types"], FF_db, Improper_flag=False)
        print('\tTotal charge on {} is {} |e|'.format(ion, sum(Ion["charges"])))

    else:
        Ion["count"] = 1
        Ion["elements"] = [element]
        Ion["geometry"] = numpy.zeros([1, 3])
        Ion["centroid"] = numpy.zeros([1, 3])
        Ion["charges"] = [charge]
        Ion["adj_mat"] = adjacency.Table_generator(Ion["elements"], Ion["geometry"])
        Ion["atom_types"] = id_types.id_types(Ion["elements"], Ion["adj_mat"])
        Ion["bonds"] = Ion["bond_types"] = Ion["bond_params"] = Ion["angles"] = Ion["angle_types"] = Ion[
            "angle_params"] = Ion["dihedrals"] = Ion["dihedral_types"] = Ion["dihedral_params"] = Ion["impropers"] = \
        Ion["improper_types"] = Ion["improper_params"] = Ion["VDW_params"] = Ion["VDW_comments"] = Ion[
            "VDW_comments"] = []
        Ion["masses"] = {Ion["atom_types"][0]: get_masses(Ion["elements"])[0]}
    return Ion


def parse_surface_cation(element, surface_cation_charge, FF_db, debug, output):
    surface_cation = {}

    if element == None:
        # Not specified, use the specified cation
        print("Surface cation not specified. Will not place any cations along top/bottom.")
        # surface_cation["bonds"] = surface_cation["bond_types"] = surface_cation["bond_params"] = surface_cation["angles"] = surface_cation["angle_types"] = surface_cation["angle_params"] = surface_cation["dihedrals"] = surface_cation["dihedral_types"] = surface_cation["dihedral_params"] = surface_cation["impropers"] = surface_cation["improper_types"] = surface_cation["improper_params"] = surface_cation["VDW_params"] = surface_cation["VDW_comments"] = surface_cation["masses"] = []
        surface_cation["bonds"] = surface_cation["bond_types"] = surface_cation["bond_params"] = surface_cation[
            "angles"] = surface_cation["angle_types"] = surface_cation["angle_params"] = surface_cation["dihedrals"] = \
        surface_cation["dihedral_types"] = surface_cation["dihedral_params"] = surface_cation["impropers"] = \
        surface_cation["improper_types"] = surface_cation["improper_params"] = surface_cation["VDW_params"] = \
        surface_cation["VDW_comments"] = surface_cation["masses"] = surface_cation['elements'] = []

    else:
        if element.lower().endswith('.xyz'):
            # An *.xyz file was specified.
            print('*.xyz file specified for the surface cation. Reading file "{}"...'.format(element))

            if not os.path.isfile(FF_db):
                print('ERROR: Specified force field file ({}) does not exist. Aborting...'.format(FF_db))
                exit()

            surface_cation["count"], surface_cation["elements"], surface_cation["geometry"], surface_cation[
                "centroid"] = read_xyz(element)
            surface_cation["adj_mat"] = adjacency.Table_generator(surface_cation["elements"],
                                                                  surface_cation["geometry"])
            surface_cation["atom_types"] = id_types.id_types(surface_cation["elements"], surface_cation["adj_mat"])

            if debug:
                with open(output + '/' + output + '_debug_surface_cation.out', 'w') as d:
                    d.write('Atom type identifier\nAtom #\tElement\tAtom type\n\n')
                    for i in range(len(surface_cation["elements"])):
                        d.write('{}\t{}\t{}\n'.format(i + 1, surface_cation["elements"][i],
                                                      surface_cation["atom_types"][i]))
                    d.write('\n\n')

            print('\tParsing force field information...')
            surface_cation["bonds"], surface_cation["bond_types"], surface_cation["bond_params"], surface_cation[
                "angles"], surface_cation["angle_types"], surface_cation["angle_params"], surface_cation["dihedrals"], \
            surface_cation["dihedral_types"], surface_cation["dihedral_params"], surface_cation["impropers"], \
            surface_cation["improper_types"], surface_cation["improper_params"], surface_cation["charges"], \
            surface_cation["masses"], surface_cation["VDW_params"], surface_cation["VDW_comments"] = Find_parameters(
                surface_cation["adj_mat"], surface_cation["geometry"], surface_cation["atom_types"], FF_db,
                Improper_flag=False)
            print('\tTotal charge on surface cation is {} |e|'.format(sum(surface_cation["charges"])))

            # Check to make sure that the total charge is unity. If not, equally distribute the difference over all atoms.
            if sum(surface_cation["charges"]) != 1.0 and sum(surface_cation["charges"]) != surface_cation_charge:
                print('\t\tAdjusting charge to unity...')
                charge_diff = 1.0 - float(sum(surface_cation["charges"]))
                charge_diff = charge_diff / float(len(surface_cation["charges"]))
                print('\t\tcorrection per atom: {}'.format(charge_diff))

                for i in range(len(surface_cation["charges"])):
                    surface_cation["charges"][i] += charge_diff

                print('\t\tTotal charge on surface cation after adjustment is now {} |e|'.format(
                    sum(surface_cation["charges"])))
            print('\tTotal number of atoms in surface cation: {}'.format(len(surface_cation["elements"])))

            if debug:
                with open(output + '/' + output + '_debug_surface_cation_bonds_elements.out', 'a') as d:
                    d.write('List of bonds with atom types\n\n')
                    d.write('{}\n'.format([i for i in surface_cation["bonds"]]))

                    d.write('\n\n')
                    d.write('List of atomic charges\n\n')
                    for i in range(len(surface_cation["elements"])):
                        d.write('{}\t{}\n'.format(surface_cation["elements"][i], surface_cation["charges"][i]))
                    d.write('\n\n')

        else:
            print('Single atom specified for the surface cation.')
            surface_cation["count"] = 1
            surface_cation["elements"] = [element]
            surface_cation["geometry"] = numpy.zeros([1, 3])
            surface_cation["centroid"] = numpy.zeros([1, 3])
            surface_cation["charges"] = [surface_cation_charge]
            surface_cation["adj_mat"] = adjacency.Table_generator(surface_cation["elements"],
                                                                  surface_cation["geometry"])
            surface_cation["atom_types"] = id_types.id_types(surface_cation["elements"], surface_cation["adj_mat"])

            surface_cation["bonds"] = surface_cation["bond_types"] = surface_cation["bond_params"] = surface_cation[
                "angles"] = surface_cation["angle_types"] = surface_cation["angle_params"] = surface_cation[
                "dihedrals"] = surface_cation["dihedral_types"] = surface_cation["dihedral_params"] = surface_cation[
                "impropers"] = surface_cation["improper_types"] = surface_cation["improper_params"] = surface_cation[
                "VDW_params"] = []
            surface_cation["masses"] = {surface_cation["atom_types"][0]: get_masses([surface_cation["elements"][0]])}

        # Center the first atom in surface_cation at (0,0,0)
        surface_cation["geometry"] -= surface_cation["geometry"][0]

        # Since the molecule has been centered at (0,0,) at the first atom (taken as the head group),
        # the coordinates of each other atom are the displacements/vectors from the first atom. Use numpy to
        # find the distances by finding the length of each row in surface_cation["geometry"] and then
        # taking the greatest value as the atom furthest from the head group.

        surface_cation["distances"] = numpy.linalg.norm(surface_cation["geometry"], axis=1)
        surface_cation["tail_atom"] = numpy.argmax(surface_cation["distances"])
        print('\tLength of surface anion: {}, index: {}'.format(max(surface_cation["distances"]),
                                                                numpy.argmax(surface_cation["distances"])))

    return surface_cation


# Bond length is the metal-anion distance, i.e., half the metal-metal bond length
def build_unit_cell(Metal, Anion, Cation, bond_length, monolayer_flag, anion_vacancy_flag, metal_vacancy_flag,
                    structure=1):
    # Counting:
    #
    # 1 metal  (A)
    # 1 cation (B) in a multilayer structure, otherwise none in a monolayer
    # 3 anions (X) in a multilayer structure, otherwise 4 anions in a monolayer

    n_cations = 0 if monolayer_flag else 1

    # Initialize unit cell lists for holding geometry and elements

    # Order is:
    # metal (at 0,0,0)
    # anions
    # [cations]
    # [ligands]

    unit_cell = {}

    if anion_vacancy_flag:
        unit_cell["count"] = (1 + 3 * Anion["count"]) if monolayer_flag else (
                    1 + 2 * Anion["count"] + n_cations * Cation["count"])
    elif metal_vacancy_flag:
        unit_cell["count"] = (4 * Anion["count"]) if monolayer_flag else (
                    3 * Anion["count"] + n_cations * Cation["count"])
    else:
        unit_cell["count"] = (1 + 4 * Anion["count"]) if monolayer_flag else (
                    1 + 3 * Anion["count"] + n_cations * Cation["count"])

    unit_cell["adj_mat"] = numpy.zeros([unit_cell["count"], unit_cell["count"]])
    unit_cell["geometry"] = numpy.zeros([unit_cell["count"], 3])
    unit_cell["elements"] = []
    unit_cell["atom_types"] = []
    unit_cell["masses"] = []
    unit_cell["charges"] = []
    unit_cell["bonds"] = unit_cell["bond_types"] = unit_cell["bond_params"] = unit_cell["angles"] = unit_cell[
        "angle_types"] = unit_cell["angle_params"] = unit_cell["dihedrals"] = unit_cell["dihedral_types"] = unit_cell[
        "dihedral_params"] = unit_cell["impropers"] = unit_cell["improper_types"] = unit_cell["improper_params"] = \
    unit_cell["VDW_params"] = []
    unit_cell["atoms_placed"] = 0

    # Add metal atom to center (B)
    if not metal_vacancy_flag:
        unit_cell["elements"] = unit_cell["elements"] + Metal["elements"]
        unit_cell["adj_mat"][0:1] = Metal["adj_mat"]
        unit_cell["atom_types"] = unit_cell["atom_types"] + Metal["atom_types"]
        unit_cell["masses"] = unit_cell["masses"] + [Metal["masses"][j] for j in Metal["atom_types"]]
        unit_cell["charges"] = unit_cell["charges"] + Metal["charges"]
        unit_cell["bonds"] = unit_cell["bonds"] + Metal["bonds"]
        unit_cell["bond_types"] = unit_cell["bond_types"] + Metal["bond_types"]
        unit_cell["angles"] = unit_cell["angles"] + Metal["angles"]
        unit_cell["angle_types"] = unit_cell["angle_types"] + Metal["angle_types"]
        unit_cell["dihedrals"] = unit_cell["dihedrals"] + Metal["dihedrals"]
        unit_cell["dihedral_types"] = unit_cell["dihedral_types"] + Metal["dihedral_types"]
        unit_cell["impropers"] = unit_cell["impropers"] + Metal["impropers"]
        unit_cell["improper_types"] = unit_cell["improper_types"] + Metal["improper_types"]
        unit_cell["atoms_placed"] += 1

    # Add the anions (A's)
    if anion_vacancy_flag:
        n = 3 if monolayer_flag else 2
    else:
        n = 4 if monolayer_flag else 3

    for i in range(n):
        unit_cell["elements"] = unit_cell["elements"] + Anion["elements"]
        unit_cell["adj_mat"][unit_cell["atoms_placed"]:(unit_cell["atoms_placed"] + len(unit_cell["elements"]))] = \
        Anion["adj_mat"]
        unit_cell["atom_types"] = unit_cell["atom_types"] + Anion["atom_types"]
        unit_cell["masses"] = unit_cell["masses"] + [Anion["masses"][j] for j in Anion["atom_types"]]
        unit_cell["charges"] = unit_cell["charges"] + Anion["charges"]
        unit_cell["bonds"] = unit_cell["bonds"] + [(j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"])
                                                   for j in Anion["bonds"]]
        unit_cell["bond_types"] = unit_cell["bond_types"] + Anion["bond_types"]
        unit_cell["angles"] = unit_cell["angles"] + [
            (j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"], j[2] + unit_cell["atoms_placed"]) for j
            in Anion["angles"]]
        unit_cell["angle_types"] = unit_cell["angle_types"] + Anion["angle_types"]
        unit_cell["dihedrals"] = unit_cell["dihedrals"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Anion["dihedrals"]]
        unit_cell["dihedral_types"] = unit_cell["dihedral_types"] + Anion["dihedral_types"]
        unit_cell["impropers"] = unit_cell["impropers"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Anion["impropers"]]
        unit_cell["improper_types"] = unit_cell["improper_types"] + Anion["improper_types"]
        unit_cell["atoms_placed"] += len(Anion["elements"])

    if anion_vacancy_flag:
        # DEFINE UNIT CELL
        # cubic
        counter = 1

        if monolayer_flag:
            loc_choice = random.choice([1, 2, 3, 4])

            if loc_choice != 1:
                unit_cell["geometry"][counter] = bond_length, 0, 0
                counter += 1
            if loc_choice != 2:
                unit_cell["geometry"][counter] = 0, -bond_length, 0
                counter += 1
            if loc_choice != 3:
                unit_cell["geometry"][counter] = 0, 0, bond_length
                counter += 1
            if loc_choice != 4:
                unit_cell["geometry"][counter] = 0, bond_length, 0

        else:
            loc_choice = random.choice([1, 2, 3])

            if loc_choice != 1:
                unit_cell["geometry"][counter] = bond_length, 0, 0
                counter += 1
            if loc_choice != 2:
                unit_cell["geometry"][counter] = 0, -bond_length, 0
                counter += 1
            if loc_choice != 3:
                unit_cell["geometry"][counter] = 0, 0, bond_length
                counter += 1

    else:
        # DEFINE UNIT CELL
        # cubic
        counter = 0 if metal_vacancy_flag else 1
        unit_cell["geometry"][counter] = bond_length, 0, 0  # Anion 1
        unit_cell["geometry"][counter + 1] = 0, -bond_length, 0  # Anion 2
        unit_cell["geometry"][counter + 2] = 0, 0, bond_length  # Anion 3

        if monolayer_flag:
            unit_cell["geometry"][counter + 3] = 0, bond_length, 0  # Add fourth anion

        # cubic unit cell:
        # unit_cell[1] = args.bond_length, args.bond_length, 0
        # unit_cell[2] = 0, args.bond_length, args.bond_length
        # unit_cell[3] = args.bond_length, 0, args.bond_length

    # Add n instances of the cation
    for i in range(n_cations):
        unit_cell["elements"] = unit_cell["elements"] + Cation["elements"]
        # unit_cell["adj_mat"][adj_count:adj_count+len(Cation["elements"])] = Cation["adj_mat"]
        # unit_cell["adj_mat"][adj_count:(adj_count+len(Cation["elements"])),adj_count:(adj_count+len(Cation["elements"]))] = Cation["adj_mat"]
        unit_cell["adj_mat"][unit_cell["atoms_placed"]:(unit_cell["atoms_placed"] + len(Cation["elements"])),
        unit_cell["atoms_placed"]:(unit_cell["atoms_placed"] + len(Cation["elements"]))] = Cation["adj_mat"]
        unit_cell["atom_types"] = unit_cell["atom_types"] + Cation["atom_types"]
        unit_cell["masses"] = unit_cell["masses"] + [Cation["masses"][j] for j in Cation["atom_types"]]
        unit_cell["charges"] = unit_cell["charges"] + Cation["charges"]
        unit_cell["bonds"] = unit_cell["bonds"] + [(j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"])
                                                   for j in Cation["bonds"]]
        unit_cell["bond_types"] = unit_cell["bond_types"] + Cation["bond_types"]
        unit_cell["angles"] = unit_cell["angles"] + [
            (j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"], j[2] + unit_cell["atoms_placed"]) for j
            in Cation["angles"]]
        unit_cell["angle_types"] = unit_cell["angle_types"] + Cation["angle_types"]
        unit_cell["dihedrals"] = unit_cell["dihedrals"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Cation["dihedrals"]]
        unit_cell["dihedral_types"] = unit_cell["dihedral_types"] + Cation["dihedral_types"]
        unit_cell["impropers"] = unit_cell["impropers"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Cation["impropers"]]
        unit_cell["improper_types"] = unit_cell["improper_types"] + Cation["improper_types"]
        unit_cell["atoms_placed"] += len(Cation["elements"])

    return unit_cell

def build_top_quasi2D(Metal, Anion, Cation, bond_length, quasi2D_flag, anion_vacancy_flag, metal_vacancy_flag,
                    structure=1):
    # Counting:
    #
    # 1 metal  (A)
    # 1 cation (B) in a multilayer structure, otherwise none in a monolayer
    # 3 anions (X) in a multilayer structure, otherwise 4 anions in a monolayer

    n_cations = 1

    # Initialize unit cell lists for holding geometry and elements

    # Order is:
    # metal (at 0,0,0)
    # anions
    # [cations]
    # [ligands]

    unit_cell = {}

    unit_cell["count"] = (1 + 4 * Anion["count"] + n_cations * Cation["count"])

    unit_cell["adj_mat"] = numpy.zeros([unit_cell["count"], unit_cell["count"]])
    unit_cell["geometry"] = numpy.zeros([unit_cell["count"], 3])
    unit_cell["elements"] = []
    unit_cell["atom_types"] = []
    unit_cell["masses"] = []
    unit_cell["charges"] = []
    unit_cell["bonds"] = unit_cell["bond_types"] = unit_cell["bond_params"] = unit_cell["angles"] = unit_cell[
        "angle_types"] = unit_cell["angle_params"] = unit_cell["dihedrals"] = unit_cell["dihedral_types"] = unit_cell[
        "dihedral_params"] = unit_cell["impropers"] = unit_cell["improper_types"] = unit_cell["improper_params"] = \
    unit_cell["VDW_params"] = []
    unit_cell["atoms_placed"] = 0

    # Add metal atom to center (B)
    if not metal_vacancy_flag:
        unit_cell["elements"] = unit_cell["elements"] + Metal["elements"]
        unit_cell["adj_mat"][0:1] = Metal["adj_mat"]
        unit_cell["atom_types"] = unit_cell["atom_types"] + Metal["atom_types"]
        unit_cell["masses"] = unit_cell["masses"] + [Metal["masses"][j] for j in Metal["atom_types"]]
        unit_cell["charges"] = unit_cell["charges"] + Metal["charges"]
        unit_cell["bonds"] = unit_cell["bonds"] + Metal["bonds"]
        unit_cell["bond_types"] = unit_cell["bond_types"] + Metal["bond_types"]
        unit_cell["angles"] = unit_cell["angles"] + Metal["angles"]
        unit_cell["angle_types"] = unit_cell["angle_types"] + Metal["angle_types"]
        unit_cell["dihedrals"] = unit_cell["dihedrals"] + Metal["dihedrals"]
        unit_cell["dihedral_types"] = unit_cell["dihedral_types"] + Metal["dihedral_types"]
        unit_cell["impropers"] = unit_cell["impropers"] + Metal["impropers"]
        unit_cell["improper_types"] = unit_cell["improper_types"] + Metal["improper_types"]
        unit_cell["atoms_placed"] += 1

    # Add the anions (A's)
    n = 4

    for i in range(n):
        unit_cell["elements"] = unit_cell["elements"] + Anion["elements"]
        unit_cell["adj_mat"][unit_cell["atoms_placed"]:(unit_cell["atoms_placed"] + len(unit_cell["elements"]))] = \
        Anion["adj_mat"]
        unit_cell["atom_types"] = unit_cell["atom_types"] + Anion["atom_types"]
        unit_cell["masses"] = unit_cell["masses"] + [Anion["masses"][j] for j in Anion["atom_types"]]
        unit_cell["charges"] = unit_cell["charges"] + Anion["charges"]
        unit_cell["bonds"] = unit_cell["bonds"] + [(j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"])
                                                   for j in Anion["bonds"]]
        unit_cell["bond_types"] = unit_cell["bond_types"] + Anion["bond_types"]
        unit_cell["angles"] = unit_cell["angles"] + [
            (j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"], j[2] + unit_cell["atoms_placed"]) for j
            in Anion["angles"]]
        unit_cell["angle_types"] = unit_cell["angle_types"] + Anion["angle_types"]
        unit_cell["dihedrals"] = unit_cell["dihedrals"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Anion["dihedrals"]]
        unit_cell["dihedral_types"] = unit_cell["dihedral_types"] + Anion["dihedral_types"]
        unit_cell["impropers"] = unit_cell["impropers"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Anion["impropers"]]
        unit_cell["improper_types"] = unit_cell["improper_types"] + Anion["improper_types"]
        unit_cell["atoms_placed"] += len(Anion["elements"])

    if anion_vacancy_flag:
        # DEFINE UNIT CELL
        # cubic
        counter = 1

        if monolayer_flag:
            loc_choice = random.choice([1, 2, 3, 4])

            if loc_choice != 1:
                unit_cell["geometry"][counter] = bond_length, 0, 0
                counter += 1
            if loc_choice != 2:
                unit_cell["geometry"][counter] = 0, -bond_length, 0
                counter += 1
            if loc_choice != 3:
                unit_cell["geometry"][counter] = 0, 0, bond_length
                counter += 1
            if loc_choice != 4:
                unit_cell["geometry"][counter] = 0, bond_length, 0

        else:
            loc_choice = random.choice([1, 2, 3])

            if loc_choice != 1:
                unit_cell["geometry"][counter] = bond_length, 0, 0
                counter += 1
            if loc_choice != 2:
                unit_cell["geometry"][counter] = 0, -bond_length, 0
                counter += 1
            if loc_choice != 3:
                unit_cell["geometry"][counter] = 0, 0, bond_length
                counter += 1

    else:
        # DEFINE UNIT CELL
        # cubic
        counter = 0 if metal_vacancy_flag else 1
        unit_cell["geometry"][counter] = bond_length, 0, 0  # Anion 1
        unit_cell["geometry"][counter + 1] = 0, -bond_length, 0  # Anion 2
        unit_cell["geometry"][counter + 2] = 0, 0, bond_length  # Anion 3
        unit_cell["geometry"][counter + 3] = 0, bond_length, 0  # Anion 4

        # cubic unit cell:
        # unit_cell[1] = args.bond_length, args.bond_length, 0
        # unit_cell[2] = 0, args.bond_length, args.bond_length
        # unit_cell[3] = args.bond_length, 0, args.bond_length

    # Add n instances of the cation
    for i in range(n_cations):
        unit_cell["elements"] = unit_cell["elements"] + Cation["elements"]
        # unit_cell["adj_mat"][adj_count:adj_count+len(Cation["elements"])] = Cation["adj_mat"]
        # unit_cell["adj_mat"][adj_count:(adj_count+len(Cation["elements"])),adj_count:(adj_count+len(Cation["elements"]))] = Cation["adj_mat"]
        unit_cell["adj_mat"][unit_cell["atoms_placed"]:(unit_cell["atoms_placed"] + len(Cation["elements"])),
        unit_cell["atoms_placed"]:(unit_cell["atoms_placed"] + len(Cation["elements"]))] = Cation["adj_mat"]
        unit_cell["atom_types"] = unit_cell["atom_types"] + Cation["atom_types"]
        unit_cell["masses"] = unit_cell["masses"] + [Cation["masses"][j] for j in Cation["atom_types"]]
        unit_cell["charges"] = unit_cell["charges"] + Cation["charges"]
        unit_cell["bonds"] = unit_cell["bonds"] + [(j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"])
                                                   for j in Cation["bonds"]]
        unit_cell["bond_types"] = unit_cell["bond_types"] + Cation["bond_types"]
        unit_cell["angles"] = unit_cell["angles"] + [
            (j[0] + unit_cell["atoms_placed"], j[1] + unit_cell["atoms_placed"], j[2] + unit_cell["atoms_placed"]) for j
            in Cation["angles"]]
        unit_cell["angle_types"] = unit_cell["angle_types"] + Cation["angle_types"]
        unit_cell["dihedrals"] = unit_cell["dihedrals"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Cation["dihedrals"]]
        unit_cell["dihedral_types"] = unit_cell["dihedral_types"] + Cation["dihedral_types"]
        unit_cell["impropers"] = unit_cell["impropers"] + [(j[0] + unit_cell["atoms_placed"],
                                                            j[1] + unit_cell["atoms_placed"],
                                                            j[2] + unit_cell["atoms_placed"],
                                                            j[3] + unit_cell["atoms_placed"]) for j in
                                                           Cation["impropers"]]
        unit_cell["improper_types"] = unit_cell["improper_types"] + Cation["improper_types"]
        unit_cell["atoms_placed"] += len(Cation["elements"])

    return unit_cell


def build_bottom(Metal, Anion, Cation, bond_length, heterojunction_flag):
    # DEFINE UNIT CELL
    # Build the capping bottom "unit cell"
    # only the metal and two anions

    bottom = {}

    bottom["count"] = 1 + 2 * Anion["count"]
    bottom["adj_mat"] = numpy.zeros([bottom["count"], bottom["count"]])
    bottom["geometry"] = numpy.zeros([bottom["count"], 3])
    bottom["elements"] = []
    bottom["atom_types"] = []
    bottom["masses"] = []
    bottom["charges"] = []
    bottom["bonds"] = bottom["bond_types"] = bottom["bond_params"] = bottom["angles"] = bottom["angle_types"] = bottom[
        "angle_params"] = bottom["dihedrals"] = bottom["dihedral_types"] = bottom["dihedral_params"] = bottom[
        "impropers"] = bottom["improper_types"] = bottom["improper_params"] = bottom["VDW_params"] = []
    bottom["atoms_placed"] = 0

    # Add metal atom to center (B):
    bottom["elements"] = bottom["elements"] + Metal["elements"]
    bottom["adj_mat"][0:1] = Metal["adj_mat"]
    bottom["atom_types"] = bottom["atom_types"] + Metal["atom_types"]
    bottom["masses"] = bottom["masses"] + [Metal["masses"][j] for j in Metal["atom_types"]]
    bottom["charges"] = bottom["charges"] + Metal["charges"]
    bottom["bonds"] = bottom["bonds"] + Metal["bonds"]
    bottom["bond_types"] = bottom["bond_types"] + Metal["bond_types"]
    bottom["angles"] = bottom["angles"] + Metal["angles"]
    bottom["angle_types"] = bottom["angle_types"] + Metal["angle_types"]
    bottom["dihedrals"] = bottom["dihedrals"] + Metal["dihedrals"]
    bottom["dihedral_types"] = bottom["dihedral_types"] + Metal["dihedral_types"]
    bottom["impropers"] = bottom["impropers"] + Metal["impropers"]
    bottom["improper_types"] = bottom["improper_types"] + Metal["improper_types"]
    bottom["atoms_placed"] += 1

    # Add the anions (A's)
    for i in range(2):
        bottom["elements"] = bottom["elements"] + Anion["elements"]
        bottom["adj_mat"][bottom["atoms_placed"]:(bottom["atoms_placed"] + len(bottom["elements"]))] = Anion["adj_mat"]
        bottom["atom_types"] = bottom["atom_types"] + Anion["atom_types"]
        bottom["masses"] = bottom["masses"] + [Anion["masses"][j] for j in Anion["atom_types"]]
        bottom["charges"] = bottom["charges"] + Anion["charges"]
        bottom["bonds"] = bottom["bonds"] + [(j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"]) for j in
                                             Anion["bonds"]]
        bottom["bond_types"] = bottom["bond_types"] + Anion["bond_types"]
        bottom["angles"] = bottom["angles"] + [
            (j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"], j[2] + bottom["atoms_placed"]) for j in
            Anion["angles"]]
        bottom["angle_types"] = bottom["angle_types"] + Anion["angle_types"]
        bottom["dihedrals"] = bottom["dihedrals"] + [(j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"],
                                                      j[2] + bottom["atoms_placed"], j[3] + bottom["atoms_placed"]) for
                                                     j in Anion["dihedrals"]]
        bottom["dihedral_types"] = bottom["dihedral_types"] + Anion["dihedral_types"]
        bottom["impropers"] = bottom["impropers"] + [(j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"],
                                                      j[2] + bottom["atoms_placed"], j[3] + bottom["atoms_placed"]) for
                                                     j in Anion["impropers"]]
        bottom["improper_types"] = bottom["improper_types"] + Anion["improper_types"]
        bottom["atoms_placed"] += len(Anion["elements"])

    # DEFINE UNIT CELL
    # orthorhombic
    if heterojunction_flag:
        bottom["geometry"][1] = -bond_length, 0, 0
        bottom["geometry"][2] = 0, 0, -bond_length
    else:
        bottom["geometry"][1] = bond_length, 0, 0
        bottom["geometry"][2] = 0, 0, bond_length

    return bottom

def build_bottom_quasi2D(Metal, Anion, Cation, bond_length, heterojunction_flag):
    # DEFINE UNIT CELL
    # Build the capping bottom "unit cell"
    # only the metal and two anions

    bottom = {}

    bottom["count"] = 1 + 3 * Anion["count"]
    bottom["adj_mat"] = numpy.zeros([bottom["count"], bottom["count"]])
    bottom["geometry"] = numpy.zeros([bottom["count"], 3])
    bottom["elements"] = []
    bottom["atom_types"] = []
    bottom["masses"] = []
    bottom["charges"] = []
    bottom["bonds"] = bottom["bond_types"] = bottom["bond_params"] = bottom["angles"] = bottom["angle_types"] = bottom[
        "angle_params"] = bottom["dihedrals"] = bottom["dihedral_types"] = bottom["dihedral_params"] = bottom[
        "impropers"] = bottom["improper_types"] = bottom["improper_params"] = bottom["VDW_params"] = []
    bottom["atoms_placed"] = 0

    # Add metal atom to center (B):
    bottom["elements"] = bottom["elements"] + Metal["elements"]
    bottom["adj_mat"][0:1] = Metal["adj_mat"]
    bottom["atom_types"] = bottom["atom_types"] + Metal["atom_types"]
    bottom["masses"] = bottom["masses"] + [Metal["masses"][j] for j in Metal["atom_types"]]
    bottom["charges"] = bottom["charges"] + Metal["charges"]
    bottom["bonds"] = bottom["bonds"] + Metal["bonds"]
    bottom["bond_types"] = bottom["bond_types"] + Metal["bond_types"]
    bottom["angles"] = bottom["angles"] + Metal["angles"]
    bottom["angle_types"] = bottom["angle_types"] + Metal["angle_types"]
    bottom["dihedrals"] = bottom["dihedrals"] + Metal["dihedrals"]
    bottom["dihedral_types"] = bottom["dihedral_types"] + Metal["dihedral_types"]
    bottom["impropers"] = bottom["impropers"] + Metal["impropers"]
    bottom["improper_types"] = bottom["improper_types"] + Metal["improper_types"]
    bottom["atoms_placed"] += 1

    # Add the anions (A's)
    for i in range(3):
        bottom["elements"] = bottom["elements"] + Anion["elements"]
        bottom["adj_mat"][bottom["atoms_placed"]:(bottom["atoms_placed"] + len(bottom["elements"]))] = Anion["adj_mat"]
        bottom["atom_types"] = bottom["atom_types"] + Anion["atom_types"]
        bottom["masses"] = bottom["masses"] + [Anion["masses"][j] for j in Anion["atom_types"]]
        bottom["charges"] = bottom["charges"] + Anion["charges"]
        bottom["bonds"] = bottom["bonds"] + [(j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"]) for j in
                                             Anion["bonds"]]
        bottom["bond_types"] = bottom["bond_types"] + Anion["bond_types"]
        bottom["angles"] = bottom["angles"] + [
            (j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"], j[2] + bottom["atoms_placed"]) for j in
            Anion["angles"]]
        bottom["angle_types"] = bottom["angle_types"] + Anion["angle_types"]
        bottom["dihedrals"] = bottom["dihedrals"] + [(j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"],
                                                      j[2] + bottom["atoms_placed"], j[3] + bottom["atoms_placed"]) for
                                                     j in Anion["dihedrals"]]
        bottom["dihedral_types"] = bottom["dihedral_types"] + Anion["dihedral_types"]
        bottom["impropers"] = bottom["impropers"] + [(j[0] + bottom["atoms_placed"], j[1] + bottom["atoms_placed"],
                                                      j[2] + bottom["atoms_placed"], j[3] + bottom["atoms_placed"]) for
                                                     j in Anion["impropers"]]
        bottom["improper_types"] = bottom["improper_types"] + Anion["improper_types"]
        bottom["atoms_placed"] += len(Anion["elements"])

    # DEFINE UNIT CELL
    # orthorhombic
    if heterojunction_flag:
        bottom["geometry"][1] = -bond_length, 0, 0
        bottom["geometry"][2] = 0, 0, -bond_length
    else:
        bottom["geometry"][1] = bond_length, 0, 0
        bottom["geometry"][2] = 0, -bond_length, 0
        bottom["geometry"][3] = 0, 0, bond_length

    return bottom

def update_simbox(x, y, z, bond_length, sim_box, sim_data, atoms_placed, unit_cell, mols_placed):
    geom = unit_cell["geometry"] + numpy.array([x * (bond_length * 2), y * (bond_length * 2), z * (bond_length * 2)])
    sim_box[atoms_placed:atoms_placed + len(unit_cell["elements"])] = geom
    sim_data["adj_mat"][atoms_placed:(atoms_placed + len(unit_cell["elements"])),
    atoms_placed:(atoms_placed + len(unit_cell["elements"]))] = unit_cell["adj_mat"]
    sim_data["elements"] = sim_data["elements"] + unit_cell["elements"]
    sim_data["masses"] = sim_data["masses"] + unit_cell["masses"]
    sim_data["charges"] = sim_data["charges"] + unit_cell["charges"]
    sim_data["atom_types"] = sim_data["atom_types"] + unit_cell["atom_types"]
    sim_data["bonds"] = sim_data["bonds"] + [(j[0] + atoms_placed, j[1] + atoms_placed) for j in
                                                unit_cell["bonds"]]
    sim_data["bond_types"] = sim_data["bond_types"] + unit_cell["bond_types"]
    sim_data["angles"] = sim_data["angles"] + [
        (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed) for j in
        unit_cell["angles"]]
    sim_data["angle_types"] = sim_data["angle_types"] + unit_cell["angle_types"]
    sim_data["dihedrals"] = sim_data["dihedrals"] + [
        (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
        in unit_cell["dihedrals"]]
    sim_data["dihedral_types"] = sim_data["dihedral_types"] + unit_cell["dihedral_types"]
    sim_data["impropers"] = sim_data["impropers"] + [
        (j[0] + atoms_placed, j[1] + atoms_placed, j[2] + atoms_placed, j[3] + atoms_placed) for j
        in unit_cell["impropers"]]
    sim_data["improper_types"] = sim_data["improper_types"] + unit_cell["improper_types"]
    sim_data["molecules"] = sim_data["molecules"] + [mols_placed] * len(unit_cell["elements"])
    mols_placed += 1

class Logger(object):
    def __init__(self, folder):
        self.terminal = sys.stdout
        self.log = open(folder + "_build.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    main(sys.argv[1:])
