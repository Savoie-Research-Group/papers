#!/bin/env python
import math
import argparse
import sys
import os
import numpy
import random
import datetime
from copy import deepcopy
from scipy.spatial.distance import cdist

# Add TAFFI Lib to path
path = os.path.dirname(os.path.abspath(__file__))
lib_path = os.path.join('/'.join(os.path.dirname(os.path.abspath(__file__)).split('/')[:-1]), 'lib')
sys.path.append(lib_path)
# sys.path.append('/Users/nianz/Desktop/Research/high-throughput-2D-Perovskite-ligand-screening-tool/MD/lib')
import adjacency
import id_types
import perovskite_builder

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

    parser.add_argument('-SO', dest='surface_cation_orientation', type=str, default='0.0 45.0 30.0',
                        help='Space-delimited string specifying the tilt, sweep, and orientation angles for surface cation insertion. All three must be specified. Default: "0.0 45.0 30.0"')

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

    output = args.output
    if os.path.isdir(output):
        pass
    else:
        os.mkdir(output)

    dims = args.dims
    dims = [int(_) for _ in dims.split()]
    charges = '1.00 2.00 -1.00'
    charges = [float(_) for _ in charges.split()]

    bond_length = args.bond_length

    cation_charge = charges[0]
    metal_charge = charges[1]
    anion_charge = charges[2]
    surface_cation_charge = charges[0]

    mixing_rule = args.mixing_rule

    perov_layers_FF = args.FF_db
    surf_cation_FF = args.surface_FF_db

    anion = args.anion
    cation = args.cation
    metal = args.metal
    Surface_cation = args.surface_cation

    quasi_2D = args.quasi_2D

    args.mixing_rule = args.mixing_rule.lower()
    if args.mixing_rule not in ['wh', 'lb', 'none']:
        print(
            'ERROR: Supplied -mixing_rule ({}) not accepted. Only "lb", "wh", or "none" are accepted. Exiting...'.format(
                args.mixing_rule))
        exit()
    if args.mixing_rule == 'none':
        print('WARNING: A mixing rule of "none" has been specified...')

    y_pad = float(args.y_padding)

    Anion = perovskite_builder.parse_ion(anion, anion_charge, perov_layers_FF, 'anion')
    Cation = perovskite_builder.parse_ion(cation, cation_charge, perov_layers_FF, 'cation')
    Metal = perovskite_builder.parse_metal(metal, metal_charge)
    surface_cation = parse_surface_cation(Surface_cation, surface_cation_charge, surf_cation_FF, False, 'surface_cation')
    
    Metal["atom_types"] = ['[82[53]]']
    Metal["masses"] = {Metal["atom_types"][0]: perovskite_builder.get_masses(Metal["elements"])[0]}

    surface_cation_orientation = args.surface_cation_orientation
    surface_cation_orientation = [float(_) for _ in surface_cation_orientation.split()]
    print(surface_cation_orientation)

    unit_cell = perovskite_builder.build_unit_cell(Metal, Anion, Cation, 3.1, False, False, False)
    unit_cell_no_cation = build_bottom_quasi2D(Metal, Anion, Cation, 3.1, False)
    unit_cell_plusAnion = build_top_quasi2D(Metal, Anion, Cation, 3.1, True, False, False)

    expected_unit_cell_count = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)

    # Calculate the total number of atoms needed...

    # Number of interior atoms
    N_interior = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1) * int(unit_cell["count"])

    # Number of surface atoms (surface is along the xz planes)
    if quasi_2D:
            N_bot = (dims[0] + 1) * (dims[2] + 1) * len(unit_cell_no_cation['elements'])
            N_top = (dims[0] + 1) * (dims[2] + 1) * len(unit_cell_plusAnion['elements'])
            N_interior = (dims[0] + 1) * (dims[1] - 1) * (dims[2] + 1) * len(unit_cell["elements"])
    else:
            N_bot = N_top = 0
            N_interior = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1) * len(unit_cell["elements"])

    # Number of surface atoms (surface is along the xz planes)
    if surface_cation != None:
        N_surface = 2 * ((dims[0] + 1) * (dims[2] + 1) * len(surface_cation["elements"]))
    else:
        N_surface = 0

    # Total up the number of atoms in the simulation
    N_atoms = N_interior + N_surface + N_bot + N_top 
    print('N_atoms:' + str(N_atoms) + ' expected_unit_cell_count:' + str(expected_unit_cell_count))

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

    sim_data["angle_params"] = {}
    sim_data["angle_params"].update(Anion["angle_params"])
    sim_data["angle_params"].update(Cation["angle_params"])
    sim_data["angle_params"].update(surface_cation["angle_params"])

    sim_data["dihedral_params"] = {}
    sim_data["dihedral_params"].update(Anion["dihedral_params"])
    sim_data["dihedral_params"].update(Cation["dihedral_params"])
    sim_data["dihedral_params"].update(surface_cation["dihedral_params"])

    sim_data["improper_params"] = {}
    sim_data["improper_params"].update(Anion["improper_params"])
    sim_data["improper_params"].update(Cation["improper_params"])
    sim_data["improper_params"].update(surface_cation["improper_params"])

    sim_data["VDW_params"] = {}
    sim_data["VDW_params"].update(Anion["VDW_params"])
    sim_data["VDW_params"].update(Cation["VDW_params"])
    sim_data["VDW_params"].update(surface_cation["VDW_params"])

    sim_data["VDW_comments"] = {}
    sim_data["VDW_comments"].update(Anion["VDW_comments"])
    sim_data["VDW_comments"].update(Cation["VDW_comments"])
    sim_data["VDW_comments"].update(surface_cation["VDW_comments"])

    sim_data["all_masses"] = {}
    sim_data["all_masses"].update(Metal["masses"])
    sim_data["all_masses"].update(Anion["masses"])
    sim_data["all_masses"].update(Cation["masses"])
    sim_data["all_masses"].update(surface_cation["masses"])

    # Internal tracking
    atoms_placed = 0
    unit_cells_placed = 0
    surface_atoms_placed = 0
    mols_placed = 0
    place_anion_vacancy = False
    place_surface_vacancy = False
    centroids_list = []

    for x in range(dims[0] + 1):
        # print(x)
        for y in range(dims[1] + 1):
            for z in range(dims[2] + 1):
                n_cations = 1
                for i in range(n_cations):
                    # do cation site #1
                    geom = deepcopy(Cation["geometry"])

                    # Randomize the cation orientation
                    # perform x rotations
                    angle = random.random() * 360
                    for count_j, j in enumerate(geom):
                        geom[count_j, :] = perovskite_builder.axis_rot(j, numpy.array([1.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]),
                                                    angle, mode='angle')

                    # perform y rotations
                    angle = random.random() * 360
                    for count_j, j in enumerate(geom):
                        geom[count_j, :] = perovskite_builder.axis_rot(j, numpy.array([0.0, 1.0, 0.0]), numpy.array([0.0, 0.0, 0.0]),
                                                    angle, mode='angle')

                    # perform z rotations
                    angle = random.random() * 360
                    for count_j, j in enumerate(geom):
                        geom[count_j, :] = perovskite_builder.axis_rot(j, numpy.array([0.0, 0.0, 1.0]), numpy.array([0.0, 0.0, 0.0]),
                                                    angle, mode='angle')

                    # set initial cation position
                    geom += numpy.array([bond_length, -bond_length, bond_length])

                    if y == dims[1]:
                        unit_cell_plusAnion["geometry"][5 + i * len(Cation["elements"]): 5 + (i + 1) * len(Cation["elements"])] = geom
                    else: 
                        unit_cell["geometry"][
                        4 + i * len(Cation["elements"]): 4 + (i + 1) * len(Cation["elements"])] = geom
                                # if .quasi_2D:

                # Translate the unit cell to the current position
                if y == 0:
                    geom, mols_placed, atoms_placed, unit_cells_placed = update_simbox(
                        x, y, z, bond_length, sim_box, sim_data, atoms_placed, unit_cell_no_cation, unit_cells_placed,mols_placed)
                    # print(geom)
                elif y == dims[1]:
                    geom, mols_placed, atoms_placed, unit_cells_placed = update_simbox(
                        x, y, z, bond_length, sim_box, sim_data, atoms_placed, unit_cell_plusAnion, unit_cells_placed, mols_placed)
                
                else:
                    geom, mols_placed, atoms_placed, unit_cells_placed = update_simbox(
                        x, y, z, bond_length, sim_box, sim_data, atoms_placed, unit_cell, unit_cells_placed, mols_placed)
                
                centroid = geom.mean(axis=0)

                for j in range(len(centroids_list)):
                        distance = numpy.around(numpy.linalg.norm(centroid - centroids_list[j]), decimals=5)
                        if distance == 0:
                            print('ERROR: OVERLAPPING CENTROIDS')

                centroids_list.append(centroid)
                

                if y == 0 or y == dims[1]:
                    if y == 0:
                        v_1 = geom[1] - geom[0]
                        v_2 = geom[3] - geom[0]
                        v_n = numpy.cross(v_1, v_2)
                        v_n = v_n / numpy.linalg.norm(v_n)
        
                    elif y == dims[1]:
                        v_1 = geom[1] - geom[0]
                        v_2 = geom[3] - geom[0]
                        v_n = numpy.cross(v_1, v_2)
                        v_n = v_n / numpy.linalg.norm(v_n)
                    
                    if surface_cation != None:
                        # Get copy of surface cation geometry
                        surface_geom = deepcopy(surface_cation["geometry"])

                        # DEFINE UNIT CELL
                        # Position the geometry relative to the unit cell
                        surface_geom += numpy.array([bond_length, 0, bond_length])

                        # Translate geometry to correct location, depending on if we are at the bottom (y = 0) or top (y = .dims[1]) of the simulation cell.
                        if y == 0:
                            place = surface_geom + numpy.array(
                                [x * (bond_length * 2), -bond_length, z * (bond_length * 2)])

                        elif y == dims[1]:
                            place = surface_geom + numpy.array(
                                [x * (bond_length * 2), (y * 2 + 1) * (bond_length),
                                    z * (bond_length * 2)])
                        
                        # print(place)
                        # for count_k, k in enumerate(place):
                        #     place[count_k, :] = perovskite_builder.axis_rot(k, numpy.array([0, 0.0, 1]), place[0],
                        #                                 180, mode='angle')
                        # v_n = -1 * v_n
                        
                        # Define long-axis vector and normalize
                        v_la = place[surface_cation["tail_atom"]] - place[0]
                        # v_la = place[1] - place[0]
                        v_la = v_la / numpy.linalg.norm(v_la)

                        # Angle to rotate by, convert to degrees
                        angle = (math.acos(numpy.dot(v_n, v_la) / (
                                    numpy.linalg.norm(v_la) * numpy.linalg.norm(v_n))) * 180.0 / math.pi)
                        
                        # debug print
                        if args.debug:
                            if y == 0:
                                print('Angle bottom: ' + str(angle))
                                print('v_n bottom: ' + str(v_n))
                            elif y == dims[1]:
                                print('Angle top: ' + str(angle))
                                print('v_n top: ' + str(v_n))
            
                        v_n_la = numpy.cross(v_n, v_la)
                        # print(v_n_la)

                        # rotate surface anion to be normal to the top/bottom
                        for count_j, j in enumerate(place):
                            if count_j != 0:
                                place[count_j, :] = perovskite_builder.axis_rot(j, v_n_la, place[0], -angle, mode='angle')

                        #if it is the top layer, we need to rotate the surface cation 180 degrees
                        if y == dims[1]:
                            for count_k, k in enumerate(place):
                                place[count_k, :] = perovskite_builder.axis_rot(k, numpy.array([0, 0, 1]), place[0], 180, mode='angle')
                        
                        # Define long-axis vector and normalize
                        # v_la = place[surface_cation["tail_atom"]] - place[0]
                        # v_la = place[1] - place[0]
                        # v_la = v_la / numpy.linalg.norm(v_la)                    

                        # Tilt relative to x axis
                        for count_k, k in enumerate(place):
                            place[count_k, :] = perovskite_builder.axis_rot(k, numpy.array([1.0, 0.0, 0.0]), place[0],
                                                            surface_cation_orientation[0], mode='angle')

                        if y == 0:
                        # Sweep relative to the y axis (sweeps around the unit cell based on the tilt angle)
                            for count_k, k in enumerate(place):
                                place[count_k, :] = perovskite_builder.axis_rot(k, numpy.array([0.0, 1.0, 0.0]), place[0],
                                                            surface_cation_orientation[1], mode='angle')
                        elif y == dims[1]:
                        # Sweep relative to the y axis (sweeps around the unit cell based on the tilt angle)
                            for count_k, k in enumerate(place):
                                place[count_k, :] = perovskite_builder.axis_rot(k, numpy.array([0.0, -1.0, 0.0]), place[0],
                                                            surface_cation_orientation[1], mode='angle')

                        # Rotate about molecular z axis (spins the molecule)
                        for count_k, k in enumerate(place):
                            place[count_k, :] = perovskite_builder.axis_rot(k, [0, 0, 1], place[0], surface_cation_orientation[2],
                                                            mode='angle')
                        
                        sim_box[atoms_placed:atoms_placed + len(surface_cation["elements"])] = place
                                
                        # Record the indices of the surface anions. These are the numpy array indices, which start numbering at 0;
                        # most molecular visualization programs start counting at 1, so when writing these out remember to add 1 to each.
                        for i in range(atoms_placed, atoms_placed + len(surface_cation["elements"])):
                            surface_indices.append(i)

                        # Record index of head atom in surface cation
                        surface_head_indices.append(atoms_placed)

                        # Update lists
                        sim_data["elements"] = sim_data["elements"] + surface_cation["elements"]

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
    surface_cation_ydim = 10
    # print(len(sim_box))
    # print(sim_box[sim_box[:,1] > 30])
    # print(max(sim_box[:, 0]), min(sim_box[:, 0]), max(sim_box[:, 1]), min(sim_box[:, 1]))

    sim_box_dims = (-bond_length, ((2 * bond_length) * dims[0]) + bond_length,
                    sim_box[:,1].min() - y_pad, sim_box[:,1].max() + y_pad,
                    -bond_length, ((2 * bond_length) * dims[2]) + bond_length)

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
    
    # Pb_indices = [count_i for count_i, i in enumerate(sim_data['elements']) if i == 'Pb']
    # N_indices = [count_i for count_i, i in enumerate(sim_data['elements']) if i == 'N']
    # # print("N_id: " + str(N_id))
    # # print("Pb_id: " + str(Pb_id))

    # N_geo = sim_box[N_indices]
    # Pb_geo = sim_box[Pb_indices]
    # closest_N_Pb = []
    # # print(N_geo, Pb_geo)
    # for point_N in N_geo:
    #     Pb_index = Pb_indices[find_closest_point(point_N, Pb_geo)]
    #     N_index = N_indices[numpy.where((N_geo == point_N).all(axis=1))[0][0]]
    #     print(Pb_index, N_index)

    total_box = sim_box
    total_adj_mat = sim_data["adj_mat"]

    sim_box_dims = (-bond_length, ((2 * bond_length) * dims[0]) + bond_length,
                    sim_box[:,1].min() - y_pad, sim_box[:,1].max() + y_pad,
                    -bond_length, ((2 * bond_length) * dims[2]) + bond_length)

    # Write out the *.xyz output file
    print('\nWriting "{}.xyz" output'.format(output))
    with open(output + '/' + output + '.xyz', 'w') as f:
        f.write('{}\n'.format(len(sim_data["elements"])))

        f.write('perovskite_{}{}{}\n'.format(anion, metal, cation))

        for i in range(len(sim_data["elements"])):
            f.write('{:<40s} {:<20.6f} {:<20.6f} {:<20.6f}\n'.format(sim_data["elements"][i], total_box[i][0],
                                                                        total_box[i][1], total_box[i][2]))

    print('\nWriting LAMMPS data file "{}.data"'.format(output))

    # Generate VDW parameters
    VDW_params = perovskite_builder.initialize_VDW(sorted(set(sim_data["atom_types"])), 1.0, 1.0, sim_data["VDW_params"], 0,
                                mixing_rule, False, False)

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

    Atom_type_dict, Bond_type_dict, Angle_type_dict, fixed_modes = perovskite_builder.Write_data(output + '/' + output,
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


    print('\nWriting atom type correspondence file "{}_correspondence.map"'.format(output))
    with open(output + '/' + output + '_correspondence.map', 'w') as f:
        f.write('{}\n'.format(len(sim_data["elements"])))

        f.write('perovskite_{}{}{}\n\n'.format(anion, metal, cation))

        for key in Atom_type_dict:
            f.write('{0:<50} {1:<3} \n'.format(key, Atom_type_dict[key]))

        f.write('\n\n')
        for i in range(len(sim_data["elements"])):
            f.write('{}\t {}\t {}\t {}\n'.format(i, sim_data["elements"][i], sim_data["atom_types"][i],
                                                    Atom_type_dict[sim_data["atom_types"][i]]))

    # Write map file to more easily post-process the trajectory
    print("Writing mapfile ({})...".format(output + '_map.map'))
    perovskite_builder.write_map(output + '/' + output, sim_data["elements"], sim_data["atom_types"], sim_data["charges"],
                sim_data["masses"], total_adj_mat, numpy.zeros([len(sim_data["elements"])]), mols_placed)

    print('Writing LAMMPS tether fix to "{}"...'.format(output + '_tether.fix'))
    with open(output + '/' + output + '_tether.fix', 'w') as f:
        f.write('group surface-heads id {}\n'.format(' '.join(str(_ + 1) for _ in surface_head_indices)))
        f.write('fix tether surface-heads spring/self 10.0\n\nunfix		tether')


    print('\nFinished!\n')




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

def update_simbox(x, y, z, bond_length, sim_box, sim_data, atoms_placed, unit_cell, unit_cells_placed, mols_placed):
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
    temp_mols_placed = mols_placed + 1
    temp_atoms_placed = atoms_placed + len(unit_cell["elements"])
    temp_unit_cells_placed = unit_cells_placed + 1
    return geom, temp_mols_placed, temp_atoms_placed, temp_unit_cells_placed

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
                "centroid"] = perovskite_builder.read_xyz(element)
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
            surface_cation["masses"], surface_cation["VDW_params"], surface_cation["VDW_comments"] = perovskite_builder.Find_parameters(
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
            surface_cation["masses"] = {surface_cation["atom_types"][0]: perovskite_builder.get_masses([surface_cation["elements"][0]])}

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

def find_closest_point(point_a, points_b):
    distances = numpy.linalg.norm(points_b - point_a, axis=1)
    closest_index = numpy.argmin(distances)
    # closest_point = points_b[closest_index]
    return closest_index


if __name__ == '__main__':
    main(sys.argv[1:])