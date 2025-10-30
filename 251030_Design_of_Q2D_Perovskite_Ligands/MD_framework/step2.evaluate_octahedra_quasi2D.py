#!/bin/env python
#author: Stephen Shiring

import argparse,math,os,sys
import numpy
import numpy as np
from copy import deepcopy

# Append root directory to system path and import common functions
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
# sys.path.append('/home/nianz/code_repo/high-throughput-2D-Perovskite-ligand-screening-tool/MD/lib')
import functions

def main(argv):
    # TODO:
    # edit this script so that if there are mixing metal and halide structures, it will still be able to parse the data
    parser = argparse.ArgumentParser(description='Parse the BX6 octahedrons\' geometry and quantify PbX 6 distortion by calculating bond length '\
                                     'quadratic elongation (<lambda>) and their bond angle variance (sigma^2). This script assume there are not any'\
                                     'mixing halide/metal structures, i.e., no Pb_x_Sn_(1-x) or all halides are the same.' )
    # mandatory arguments
    parser.add_argument('metal', type=str, help = 'Metal atom name (e.g., Pb)')
    
    parser.add_argument('halide', type=str, help = 'Halide atom name (e.g., I)')
    
    parser.add_argument('n', type=int, help = 'Number of inorganic layers in the perovskite structure, e.g. 3')
    
    # positional arguments
    parser.add_argument('-d', type=str, dest='dirs', default='',
                        help = 'Space-delimited string of directories to operate on. If empty, operate on all discovered directories. Default: "" (empty)')
    
    parser.add_argument('-traj', type=str, dest='traj_file', default='0.nvt.lammpstrj',
                        help = 'Name of trajectory to parse. Default: 0.nvt.lammpstrj')
    
    parser.add_argument('-s', type=str, dest='success_file', default='0.success',
                        help = 'Name of corresponding success file for trajectory to parse. Default: 0.success')
    
    parser.add_argument('-f_start', type=int, dest='f_start', default=0,
                        help = 'Index of the frame to start on (0-indexing, inclusive, with the first frame being 0 irrespective of timestamp) Default: 0')

    parser.add_argument('-f_end',   type=int, dest='f_end', default=100,
                        help = 'Index of the frame to end on (0-indexing, inclusive, with the first frame being 0 irrespective of timestamp) Default: 100')

    parser.add_argument('-f_every', type=int, dest='f_every', default=1,         
                        help = 'Frequency of frames to parse Default: 1, every frame is parsed')

    parser.add_argument('-o',       type=str, dest='output', default='stability.out',
                        help='Name of output file containing stabilities. Default: stability.out')
    
    parser.add_argument('--disordered',  dest='flag_disordered',        action='store_const', const=True, default=False,
                        help = 'Invoke for systems that are disordered, i.e., not an ideal perovskite framework.')
    
    parser.add_argument('--debug',  dest='flag_debug',        action='store_const', const=True, default=False,
                        help = 'When invoked, print out diagnostic info to log file.')
    
    parser.add_argument('-exclude_dirs', dest='exclude_dirs', type=str, default='Stability',
                        help='Space-delimited string specifying directories to avoid processing. Default: Stability')
    
    
    args = parser.parse_args()
    if args.dirs == '':
        log_file = start_tee(f'evaluate_octahedra_current_dir.log')
    else:
        log_file = start_tee(f'evaluate_octahedra_{str(args.dirs).split("/")[-2]}.log')
    print("{}\n\nPROGRAM CALL: python evaluate_octahedra.py {}\n".format('-'*150, ' '.join([ i for i in argv])))
    
    # Infer metal and halide(s) of the simulation from the master directory name
    # Expects master directory name to be [metal]_[halide]_[perov_geom] (Pb_I_cubic)
    Info = {}
    Info['metal']      = args.metal
    Info['halide']     = args.halide
    Info['elements']   = ' '.join(sorted([Info['metal'], Info['halide']]))

    inorg_n = args.n
    if inorg_n <= 1:
        print("For Quasi2D structure, the number of inorganic layers must be greater than 1, exiting...")
        sys.exit(1)
    
    traj_file = args.traj_file
    success_file = args.success_file
    
    # this script must be in the same directory as the target directories
    # args.dirs = [ './'+_ for _ in args.dirs.split() ]
    if len(args.dirs) == 0:
        # Look over all directories in current directory
        # path = './'
        path = os.getcwd()
        dirs = sorted([os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o)) and o not in args.exclude_dirs and o.startswith('NH3')])
    else:
        path = str(args.dirs)
        dirs = sorted([os.path.join(path, o) for o in os.listdir(path) if os.path.isdir(os.path.join(path, o)) and o not in args.exclude_dirs and o.startswith('NH3')])
    if args.flag_debug:
        dirs = dirs[:1]
        
    print("all operating directories:")
    for d in dirs:
        print(d)
    print('\n')
    
    
    for d in dirs:
        print('Operating on {}'.format(' '.join([str(i) for i in d.split('/')[-2:]])))
        
        # Move into operating directory
        with functions.cd(d):
            
            # If there are no run dirs, skip
            # path = d
            run_dirs = sorted([o for o in os.listdir() if os.path.isdir(o) and 'run' in o])
            # if args.flag_debug:
            #     run_dirs = run_dirs[:1]
            
            
            if len(run_dirs) == 0:
                print('   ERROR: No run directories found, skipping...')
                continue
            
            data_file = [o for o in os.listdir() if os.path.isfile(o) and o.endswith('.data')]
            # read the data_file and check the atom types
            
            lmpatom = read_lmpdata(data_file[0])
            # check if info[metal] and info[halide] are in the data file
            halide_flag, metal_flag = False, False
            for num_atom, char_atom in enumerate(lmpatom):
                if char_atom == Info['metal']:
                    metal_flag = True
                elif char_atom == Info['halide']:
                    halide_flag = True
            if all((metal_flag, halide_flag)):
                print("Metal and/or halide specified in the propmt not exist, exiting...")
                sys.exit(1)
            
            
            
            # Data dictionary, holds run/bond_lqe/angle_var for the runs
            Data = {}
            Data['runs']       = []
            Data['bond_lqe']   = []    # bond length quadratic elongation
            Data['angle_var']  = []    # angle variance
            Data['drifting_halides'] = [] # number of drifting halides
            Data['metal_halide_metal_angle'] = [] # metal-halide-metal angle

            # if os.path.isfile(args.output):
            #    continue
            
            # Loop over run directories
            for run in run_dirs:
                with functions.cd(run):           
                    # Check that both the LAMMPS trajectory and success files are present
                    if not os.path.isfile(traj_file):
                        print('   ERROR: Missing trajectory file in {}, skipping...'.format(run[2:]))
                        continue
                    if not os.path.isfile(success_file):
                        print('   ERROR: LAMMPS simulation for {} is still running or failed, skipping...'.format(run[2:]))
                        continue
                    
                    # at_indices: key:element, value: elements' indicies
                    coord,vel,box_lo,box_hi,at_indices = read_lmptrj(traj_file,Info['elements'].split(),lmpatom)
                    # for quasi2D perovskite, when determining the stability, only need to consider the surface metal atoms
                    for time in coord:
                        metal_coords = []
                        for metal in at_indices[Info['metal']]:
                            metal_coords.append(coord[time][metal+1])
                        metal_coords = np.array(metal_coords)
                        
                        # zip the metal_coords and the corresponding index, sort by the y-coordinate
                        metal_coords = sorted(zip(metal_coords, at_indices[Info['metal']]), key=lambda x: x[0][1])
                        num_surf_metal = len(metal_coords) // inorg_n
                        top_surface_metal = metal_coords[:num_surf_metal]
                        bottom_surface_metal = metal_coords[-num_surf_metal:]
                        at_indices[Info['metal']] = [i[1] for i in top_surface_metal] + [i[1] for i in bottom_surface_metal]
                    
                    
                    bond_lqes, angle_vars, drifting_ions, metal_halide_metal_angle = eval_octahedra_quasi2D(Info['metal'], Info['halide'], at_indices, traj_file, args.f_start, args.f_end, args.f_every, debug=args.flag_debug)
                    # print('\n')
                    Data['runs'].append(run[2:])
                    Data['bond_lqe'].append(numpy.mean(numpy.array(bond_lqes)))
                    Data['angle_var'].append(numpy.mean(numpy.array(angle_vars)))
                    Data['drifting_halides'].append(numpy.mean(numpy.array(drifting_ions)))
                    
                    Data['metal_halide_metal_angle'].append(numpy.mean(numpy.array(metal_halide_metal_angle)))
           
            write_data(Data, fname=args.output)
            print('\n')

    print('\nFinished!')
    stop_tee(log_file)
    return

# Write out data file
def write_data(Data, fname='stability.out'):
    with open(fname, 'w') as o:
        o.write('{:15} {:^15} {:^15}\n'.format('run', 'bond_lqe', 'angle_var'))
        for i,r in enumerate(Data['runs']):
                o.write('{:15} {:< 15.12f} {:< 15.12f}\n'.format(r.split('/')[-1], Data['bond_lqe'][i], Data['angle_var'][i]))
        
        o.write('\n')
        o.write('{:15} {:^15} {:^15}\n'.format('run', 'drifting_halides', 'metal_halide_metal_angle'))
        for i,r in enumerate(Data['runs']):
            o.write('{:15} {:< 15.12f} {:< 15.12f}\n'.format(r.split('/')[-1], Data['drifting_halides'][i], Data['metal_halide_metal_angle'][i]))

def eval_octahedra_quasi2D(metals, halides, at_indices, traj_file, f_start, f_end, f_every, debug=False):
    # Define unit vectors to check halide positions. Defined here to avoid repetition and just passed to each function.
    # order: Top axial position [0], Bottom axial position [1], +x (x1) equitorial position [2],
    #        -x (x2) equitorial position [3], +z (z1) equitorial position [4], -z (z2) equitorial position [5]
    site_vectors = [ (0,1,0), (0,-1,0), (1,0,0), (-1,0,0), (0,0,1), (0,0,-1) ]
    
    bond_lqes    = []
    angle_vars   = []
    drifting_halides = []
    metal_halide_metal_angles = []
    # debug
    if debug:
        BOND_RECORD = {}
    
    if type(metals) == str:
        metals = metals.split() # assuming that metal is space delimited strings, e.g., 'Pb Sn'
    if type(halides) == str:
        halides = halides.split()
    
    for geo,ids,types,timestep,box in frame_generator(traj_file, f_start, f_end, f_every, unwrap=False):
        # debug
        if debug:
            BOND_RECORD[timestep] = []
                
        
        count_drifting_halides = 0
        for metal in metals: # Pb, Sn
            # Collect all target indices
            # all indices correspond to LAMMPS ids (i.e., counting starts at 1. to visualize in VMD, subtract 1)
            for halide in halides: # I, Br, Cl
                if len(at_indices[halide]) == 0 or len(at_indices[metal]) == 0:
                    print('   ERROR: No {} or {} atoms found in the trajectory, skipping...'.format(metal, halide))
                    continue
    
                # Take subset of geometries                            
                metal_geos      = deepcopy(numpy.take(geo, at_indices[metal], axis=0))
                halide_geos     = deepcopy(numpy.take(geo, at_indices[halide], axis=0))
                
                # use top 10 metal atoms to define the perovskite plane
                if len(metal_geos) < 10:
                    print('   ERROR: Less than 10 metal atoms found, skipping...')
                    continue
                m_X, m_Y, m_Z = metal_geos[:10,0], metal_geos[:10,1], metal_geos[:10,2]
                A = numpy.c_[m_X, m_Y, m_Z, numpy.ones(m_X.shape)]
                U, S, Vt = numpy.linalg.svd(A)
                v_n_general = Vt[-1,:3]
                v_n_general = v_n_general / numpy.linalg.norm(v_n_general)
                plane_coeff = Vt[-1,:]
                
                # Loop over each metal site: 
                for idx, metal_coord in enumerate(metal_geos):
                    
                    ##
                    # Define perovskite plane by the current metal and its 2 nearest neighbors
                    ##
                    
                    # Get 3 nearest metal atoms
                    dist_metal_metal    = numpy.zeros([len(metal_geos), 2])     # Holds distance, metal index
                    geo_unwraped_metals = numpy.zeros([len(metal_geos), 3])     # Holds distance, metal index
                    for j in range(len(metal_geos)):
                        unwraped_metal_coord = unwrap_pbc(metal_coord, metal_geos[j], box) 
                        geo_unwraped_metals[j] = unwraped_metal_coord
                        dist = calc_dist(metal_coord, unwraped_metal_coord)
                        dist_metal_metal[j] = dist,j
                        
                    # Sort by distance (column 0)            
                    # .argsort() returns an numpy.array of indices that sort the given numpy.array. 
                    # Now call .argsort() on the column to sort, and it will give an array of row indices that sort that particular column to pass as an index to the original array.
                    dist_metal_metal = dist_metal_metal[numpy.argsort(dist_metal_metal[:, 0])]
                    
                    # Define the perovskite plane by the current metal site and its 2 closest metal atoms.
                    # The finding nearest neighbors algorithm above will return the current metal site as the first element in the list (since it's not excluded and its distance is 0),
                    # so just its coordinates as is from the list
                    # Find the vector normal to this plane.            
                    v_1 = geo_unwraped_metals[int(dist_metal_metal[1,1])] - geo_unwraped_metals[int(dist_metal_metal[0,1])]
                    v_2 = geo_unwraped_metals[int(dist_metal_metal[2,1])] - geo_unwraped_metals[int(dist_metal_metal[0,1])]
                    v_n = numpy.cross(v_1, v_2)
                    if numpy.linalg.norm(v_n) < 1.5: # manually set a threshold to avoid the case where the 3 metal atoms are in a line
                        # print('   ERROR: 3 metal atoms are in a line, using the general normal vector...')
                        v_n = v_n_general
                    
                    v_n = v_n / numpy.linalg.norm(v_n)
                    
                    
                    # if debug:
                    #     print(geo_unwraped_metals[int(dist_metal_metal[0,1])])
                    #     print(geo_unwraped_metals[int(dist_metal_metal[1,1])])
                    #     print(geo_unwraped_metals[int(dist_metal_metal[2,1])])
                    #     print(v_1, v_2)
                    #     print(v_n)
                    
                    '''
                    Zhichen Edited: calculate the plane equation
                    '''
                    plane_eq = [v_n[0], v_n[1], v_n[2], -numpy.dot(v_n, metal_coord)] # Ax1 + By1 + Cz1 = -D
                    # find four metal closest to the current metal
                    equator_metal_idx = dist_metal_metal[1:5,1]
                    equator_metal_coords = [geo_unwraped_metals[int(i)] for i in equator_metal_idx]
                    
                    
                    ##
                    # Determine halide sites
                    # if, for whatever reason, a metal site doesn't have all of its halides / they can't be
                    # identified, then skip that site.
                    ##
                    axial_sites      = [0, 0]   # top, bottom
                    equitorial_sites = [0, 0, 0, 0]   # +x, -x, +z, -z
                    
                    found_axial_sites      = [False, False]   # top, bottom
                    found_equitorial_sites = [False, False, False, False]   # +x, -x, +z, -z
                    
                    working_coords = numpy.zeros([11,3])
                    working_coords[0] = deepcopy(metal_coord)
                    
                    # Get 10 nearest halide atoms. Fully occupied it has 6 neighbors, but in some frames there is a transient bond elongation along the apical direction
                    # that the parser may miss. Including additional neighbors ensures that we catch that, since the dot product will weed out any halides on 
                    # adjacent sites.
                    halide_neighbors = numpy.zeros([len(halide_geos), 3])     # Holds distance, index, atom index in sim
                    geo_unwrapped_halides = numpy.zeros([len(halide_geos), 3])     # Holds unwrapped coordinates
                                      
                    
                    
                    for i in range(len(halide_geos)):
                        unwrapped_halide_coord = unwrap_pbc(metal_coord, halide_geos[i], box) 
                        geo_unwrapped_halides[i] = unwrapped_halide_coord
                        dist = calc_dist(metal_coord, unwrapped_halide_coord)
                        halide_neighbors[i] = dist,i,at_indices[halide][i]
                    halide_neighbors = halide_neighbors[numpy.argsort(halide_neighbors[:, 0])]
                    
                    '''
                    Zhichen Edited: find the metal-halide-metal angle.
                    '''
                    middle_halide_geo = []
                    current_metal_halide_metal_angle = 0
                    correct_emc = []
                    for emc in equator_metal_coords:
                        mid_dis = 1000
                        for guh in geo_unwrapped_halides:
                            distance_1 = calc_dist(guh, emc)
                            distance_2 = calc_dist(guh, metal_coord)
                            if distance_1 + distance_2 < mid_dis:
                                mid_dis = distance_1 + distance_2
                                middle_halide_geo = guh
                                # if debug:
                                #     print("updating mhg", middle_halide_geo)
                                
                        
                        
                        # calculate the angle between the metal-halide-metal
                        if not len(middle_halide_geo) == 3 and mid_dis < 8:
                            print('ERROR: middle halide not found, skipping...')
                            sys.exit(1)
                            
                        # project the middle halide to the plane
                        # find the projection of the middle halide to the plane
                        # the projection of a vector a to the plane is a - (a.n)n
                        # where n is the normal vector of the plane
                        t = (numpy.dot(v_n_general, middle_halide_geo) + plane_eq[3]) / numpy.dot(v_n_general, v_n_general)
                        proj_middle_halide = middle_halide_geo - t * v_n_general
                        # if debug:
                        #     if calc_angle(emc, proj_middle_halide, metal_coord)*180.0/numpy.pi < 90:
                        #         print('ERROR: angle less than 150, skipping...')
                        #         print('mid_dis:', mid_dis)
                        #         print('current metal:', metal_coord)
                        #         print('middle halide:', middle_halide_geo)
                        #         print('projected middle halide:', proj_middle_halide)
                        #         print('equator metal:', emc)
                        #         print('calc_angle:', calc_angle(emc, proj_middle_halide, metal_coord)*180.0/numpy.pi)
                        current_metal_halide_metal_angle += calc_angle(emc, middle_halide_geo, metal_coord)
                        
                    
                    # if debug:
                    #     print('current_metal_halide_metal_angle:', current_metal_halide_metal_angle * 180.0 / numpy.pi)
                    current_metal_halide_metal_angle = current_metal_halide_metal_angle / 4
                    metal_halide_metal_angles.append(current_metal_halide_metal_angle * 180.0 / numpy.pi)
                            
                        
                            
                    for i in range(10):
                        working_coords[i+1] = geo_unwrapped_halides[int(halide_neighbors[i][1])]
                        
                    working_coords -= working_coords[0]             # Center about the metal atom
                    # Loop over the halide positions, compute dot product between its vector and the top axial position vector
                    # use a cutoff to determine position; if dot product value is (+), points towards top axial position, while (-) will point towards bottom axial position.
                    # if dot product is 0 (or near 0), they are orthogonal, i.e. in an equitorial position.
                    # reordered to check all primary axes first, then the off diagonal axial positions
                    # only rounded (originally) the axial positions
                    # not tracking an individual occupancy in the equitorial position, but rather just total occupancy.
                    
                    for wc_i, wc in enumerate(working_coords[1:]):
                        ref_wc = deepcopy(wc)
                        wc     = wc / numpy.linalg.norm(wc)
                
                        for vec_i, vec in enumerate(site_vectors):
                            if vec_i == 0 or vec_i == 1:
                                if round(numpy.dot(wc, vec),4) >= 0.8500 and not found_axial_sites[vec_i]:
                                    if vec_i == 0:
                                        axial_sites[0]       = deepcopy(ref_wc)
                                        found_axial_sites[0] = True
                                        break
                                    elif vec_i == 1:
                                        axial_sites[1]       = deepcopy(ref_wc)
                                        found_axial_sites[1] = True
                                        break
                            else:
                                if round(numpy.dot(wc, vec),4) >= 0.7500 and not found_equitorial_sites[vec_i-2]:
                                    if vec_i == 2:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[0]       = deepcopy(ref_wc)
                                            found_equitorial_sites[0] = True
                                            break
                                    elif vec_i == 3:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[1]       = deepcopy(ref_wc)
                                            found_equitorial_sites[1] = True
                                            break
                                    elif vec_i == 4:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[2]       = deepcopy(ref_wc)
                                            found_equitorial_sites[2] = True
                                            break
                                    elif vec_i == 5:
                                        if calc_dist(working_coords[0], ref_wc) <= 4.5:
                                            equitorial_sites[3]       = deepcopy(ref_wc)
                                            found_equitorial_sites[3] = True
                                            break
    
                    # There are 12 unique 90-degree angles within the perovskite octahedron that we need to check
                    # these are:
                    # apical1/top - axial1, apical1 - axial2, apical1 - axial3, apical1 - axial4
                    # apical2/bottom - axial1, apical2 - axial2, apical2 - axial3, apical2 - axial4
                    # axial1 - axial2, axial1 - axial4, axial3 - axial2, axial3 - axial4
                    # (+x,+z), (+x,-z), (-x,+z), (-x,-z)
                    # store the indices to loop over
                    axial_check_sites = [(0,0),(0,1),(0,2),(0,3),(1,0),(1,1),(1,2),(1,3)]  # apical, equitorial
                    equitorial_check_sites = [(0,2),(0,3),(1,2),(1,3)] #equitorial, equitorial
                    
                    
                    # debug
                    if debug:
                        # print(axial_sites)
                        # print(equitorial_sites)
                        # check if any axial sites has element that is greater than 4.5
                        for i,site in enumerate(axial_sites):
                            if site is not False:
                                if calc_dist(working_coords[0], site) > 4.5:
                                    print('   ERROR: Axial site {} is greater than 4.5, skipping...'.format(i))
                                    print(working_coords[0], site)
                                    # change the values in working_coords to 2 decimal places
                                    working_coords = numpy.round(working_coords, 2)
                                    print(working_coords)
                                    continue
                    
                    Bonds = []
                    Angles = []
                    
                    counter = 0                        
                    for site in axial_check_sites:
                        if found_axial_sites[site[0]] is False or found_equitorial_sites[site[1]] is False:
                            # skip if one of the sites wasn't found
                            continue
                        else:
                            Angles.append( calc_angle( axial_sites[site[0]], working_coords[0], equitorial_sites[site[1]] )*180.0/numpy.pi )
                            
                        counter += 1
                    
                    for site in equitorial_check_sites:
                        if found_equitorial_sites[site[0]] is False or found_equitorial_sites[site[1]] is False:
                            continue
                        else:
                            Angles.append( calc_angle( equitorial_sites[site[0]], working_coords[0], equitorial_sites[site[1]] )*180.0/numpy.pi )
                        counter += 1
                    
                    for i,site in enumerate(found_axial_sites):
                        if site is not False:
                            Bonds.append( calc_bond( working_coords[0], axial_sites[i] ) )
                    
                    for i,site in enumerate(found_equitorial_sites):
                        if site is not False:
                            Bonds.append( calc_bond( working_coords[0], equitorial_sites[i] ) )
                    
                    Bonds = numpy.array(Bonds)
                    Angles = numpy.array(Angles)
                    
                    # debug
                    if debug:
                        BOND_RECORD[timestep].append(Bonds)
                    
                    if len(Bonds) > 0:
                        bond_lqe = 0
                        bond_avg = numpy.mean(Bonds)
                        for b in Bonds:
                            bond_lqe += (b/bond_avg)**2
                            if b/bond_avg > 1.5:
                                count_drifting_halides += 1
                                
                        bond_lqe = bond_lqe / 6
                        bond_lqes.append(bond_lqe)
                    
                    # also if the len(bond) is less than 6, then the halide is drifting
                    count_drifting_halides += 6 - len(Bonds)
                    
                    if len(Angles) > 0:
                        angle_var = 0
                        for a in Angles:
                            angle_var += (a - 90.0)**2
                        angle_var = angle_var / 11
                        angle_vars.append(angle_var)

        drifting_halides.append(count_drifting_halides)
    
    # debug   
    # for timestep in BOND_RECORD:
    #     print(timestep)
    #     for bonds in BOND_RECORD[timestep]:
    #         print(bonds)
    print(len(bond_lqes), len(angle_vars), len(drifting_halides), len(metal_halide_metal_angles))
    return bond_lqes, angle_vars, drifting_halides, metal_halide_metal_angles

# Loop for parsing the mapfile information
def parse_map(map_file, atoms_list):

    at_indices = {}
    at_counts  = {}
    for atom in atoms_list:
        at_indices[atom] = []
        at_counts[atom]  = 0
    
    atomtypes  = []
    elements   = []
    masses     = []
    charges    = []
    adj_list   = []
    masses_dict = {}
    
    with open(map_file,'r') as f:
        for lc,lines in enumerate(f):
            fields = lines.split()
            if lc > 1 and len(fields) > 4:
                atomtypes += [fields[0]]
                elements  += [fields[1]]
                masses    += [float(fields[3])]
                charges   += [float(fields[4])]
                adj_list  += [ [int(_) for _ in fields[5:] ] ]
                
                if fields[1] in atoms_list:
                    at_indices[fields[1]].append(lc-2)
                    at_counts[fields[1]] += 1
                
                if str(fields[3]) not in masses_dict:
                    masses_dict[str(fields[3])] =  fields[1]
                
    return atomtypes, elements, masses, charges, adj_list, masses_dict, at_indices, at_counts


def read_lmpdata(dataname):
    """
    Args:
        dataname (str): the name of the lammps data file

    Returns:
        lammps atom type, determined by its mass
    """
    
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}
    lmpatom = {} # key:lammps atom type(just a number) value:Element
    mass_flag = False
    with open(dataname,'r') as f:
        for l in f:
            temp = l.split()
            if len(temp) == 1 and temp[0] == "Masses":
                mass_flag = True
                continue
            if len(temp) == 2 and mass_flag:
                numeric_ele, mass = temp
                mass = float(mass)
                for char_ele in mass_dict:
                    char_ele_mass = mass_dict[char_ele]
                    if abs(mass - char_ele_mass) < 0.1:
                        lmpatom[int(numeric_ele)] = char_ele
                
    #   for lc,lines in enumerate(f):
    #      fields = lines.split()
    #      if len(fields)>0  and fields[0] == 'Masses':
    #         flag = 1 
    #         tag = 0
    #         continue
    #      if flag == 1:
    #         if len(fields) == 0: 
    #            tag += 1
    #            if tag == 2: 
    #               flag = 0
    #            continue
    #         mass = float(fields[1]) 
    #         for key in mass_dict:
    #            if (abs(mass-mass_dict[key])) < 0.01:
    #               ele = key
    #               break
    #         lmpatom[int(fields[0])] = ele
    #         continue

    return lmpatom 


def read_lmptrj(lmptrj_file,atoms_list,lmpatom):
    """only read the 

    Args:
        data_name (_type_): _description_
        atoms_list (_type_): _description_
        lmpatom (_type_): _description_

    Returns:
        _type_: _description_
    """
    at_indices = {}
    for atom in atoms_list:
        at_indices[atom] = []
    # print(at_indices)
    
    coord = {}
    vel = {}
    box_lo = {}
    box_hi = {}
    print('reading {}...'.format(lmptrj_file))# , end='\r')
    with open(lmptrj_file,'r') as f:
        flag = 0
        for lc,lines in enumerate(f):
            fields = lines.split()
            if flag == 0 and len(fields) < 2: continue
            if fields[0] == 'ITEM:' and fields[1] == 'TIMESTEP':
                flag = 1
                continue

            if fields[0] == 'ITEM:' and fields[1] == 'ATOMS' and fields[2] == 'id':
                flag  = 2
                continue

            if fields[0] == 'ITEM:' and fields[1] == 'BOX' and fields[2] == 'BOUNDS':
                flag  = 3
                box_lo[time]  = []
                box_hi[time] = []
                continue

            if flag == 1:
                time = float(fields[0])
                if len(list(coord.keys())) == 1: 
                    break
                coord[time] = {}
                vel[time] = {}
                flag = 0
                continue 
            
            if flag == 2:
                index = int(fields[0])
                atype = int(fields[1])
                for _ in atoms_list:    
                    if lmpatom[atype] == _:
                        at_indices[_].append(index-1) 
                coord[time][index] = [ float(i) for i in fields[2:5]]
                vel[time][index] = [ float(i) for i in fields[5:8]]
                continue

            if flag == 3:
                if len(box_lo[time]) == 3: 
                    flag = 0
                    continue
                box_lo[time].append(fields[0])
                box_hi[time].append(fields[1])
                continue

    return coord,vel,box_lo,box_hi,at_indices 
         

# Generator function that yields the geometry, atomids, and atomtypes of each frame
# with a user specified frequency
def frame_generator(name,start,end,every,unwrap=True,adj_list=None):

    if unwrap is True and adj_list is None:
        print("ERROR in frame_generator: unwrap option is True but no adjacency_list is supplied. Exiting...")
        quit()

    # Parse data for the monitored molecules from the trajectories
    # NOTE: the structure of the molecule based parse is almost identical to the type based parse
    #       save that the molecule centroids and charges are used for the parse
    # Initialize subdictionary and "boxes" sub-sub dictionary (holds the box dimensions for each parsed frame)

    # Parse Trajectories
    frame       = -1                                                  # Frame counter (total number of frames in the trajectory)
    frame_count = -1                                                  # Frame counter (number of parsed frames in the trajectory)
    frame_flag  =  0                                                  # Flag for marking the start of a parsed frame
    atom_flag   =  0                                                  # Flag for marking the start of a parsed Atom data block
    N_atom_flag =  0                                                  # Flag for marking the place the number of atoms should be updated
    atom_count  =  0                                                  # Atom counter for each frame
    box_flag    =  0                                                  # Flag for marking the start of the box parse
    box_count   = -1                                                  # Line counter for keeping track of the box dimensions.

    # Open the trajectory file for reading
    with open(name,'r') as f:

        # Iterate over the lines of the original trajectory file
        for lines in f:

            fields = lines.split()

            # Find the start of each frame and check if it is included in the user-requested range
            if len(fields) == 2 and fields[1] == "TIMESTEP":
                frame += 1
                if frame >= start and frame <= end and (frame-start) % every == 0:
                    frame_flag = 1
                    frame_count += 1
                elif frame > end:
                    break
            # Parse commands for when a user-requested frame is being parsed
            if frame_flag == 1:

                # Header parse commands
                if atom_flag == 0 and N_atom_flag == 0 and box_flag == 0:
                    if len(fields) > 2 and fields[1] == "ATOMS":
                        atom_flag = 1
                        id_ind   = fields.index('id')   - 2
                        type_ind = fields.index('type') - 2
                        x_ind    = fields.index('x')    - 2
                        y_ind    = fields.index('y')    - 2
                        z_ind    = fields.index('z')    - 2
                        continue
                    if len(fields) > 2 and fields[1] == "NUMBER":                        
                        N_atom_flag = 1
                        continue

                    if len(fields) > 2 and fields[1] == "BOX":
                        box      = numpy.zeros([3,2])
                        box_flag = 1
                        continue
                    
                    if len(fields) == 1:
                        timestep = fields[0]
                        continue

                # Update the number of atoms in each frame
                if N_atom_flag == 1:

                    # Intialize total geometry of the molecules being parsed in this frame
                    # Note: from here forward the N_current acts as a counter of the number of atoms that have been parsed from the trajectory.
                    N_atoms     = int(fields[0])
                    geo         = numpy.zeros([N_atoms,3])                    
                    ids         = [ -1 for _ in range(N_atoms) ]
                    types       = [ -1 for _ in range(N_atoms) ]
                    N_atom_flag = 0
                    continue

                # Read in box dimensions
                if box_flag == 1:
                    box_count += 1
                    box[box_count] = [float(fields[0]),float(fields[1])]

                    # After all box data has been parsed, save the box_lengths/2 to temporary variables for unwrapping coordinates and reset flags/counters
                    if box_count == 2:
                        box_count = -1
                        box_flag = 0
                    continue

                # Parse relevant atoms
                if atom_flag == 1:
                    geo[atom_count]   = numpy.array([ float(fields[x_ind]),float(fields[y_ind]),float(fields[z_ind]) ])
                    ids[atom_count]   = int(fields[id_ind])
                    types[atom_count] = int(fields[type_ind])                    
                    atom_count += 1

                    # Reset flags once all atoms have been parsed
                    if atom_count == N_atoms:

                        frame_flag = 0
                        atom_flag  = 0
                        atom_count = 0       

                        # Sort based on ids
                        ids,sort_ind =  list(zip(*sorted([ (k,count_k) for count_k,k in enumerate(ids) ])))
                        geo = geo[list(sort_ind)]
                        types = [ types[_] for _ in sort_ind ]
                        
                        # Upwrap the geometry
                        if unwrap is True:
                            geo = unwrap_geo(geo,adj_list,box)

                        yield geo,ids,types,timestep,box

# Unwrap the PBC between a given ref (3x1 array) and target (3x1) geom. box is actual box dimensions
def unwrap_pbc(ref_coord,target_coord,box):
    bx_2 = ( box[0,1] - box[0,0] ) / 2.0
    by_2 = ( box[1,1] - box[1,0] ) / 2.0
    bz_2 = ( box[2,1] - box[2,0] ) / 2.0
    
    if (target_coord[0] - ref_coord[0])   >  bx_2: target_coord[0] -= (bx_2*2.0) 
    elif (target_coord[0] - ref_coord[0]) < -bx_2: target_coord[0] += (bx_2*2.0) 
    if (target_coord[1] - ref_coord[1])   >  by_2: target_coord[1] -= (by_2*2.0) 
    elif (target_coord[1] - ref_coord[1]) < -by_2: target_coord[1] += (by_2*2.0) 
    if (target_coord[2] - ref_coord[2])   >  bz_2: target_coord[2] -= (bz_2*2.0) 
    elif (target_coord[2] - ref_coord[2]) < -bz_2: target_coord[2] += (bz_2*2.0) 
    
    return target_coord

# Description: Performed the periodic boundary unwrap of the geometry
def unwrap_geo(geo,adj_list,box):

    bx_2 = ( box[0,1] - box[0,0] ) / 2.0
    by_2 = ( box[1,1] - box[1,0] ) / 2.0
    bz_2 = ( box[2,1] - box[2,0] ) / 2.0

    # Unwrap the molecules using the adjacency matrix
    # Loops over the individual atoms and if they haven't been unwrapped yet, performs a walk
    # of the molecular graphs unwrapping based on the bonds. 
    unwrapped = []
    for count_i,i in enumerate(geo):

        # Skip if this atom has already been unwrapped
        if count_i in unwrapped:
            continue

        # Proceed with a walk of the molecular graph
        # The molecular graph is cumulatively built up in the "unwrap" list and is initially seeded with the current atom
        else:
            unwrap     = [count_i]    # list of indices to unwrap (next loop)
            unwrapped += [count_i]    # list of indices that have already been unwrapped (first index is left in place)
            for j in unwrap:

                # new holds the index in geo of bonded atoms to j that need to be unwrapped
                new = [ k for k in adj_list[j] if k not in unwrapped ] 

                # unwrap the new atoms
                for k in new:
                    unwrapped += [k]
                    if (geo[k][0] - geo[j][0])   >  bx_2: geo[k,0] -= (bx_2*2.0) 
                    elif (geo[k][0] - geo[j][0]) < -bx_2: geo[k,0] += (bx_2*2.0) 
                    if (geo[k][1] - geo[j][1])   >  by_2: geo[k,1] -= (by_2*2.0) 
                    elif (geo[k][1] - geo[j][1]) < -by_2: geo[k,1] += (by_2*2.0) 
                    if (geo[k][2] - geo[j][2])   >  bz_2: geo[k,2] -= (bz_2*2.0) 
                    elif (geo[k][2] - geo[j][2]) < -bz_2: geo[k,2] += (bz_2*2.0) 

                # append the just unwrapped atoms to the molecular graph so that their connections can be looped over and unwrapped. 
                unwrap += new

    return geo

def calc_dist(ref,target):
    return math.sqrt((target[0]-ref[0])**2+(target[1]-ref[1])**2+(target[2]-ref[2])**2)

def calc_bond(atom_1,atom_2):
    return numpy.linalg.norm(atom_2 - atom_1)

def calc_angle(atom_1,atom_2,atom_3):
        # if numpy.arccos(numpy.dot(atom_1-atom_2,atom_3-atom_2)/(numpy.linalg.norm(atom_1-atom_2)*numpy.linalg.norm(atom_3-atom_2)))<20.0*numpy.pi/180.0:
        #         print(atom_1,atom_2, atom_3)
        #         print()
        return numpy.arccos(numpy.dot(atom_1-atom_2,atom_3-atom_2)/(numpy.linalg.norm(atom_1-atom_2)*numpy.linalg.norm(atom_3-atom_2)))

class Tee:
    def __init__(self, *files):
        """Initialize with a list of file-like objects."""
        self.files = files

    def write(self, text):
        """Write text to all file-like objects."""
        for f in self.files:
            f.write(text)
            f.flush()  # Ensure output is written immediately

    def flush(self):
        """Handle flush to avoid I/O buffering issues."""
        for f in self.files:
            f.flush()

# Function to start teeing the output
def start_tee(log_file_path):
    """Start redirecting stdout to both the console and a log file."""
    log_file = open(log_file_path, 'w')
    tee = Tee(sys.stdout, log_file)  # Send output to both console and file
    sys.stdout = tee  # Redirect stdout to the tee instance
    return log_file

# Function to stop teeing the output
def stop_tee(log_file):
    """Stop the redirection and restore stdout."""
    log_file.close()
    sys.stdout = sys.__stdout__  # Restore the original stdout
   
if  __name__ == '__main__': 
    main(sys.argv[1:])
