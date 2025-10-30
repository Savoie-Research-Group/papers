#!/bin/env python
# Author: Brett Savoie (brettsavoie@gmail.com)

# Add TAFFY Lib to path
import sys,os
import io
from subprocess import call,check_output,STDOUT,Popen,PIPE
import subprocess
from numpy import *
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy.linalg import norm
from file_parsers import *
from write_functions import *
from adjacency import *
from copy import deepcopy

def main(argv):

    # Sanity check on the parameters
    if len(argv) < 1:
        print("ERROR: transify.py expects at least a filename. Exiting...")
        quit()

    # Assign output name
    if len(argv) > 1:
        name=argv[1]
    else:
        name="straightened.xyz"

    # Extract Element list and Coord list from the file
    elements,geo = xyz_parse(argv[0])
    adj_mat = Table_generator(elements,geo)
    geo = transify(geo,adj_mat,opt_terminals=True,opt_final=True,elements=elements)
#    geo = transify(geo,adj_mat,opt_terminals=False,opt_final=False,elements=elements)

    # Save the straightened geometry    
    with open(name,'w') as f:
        f.write("{}\n\n".format(len(elements)))
        for count_i,i in enumerate(elements):
            f.write("{} {}\n".format(i," ".join([ "{:< 20.8f}".format(j) for j in geo[count_i] ])))

# This function takes a geometry and adjacency matrix and returns an "all-trans" aligned geometric conformer                
def transify(geo,adj_mat,start=-1,end=-1,elements=None,opt_terminals=False,opt_final=False):

    # Sanity checks
    if start < -1 or start >= len(geo): print("ERROR in transify: start must be set to an index within geo or -1. Exiting..."); quit()
    if end   < -1 or end   >= len(geo): print("ERROR in transify: end must be set to an index within geo or -1. Exiting..."); quit()
    if len(geo) != len(adj_mat): print("ERROR in transify: geo and adj_mat must have the same dimensions. Exiting..."); quit()
    if opt_terminals == True and elements is None: print("ERROR in transify: element information must be supplied for terminals to be optimized. Exiting..."); quit()
    if opt_terminals == True and elements is not None and len(elements) != len(geo): print("ERROR in transify: length of elements is not equal to the length of geo. Exiting..."); quit()
    if opt_final == True and elements is None: print("ERROR in transify: element information must be supplied for the final structure to be optimized. Exiting..."); quit()
    if opt_final == True and elements is not None and len(elements) != len(geo): print("ERROR in transify: length of elements is not equal to the length of geo. Exiting..."); quit()

    # Return the geometry if it is empty
    if len(geo) == 0: 
        return geo

    # Calculate the pair-wise graphical separations between all atoms
    seps    = graph_seps(adj_mat)

    # Find the most graphically separated pair of atoms. 
    if start == -1 and end == -1:
        max_ind = where(seps == seps.max())
        start,end =  max_ind[0][0],max_ind[1][0]

    # Find the most graphically separated atom from the start atom
    elif end == -1:
        end = argmax(seps[start])        

    # Find the shortest pathway between these points
    pathway = Dijkstra(adj_mat,start,end)

    # If the molecule doesn't have any dihedrals then return
    if len(pathway) < 4:
        return geo

    # Initialize the list of terminal atoms and ring atoms (used in a couple places so initialized here relatively early)
    terminals = set([ count_i for count_i,i in enumerate(adj_mat) if sum(i) == 1 ])
    rings = set([ count_i for count_i,i in enumerate(geo) if ring_atom(adj_mat,count_i) == True ]) 

    # Work through the backbone dihedrals and straighten them out
    for i in range(1,len(pathway)-2):

        if pathway[i] in rings: continue

        # Collect the atoms that are connected to the 2 atom of the dihedral but not to the 3 atom of the dihedral (and vice versus)
        group_1 = return_connected(adj_mat,start=pathway[i],avoid=[pathway[i+1]])   # not used since only the forward portion of the pathway gets rotated
        group_2 = return_connected(adj_mat,start=pathway[i+1],avoid=[pathway[i]])

        # Skip if the two groups are equal (happens in the case of rings)
        if group_1 == group_2: continue

        # Calculate the rotation vector and angle
        rot_vec = geo[pathway[i+1]] - geo[pathway[i]]
        stationary = geo[pathway[i]]
        theta = pi - dihedral_calc(geo[pathway[i-1:i+3]])                
        
        # Perform the rotation
        for j in group_2:
            geo[j] = axis_rot(geo[j],rot_vec,stationary,theta,mode='radian')
    
        # Check for overlaps
        theta = pi - dihedral_calc(geo[pathway[i-1:i+3]])                
        
        # Consider including a check for overlaps and performing a gauche rotation as an alternative.
        # if len(where(cdist(geo,geo) < 1.0 & seps > 3)) > 0:            

    # Identify the branch points
    branches  = [ count_i for count_i,i in enumerate(adj_mat) if len([ j for count_j,j in enumerate(i) if j == 1 and count_j not in terminals and count_j not in rings]) > 2 ]
    avoid_list = terminals

    # Loop over the branch points and correct their dihedrals
    for count_i,i in enumerate(branches):
        
        branching_cons = [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j not in pathway ]

        for c in branching_cons:

            # Find the connections to this branch point that are terminal atoms
            conn_terminals = [ j for j in branching_cons if j != c ]

            # If a terminal connection exists then it is used to orient the branch by prepending it to the branch index list
            if len(conn_terminals) > 0:
                branch_ind = [conn_terminals[0]] + list(return_connected(adj_mat,start=i,avoid=[ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j in pathway ]+conn_terminals))
                geo[branch_ind[1:]] = transify(geo[branch_ind],adj_mat[branch_ind,:][:,branch_ind],start=0)[1:]

            # If no terminal connections exists then the first dihedral of the branch is not adjusted. The logic is that the sp2/sp3 alignment is better judged by the initial guess. 
            else:
                branch_ind = list(return_connected(adj_mat,start=i,avoid=[ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j in pathway ]+conn_terminals))
                geo[branch_ind] = transify(geo[branch_ind],adj_mat[branch_ind,:][:,branch_ind],start=next( count_j for count_j,j in enumerate(branch_ind) if j == i))

    # If optimize structure is set to True then the transified structure is relaxed using obminimize
    if opt_final == True:
        opt_geo(geo,adj_mat,elements,ff='uff')

    # If optimize terminals is set to true then a conformer search is performed over the terminal groups
    if opt_terminals == True:
        geo = opt_terminal_centers(geo,adj_mat,elements,ff='uff')

    return geo

# Description: This function calls obminimize (open babel geometry optimizer function) to optimize the current geometry
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_geo(geo,adj_mat,elements,ff='uff'):

    # Write a temporary molfile for obminimize to use
    tmp_filename = '.tmp.mol'
    count = 0
    while os.path.isfile(tmp_filename):
        count += 1
        if count == 10:
            print("ERROR in opt_geo: could not find a suitable filename for the tmp geometry. Exiting...")
            return geo
        else:
            tmp_filename = ".tmp" + tmp_filename            

    # Use the mol_write function imported from the write_functions.py 
    # to write the current geometry and topology to file
    mol_write(tmp_filename,elements,geo,adj_mat,append_opt=False)
    
    # Popen(stdout=PIPE).communicate() returns stdout,stderr as strings
    output = Popen('obminimize -sd -c 1e-20 -n 100000 -ff {} {}'.format(ff,tmp_filename).split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,bufsize=-1).communicate()[0]

    # Parse the optimized *.xyz geometry
    counter = 0
    for count_i,i in enumerate(output.split('\n')[2:]):
        fields = i.split()
        if len(fields) != 4: continue                    
        else: 
            geo[counter] = [ float(j) for j in fields[1:] ]
            counter += 1

    # Remove the tmp file that was read by obminimize
    try:
        os.remove(tmp_filename)
    except:
        pass
    return geo

# Description: This function instantiates all conformers involving adjustments of terminal centers (non-terminal atoms with
#              only one non-terminal connection, such as a methyl carbon). The function performs repeated calls to obminimize
#              (open babel geometry optimizer function) to minimize each conformer and returns the lowest energy geometry.
#
# Inputs:      geo:      Nx3 array of atomic coordinates
#              adj_mat:  NxN array of connections
#              elements: N list of element labels
#              ff:       force-field specification passed to obminimize (uff, gaff)
#
# Returns:     geo:      Nx3 array of optimized atomic coordinates
# 
def opt_terminal_centers(geo,adj_mat,elements,ff='uff'):

    # Generate lists of rotation centers, the terminal and non-terminal atoms connected to each center and the rotation vectors for each center
    centers = terminal_centers(adj_mat)
    terminal_cons = [ [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and sum(adj_mat[count_j]) == 1 ] for i in centers ]
    nonterminal_cons = [ next( count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 and sum(adj_mat[count_j]) != 1 ) for i in centers ]
    rot_vecs = [ geo[nonterminal_cons[count_i]]-geo[i] for count_i,i in enumerate(centers) ]

    # Print a warning if there are a large number of centers
    if len(centers) > 5:
        print("WARNING in opt_terminal_centers: Iterating through all terminal center conformers. There are {} centers, so this might take a while.".format(len(centers))) 

    # Collect the rotation operations
    ops = []    
    for i in terminal_cons:

        if len(i) == 1:
            ops += [[ 0.0 ]]
        elif len(i) == 2 and False in [elements[i[0]] == elements[j] for j in i ]:
            ops += [[ j*90.0 for j in range(4) ]]
        elif len(i) == 2:
            ops += [[ j*90.0 for j in range(2) ]]
        elif len(i) == 3 and False in [elements[i[0]] == elements[j] for j in i ]:
            ops += [[ j*60.0 for j in range(6) ]]
        elif len(i) == 3:
            ops += [[ j*60.0 for j in range(2) ]]

    # Loop over all combinations of rotation operations and retain the lowest energy conformer
    E_min   = None
    geo_min = None    
    min_num = None
    for count_i,i in enumerate(combos(ops)):
        new_geo = deepcopy(geo)
        for count_j,j in enumerate(i):            
            for k in terminal_cons[count_j]:
                new_geo[k] = axis_rot(new_geo[k],rot_vecs[count_j],v2=new_geo[centers[count_j]],theta=j)

        # UNCOMMENT TO KEEP ALL CONFORMERS: For debugging it can be useful to see the individual conformers. 
        # mol_write("appended.mol",elements,new_geo,adj_mat,append_opt=True)

        # Optimize the current geometry
        new_geo = opt_geo(new_geo,adj_mat,elements,ff='uff')        

        # Generate a temporary filename for opt_terminal_centers to use
        tmp_filename = '.tmp.mol'
        count = 0
        while os.path.isfile(tmp_filename):
            count += 1
            if count == 10:
                print("ERROR in opt_terminal_centers: could not find a suitable filename for the tmp geometry. Exiting...")
                return geo
            else:
                tmp_filename = ".tmp" + tmp_filename            

        # Calculate the energy of the optimized conformer
        # use os.devnull to suppress stderr from the obminimize call
        mol_write(tmp_filename,elements,new_geo,adj_mat,append_opt=True)

        # Popen(stdout=PIPE).communicate() returns stdout,stderr as strings
        output = Popen('obenergy -ff {} {}'.format(ff,tmp_filename).split(), stdin=PIPE, stdout=PIPE, stderr=PIPE,bufsize=-1).communicate()[0]
        
        # Grab the energy from the output
        Energy = next( float(j.split()[3]) for j in output.split('\n') if len(j.split()) > 4 and j.split()[0] == "TOTAL" and j.split()[1] == "ENERGY" and j.split()[2] == "=" )
        if E_min is None or Energy < E_min:
            E_min = Energy
            geo_min = deepcopy(new_geo)
            min_num = count_i
        os.remove(tmp_filename)

    # UNCOMMENT TO OPTIMIZE ALL CONFORMERS: For debugging it can be useful to see the individual optimized conformers. 
    # with open('appended.opt.xyz','w') as f:
    #     with open(os.devnull,'w') as devnull:
    #         call('obminimize -sd -c 1e-10 -ff uff appended.mol'.split(),stdout=f,stderr=devnull)

    return geo_min

# Description: Generates all unique combinations (independent of order) of objects in a list of lists (x)
#
# Inputs:        x: a list of lists
#        
# Returns:       an iterable list of objects that yields a unique combination of objects (one from each sublist in x)
#                from the x.
def combos(x):
    if len(x) == 0:
        yield []
    elif hasattr(x[0], '__iter__'):
        for i in x[0]:
            for j in combos(x[1:]):
                yield [i]+j
    else:
        for i in x:
            yield [i]

# Description: Calculates the dihedral angle (in radians) for
#              a quadruplet of atoms
#
# Inputs:      xyz: a 4x3 numpy array, where each row holds the 
#                   cartesian position of each atom and row indices
#                   determine placement in the dihedral ( i.e, which
#                   atoms correspond to 1-2-3-4 in the dihedral )
# 
# Returns:    angle: dihedral angle in radians
#
def dihedral_calc(xyz):
    
    # Calculate the 2-1 vector           
    v1 = (xyz[1]-xyz[0]) 
                                                             
    # Calculate the 3-2 vector           
    v2 = (xyz[2]-xyz[1]) 
                                                             
    # Calculate the 4-3 bond vector      
    v3 = (xyz[3]-xyz[2]) 

    # Calculate dihedral (funny arctan2 formula avoids the use of arccos and retains the sign of the angle)
    angle = arctan2( dot(v1,cross(v2,v3))*(dot(v2,v2))**(0.5) , dot(cross(v1,v2),cross(v2,v3)) )
    
    return angle

# Description: Relaxes the geometry using steepest descent and a heuristic set of LJ parameters between all pairs 
#              that are graphically separated by more than 3 bonds. 
def LJ_relax(geo,adj_mat,sigma=0.5,eps=0.01,max_steps=10,elements=None):

    # Initialize some basic variables for the optimization
    N_atoms = len(geo)
    max_disp = 0.1
    step_scale = 0.2

    # Calculate the pair-wise graphical separations between all atoms
    seps    = graph_seps(adj_mat)
    disps   = cdist(geo,geo)

    # Initialize matrices for bonding force evaluations
    # NOTE: the r0 values for the bonds are set to the lengths from the supplied geo 
    bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 ]
    bond_ind = where(adj_mat == 1)
    k_bond_mat                   = zeros([N_atoms,N_atoms])
    r0_bond_mat                  = zeros([N_atoms,N_atoms])
    k_bond_mat[bond_ind]         = 1.0
    r0_bond_mat[bond_ind]        = disps[bond_ind]                

    # Initialize matrices for LJ force evaluations
    eps_LJ_mat                   = zeros([N_atoms,N_atoms])
    sigma_LJ_mat                 = zeros([N_atoms,N_atoms])
    pair_ind                     = where(seps > 3 )
    eps_LJ_mat[pair_ind]         = eps 
    sigma_LJ_mat[pair_ind]       = sigma
    geo[30] = geo[30] - array([-20.,0,0])
    if elements is not None:
        f = open('test_relax.xyz','w')
        f.write("{:d}\n\n".format(N_atoms))
        for count_j,j in enumerate(geo):
            f.write("{:20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(elements[count_j],j[0],j[1],j[2]))

    # Perform relaxation until max_steps is reach or a break condition is met.
    F = zeros([N_atoms,3])
    for i in range(max_steps):

        # r_mat holds the vectorial displacements between atom i j at index i,j (i.e. r_mat[i][j] returns a vector pointing from j to i)
        x, y  = triu_indices(N_atoms, -N_atoms)
        r_mat = geo[x] - geo[y]
        r_mat = r_mat.reshape(N_atoms,N_atoms,3)
        r_mat[x,x] = [1.0,0.0,0.0]  # self terms are set to [1.0,0.0,0.0] to avoid a division by 0 zero. These terms are never used in the force evaluations so the relaxation is unaffected
        r_mat = r_mat / sqrt((r_mat ** 2).sum(-1))[..., newaxis] # Normalize 
        r_mat[isnan(r_mat)] = 0.0 

        # calculate gradient using simple harmonic connections between all atoms
        dist_2D = cdist(geo,geo)
        fill_diagonal(dist_2D,0.001)

        # Calculate forces (i.e. -gradient of the potential)
        F = - sum((k_bond_mat*(dist_2D-r0_bond_mat))[:,:,None]*r_mat,axis=1) 
        for count_j,j in enumerate(F):
            print("{} {}".format(count_j,j))
        print(" ")
        # Add LJ component for clashes
#        F = F - sum((24.0*(eps_LJ_mat/dist_2D)*(2.0*(sigma_LJ_mat/dist_2D)**(12.0) - (sigma_LJ_mat/dist_2D)**(6.0)))[:,:,None]*r_mat,axis=1)
        
        # Update geometry
        update = F*step_scale
        for count_j,j in enumerate(update):
            print("{} {}".format(count_j,j))
        print(" ")

#        update[where(abs(update) > max_disp)] = max_disp
        update = update - mean(update,axis=0)
        for count_j,j in enumerate(update):
            print("{} {}".format(count_j,j))
        print(" ")


        geo = geo+update
        if elements is not None and i % 1 == 0:                
            f.write("{:d}\n\n".format(N_atoms))
            for count_j,j in enumerate(geo):
                f.write("{:20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(elements[count_j],j[0],j[1],j[2]))

        print("F/atom: {}".format(sum(norm(F,axis=1))/float(N_atoms)))

    f.close
    return geo

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
def axis_rot(Point,v1,v2=[0.0,0.0,0.0],theta=0.0,mode='angle'):

    # Temporary variable for performing the transformation
    rotated=array([Point[0],Point[1],Point[2]])

    # If mode is set to 'angle' then theta needs to be converted to radians to be compatible with the
    # definition of the rotation vectors
    if mode == 'angle':
        theta = theta*pi/180.0

    # Rotation carried out using formulae defined here (11/22/13) http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/)
    # Adapted for the assumption that v1 is the direction vector and v2 is a point that v1 passes through
    a = v2[0]
    b = v2[1]
    c = v2[2]
    u = v1[0]
    v = v1[1]
    w = v1[2]
    L = u**2 + v**2 + w**2

    # Rotate Point
    x=rotated[0]
    y=rotated[1]
    z=rotated[2]

    # x-transformation
    rotated[0] = ( a * ( v**2 + w**2 ) - u*(b*v + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - cos(theta) ) + L*x*cos(theta) + L**(0.5)*( -c*v + b*w - w*y + v*z )*sin(theta)

    # y-transformation
    rotated[1] = ( b * ( u**2 + w**2 ) - v*(a*u + c*w - u*x - v*y - w*z) )\
             * ( 1.0 - cos(theta) ) + L*y*cos(theta) + L**(0.5)*(  c*u - a*w + w*x - u*z )*sin(theta)

    # z-transformation
    rotated[2] = ( c * ( u**2 + v**2 ) - w*(a*u + b*v - u*x - v*y - w*z) )\
             * ( 1.0 - cos(theta) ) + L*z*cos(theta) + L**(0.5)*( -b*u + a*v - v*x + u*y )*sin(theta)

    rotated = rotated/L
    return rotated

if __name__ == "__main__": 
    main(sys.argv[1:])
    
