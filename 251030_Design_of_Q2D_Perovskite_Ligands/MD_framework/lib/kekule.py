#!/bin/env python                                                                                                                                                             
# Author: Brett Savoie (brettsavoie@gmail.com)

from numpy import *
from math import sqrt,sin,cos,tan,factorial,acos
from copy import deepcopy
from scipy import *
from scipy.spatial.distance import *
from numpy.linalg import *

from id_types import *
from file_parsers import *
from adjacency import *
from transify import *
    
import matplotlib    
matplotlib.use('Agg') # Needed for cluster image generation    
from pylab import *

def main(argv):

    # Set output name
    input_name = argv.pop(0)
    if "-o" in argv and len(argv) > argv.index('-o')+1:
        output = argv[argv.index('-o')+1]
    else:
        output = "2d_drawing.pdf"

    # Extract Element list and Coord list from the file
    elements,geo = xyz_parse(input_name)
    adj_mat = Table_generator(elements,geo)
    geo = transify(geo,adj_mat)
    hybridizations = Hybridization_finder(elements,adj_mat)
    atom_types = id_types(elements,adj_mat,2,hybridizations,geo)
    
    geo_2D = kekule(elements,atom_types,geo,adj_mat)

    # Rescale molecule to lie within 0 1
    min_x,max_x,min_y,max_y = (min(geo_2D[:,0]),max(geo_2D[:,0]),min(geo_2D[:,1]),max(geo_2D[:,1]))
    x_scale = 1.0/(max_x-min_x)
    y_scale = 1.0/(max_y-min_y)
    scale = min(x_scale,y_scale)
    coords = (geo_2D-array([min_x,min_y]))*scale    

    draw_scale = 1.0/(3.0*(max(coords[:,1])-min(coords[:,1])))   # The final drawing ends up in the 1:3 box, so the scaling of the lines/labels should match the eventual scaling of the drawing.
    if draw_scale > 1.0: draw_scale = 1.0
    draw_scale = 1.0
    fig = plt.figure()
    fig.figsize=(4, 4)
    f, ax  = plt.subplots(1,1)

    # Add bonds
    for count_i,i in enumerate(adj_mat):
        for count_j,j in enumerate(i):
            if count_j > count_i:
                if j == 1:
                    ax.add_line(matplotlib.lines.Line2D([coords[count_i][0],coords[count_j][0]],[coords[count_i][1],coords[count_j][1]],color=(0,0,0),linewidth=2.0*draw_scale))

    # Add atom labels
    for count_i,i in enumerate(coords):
        ax.text(i[0], i[1], elements[count_i], style='normal', fontweight='bold', fontsize=16*draw_scale, ha='center', va='center',bbox={'facecolor':'white', 'edgecolor':'None','pad':1.0*draw_scale})
    ax.axis('image')
    ax.axis('off')
    savefig(output, bbox_inches='tight',dpi=300)
    close(fig)

    return

# Heuristic based construction of the 2D structure from the adjacency matrix, followed by force-field based relaxation
def kekule(elements_0,atomtypes_0,geo_0,adj_mat_0,align='pc'):

    # Check for valid alignment option
    if align not in ["pc","longest"]:
        print("ERROR: kekule only accepts 'pc' (principle-component) and 'longest' (longest pair) as alignment options. Exiting...")
        quit()

    # Create deep copies of the input arrays to avoid modifying mutable lists
    elements  = deepcopy(elements_0)
    atomtypes = deepcopy(atomtypes_0)
    geo       = deepcopy(geo_0)
    adj_mat   = deepcopy(adj_mat_0)

    # Remove terminal atoms from rings (added back at the end after the ring structure is established via recursive call to kekule)
    ridx = [ count_i for count_i,i in enumerate(atomtypes) if "R" in i ]    
    del_ring_list = sorted(set([ count_j for i in ridx for count_j,j in enumerate(adj_mat[i]) if j == 1 and sum(adj_mat[count_j]) == 1 ]))
    keep_list = [ i for i in range(len(elements)) if i not in del_ring_list ]
    del_tuples = []
    if len(del_ring_list) > 0:
        del_tuples  = [ (i,elements[i],atomtypes[i],[ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 ]) for i in del_ring_list ]
        elements    = [ elements[i] for i in keep_list ]
        atomtypes   = [ atomtypes[i] for i in keep_list ]
        adj_mat     = adj_mat[keep_list,:][:,keep_list]
        geo         = geo[keep_list,:]

    # Add hydrogens to any undercoordinated non-carbon atoms (helps linearize the backbone)
    tmp_adj_mat = deepcopy(adj_mat)
    N0 = len(geo)
    for count_i,i in enumerate(adj_mat):
        if sum(i) > 1 and ( "R" not in atomtypes[count_i] ):
            for j in range(4-int(round(sum(i)))):
                new = zeros([len(tmp_adj_mat)+1,len(tmp_adj_mat)+1])
                new[:-1,:-1] = tmp_adj_mat
                new[count_i,-1] = 1
                new[-1,count_i] = 1
                tmp_adj_mat = new
                elements += ["H"]
                atomtypes += ["[1]"]
                if j == 0:
                    geo = vstack([geo,geo[count_i]+array([1.,0.,0.])])
                elif j == 1:
                    geo = vstack([geo,geo[count_i]+array([0.5,0.,0.])])
    N_dummy = len(geo) - N0
    adj_mat = tmp_adj_mat

    # Calculate the number of bonds separating each pair of atoms (topological distance). 
    Sep_Mat = zeros([len(geo),len(geo)])
    pairs   = triu_indices(len(Sep_Mat),k=1)
    pairs   = [ (i,pairs[1][count_i]) for count_i,i in enumerate(pairs[0]) ]
    tmp_am  = deepcopy(adj_mat)
    for i in range(len(elements)):               # The maximum distance between two atoms is N_atoms
        del_list = []                            # initialize empty list
        for count_j,j in enumerate(pairs):       # Iterate over the pairs that are unassigned
            if tmp_am[j[0],j[1]] != 0:           # if the adj_mat**(i) entry for the pair is non-zeros then assign the separation to (i+1)
                del_list += [count_j]            # Add to del_list so that the pair is treated as assigned in subsequent loops. 
                Sep_Mat[j[0],j[1]] = i+1         # Update the element(s) in Sep_Mat
                Sep_Mat[j[1],j[0]] = i+1         #
        pairs = [ j for count_j,j in enumerate(pairs) if count_j not in del_list ]        
        tmp_am = dot(tmp_am,adj_mat)

    # Find the pair with the largest separation (L_P), this is used to align the molecules. 
    L_max = max(Sep_Mat.flatten())
    L_P   = list(Sep_Mat.flatten()).index(L_max) 
    L_P   = (L_P/len(geo),L_P-(L_P/len(geo))*len(geo))

    # Starting with the first element in the L_P start constructing the geometry
    new    = [L_P[0]]
    placed = []
    locs   = { i:[] for i in range(len(geo)) } 

    # Build the kekule structure (without hydrogens on the ring atoms)
    for i in new:

        # Avoid redundant placement
        if i in placed: continue

        # Add sites
        geo,locs = add_locs(i,geo,adj_mat,atomtypes,locs,L_P,Sep_Mat)
        new += [ j[0] for j in locs[i] ]
        placed += [i]

    # Relax the geometry with a repulsive interaction enabled (k_rep) which accelerates topology formation
    # NOTE: This is only applied to structures with rings
    if len(del_ring_list) > 0:
        geo = relax_2D(geo,adj_mat,k_rep=1.0,break_cond=0.1)

    # Relax the geometry (1-2, 1-2-3, and LJ interactions; no repulsive interaction)
#    geo = relax_2D(geo,adj_mat,eps=0.05,sigma=0.6)   # This has caused problems for crowded geometries
    geo = relax_2D(geo,adj_mat,eps=0.05,sigma=0.0)

    # For molecules with rings, the terminal atoms that were removed are added back in (now that the ring topology has been established)
    # and the entire structure is reoptimized. 
    if len(del_ring_list) > 0:
        tmp = [ [] for i in list(locs.keys()) ]
        for i in del_tuples:
            N = len(adj_mat)+1
            tmp = zeros([N,N])
            not_i = array([ j for j in range(N) if j != i[0] ])
            tmp[not_i.reshape(-1,1), not_i] = adj_mat
            tmp[i[0],i[3]] = 1.0
            tmp[i[3],i[0]] = 1.0
            adj_mat = tmp
            tmp = zeros([N,3])
            tmp[not_i] = geo[:]
            geo = tmp
            elements.insert(i[0],i[1])
            atomtypes.insert(i[0],i[2])

        # (Re)populate the locs dictionary with the added atoms
        locs = { i:[] for i in range(len(adj_mat)) }
        for j in [ i for i in range(len(adj_mat)) if i not in del_ring_list ]:
            for m in [ count_k for count_k,k in enumerate(adj_mat[j]) if k == 1 and count_k not in del_ring_list ]:
                locs[j] += [(m,geo[m])]

        # (Re)calculate the number of bonds separating each pair of atoms (topological distance). 
        Sep_Mat = zeros([len(geo),len(geo)])
        pairs   = triu_indices(len(Sep_Mat),k=1)
        pairs   = [ (i,pairs[1][count_i]) for count_i,i in enumerate(pairs[0]) ]
        tmp_am  = deepcopy(adj_mat)
        for i in range(len(elements)):               # The maximum distance between two atoms is N_atoms
            del_list = []                            # initialize empty list
            for count_j,j in enumerate(pairs):       # Iterate over the pairs that are unassigned
                if tmp_am[j[0],j[1]] != 0:           # if the adj_mat**(i) entry for the pair is non-zeros then assign the separation to (i+1)
                    del_list += [count_j]            # Add to del_list so that the pair is treated as assigned in subsequent loops. 
                    Sep_Mat[j[0],j[1]] = i+1         # Update the element(s) in Sep_Mat
                    Sep_Mat[j[1],j[0]] = i+1         #
            pairs = [ j for count_j,j in enumerate(pairs) if count_j not in del_list ]        
            tmp_am = dot(tmp_am,adj_mat)

        # (Re)find the pair with the largest separation (L_P), this is used to align the molecules. 
        L_max = max(Sep_Mat.flatten())
        L_P   = list(Sep_Mat.flatten()).index(L_max) 
        L_P   = (L_P/len(geo),L_P-(L_P/len(geo))*len(geo))

        ring_atoms = sorted(set([ j for i in del_tuples for j in i[3] ]))
        for i in ring_atoms:

            # Add sites
            geo,locs = add_locs(i,geo,adj_mat,atomtypes,locs,L_P,Sep_Mat)

        # Final Relaxation
        geo = relax_2D(geo,adj_mat,k_angle=0.2,break_cond=0.001)

    # Reduce to 2D and center the molecule at the origin
    geo = geo[:,:2]                                             # Only retain the x-y components
    geo = geo - mean(geo,axis=0)                                # Center the molecule

    # Align the molecule based on the largest principle component
    if align == 'pc':
        e,v = eig(dot(geo.T,geo))                                   # Calculate the principle component
        idx = argsort(e)[::-1]                                      # Sort the eigenvals/vecs
        e = e[idx]                                                  #
        v = v[:,idx]                                                #
        long    = array([v[0,0],v[1,0],0.0])                        # principle component vector
        x_axis  = array([1.0,0.0,0.0])                              # alignment vector
        z_axis  = array([0.0,0.0,1.0])                              # rotation vector
        theta   = acos(dot(long,x_axis))*180.0/pi                   # angle of the rotation
        if dot(z_axis,cross(long,x_axis)) < 0:                      # check for the sign of the rotation via cross product with normal
            theta = theta * -1.0                                    # reverse the sign if necessary
        for count_i,i in enumerate(geo):                            # rotate the atoms
            old = deepcopy(geo[count_i])
            geo[count_i] = axis_rot(array([geo[count_i][0],geo[count_i][1],0.0]),z_axis,array([0.0,0.0,0.0]),theta,mode='angle')[:2]

    # Align the molecule based on the longest atomic pair
    elif align == 'longest':
        dist_2D = cdist(geo,geo)                                    
        max_ind = unravel_index(dist_2D.argmax(), dist_2D.shape)    # find index in dist_2D of the most separated pair of atoms
        long    = normalize(geo[max_ind[1]]-geo[max_ind[0]])        # define the normalized long axis 
        x_axis  = array([1.0,0.0,0.0])                              # alignment vector
        z_axis  = array([0.0,0.0,1.0])                              # rotation vector
        theta   = acos(dot(long,x_axis[:2]))*180.0/pi               # angle of the rotation
        if dot(z_axis,cross(long,x_axis)) < 0:                      # check for the sign of the rotation via cross product with normal
            theta = theta * -1.0                                    # reverse the sign if necessary
        for count_i,i in enumerate(geo):                            # rotate the atoms
            old = deepcopy(geo[count_i])
            geo[count_i] = axis_rot(array([geo[count_i][0],geo[count_i][1],0.0]),z_axis,array([geo[max_ind[0]][0],geo[max_ind[0]][1],0.0]),theta,mode='angle')[:2]   # Line for use with longest vector algirihtm of alignment

    # Return the kekule geometry less any dummy atoms
    if N_dummy > 0:
        return geo[:-N_dummy]
    else:
        return geo

# This function is called by kekule to "grow" molecules with structures that emulate kekule diagrams. 
# Kekule calls this program on a specific atom, this function then checks the geometry for which atoms are connected
# with the current atom (idx), and places the new atoms with consideration for the number of atoms still needing to be
# placed and the structural features of the molecule. 
def add_locs(idx,geo,adj_mat,atomtypes,locs,L_P,Sep_Mat,long=1.5,short=1.0):

    # Keep stock of which connections have already been placed
    old_conn = [ i[0] for i in locs[idx] ]
    new_conn = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in old_conn ]

    # Order the list based on increasing graphical distance to the terminal index.
    non_term_new = [ j[1] for j in sorted([ (Sep_Mat[i][L_P[1]],i) for i in new_conn if sum(adj_mat[i]) > 1 ])[::-1] ]
    term_new     = [ j[1] for j in sorted([ (Sep_Mat[i][L_P[1]],i) for i in new_conn if sum(adj_mat[i]) == 1 ])[::-1] ]

    # For ring structures, the non-terminal atoms should be grouped together, this is easily accomplished by putting them first in the list combination.
    if "R" in atomtypes[idx]:        
        ring_idx = [ i for i in non_term_new if "R" in atomtypes[i] ]
        if len([ i[0] for i in locs[idx] if "R" in atomtypes[i[0]] ]) == 0 and len(ring_idx) > 1:
            non_term_new = [ i for i in non_term_new if i not in ring_idx[1:] ]
            
        new_conn = non_term_new + term_new

    # Non non-ring structures, the following algorithm sorts the list so that the highest priority groups are added opposite of the already placed groups. 
    else:

        # list of variables
        N_conn       = float(sum(adj_mat[idx]))
        list_of_ind  = []
        N_new        = N_conn - len(old_conn)
        start        = int(round(N_conn/2.0+0.1)) - len(old_conn)
        posneg       = [ k for j in [ [i,-1*i] for i in range(int(N_new)+1) ] for k in j ][1:int(N_new)/2+2] 
        list_of_ind  = [ start+2*i for i in posneg ]

        # Place the highest priority atom in the "central slot", 
        # next place the first non-terminal atom alongside the high priority atom
        # At each subsequent step, the next highest priority atom is placed along with the next terminal atom, until all atoms have been placed.     
        for i in list_of_ind:

            if len(non_term_new) > 0 and len(new_conn) > i:
                new_conn[i] = non_term_new.pop(0)
            elif len(term_new) > 0 and len(new_conn) > i:
                new_conn[i] = term_new.pop(0)            

            if len(new_conn) <= (i - 1) or ( i - 1 ) < 0:
                continue
            elif len(term_new) > 0:
                new_conn[i-1] = term_new.pop(0)
            elif len(non_term_new) > 0:
                new_conn[i-1] = non_term_new.pop(0)

    # Iterate over the new connections
    new_locs = []
    N_conn = sum([ i for i in adj_mat[idx] if i == 1 ])
    wide_angle = 170.0
    for count_i,i in enumerate(new_conn):

        # Use the short length
        if sum(adj_mat[idx]) == 1 or sum(adj_mat[i]) == 1:
            length = short
        else:
            length = long
        
        # If this is the first site being placed then the site is generated by a displacment along x. 
        if len(old_conn) == 0 and count_i == 0:            
            geo[i] = geo[idx]+array([1.0,0.0,0.0])*length

        # If this isn't the first site placed then the new next site is placed relative to the last based on a rotation of the previous site. 
        else:

            # Get the location of the last connected atom in the group
            last = locs[idx][-1][1]

            # For ring atoms a special algorithm is used to generate wide angles between the non-terminal atoms
            if "R" in atomtypes[idx]:
                new = (last - geo[idx])/norm(last-geo[idx]) * length + geo[idx]

                # Get the direction for the new location by performing a wide rotation (ring-ring)
                if "R" in atomtypes[idx] and "R" in atomtypes[i] and "R" in atomtypes[locs[idx][-1][0]]:

                    if len(new_conn) >= 1:
                        if min([ norm(j[1]-axis_rot(new,array([0.0,0.0,1.0]),geo[idx],wide_angle,mode='angle')) for j in locs[idx] ]) > \
                           min([ norm(j[1]-axis_rot(new,array([0.0,0.0,1.0]),geo[idx],-wide_angle,mode='angle')) for j in locs[idx] ]) + 0.1:
                            geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],wide_angle,mode='angle')
                        else:
                            geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],-wide_angle,mode='angle')
                    else:
                        geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],wide_angle,mode='angle')

                # For non-terminal atoms that branch off the ring a long bond length is used as a safeguard that the ring has room to close before the bond shortens
                elif "R" in atomtypes[idx] and i in non_term_new and "R" in atomtypes[locs[idx][-1][0]]:
                    new = (last - geo[idx])/norm(last-geo[idx]) * length + geo[idx]   
                    geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],wide_angle/2.0,mode='angle')                    

                # Get the direction for the new location by performing a narrow rotation (nonring-ring/nonring-nonring)
                else:

                    # If the two ring atoms have already been placed, then place the non-terminal atoms in relation to their bisector.
                    if len(old_conn) >= 2:
                        bisector = (locs[idx][0][1] - geo[idx]) + (locs[idx][1][1] - geo[idx])
                        a = (locs[idx][0][1] - geo[idx])
                        a = a/norm(a)
                        b = (locs[idx][1][1] - geo[idx])
                        b = b/norm(b)
                        bisector = a+b
                        bisector = bisector/norm(bisector)
                        bisector = axis_rot(bisector,array([0.0,0.0,1.0]),array([0.0,0.0,0.0]),180.0+2.0*count_i,mode='angle')  # avoid overlapping atoms by making an additional small displacement based on the index number
                        geo[i] = geo[idx] + bisector*length

                    # For the last ring atom the sign of the rotation is reversed
                    elif len(new_conn) == 1:
                        geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],(360.0-wide_angle)/(N_conn-1.0),mode='angle')

                    # All other ring atoms
                    else:

                        if len(new_conn) >= 1:
                            if min([ norm(j[1]-axis_rot(new,array([0.0,0.0,1.0]),geo[idx], (360.0-wide_angle)/(N_conn-1.0),mode='angle')) for j in locs[idx] ]) > \
                               min([ norm(j[1]-axis_rot(new,array([0.0,0.0,1.0]),geo[idx],-(360.0-wide_angle)/(N_conn-1.0),mode='angle')) for j in locs[idx] ]):
                                geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],(360.0-wide_angle)/(N_conn-1.0),mode='angle')
                            else:
                                geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],-(360.0-wide_angle)/(N_conn-1.0),mode='angle')
                        else:
                            geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],(360.0-wide_angle)/(N_conn-1.0),mode='angle')

            # For non-ring atoms equil rotations are performed to obtain all sites
            else:
                # Get the direction for the new location
                if "R" in atomtypes[i]:
                    new = (last - geo[idx])/norm(last-geo[idx]) * length + geo[idx]
                else:
                    new = (last - geo[idx])/norm(last-geo[idx]) * length + geo[idx]
                geo[i] = axis_rot(new,array([0.0,0.0,1.0]),geo[idx],360.0/N_conn,mode='angle')

        locs[idx] += [(i,geo[i])]

    # Update the locs dictionary                
    for i in new_conn:
        if idx not in [ j[0] for j in locs[i] ]:
            locs[i] += [(idx,geo[idx])]
        for j in [ count_k for count_k,k in enumerate(adj_mat[i]) if k == 1 and count_k != idx ]:
            if i not in [ k[0] for k in locs[j] ]:
                locs[j] += [(i,geo[i])]

    return geo,locs

# 2D geometry relaxation via steepest descent
def relax_2D(geo,adj_mat,elements=[],r_short=1.0,r_long=1.5,k_bond=1.0,k_angle=0.1,eps=0.1,sigma=0.1,max_steps=10000,min_steps=100,step_size=0.1,max_disp=0.1,break_cond=0.01,deriv_cond=1.0E-7,k_rep=0.0,name='relax.xyz'):

    # Set the geometry to 2D if it isn't already
    init_dim = len(geo[0])
    if init_dim < 2: print("ERROR: the relax_2D function expects a geometry in at least two dimensions. Exiting..."); quit()
    geo = geo[:,:2]

    # Initialize useful arrays
    N_atoms = len(geo)
    dist_0 = cdist(geo,geo)    
    bond_ind = where(adj_mat == 1)
    nonbond_ind = where(adj_mat == 0)

    # find 1-3 and 1-4 interactions
    adj_mat_2 = dot(adj_mat,adj_mat)
    adj_mat_3 = dot(adj_mat_2,adj_mat)
    fill_diagonal(adj_mat_2,0)
    fill_diagonal(adj_mat_3,0)
    one_three_ind = where(adj_mat_2 > 0 )
    one_four_ind  = where(adj_mat_3 == 1 )

    # identify terminal atoms
    terminal_atoms = [ count_i for count_i,i in enumerate(adj_mat) if sum(i) == 1 ]
    terminal_ind   = [ (i,one_three_ind[1][count_i]) for count_i,i in enumerate(one_three_ind[0]) if i in terminal_atoms and one_three_ind[1][count_i] in terminal_atoms ]    
    terminal_ind   = (array([ i[0] for i in terminal_ind ],dtype=int),array([ i[1] for i in terminal_ind ],dtype=int))
    short_bonds    = (array([ i for count_i,i in enumerate(bond_ind[0]) if i in terminal_atoms or bond_ind[1][count_i] in terminal_atoms ],dtype=int),\
                      array([ i for count_i,i in enumerate(bond_ind[1]) if i in terminal_atoms or bond_ind[0][count_i] in terminal_atoms ],dtype=int)) 

    # Set bond lengths for connected atoms and short bonds for relaxing hydrogens
    dist_0[bond_ind] = r_long
    dist_0[short_bonds] = r_short

    # parse angles from the adjmat
    bonds = [ (count_i,count_j) for count_i,i in enumerate(adj_mat) for count_j,j in enumerate(i) if j == 1 ]
    angles = [ (i[0],i[1],count_j) for i in bonds for count_j,j in enumerate(adj_mat[i[1]]) if j == 1 and count_j != i[0] ]
    angles = [ i for count_i,i in enumerate(angles) if (i[2],i[1],i[0]) not in angles[count_i:] ] 
    term_angles = [ i for i in angles if i[0] in terminal_atoms and i[2] in terminal_atoms ]                                          # Angles involving terminal atoms
    nonterm_angles = [ i for i in angles if i[0] not in terminal_atoms and i[2] not in terminal_atoms ]                               # Angles not involving terminal atoms

    # initialize force constants for the various interactions
    k_bond_mat                   = zeros([N_atoms,N_atoms])
    k_bond_mat[bond_ind]         = k_bond
    k_eps                        = ones([N_atoms,N_atoms])*eps
    k_eps[bond_ind]              = 0.0
    k_eps[one_three_ind]         = 0.0
    k_eps[one_four_ind]          = 0.0
    fill_diagonal(k_eps,0.0)
    k_rep                        = ones([N_atoms,N_atoms])*k_rep
    k_rep[bond_ind]              = 0.0
    k_rep[one_three_ind]         = 0.0
    k_rep[one_four_ind]          = 0.0
    fill_diagonal(k_rep,0.0)
    x, y  = triu_indices(N_atoms, -N_atoms)
    old_F = 1E10
    step_scale = 0.1

    # Open xyz file
    if elements != []:
        with open(name,'w') as f:
            f.write("{:d}\n\n".format(N_atoms))
            for count_j,j in enumerate(geo):
                f.write("{:20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(elements[count_j],j[0],j[1],0.0))

            for i in range(max_steps):

                # Calculate forces from 1-3 interactions (k_angle/theta_ijk * tangential vector to the ij pointing away from k )
                F = zeros([N_atoms,2])
                if i > -1:
                    for count_j,j in enumerate(angles):
                        try:                            
                            theta = acos( dot(normalize(geo[j[0]]-geo[j[1]]),normalize(geo[j[2]]-geo[j[1]])) ) - pi 
                        except:
                            theta = 0.0
                        v0 = normalize(cross(geo[j[0]]-geo[j[1]],array([0.0,0.0,cross(geo[j[0]]-geo[j[1]],geo[j[2]]-geo[j[1]])])))[:2] 
                        v2 = normalize(cross(geo[j[1]]-geo[j[2]],array([0.0,0.0,cross(geo[j[0]]-geo[j[1]],geo[j[2]]-geo[j[1]])])))[:2]
                        F[j[0]] += -2.0*k_angle*theta*v0/(norm(geo[j[0]]-geo[j[1]]))
                        F[j[2]] += -2.0*k_angle*theta*v2/(norm(geo[j[2]]-geo[j[1]]))
                        F[j[1]] += 2.0*k_angle*theta*v0/(norm(geo[j[0]]-geo[j[1]])) + 2.0*k_angle*theta*v2/(norm(geo[j[2]]-geo[j[1]]))

                # calculate gradient using simple harmonic connections between all atoms
                dist_2D = cdist(geo,geo)
                fill_diagonal(dist_2D,0.001)

                # r_mat holds the vectorial displacements between atom i j at index i,j (i.e. r_mat[i][j] returns a vector pointing from j to i)
                x, y  = triu_indices(N_atoms, -N_atoms)
                r_mat = geo[x] - geo[y]
                r_mat = r_mat.reshape(N_atoms,N_atoms,2)
                r_mat[x,x] = [1.0,0.0]  # self terms are set to [1.0,0] to avoid a division by 0 zero. These terms are never used in the force evaluations so the relaxation is unaffected
                r_mat = r_mat / sqrt((r_mat ** 2).sum(-1))[..., newaxis] # Normalize 
                r_mat[isnan(r_mat)] = 0.0 

                # Calculate forces (i.e. -gradient of the potential)
                F = F - sum((k_bond_mat*(dist_2D-dist_0))[:,:,None]*r_mat,axis=1) 

                # Add LJ component for clashes
                if i > -1:
                    F = F - sum((24.0*(k_eps/dist_2D)*(2.0*(sigma/dist_2D)**(12.0) - (sigma/dist_2D)**(6.0)))[:,:,None]*r_mat,axis=1)

                # Add repulsive component
                if i > -1:
                    F = F + sum(((k_rep/dist_2D)**(2.0))[:,:,None]*r_mat,axis=1)

                # Update geometry
                update = F*step_scale
                update[where(update > max_disp)] = max_disp
                geo = geo+F*step_scale
                if i % 1 == 0:
                    f.write("{:d}\n\n".format(N_atoms))
                    for count_j,j in enumerate(geo):
                        f.write("{:20s} {:< 20.8f} {:< 20.8f} {:< 20.8f}\n".format(elements[count_j],j[0],j[1],0.0))

                # Calculate total forces
                new_F = sum(norm(F,axis=1))
                print("cycle {}: {}".format(i,new_F))

                # Check break condition
                if new_F < break_cond and i > min_steps:
                    print("Break condition met: {:6.4} < {:6.4g}".format(new_F,break_cond))
                    break
                elif abs(old_F - new_F) < deriv_cond and i > min_steps:
                    print("Derivative break condition met: {:6.4} < {:6.4g}".format(abs(old_F-new_F),deriv_cond))
                    break
                elif isnan(new_F) == True:
                    print("Unphysical behavior...Breaking...")
                    break
                old_F = new_F

    # Without writing a relaxation file
    else:
        for i in range(max_steps):

            # Calculate forces from 1-3 interactions (k_angle/theta_ijk * tangential vector to the ij pointing away from k )
            F = zeros([N_atoms,2])
            if i > -1:
                for count_j,j in enumerate(angles):
                    try:
                        theta = acos( dot(normalize(geo[j[0]]-geo[j[1]]),normalize(geo[j[2]]-geo[j[1]])) ) - pi 
                    except:
                        theta = 0.0                    
                    v0 = normalize(cross(geo[j[0]]-geo[j[1]],array([0.0,0.0,cross(geo[j[0]]-geo[j[1]],geo[j[2]]-geo[j[1]])])))[:2] 
                    v2 = normalize(cross(geo[j[1]]-geo[j[2]],array([0.0,0.0,cross(geo[j[0]]-geo[j[1]],geo[j[2]]-geo[j[1]])])))[:2]
                    F[j[0]] += -2.0*k_angle*theta*v0/(norm(geo[j[0]]-geo[j[1]]))
                    F[j[2]] += -2.0*k_angle*theta*v2/(norm(geo[j[2]]-geo[j[1]]))
                    F[j[1]] += 2.0*k_angle*theta*v0/(norm(geo[j[0]]-geo[j[1]])) + 2.0*k_angle*theta*v2/(norm(geo[j[2]]-geo[j[1]]))

            # calculate gradient using simple harmonic connections between all atoms
            dist_2D = cdist(geo,geo)
            fill_diagonal(dist_2D,0.001)

            # r_mat holds the vectorial displacements between atom i j at index i,j (i.e. r_mat[i][j] returns a vector pointing from j to i)
            x, y  = triu_indices(N_atoms, -N_atoms)
            r_mat = geo[x] - geo[y]
            r_mat = r_mat.reshape(N_atoms,N_atoms,2)
            r_mat[x,x] = [1.0,0.0]  # self terms are set to [1.0,0] to avoid a division by 0 zero. These terms are never used in the force evaluations so the relaxation is unaffected
            r_mat = r_mat / sqrt((r_mat ** 2).sum(-1))[..., newaxis] # Normalize 
            r_mat[isnan(r_mat)] = 0.0 

            # Calculate forces (i.e. -gradient of the potential)
            F = F - sum((k_bond_mat*(dist_2D-dist_0))[:,:,None]*r_mat,axis=1) 

            # Add LJ component for clashes
            if i > -1:
                F = F - sum((24.0*(k_eps/dist_2D)*(2.0*(sigma/dist_2D)**(12.0) - (sigma/dist_2D)**(6.0)))[:,:,None]*r_mat,axis=1)

            # Add repulsive component
            if i > -1:
                F = F + sum(((k_rep/dist_2D)**(2.0))[:,:,None]*r_mat,axis=1)

            # Update geometry
            update = F*step_scale
            update[where(update > max_disp)] = max_disp
            geo = geo+F*step_scale

            # Calculate total forces
            new_F = sum(norm(F,axis=1))

            # Check break condition
            if new_F < break_cond and i > min_steps:                
                break
            elif abs(old_F - new_F) < deriv_cond and i > min_steps:
                break
            elif isnan(new_F) == True:
                break
            old_F = new_F

    while len(geo[0]) < init_dim:
        geo = hstack([geo,zeros(len(geo)).reshape(len(geo[:,0]),1)])
    return geo

def Structure_finder(Adj_mat):

    # The Structure variable holds the highest structure factor for each atom
    # (atoms might be part of several rings, only the largest is documented)
    # Values correspond to the following structural features:
    # 0: terminal (e.g. hydrogens)
    # 1: chain (e.g. methylene)
    # 2: branch point (e.g. Carbon attached to 3 or more other carbons; -2 indicates a possible chiral center based on coordination)
    # 3: 3-membered ring
    # 4: 4-membered ring
    # 5: 5-membered ring
    # 6: 6-membered ring
    # 7: 7-membered ring
    # 8: 8-membered ring
    Structure = array([-1]*len(Adj_mat))
    # Remove terminal sites (sites with only a single length 2
    # self walk). Continue until all of the terminal structure 
    # has been removed from the topology. Avoid deleting head and tail
    Adj_trimmed = copy(Adj_mat)
    ind_trim = [ count for count,i in enumerate(diag(dot(Adj_trimmed,Adj_trimmed))) if i == 1 and count not in [0,len(Adj_trimmed)-1] ]
    Structure[ind_trim]=0

    # Remove terminal sites
    Adj_trimmed[:,ind_trim] = 0
    Adj_trimmed[ind_trim,:] = 0
    ind_trim = []

    # Find branch points (at this point all hydrogens have been removed, all remaining atoms with
    # over two connected neighbors are at least branches). The first and last atoms are treated separately
    # since an accounting must be made for the fact that as polymerization sites they have an implict bond
    branch_ind = [ count for count,i in enumerate(diag(dot(Adj_trimmed,Adj_trimmed))) if i > 2 and count not in [0,len(Adj_trimmed)-1] ]

    # Check if first and last atoms are branch points
    if diag(dot(Adj_trimmed,Adj_trimmed))[0]+1 > 2:
        branch_ind += [0]
    if diag(dot(Adj_trimmed,Adj_trimmed))[len(Adj_trimmed)-1]+1 > 2:
        branch_ind += [len(Adj_trimmed)-1]

    while( len(ind_trim) > 0 ):

        # Remove remaining terminal sites to reveal cyclic structures (This time, the head and tail can be removed
        ind_trim = ind_trim + [ count for count,i in enumerate(diag(dot(Adj_trimmed,Adj_trimmed))) if i == 1 ]
        Structure[ind_trim] = 1

        # Remove terminal sites
        Adj_trimmed[:,ind_trim] = 0
        Adj_trimmed[ind_trim,:] = 0
        ind_trim = []

    # Label branches. This has to be done here otherwise it would get overwritten during the while loop
    Structure[branch_ind] = 2

    # Label possible chiral centers (narrow down to 4-centered branch sites (remove sp2 carbon type branches))
    Chiral_ind = [ i for i in branch_ind if sum(Adj_mat[i]) == 4 ]
    Structure[Chiral_ind] = -2

    # Find non-repeating looping walks of various lengths to identify rings
    # Algorithm: Non-repeating walks are conducted over the trimmed adjacency matrix to 
    tmp=zeros([len(Adj_trimmed),len(Adj_trimmed)])
    for i in range(len(Adj_trimmed)):

        # Instantiate generation lists. These hold tupels where the first entry is the connected
        # vertex and the second site is the previous vertex) 
        Gen_1 = []
        Gen_2 = []
        Gen_3 = []
        Gen_4 = []
        Gen_5 = []
        Gen_6 = []
        Gen_7 = []
        Gen_8 = []
        Gen_9 = []
        Gen_10 = []

        # Find 1st generation connections to current atom. (connection site, previous site)
        Gen_1 = [ (count_z,i) for count_z,z in enumerate(Adj_trimmed[i,:]) if z == 1 ]

        # Loop over the 1st generation connections and find the connected atoms. Avoid back hops using the previous site information 
        for j in Gen_1:
            Gen_2 = Gen_2 +  [ (count_z,j[0],j[1]) for count_z,z in enumerate(Adj_trimmed[j[0],:]) if (z == 1 and count_z not in j[0:-1]) ]
            
        # Loop over the 2nd generation connections and find the connected atoms. Avoid back hops using the previous site information
        for k in Gen_2:
            Gen_3 = Gen_3 + [ (count_z,k[0],k[1],k[2]) for count_z,z in enumerate(Adj_trimmed[k[0],:]) if (z == 1 and count_z not in k[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_k for count_k,k in enumerate(Gen_3) if k[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=3
            Gen_3 = [ z for count_z,z in enumerate(Gen_3) if count_z not in del_ind ]

        # Loop over the 3rd generation connections and find the connected atoms. Avoid back hops using the previous site information        
        for l in Gen_3:
            Gen_4 = Gen_4 + [ (count_z,l[0],l[1],l[2],l[3]) for count_z,z in enumerate(Adj_trimmed[l[0],:]) if (z == 1 and count_z not in l[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_l for count_l,l in enumerate(Gen_4) if l[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=4
            Gen_4 = [ z for count_z,z in enumerate(Gen_4) if count_z not in del_ind ]

        # Loop over the 4th generation connections and find the connected atoms. Avoid back hops using the previous site information
        for m in Gen_4:
            Gen_5 = Gen_5 + [ (count_z,m[0],m[1],m[2],m[3],m[4]) for count_z,z in enumerate(Adj_trimmed[m[0],:]) if (z == 1 and count_z not in m[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_m for count_m,m in enumerate(Gen_5) if m[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=5
            Gen_5 = [ z for count_z,z in enumerate(Gen_5) if count_z not in del_ind ]

        # Loop over the 5th generation connections and find the connected atoms. Avoid back hops using the previous site information
        for n in Gen_5:
            Gen_6 = Gen_6 + [ (count_z,n[0],n[1],n[2],n[3],n[4],n[5]) for count_z,z in enumerate(Adj_trimmed[n[0],:]) if (z == 1 and count_z not in  n[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_n for count_n,n in enumerate(Gen_6) if n[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=6
            Gen_6 = [ z for count_z,z in enumerate(Gen_6) if count_z not in del_ind ]

        # Loop over the 6th generation connections and find the connected atoms. Avoid back hops using the previous site information
        for o in Gen_6:
            Gen_7 = Gen_7 + [ (count_z,o[0],o[1],o[2],o[3],o[4],o[5],o[6]) for count_z,z in enumerate(Adj_trimmed[o[0],:]) if (z == 1 and count_z not in o[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_o for count_o,o in enumerate(Gen_7) if o[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=7
            Gen_7 = [ z for count_z,z in enumerate(Gen_7) if count_z not in del_ind ]

        # Loop over the 7th generation connections and find the connected atoms. Avoid back hops using the previous site information
        for p in Gen_7:
            Gen_8 = Gen_8 + [ (count_z,p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7]) for count_z,z in enumerate(Adj_trimmed[p[0],:]) if (z == 1 and count_z not in p[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_p for count_p,p in enumerate(Gen_8) if p[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=8
            Gen_8 = [ z for count_z,z in enumerate(Gen_8) if count_z not in del_ind ]

        # Loop over the 8th generation connections and find the connected atoms. Avoid back hops using the previous site information
        for q in Gen_8:
            Gen_9 = Gen_9 + [ (count_z,q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8]) for count_z,z in enumerate(Adj_trimmed[q[0],:]) if (z == 1 and count_z not in q[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_q for count_q,q in enumerate(Gen_9) if q[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=9
            Gen_9 = [ z for count_z,z in enumerate(Gen_9) if count_z not in del_ind ]

        # Loop over the 9th generation connections and find the connected atoms. Avoid back hops using the previous site information
        for r in Gen_9:
            Gen_10 = Gen_10 + [ (count_z,r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8],r[9]) for count_z,z in enumerate(Adj_trimmed[r[0],:]) if (z == 1 and count_z not in r[0:-1]) ]
        # Find complete loops, store structure factor, and remove looping sequences from Gen_3 (avoids certain fallacious loops)
        del_ind = [ count_r for count_r,r in enumerate(Gen_10) if r[0] == i ]
        if len(del_ind) > 0:
            Structure[i]=10
            Gen_10 = [ z for count_z,z in enumerate(Gen_10) if count_z not in del_ind ]

    # Any remaining atoms with unassigned structure must be chain atoms connecting cyclic units. 
    Structure[Structure==-1] = 1 

    return Structure.tolist()

# Description:
# Rotate Point by an angle, theta, about the vector with an orientation of v1 passing through v2. 
# Performs counter-clockwise rotations (i.e., if the direction vector were pointing
# at the spectator, the rotations would appear counter-clockwise)
# For example, a 90 degree rotation of a 0,0,1 about the canonical 
# y-axis results in 1,0,0.
#
# Point: 1x3 array, coordinates to be rotated
# v1: 1x3 array, point the rotation passes through
# v2: 1x3 array, rotation direction vector
# theta: scalar, magnitude of the rotation (defined by default in degrees)
def axis_rot(Point,v1,v2,theta,mode='angle'):

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

# Simple normalization function with a safety for avoiding small normalizations (norm(x)~0)
def normalize(a):                                                           
    if norm(a) > 0.0001:
        return a/norm(a)
    else:
        return a

if __name__ == "__main__": 
    main(sys.argv[1:])
