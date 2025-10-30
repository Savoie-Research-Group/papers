#!/bin/env python
# Author: Brett Savoie (brettsavoie@gmail.com)

import sys
from file_parsers import *
from adjacency import *
from numpy import *
from itertools import combinations
from time import time
def main(argv):

    # Extract Element list and Coord list from the file
    elements,geo = xyz_parse(argv[0])
    adj_mat = Table_generator(elements,geo)    
    hybridizations = Hybridization_finder(elements,adj_mat,File=argv[0])
    adj_list = [ [ count_j for count_j,j in enumerate(adj_mat[i]) if j == 1 ] for i in range(len(adj_mat)) ]
    atom_types = id_types(elements,adj_list,gens=2,algorithm='list')
    bond_mat = find_lewis(atom_types,adj_mat, q_tot=0, b_mat_only=True,verbose=False)
    lone_electrons,bonding_electrons,core_electrons = check_lewis(atom_types,adj_mat,q_tot=0)

    # Print out molecular diagnostics
    print("idx {:20s} {:20s} {:20s}".format("Element","hybridization","atomtype"))
    for count_i,i in enumerate(atom_types):
        print("{:<2d}: {:20s} {:20s} {:40s}".format(count_i,elements[count_i],hybridizations[count_i],i))
    print("\nadj_mat:")
    for i in adj_mat:
        print(i)
    print("\nbond_mat(s):")
    for i in bond_mat:
        print(i)
        print(" ")
    print("\nlewis structure:")
    print("idx {:20s} {:20s} {:20s} {:20s}".format("Element","lone_electrons","bonding_electrons","core_electrons"))
    for count_i,i in enumerate(elements):
        print("{:<2d}: {:<20s} {:<20.2f} {:<20.2f} {:<20.2f}".format(count_i,i,lone_electrons[count_i],bonding_electrons[count_i],core_electrons[count_i]))
    return

# identifies the taffi atom types from an adjacency matrix/list (A) and element identify. 
def id_types(elements,A,gens=2,which_ind=None,avoid=[],geo=None,hybridizations=[],algorithm="matrix"):

    # On first call initialize dictionaries
    if not hasattr(id_types, "mass_dict"):

        # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
        # NOTE: It's inefficient to reinitialize this dictionary every time this function is called
        id_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                             'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                              'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                             'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                             'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                             'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                             'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                             'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    if algorithm == "matrix":

        # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
        masses = [ id_types.mass_dict[i] for i in elements ]
        atom_types = [ "["+taffi_type(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

        # Add ring atom designation for atom types that belong are intrinsic to rings 
        # (depdends on the value of gens)
        for count_i,i in enumerate(atom_types):
            if ring_atom_new(A,count_i,ring_size=(gens+2)) == True:
                atom_types[count_i] = "R" + atom_types[count_i]            

    elif algorithm == "list":

        # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
        masses = [ id_types.mass_dict[i] for i in elements ]
        atom_types = [ "["+taffi_type_list(i,elements,A,masses,gens)+"]" for i in range(len(elements)) ]

        # Add ring atom designation for atom types that belong are intrinsic to rings 
        # (depdends on the value of gens)
        for count_i,i in enumerate(atom_types):
            if ring_atom_list(A,count_i,ring_size=(gens+2)) == True:
                atom_types[count_i] = "R" + atom_types[count_i]            

    return atom_types

# adjacency matrix based algorithm for identifying the taffi atom type
def taffi_type(ind,elements,adj_mat,masses,gens=2,avoid=[]):

    # On first call initialize dictionaries
    if not hasattr(taffi_type, "periodic"):

        # Initialize periodic table
        taffi_type.periodic = { "h": 1,  "he": 2,\
                               "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                               "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                               "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                               "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Find connections, avoid is used to avoid backtracking
    cons = [ count_i for count_i,i in enumerate(adj_mat[ind]) if i == 1 and count_i not in avoid ]

    # Sort the connections based on the hash function 
    if len(cons) > 0:
        cons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in cons ])[::-1]))[1]

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        subs = []
    else:
        subs = [ taffi_type(i,elements,adj_mat,masses,gens=gens-1,avoid=[ind]) for i in cons ]

    return "{}".format(taffi_type.periodic[elements[ind].lower()]) + "".join([ "["+i+"]" for i in subs ])

# adjacency_list based algorithm for identifying the taffi atom type
def taffi_type_list(ind,elements,adj_list,masses,gens=2,avoid=[]):

    # On first call initialize dictionaries
    if not hasattr(taffi_type_list, "periodic"):

        # Initialize periodic table
        taffi_type_list.periodic = { "h": 1,  "he": 2,\
                                    "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                                    "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                     "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                                    "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                                    "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Find connections, avoid is used to avoid backtracking
    cons = [ i for i in adj_list[ind] if i  not in avoid ]

    # Sort the connections based on the hash function 
    if len(cons) > 0:
        cons = list(zip(*sorted([ (atom_hash_list(i,adj_list,masses,gens=gens-1),i) for i in cons ])[::-1]))[1]

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        subs = []
    else:
        subs = [ taffi_type_list(i,elements,adj_list,masses,gens=gens-1,avoid=[ind]) for i in cons ]

    return "{}".format(taffi_type_list.periodic[elements[ind].lower()]) + "".join([ "["+i+"]" for i in subs ])

def ring_atom_new(adj_mat,idx,start=None,ring_size=10,counter=0,avoid=[]):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    
    # Loop over connections and recursively search for idx
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom_new(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid=[idx]) == True:
                return True
        return False

# Return true if idx is a ring atom
def ring_atom(adj_mat,idx,start=None,ring_size=10,counter=0,avoid_set=None):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    if avoid_set is None:
        avoid_set = set([])

    # Trick: The fact that the smallest possible ring has three nodes can be used to simplify
    #        the algorithm by including the origin in avoid_set until after the second step
    if counter >= 2 and start in avoid_set:
        avoid_set.remove(start)    
    elif counter <= 2 and start not in avoid_set:
        avoid_set.add(start)

    # Update the avoid_set with the current idx value
    avoid_set.add(idx)    

    # Loop over connections and recursively search for idx
    status = 0
    cons = [ count_i for count_i,i in enumerate(adj_mat[idx]) if i == 1 and count_i not in avoid_set ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom(adj_mat,i,start=start,ring_size=ring_size,counter=counter+1,avoid_set=avoid_set) == True:
                return True
        return False

# identifies ring atom types using an adjacency list based algorithm
def ring_atom_list(adj_list,idx,start=None,ring_size=10,counter=0,avoid=[]):

    # Consistency/Termination checks
    if ring_size < 3:
        print("ERROR in ring_atom: ring_size variable must be set to an integer greater than 2!")
    if counter == ring_size:
        return False

    # Automatically assign start to the supplied idx value. For recursive calls this is set manually
    if start is None:
        start = idx
    
    # Loop over connections and recursively search for idx
    cons = [ i for i in adj_list[idx] if i not in avoid ]
    if len(cons) == 0:
        return False
    elif start in cons:
        return True
    else:
        for i in cons:
            if ring_atom_list(adj_list,i,start=start,ring_size=ring_size,counter=counter+1,avoid=[idx]) == True:
                return True
        return False
    
# Returns the list of connected nodes to the start node, while avoiding any connections through nodes in the avoid list.
def return_connected(adj_mat,start=0,avoid=[]):

    # Initialize the avoid list with the starting index
    avoid = set(avoid+[start])

    # new_0 holds the most recently encountered nodes, beginning with start
    # new_1 is a set holding all of the encountered nodes
    new_0 = [start]
    new_1 = set([start])

    # keep looping until no new nodes are encountered
    while len(new_0) > 0:

        # reinitialize new_0 with new connections
        new_0 = [ count_j for i in new_0 for count_j,j in enumerate(adj_mat[i]) if j == 1 and count_j not in avoid ]

        # update the new_1 set and avoid list with the most recently encountered new nodes
        new_1.update(new_0)
        avoid.update(new_0)

    # return the set of encountered nodes
    return new_1

def rank_nested_lists(lol):    

    # matrix based sort algorithm. This is knowingly suboptimal, but it 
    # gives much more flexibility during the sort than standard algorithms. 
    order=zeros([len(lol),len(lol)])
    rank = [0]*len(lol)
    terminal_list = []
    for count_i,i in enumerate(lol):
        
        # If there were no connections in the current generation then comparisons have 
        # been exhausted and ranking should be set in this round.
        if len(flatten(i)) == 0:
            terminal_list += [count_i]
            order[count_i,:] = 0
            continue

        for count_j,j in enumerate(lol):

            if count_j > count_i:

                flag=0

                # iterate over sublists in i while performing
                # comparisons with corresponding sublists in j
                for count_k,k in enumerate(i):

                    if flag == 1:
                        break

                    # If there are no sublists left in j and difference has been found, i has priority
                    elif count_k > len(j)-1:
                        order[count_i,count_j]=1
                        order[count_j,count_i]=0
                        break

                    for count_m,m in enumerate(k):

                        # If the sublist in j has run out of values, i has priority
                        if count_m > len(j[count_k])-1:
                            order[count_i,count_j]=1
                            order[count_j,count_i]=0
                            flag = 1
                            break                       

                        # If the sublist element is an empty node (with branching empty nodes can be mixed with full nodes)
                        if i[count_k][count_m] == [] and j[count_k][count_m] != []:
                            order[count_i,count_j]=0
                            order[count_j,count_i]=1
                            flag = 1
                            break

                        # If the sublist element is an empty node (with branching empty nodes can be mixed with full nodes) (inverse)
                        if i[count_k][count_m] != [] and j[count_k][count_m] == []:
                            order[count_i,count_j]=0
                            order[count_j,count_i]=1
                            flag = 1
                            break

                        # If both sublist elements are empty nodes (with branching empty nodes can be mixed with full nodes)
                        if i[count_k][count_m] == [] and j[count_k][count_m] == []:
                            continue

                        # If there is a difference in the sublist elements assign priority
                        elif i[count_k][count_m][0] > j[count_k][count_m][0]:
                            order[count_i,count_j]=1
                            order[count_j,count_i]=0
                            flag = 1
                            break

                        # If there is a difference in the sublist elements assign priority (inverse)
                        elif i[count_k][count_m][0] < j[count_k][count_m][0]:
                            order[count_i,count_j]=0
                            order[count_j,count_i]=1
                            flag = 1
                            break

                        # If the sublist in i has run out of values and j still has attachments, j has priority
                        elif count_m == len(i[count_k])-1 and len(j[count_k]) > len(i[count_k]):
                            order[count_i,count_j]=0
                            order[count_j,count_i]=1
                            flag = 1
                            break
                            
                    # i has run out of sublists without finding a difference and j still has lists, j has priority
                    if count_k == len(i)-1 and len(j) > len(i) and flag == 0:
                        order[count_i,count_j]=0
                        order[count_j,count_i]=1                        
                        break

                    # if both i and j have run out of sublists without finding a difference then they are identical up to this point
                    elif count_k == len(i)-1 and len(i) == len(j) and flag == 0:
                        order[count_i,count_j]=1
                        order[count_j,count_i]=1
                        break
                    
    # Assign ranks based on entries in the order matrix
    for count_i,i in enumerate(order):
        rank[count_i]=int(len(order)-sum(i)-1)

    # Assign unique ranks for branches that are at their terminus
    # Reverse order shouldn't matter, but it guarrantees that the
    # first discovered gets higher priority
    for count_i,i in enumerate(terminal_list[::-1]):
        rank[i]=rank[i]-count_i

    return rank

def find_connections(tree,loc):
    if len(tree) > loc[0]+1:
        
        # Find connections
        connections = [ [i[2],[loc[0]+1,count_i]] for count_i,i in enumerate(tree[loc[0]+1]) if i[0] == tree[loc[0]][loc[1]][1] ]

        # If no connections are found then a 0-entry is returned
        if connections == []:
            connections = [[]]
        # If connections are found then proceed with sort
        else:
            connections.sort(key=lambda x:x[0],reverse=True)

    # If a next generation doesn't exist then a 0-entry is returned
    else:
        connections = [[]]

    return connections

def flatten(lol):
    flat = []
    for i in lol:
        if type(i) == type([]):
            flat += flatten(i)
        else:
            flat += [i]
    return flat
    
# lol: nested list of lists, where each element is a 3-element tuple
# This function returns a list of lists with the same structure as lol, except 
# that each tuple is replaced by its third element (atomic number in this case)
def strip_trees(lol):
    tmp = []
    for count_i,i in enumerate(lol):
        if type(i) == type([]):
            tmp += [strip_trees(i)]
        else:
            tmp += [i[2]]
    return tmp

# Description: iterate over the nodes in each generation and search the previous generation for matches (using find_loc function)
# insert all matches into the label as a list in the index following the previous generation node.
# Operates on the total tree structure (so, trees is a list of trees)
def gen_nested_trees(trees):
    nested_trees = [ [] for i in range(len(trees)) ]
    for count_i,i in enumerate(trees):

        # Count the number of occupied generations
        num_gens = len([ count_j for count_j,j in enumerate(i) if len(j)>0])

        # Intiialize the nested list with the first generation atom
        nest = [i[0][0]]

        # Loop over generations and unwind the topology
        for j in range(1,num_gens):

            # Reversed list keeps the ordering so that highest priority ends up at the front
            for k in i[j][::-1]:         
                loc = find_loc(k[0],nest)
                if len(loc) == 1:
                    nest.insert(loc[0]+1,[k])
                elif len(loc) == 2:
                    nest[loc[0]].insert(loc[1]+1,[k])
                elif len(loc) == 3:
                    nest[loc[0]][loc[1]].insert(loc[2]+1,[k])
                elif len(loc) == 4:
                    nest[loc[0]][loc[1]][loc[2]].insert(loc[3]+1,[k])
                elif len(loc) == 5:
                    nest[loc[0]][loc[1]][loc[2]][loc[3]].insert(loc[4]+1,[k])
                elif len(loc) == 6:
                    nest[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]].insert(loc[5]+1,[k])
                elif len(loc) == 7:
                    nest[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]][loc[5]].insert(loc[6]+1,[k])
                elif len(loc) == 8:
                    nest[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]][loc[5]][loc[6]].insert(loc[7]+1,[k])
                elif len(loc) == 9:
                    nest[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]][loc[5]][loc[6]][loc[7]].insert(loc[8]+1,[k])
                elif len(loc) == 10:
                    nest[loc[0]][loc[1]][loc[2]][loc[3]][loc[4]][loc[5]][loc[6]][loc[7]][loc[8]].insert(loc[9]+1,[k])
                else:
                    print("Nesting is too deep, ensure that the requested generations do not exceed 10")
                    quit()
        nested_trees[count_i] = nest

    return nested_trees

# Description: Recursive search to id location of the matching node in the list of lists.
#              Returns a list of indices corresponding to the position within the list of
#              lists of the matching element. (e.g., loc=[2,4,3] means lol[2][4][3] is the match)
#
# Inputs:      idx: scalar, sought match for the second element in the tuples held within the list of lists.
#              lol: list of lists (arbitrary nesting) whose lowest level elements are 3-tuples. 
#
# Returns:     loc: a list of indices specifying the location in lol of the matching tuple
def find_loc(idx,lol):
    loc = []                                   # Initialize list to hold the location indices
    for count_i,i in enumerate(lol):           # Iterate over the elements in the list
        if type(i) == type([]):                # If a list is encountered a recursion is initiated
            tmp = find_loc(idx,i)              # recursive search for a match (yields [] is no match is found)
            if tmp != []:                      # If the recursion yields a match then we have a winner
                loc = loc + [count_i] + tmp    # append the list index (count_i) and the loc list returned by the recursion
        elif idx == i[1]:                      # If a match is found in the element-wise search append its location
            loc += [count_i]      
    return loc

def find_EZ(trees,hybridizations,geo):
    
    EZ_labels = ['']*len(trees)
    for count_i,i in enumerate(trees):

        # Check if the anchor atom is a potential stereocenter
        # NOTE: in the current implementation only C,N,Si,and P are considered potential EZ centers)
        if hybridizations[count_i] != 'sp2' or i[0][0][2] not in [6,7,14,15]:
            continue

        # Find indices of potential EZ bonds
        idx = [ j[1] for j in i[1] if hybridizations[j[1]] == 'sp2' and j[2] in [6,7,14,15] ]

        # Iterate over all of the EZ bonds and determine stereochemistry
        for j in idx:

            # Find anchor atom's highest priority connection (excluding the sp2 bonded group)      
            ind_1 = [ k[1] for k in i[1] if k[1] != j ][0]
            
            # Find bonded atom's highest priority connection (excluding the sp2 bonded group)
            ind_4 = [ k[1] for k in trees[j][1] if k[1] != count_i ][0] 

            # Intialize the coordinate array of the relevant dihedral
            xyz = zeros([4,3])
            xyz[0] = geo[ind_1,:]
            xyz[1] = geo[count_i,:]
            xyz[2] = geo[j,:]
            xyz[3] = geo[ind_4,:]

            # Calculate dihedral using call to dihedral_calc
            angle = 180.0*dihedral_calc(xyz)/pi         # Calculate dihedral
            angle = int(180*round(abs(angle)/180.0))    # Round to 0 or 180

            # ID E/Z based on angle
            if angle == 0:
                EZ_labels[count_i] += "Z"
            elif angle == 180:
                EZ_labels[count_i] += "E"

    return EZ_labels

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

# Description: This functions uses some simple heuristics to determine each atom's
#              hybridization based on the elements types and adjacency matrix
# 
# Inputs:      Elements: a list of the atomic elements elements
#              Adj_mat:  the adjacency matrix indexed to the atoms
#              File:     an optional filename used when printing errors
def Hybridization_finder(Elements,Adj_mat,File=None,force_opt=False):

    Hybridizations = ['X']*len(Elements)
    fail_flag = 0
    
    for i in range(len(Hybridizations)):

        # Find connected atoms and coordination number
        Coord_ind = [ count_j for count_j,j in enumerate(Adj_mat[i]) if j == 1 ]
        Coord_number = len(Coord_ind)

        # Based upon element and coordination number determine hybridization
        if Elements[i] in ['B','Al','C','Si','Ge']:
            if Coord_number == 2:
                Hybridizations[i] = 'sp'
            elif Coord_number == 3:
                Hybridizations[i] = 'sp2'
            elif Coord_number == 4:
                Hybridizations[i] = 'sp3'

        elif Elements[i] in ['N','P','As']:
            if Coord_number == 1:
                Hybridizations[i] = 'sp'
            elif Coord_number == 2:
                Hybridizations[i] = 'sp2'
            elif Coord_number == 3:
                Hybridizations[i] = 'sp3'
            elif Coord_number == 4:
                Hybridizations[i] = 'sp3'

        elif Elements[i] in ['O','S','Se']:
            if Coord_number == 1:
                Hybridizations[i] = 'sp2'
            elif Coord_number == 2:
                Hybridizations[i] = 'sp3'
            elif Elements[i] in ['S', 'Se'] and Coord_number == 3:
                Hybridizations[i] = 'sp2'
            elif Elements[i] in ['S', 'Se'] and Coord_number == 4:
                Hybridizations[i] = 'sp3'

        elif Elements[i] in ['H','Li','Na','K','Rb','Cs','Fr','Be','Mg','Ca','Sr','Ba','Ra']:
            if Coord_number == 1:
                Hybridizations[i] = 's' 

        elif Elements[i] in ['F','Cl','Br','I']:
            if Coord_number == 1:
                Hybridizations[i] = 'sp3'

        if Hybridizations[i] == "X":
            if force_opt is True:
                if Coord_number == 2:
                    Hybridizations[i] = 'sp'
                elif Coord_number == 3:
                    Hybridizations[i] = 'sp2'
                elif Coord_number == 4:
                    Hybridizations[i] = 'sp3'
            else:
                if File is None:
                    print("ERROR in Hybridization_finder: hybridization of atom {} ({};{} connections) could not be determined! Exiting...".format(i,Elements[i],Coord_number))
                else:
                    print("ERROR in Hybridization_finder: parsing {}, hybridization of atom {} ({};{} connections) could not be determined! Exiting...".format(File,i,Elements[i],Coord_number))
                fail_flag = 1
                quit()

    return Hybridizations

# Description: This functions uses some simple heuristics to determine each atom's
#              hybridization based on the elements types and adjacency matrix
# 
# Inputs:      Elements: a list of the atomic elements elements
#              Adj_mat:  the adjacency matrix indexed to the atoms
#              File:     an optional filename used when printing errors
def Hybridization_of_type(type):

    # Find connected atoms and coordination number
    Coord_number = int(sum([ i for i in type_adjmat(type)[0][0,:] ]))
    fail_flag = 0
    atom = type.split('[')[1].split(']')[0]    
    hybridization = 'X'
    
    # Based upon element and coordination number determine hybridization
    if atom in ['5','13','6','14','32']:
        if Coord_number == 2:
            hybridization = 'sp'
        elif Coord_number == 3:
            hybridization = 'sp2'
        elif Coord_number == 4:
            hybridization = 'sp3'

    elif atom in ['7','15','33']:
        if Coord_number == 1:
            hybridization = 'sp'
        elif Coord_number == 2:
            hybridization = 'sp2'
        elif Coord_number == 3:
            hybridization = 'sp3'
        elif Coord_number == 4:
            hybridization = 'sp3'

    elif atom in ['8','16','34']:
        if Coord_number == 1:
            hybridization = 'sp2'
        elif Coord_number == 2:
            hybridization = 'sp3'
        elif atom in ['16', '34'] and Coord_number == 3:
            hybridization = 'sp2'

    elif atom in ['1','3','11','19','37','55','87','4','12','20','38','56','88']:
        if Coord_number == 1:
            hybridization = 's' 

    elif atom in ['9','17','35','53']:
        if Coord_number == 1:
            hybridization = 'sp3'

    if hybridization == "X":
        print("ERROR in Hybridization_finder: hybridization of atom {} ({};{} connections) could not be determined! Exiting...".format(i,atom,Coord_number))
        fail_flag = 1
        quit()

    return hybridization




# Description: This subprogram finds the chirality of bond-priority ordering of the 
# various branch points in the topology
# XXX INCOMPLETE FUNCTION? OLD FRAGMENT RETAINED FOR THE TIME BEING. TASK IS BETTER ACCOMPLISHED USING GRAPH SEP FUNCTION
def Find_centers(Elements,Adj_mat,Structure):

    # Dictionary of masses are needed to apply the CIP rules 
    Masses = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
            'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,'Ge':72.61,'As':74.92159,\
             'Se':78.96,'Br':79.904,'I':126.90447}

    # Find branches based on Structure variable
    Chiral_indices = [ count_i for count_i,i in enumerate(Structure) if i == 2 ]
    
    # Narrow down to 4-centered branch sites (remove sp2 carbon type branches)
    Chiral_indices = [ i for i in Chiral_indices if sum(Adj_mat[i] == 4) ]
        
    # The hybridization of each atom is needed to determine priority ordering
    # According to CIP rules, double bonded atoms are listed twice.
    Hybridizations = ['X']*len(Adj_mat)

    # At each step in the chirality finding algorithm it is necessary to write the 
    # Neighboring elements to variable. For convenience a list of len(Adj_mat) is 
    # used for that the elements can be stored to their
    Finished = [0]*len(Chiral_indices)
    
    for i in Chiral_indices:

        # Find connected atoms and coordination number
        Coord_ind = [ count_k for count_k,k in enumerate(Adj_mat[j]) if k == 1 ]
        Coord_number = len(Coord_ind)

        # The algorithm follows each of the four branches and records their
        # priority lists (i.e. first element, followed by coordinated elements
        # in order of priority). 
        Priority_list_1 = [Elements[Coord_ind[0]]]
        Priority_list_2 = [Elements[Coord_ind[1]]]
        Priority_list_3 = [Elements[Coord_ind[2]]]
        Priority_list_4 = [Elements[Coord_ind[3]]]

        for chain in range(4):

            

            while (completion_flag == 0):

                # Find the hybridization of the current atom                                                      
                # based upon element and coordination number
                if Elements[i] in ['C','Si','Ge']:
                    if Coord_number == 2:
                        Hybridizations[i] = 'sp'
                    elif Coord_number == 3:
                        Hybridizations[i] = 'sp2'
                    elif Coord_number == 4:
                        Hybridizations[i] = 'sp3'

                elif Elements[i] in ['N','P','As']:
                    if Coord_number == 1:
                        Hybridizations[i] = 'sp'
                    elif Coord_number == 2:
                        Hybridizations[i] = 'sp2'
                    elif Coord_number == 3:
                        Hybridizations[i] = 'sp3'

                elif Elements[i] in ['O','S','Se']:
                    if Coord_number == 1:
                        Hybridizations[i] = 'sp2'
                    elif Coord_number == 2:
                        Hybridizations[i] = 'sp3'
                    elif Coord_number == 3:
                        Hybridizations[i] = 'sp3'

                elif Elements[i] in ['H']:
                    if Coord_number == 1:
                        Hybridizations[i] = 's' 

                    else: 
                        print("ERROR: hybridization of {} atom could not be determined! Exiting...".format(j))


        
        neighbors = [ count_j for count_j,j in enumerate(Adj_mat[i]) if j == 1 ]
        
        # Arrange in order of priority
        neighbors =  neighbors[array([ Masses[Elements[j]] for j in neighbors ]).argsort()]

        # Find ties
        if Masses[Elements[neighbors[0]]] != Masses[Elements[neighbors[1]]]:
            Finished[0] = 1
        if Masses[Elements[neighbors[1]]] != Masses[Elements[neighbors[2]]]:
            Finished[1] = 1
        if Masses[Elements[neighbors[2]]] != Masses[Elements[neighbors[3]]]:
            Finished[2] = 1
        if Masses[Elements[neighbors[3]]] != Masses[Elements[neighbors[4]]]:
            Finished[3] = 1

    # XXX INCOMPLETE FUNCTION? OLD FRAGMENT RETAINED FOR THE TIME BEING
    return

# Description:   Checks is the supplied geometry corresponds to the minimal structure for the atomtype
# 
# Inputs:        atomtype:      The taffi atomtype being checked
#                geo:           Geometry of the molecule
#                elements:      elements, indexed to the geometry 
#                adj_mat:       adj_mat, indexed to the geometry (optional)
#                atomtypes:     atomtypes, indexed to the geometry (optional)
#                gens:          number of generations for determining atomtypes (optional, only used if atomtypes are not supplied)
# 
# Outputs:       Boolean:       True if geo is the minimal structure for the atomtype, False if not.
def minimal_structure(atomtype,geo,elements,adj_mat=None,atomtypes=None,gens=2):

    # If required find the atomtypes for the geometry
    if atomtypes is None or adj_mat is None:
        if len(elements) != len(geo):
            print("ERROR in minimal_structure: While trying to automatically assign atomtypes, the elements argument must have dimensions equal to geo. Exiting...")
            quit()

        # Generate the adjacency matrix
        # NOTE: the units are converted back angstroms
        adj_mat = Table_generator(elements,geo)

        # Generate the atomtypes
        atomtypes = id_types(elements,adj_mat,gens,Hybridization_finder(elements,adj_mat),geo)
        
    # Check minimal conditions
    for count_i,i in enumerate(atomtypes):

        # If the current atomtype matches the atomtype being searched for then proceed with minimal geo check
        if i == atomtype:

            # Initialize lists for holding indices in the structure within "gens" bonds of the seed atom (count_i)
            keep_list = [count_i]
            new_list  = [count_i]
            
            # Carry ount a "gens" bond deep search
            for j in range(gens):

                # Id atoms in the next generation
                tmp_new_list = []                
                for k in new_list:
                    tmp_new_list += [ count_m for count_m,m in enumerate(adj_mat[k]) if m == 1 and count_m not in keep_list ]

                # Update lists
                tmp_new_list = list(set(tmp_new_list))
                if len(tmp_new_list) > 0:
                    keep_list += tmp_new_list
                new_list = tmp_new_list
            
            # Check for the minimal condition
            keep_list = set(keep_list)
            if False in [ elements[j] == "H" for j in range(len(elements)) if j not in keep_list ]:
                minimal_flag = False
            else:
                minimal_flag = True
        
    return minimal_flag

if __name__ == "__main__": 
    main(sys.argv[1:])
    
