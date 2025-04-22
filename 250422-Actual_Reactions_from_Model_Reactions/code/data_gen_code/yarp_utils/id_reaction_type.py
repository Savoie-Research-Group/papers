import sys,os
import numpy as np
from copy import deepcopy
from itertools import combinations,chain
from math import sqrt,sin,cos,tan,factorial,acos,asin


# Load modules in same folder   
this_script_dir = os.path.dirname(os.path.abspath(__file__))     
sys.path.append(this_script_dir)
from taffi_functions import *
from utility import *

def main(argv):

    # Extract Element list and Coord list from the file
    canonical = False   # apply canonical or not
    E,RG,PG = parse_input(argv[0])
    adj_mat_1 = Table_generator(E,RG)    
    adj_mat_2 = Table_generator(E,PG)    
    
    # apply find lewis
    lone_1,_,_,bond_mat_1,fc_1 = find_lewis(E,adj_mat_1,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)
    lone_2,_,_,bond_mat_2,fc_2 = find_lewis(E,adj_mat_2,q_tot=0,keep_lone=[],return_pref=False,return_FC=True)

    # locate radical positions
    keep_lone_1  = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_1]
    keep_lone_2  = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_2]

    # contruct BE matrix
    BE_1   = np.diag(lone_1[0])+bond_mat_1[0]

    # loop over possible BE matrix of product 
    diff_list = []
    for ind in range(len(bond_mat_2)):
        BE_2   = np.diag(lone_2[ind])+bond_mat_2[ind]
        BE_change = BE_2 - BE_1
        diff_list += [np.abs(BE_change).sum()]

    # determine the BE matrix leads to the smallest change
    ind = diff_list.index(min(diff_list))
    BE_2   = np.diag(lone_2[ind])+bond_mat_2[ind]
    BE_change = BE_2 - BE_1
    
    # determine bonds break and bonds form from Reaction matrix
    bond_break = []
    bond_form  = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] == -1:
                bond_break += [(i,j)]
            if BE_change[i][j] == 1:
                bond_form += [(i,j)]

    # first generate model reaction 
    MR = gen_model_reaction(E,RG,adj_mat_1,bond_mat_1,bond_break,bond_form,fc=fc_1[0],keep_lone=keep_lone_1[0],gens=1,canonical=canonical)
    reaction_types,seq,bond_dis_all = return_reaction_types(MR['E'],MR['R_geo'],MR['P_geo'],Radj_mat=MR['R_adj'],Padj_mat=MR['P_adj'])
    xyz_write('MR.xyz', MR['E'],MR['R_geo'])
    xyz_write('MR.xyz', MR['E'],MR['P_geo'],append_opt=True)
    # then identify reaction type from the model reaction
    print("The bond distance in the reactant and product are {}/{}, respectively".format(bond_dis_all[0],bond_dis_all[1]))
    print("The reaction type for the given reaction is : {}".format(reaction_types))

    return

# identifies the taffi atom types from an adjacency matrix/list (A) and element identify. 
def id_atom_types(elements,adj_mat,bond_mat,gens=2,avoid=[],fc=None,keep_lone=None,return_index=False,algorithm="matrix"):

    # On first call initialize dictionaries
    if not hasattr(id_atom_types, "mass_dict"):

        # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
        # NOTE: It's inefficient to reinitialize this dictionary every time this function is called
        id_atom_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                             'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                              'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                             'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                             'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                             'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                             'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                             'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

        id_atom_types.e_dict = {1:'H',2:'He',3:'Li',4:'Be',5:'B',6:'C',7:'N',8:'O',9:'F',10:'Ne',\
                           11:'Na',12:'Mg',13:'Al',14:'Si',15:'P',16:'S',17:'Cl',18:'Ar',\
                           19:'K',20:'Ca',21:'Sc',22:'Ti',23:'V',24:'Cr',25:'Mn',26:'Fe',27:'Co',28:'Ni',29:'Cu',30:'Zn',31:'Ga',32:'Ge',33:'As',34:'Se',35:'Br',36:'Kr',\
                           37:'Rb',38:'Sr',39:'Y',40:'Zr',41:'Nb',42:'Mo',43:'Tc',44:'Ru',45:'Rh',46:'Pd',47:'Ag',48:'Cd',49:'In',50:'Sn',51:'Sb',52:'Te',53:'I',54:'Xe',\
                           55:'Cs',56:'Ba',57:'La',72:'Hf',73:'Ta',74:'W',75:'Re',76:'Os',77:'Ir',78:'Pt',79:'Au',80:'Hg',81:'Tl',82:'Pb',83:'Bi',84:'Po',85:'At',86:'Rn'}

    # If atomic numbers are supplied in place of elements
    try:
        elements = [ id_atom_types.e_dict[int(_)] for _ in elements ]
    except:
        pass

    # Initialize fc if undefined by user
    if fc is None and keep_lone is None:
        fc = [[]]
        fc[0] = [0]*len(elements)
        keep_lone=[[]]

    elif keep_lone is None:
        keep_lone=[[] for i in range(len(fc))]

    elif fc is None:
        fc = [[0]*len(elements) for i in range(len(keep_lone))] 
        
    if len(fc[0]) != len(elements):
        print("ERROR in id_atom_types: fc must have the same dimensions as elements and A")
        quit()

    # bonding index: refer to which bonding/fc it uses to determine atomtypes
    bond_index = [range(len(fc))]*len(elements) 
        
    # check duplication:
    set_fc        = list(map(list,set(map(tuple,fc))))
    set_keep_lone = list(map(list,set(map(tuple,keep_lone))))

    total_fc = [ fc[i] + keep_lone[i] for i in range(len(fc))]
    set_total = list(map(list,set(map(tuple,total_fc))))
    keep_ind = sorted([next( count_m for count_m,m in enumerate(total_fc) if m == j ) for j in set_total] )

    fc_0 = deepcopy(fc)
    keeplone_0 = deepcopy(keep_lone)
    
    if max(len(set_fc),len(set_keep_lone)) == 1:

        fc_0 = fc_0[0]
        keep_lone = keeplone_0[0]
        bond_mat = bond_mat[0]
        
        # Calculate formal charge terms and radical terms
        fc_s = ['' for i in range(len(elements))]
        for i in range(len(elements)):
            if i in keep_lone: fc_s[i] += '*'
            if fc_0[i] > 0   : fc_s[i] += abs(fc_0[i])*"+"
            if fc_0[i] < 0   : fc_s[i] += abs(fc_0[i])*"-" 
        fc_0 = fc_s

        # Assemble prerequisite masses and Loop over the inidices that need to be id'ed
        masses = [ id_atom_types.mass_dict[i] for i in elements ]
        N_masses = deepcopy(masses)
        for i in range(len(elements)):
            N_masses[i] += (fc_0[i].count('+') * 100.0 + fc_0[i].count('-') * 90.0 + fc_0[i].count('*') * 80.0)
                                    
        if algorithm == "matrix": atom_types = [ "["+taffi_type(i,elements,adj_mat,bond_mat,N_masses,gens,fc=fc_0)+"]" for i in range(len(elements)) ]
        elif algorithm == "list": atom_types = [ "["+taffi_type_list(i,elements,adj_mat,N_masses,gens,fc=fc_0)+"]" for i in range(len(elements)) ]

    #resonance structure appear, identify special atoms and keep both formal charge information (now only support matrix input)
    else:

        # Assemble prerequisite masses 
        masses = [ id_atom_types.mass_dict[i] for i in elements ]
        charge_atoms = [[index for index, charge in enumerate(fc_i) if charge !=0] for fc_i in fc_0 ]  # find charge contained atoms
        CR_atoms = [charge_atoms[i] + keeplone_0[i] for i in range(len(fc_0))]
        keep_CR_atoms = [charge_atoms[i] + keeplone_0[i] for i in keep_ind]                             # equal to set of CR_atom  
        special_atoms= [index for index in list(set(chain.from_iterable(keep_CR_atoms))) if list(chain.from_iterable(keep_CR_atoms)).count(index) < len(set_fc)*len(set_keep_lone)]  # find resonance atoms
        normal_atoms =[ind for ind in range(len(elements)) if ind not in special_atoms]  
        atom_types = []
        
        # Calculate formal charge terms
        for l in range(len(fc_0)):
            fc_s = ['' for i in range(len(elements))]
            for i in range(len(elements)):
                if i in keeplone_0[l]: fc_s[i] += '*'
                if fc_0[l][i] > 0    : fc_s[i] += abs(fc_0[l][i])*"+"
                if fc_0[l][i] < 0    : fc_s[i] += abs(fc_0[l][i])*"-" 
            fc_0[l] = fc_s
        
        for ind in range(len(elements)):
            if ind in normal_atoms:
                # Graphical separations are used for determining which atoms and bonds to keep
                gs = graph_seps(adj_mat)
                
                # all atoms within "gens" of the ind atoms are kept
                keep_atoms = list(set([ count_j for count_j,j in enumerate(gs[ind]) if j <= gens ]))  
                contain_special = [N_s for N_s in keep_atoms if N_s in special_atoms]
                N_special = len(contain_special)
        
                # if N_special = 0 select first formal charge
                if N_special == 0:
                    fc = fc_0[0]
                    bond_mat = bond_mat[0]

                # if N_special = 1, select formal charge for that special atom
                elif N_special == 1:
                    bond_ind = [ l for l in range(len(fc_0)) if contain_special[0] in CR_atoms[l]]
                    fc = fc_0[bond_ind[0]]
                    bond_mat = bond_mat[bond_ind[0]]
                    bond_index[ind]=sorted(bond_ind)
                
                # if N_special >= 2, introduce additional Criteria to determine the bond matrix 
                else:
                    fc_criteria = [0]*len(fc_0)
                    # find the nearest special atom
                    nearest_special_atom = [N_s for N_s in special_atoms if adj_mat[ind][N_s] == 1]
                    for l in range(len(fc_0)): 
                        fc_criteria[l] = -len([index for index in nearest_special_atom if index in CR_atoms[l]]) - 0.1 * len([index for index in contain_special if index not in nearest_special_atom and index in CR_atoms[l]])
                    
                    bond_ind = [bind for bind, cr in enumerate(fc_criteria) if cr == min(fc_criteria)]
                    fc = fc_0[bond_ind[0]]
                    bond_mat = bond_mat[bond_ind[0]]
                    bond_index[ind]=sorted(bond_ind)
                    
            else:
                bond_ind = [l for l in range(len(fc_0)) if ind in CR_atoms[l]]
                fc = fc_0[bond_ind[0]]
                bond_mat = bond_mat[bond_ind[0]]
                bond_index[ind]=bond_ind

            # add charge to atom_type sorting
            N_masses = deepcopy(masses)
            for i in range(len(elements)):
                N_masses[i] += (fc[i].count('+') * 100.0 + fc[i].count('-') * 90.0 + fc[i].count('*') * 80.0) 
                
            atom_types += [ "["+taffi_type(ind,elements,adj_mat,bond_mat,N_masses,gens,fc=fc)+"]" ]

    if return_index:
        return atom_types,bond_index
    else:
        return atom_types

# adjacency matrix based algorithm for identifying the taffi atom type
def taffi_type(ind,elements,adj_mat,bond_mat,masses,gens=2,avoid=[],fc=[]):    

    # On first call initialize dictionaries
    if not hasattr(taffi_type, "periodic"):

        # Initialize periodic table
        taffi_type.periodic = { "h": 1,  "he": 2,\
                               "li":3,  "be":4,                                                                                                      "b":5,    "c":6,    "n":7,    "o":8,    "f":9,    "ne":10,\
                               "na":11, "mg":12,                                                                                                     "al":13,  "si":14,  "p":15,   "s":16,   "cl":17,  "ar":18,\
                                "k":19,  "ca":20,  "sc":21,  "ti":22,  "v":23,  "cr":24,  "mn":25,  "fe":26,  "co":27,  "ni":28,  "cu":29,  "zn":30,  "ga":31,  "ge":32,  "as":33,  "se":34,  "br":35,  "kr":36,\
                               "rb":37, "sr":38,  "y":39,   "zr":40,  "nb":41, "mo":42,  "tc":43,  "ru":44,  "rh":45,  "pd":46,  "ag":47,  "cd":48,  "in":49,  "sn":50,  "sb":51,  "te":52,  "i":53,   "xe":54,\
                               "cs":55, "ba":56,            "hf":72,  "ta":73, "w":74,   "re":75,  "os":76,  "ir":77,  "pt":78,  "au":79,  "hg":80,  "tl":81,  "pb":82,  "bi":83,  "po":84,  "at":85,  "rn":86}

    # Check fc condition
    if len(fc) == 0:
        fc = ['']*len(elements)

    if len(fc) != len(elements):
        print("ERROR in taffi_type: fc must have the same dimensions as elements and A")
        quit()
        
    # Find connections, avoid is used to avoid backtracking
    scons = [ count_i for count_i,i in enumerate(bond_mat[ind]) if i == 1 and count_i not in avoid ]
    dcons = [ count_i for count_i,i in enumerate(bond_mat[ind]) if i == 2 and count_i not in avoid ]
    tcons = [ count_i for count_i,i in enumerate(bond_mat[ind]) if i == 3 and count_i not in avoid ]

    # Sort the connections based on the hash function 
    if len(scons) > 0:
        scons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in scons ])[::-1]))[1] 

    if len(dcons) > 0:
        dcons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in dcons ])[::-1]))[1] 

    if len(tcons) > 0:
        tcons = list(zip(*sorted([ (atom_hash(i,adj_mat,masses,gens=gens-1),i) for i in tcons ])[::-1]))[1] 

    # Calculate the subbranches
    # NOTE: recursive call with the avoid list results 
    if gens == 0:
        ssubs = dsubs = tsubs = []

    else:
        ssubs = [ taffi_type(i,elements,adj_mat,bond_mat,masses,gens=gens-1,avoid=[ind],fc=fc) for i in scons ]
        dsubs = [ taffi_type(i,elements,adj_mat,bond_mat,masses,gens=gens-1,avoid=[ind],fc=fc) for i in dcons ]
        tsubs = [ taffi_type(i,elements,adj_mat,bond_mat,masses,gens=gens-1,avoid=[ind],fc=fc) for i in tcons ]

    # Calculate formal charge terms
    return "{}".format(taffi_type.periodic[elements[ind].lower()]) + fc[ind] + "".join(["("+i+")" for i in tsubs] + ["{"+i+"}" for i in dsubs] + [ "["+i+"]" for i in ssubs])

# function to identify reaction types
# adj_mats contains adj matrix for reactant and product
# bond_changes is a list of two lists, in the first list [(a,b),(c,d)] refers to which bonds break and in the second [(a,c),(b,d)] refers to which bonds form
def id_reaction_types(elements,adj_mats,bond_changes,gens=1,algorithm="matrix",return_bond_dis=False):
    
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    # NOTE: It's inefficient to reinitialize this dictionary every time this function is called
    id_reaction_types.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                                   'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                                   'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                                   'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                                   'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                                   'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                                   'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                                   'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    if len(bond_changes) != 2:
        print("bond_changes is a list of two lists, in the first list [(a,b),(c,d)] refers to which bond break and in the second [(a,c),(b,d)] refers to which bond form, exit...")
        quit()
    
    if len(adj_mats) != 2:
        print("adj_mats is a list of two adj_mat, exit...")
        quit()

    # Apply find_lweis to both two ends
    lone_electrons_1,bonding_electrons_1,core_electrons_1,bond_mat_1,fc_1 = find_lewis(elements,adj_mats[0],q_tot=0,keep_lone=[],return_pref=False,verbose=False,return_FC=True)
    keep_lone_1  = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_electrons_1]
    
    lone_electrons_2,bonding_electrons_2,core_electrons_2,bond_mat_2,fc_2 = find_lewis(elements,adj_mats[1],q_tot=0,keep_lone=[],return_pref=False,verbose=False,return_FC=True)
    keep_lone_2  = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone_electrons_2]
    
    # first get atom hash value to determine canonical atom sequence
    masses = [ id_reaction_types.mass_dict[i] for i in elements]
    hash_value_1,atom_sorting_1 = list(zip(*sorted([ (atom_hash(i,adj_mats[0],masses,gens=10),i) for i in range(len(elements))])[::-1]))
    hash_value_2,atom_sorting_2 = list(zip(*sorted([ (atom_hash(i,adj_mats[1],masses,gens=10),i) for i in range(len(elements))])[::-1]))

    # use atom sequence to determine bond types
    atom_types_1 = id_atom_types(elements,adj_mats[0],bond_mat_1,gens=gens,algorithm='matrix',fc=fc_1,keep_lone=keep_lone_1,return_index=False)
    atom_types_2 = id_atom_types(elements,adj_mats[1],bond_mat_2,gens=gens,algorithm='matrix',fc=fc_2,keep_lone=keep_lone_2,return_index=False)

    bond_types_1 = [(hash_value_1[atom_sorting_1.index(ind[0])]+hash_value_1[atom_sorting_1.index(ind[1])],"{}-{}".format(atom_types_1[ind[0]],atom_types_1[ind[1]])) \
                    if hash_value_1[atom_sorting_1.index(ind[0])] >= hash_value_1[atom_sorting_1.index(ind[1])] \
                    else (hash_value_1[atom_sorting_1.index(ind[0])]+hash_value_1[atom_sorting_1.index(ind[1])],"{}-{}".format(atom_types_1[ind[1]],atom_types_1[ind[0]])) for ind in bond_changes[0]]

    bond_types_2 = [(hash_value_2[atom_sorting_2.index(ind[0])]+hash_value_2[atom_sorting_2.index(ind[1])],"{}-{}".format(atom_types_2[ind[0]],atom_types_2[ind[1]])) \
                    if hash_value_2[atom_sorting_2.index(ind[0])] >= hash_value_2[atom_sorting_2.index(ind[1])] \
                    else (hash_value_2[atom_sorting_2.index(ind[0])]+hash_value_2[atom_sorting_2.index(ind[1])],"{}-{}".format(atom_types_2[ind[1]],atom_types_2[ind[0]])) for ind in bond_changes[1]]

    # sort two/multiple break/form bonds
    bond_hash_1,bond_types_1 = zip(*sorted(bond_types_1))
    bond_hash_2,bond_types_2 = zip(*sorted(bond_types_2))
    reaction_hash = 100 * sum([h*10**count_h for count_h,h in enumerate(sorted(bond_hash_1))]) + sum([h*10**count_h for count_h,h in enumerate(sorted(bond_hash_2))])
    
    # classify the reaction category
    if max([len(i) for i in bond_changes]) - min([len(i) for i in bond_changes]) > 1:
        print("YARP model reaction only supports bnfm reaction where |n-m|<=1, exit...")
        quit()
        
    if max([len(i) for i in bond_changes]) > 3:
        print("YARP model reaction only supports the reaction within 6 bond changes, exit...")
        quit()

    elif max([len(i) for i in bond_changes]) == 1:
        # deal with b1f1 reactions
        if min([len(i) for i in bond_changes]) == 1:

            # assume (a.b) bond breaks and (a,c) bond forms, identify the atomtype of c and b in the reactant and product
            R_atomind  = [ind for ind in bond_changes[1][0] if ind not in bond_changes[0][0]][0] 
            R_atomtype = atom_types_1[R_atomind]
            P_atomind  = [ind for ind in bond_changes[0][0] if ind not in bond_changes[1][0]][0] 
            P_atomtype = atom_types_2[P_atomind]

            # compute bond-atom distance
            bond_dis_1 = atom_bond_dis(R_atomind,bond_changes[0][0],adj_mats[0])
            bond_dis_2 = atom_bond_dis(P_atomind,bond_changes[1][0],adj_mats[1])

            # determine reaction types (first based on the number of bond breaks, then smallest bond_dis, then bond hash
            if bond_dis_1 != bond_dis_2:
                seq = [sorted([bond_dis_1,bond_dis_2]).index(bond_dis_1),sorted([bond_dis_1,bond_dis_2]).index(bond_dis_2) ]
            else:
                if sum(bond_hash_1) < sum(bond_hash_2): seq = [0, 1]
                else: seq = [1, 0]

            if seq == [0,1]:
                R_type = "b1f1-({},{})-({},{})-D{}".format(bond_types_1[0],R_atomtype,bond_types_2[0],P_atomtype,bond_dis_1)
                bond_dis_all = [bond_dis_1,bond_dis_2]
            else:
                R_type = "b1f1-({},{})-({},{})-D{}".format(bond_types_2[0],P_atomtype,bond_types_1[0],R_atomtype,bond_dis_2)
                bond_dis_all = [bond_dis_2,bond_dis_1]

        else:
            print("YARP model reaction only supports bnfm reaction where n,m > 0 (at least one bond breaks and forms), exit...")
            quit()
            
    elif max([len(i) for i in bond_changes]) == 2:
        if max([len(i) for i in bond_changes]) == min([len(i) for i in bond_changes]):
            # deal with b2f2 reactions
            bond_dis_1 = bond_dis(bond_changes[0],adj_mats[0])
            bond_dis_2 = bond_dis(bond_changes[1],adj_mats[1])

            # determine reaction types (first based on the number of bond breaks, then smallest bond_dis, then bond hash
            if bond_dis_1 != bond_dis_2:
                seq = [sorted([bond_dis_1,bond_dis_2]).index(bond_dis_1),sorted([bond_dis_1,bond_dis_2]).index(bond_dis_2) ]
            else:
                if sum(bond_hash_1) < sum(bond_hash_2): seq = [0, 1]
                else: seq = [1, 0]

            if seq == [0,1]:
                R_type = "b2f2-({},{})-({},{})-D{}".format(bond_types_1[0],bond_types_1[1],bond_types_2[0],bond_types_2[1],bond_dis_1)
                bond_dis_all = [bond_dis_1,bond_dis_2]
            else:
                R_type = "b2f2-({},{})-({},{})-D{}".format(bond_types_2[0],bond_types_2[1],bond_types_1[0],bond_types_1[1],bond_dis_2)
                bond_dis_all = [bond_dis_2,bond_dis_1]
        else:
            # deal with b2f1 and b1f2 cases, start from '2' side
            if len(bond_changes[0]) == 2:
                seq   = [0,1]
                bond_dis_1 = bond_dis(bond_changes[0],adj_mats[0])
                R_type = "b2f1-({},{})-({})-D{}".format(bond_types_1[0],bond_types_1[1],bond_types_2[0],bond_dis_1)
                bond_dis_all = [bond_dis_1,0]
            else:
                seq   = [1,0]
                bond_dis_1 = bond_dis(bond_changes[1],adj_mats[1])
                R_type = "b2f1-({},{})-({})-D{}".format(bond_types_2[0],bond_types_2[1],bond_types_1[0],bond_dis_1)
                bond_dis_all = [bond_dis_1,0]
                
    elif max([len(i) for i in bond_changes]) == 3:
        if max([len(i) for i in bond_changes]) == min([len(i) for i in bond_changes]):
            # deal with b3f3 reactions, first, loop over possible pairs of bonds
            bond_dis_1 = [bond_dis(bonds,adj_mats[0]) for bonds in combinations(bond_changes[0],2)]
            bond_dis_2 = [bond_dis(bonds,adj_mats[1]) for bonds in combinations(bond_changes[1],2)]

            if sum(bond_dis_1) != sum(bond_dis_2): 
                seq = [sorted([sum(bond_dis_1),sum(bond_dis_2)]).index(sum(bond_dis_1)),sorted([sum(bond_dis_1),sum(bond_dis_2)]).index(sum(bond_dis_2)) ]
            else:
                if sum(bond_hash_1) < sum(bond_hash_2): seq = [0, 1]
                else: seq = [1, 0]

            if seq == [0,1]:
                R_type = "b3f3-({},{},{})-({},{},{})".format(bond_types_1[0],bond_types_1[1],bond_types_1[2],bond_types_2[0],bond_types_2[1],bond_types_2[2])
                bond_dis_all = [sum(bond_dis_1),sum(bond_dis_2)]
            else:
                R_type = "b3f3-({},{},{})-({},{},{})".format(bond_types_2[0],bond_types_2[1],bond_types_2[2],bond_types_1[0],bond_types_1[1],bond_types_1[2])
                bond_dis_all = [sum(bond_dis_2),sum(bond_dis_1)]
                
        else:
            # deal with b2f3/b3f2 reactions
            if len(bond_changes[0]) == 3:
                bond_dis_1 = [bond_dis(bonds,adj_mats[0]) for bonds in combinations(bond_changes[0],2)]
                bond_dis_2 = bond_dis(bond_changes[1],adj_mats[1])
                seq   = [0,1]
                R_type = "b3f2-({},{},{})-({},{})".format(bond_types_1[0],bond_types_1[1],bond_types_1[2],bond_types_2[0],bond_types_2[1])
                bond_dis_all = [sum(bond_dis_1),bond_dis_2]
                
            else:
                bond_dis_1 = bond_dis(bond_changes[0],adj_mats[0])
                bond_dis_2 = [bond_dis(bonds,adj_mats[1]) for bonds in combinations(bond_changes[1],2)]
                seq   = [1,0]
                R_type = "b3f2-({},{},{})-({},{})".format(bond_types_2[0],bond_types_2[1],bond_types_2[2],bond_types_1[0],bond_types_1[1])
                bond_dis_all = [sum(bond_dis_2),bond_dis_1]

    if return_bond_dis:
        return R_type,seq,bond_dis_all
    else:
        return R_type,seq

# Return the distance of two bonds 
def bond_dis(bonds,adj_mat,max_d=3):
    
    # check whether two bonds are given
    if len(bonds) != 2:
        print("Error! input bonds must contain two bonds, exit...")
        quit()

    # if contains same element, distance will be 0
    if len(list(set(bonds[0]).intersection(bonds[1]))) > 0:
        return 0

    else:
        gs = graph_seps(adj_mat)
        dis_list = [gs[i][j] for i in bonds[0] for j in bonds[1]]
        dis = int(min(dis_list))
        if dis == -1 or dis >= max_d: 
            dis = 10
        return dis

# Return the distance of an atom and a bond
def atom_bond_dis(atom,bond,adj_mat,max_d=3):
    
    gs = graph_seps(adj_mat)
    dis_list = [gs[i][atom] for i in bond]
    dis = int(min(dis_list))
    if dis == -1 or dis >= max_d: 
        dis = 10
    return dis

# Function to generate model reaction based on :
# 1. Elements, geometry adj and bond maytix of the initial reactant 
# 2. The bonds that are formed and broken
def gen_model_reaction(E,G,adj_mat,bond_mat,bond_break,bond_form,fc=None,keep_lone=[],gens=1,canonical=False):
    
    # set default value for fc
    if fc is None:
        fc = [0]*len(E)

    # If bond distance is infinity (>2), generate two reactants separately
    bond_change = bond_break+bond_form
    keep_ind = list(set([item for sublist in bond_change for item in sublist]))
            
    mode_ind,Fgeo,Fadj_mat,Felements,_,_,M_dict,change_ind = mode_frag(keep_ind,G,adj_mat,bond_mat,E,q_tot=sum(fc),gens=1,fc=fc,keep_lone=keep_lone,return_M_dict=True,return_remove=True)
    F_geo    = opt_geo(deepcopy(Fgeo),Fadj_mat,Felements,ff='mmff94',step=1000)
    change_inds = change_ind[0]

    # sort atoms based on the input sequence
    current_seq = change_inds + list(range(len(E),len(E)+len(change_ind[1])))
    new_seq = [current_seq.index(ind) for ind in sorted(current_seq)]
    
    # Update lists/arrays based on the new sequence
    F_geo    = F_geo[new_seq]
    Felements= [Felements[ind] for ind in new_seq]
    Fadj_mat = Fadj_mat[new_seq]
    Fadj_mat = Fadj_mat[:,new_seq]
    M_dict   = {ind:new_seq.index(M_dict[ind]) for ind in M_dict.keys()}
    
    # determine bond break and bond form
    N_bond_break = [tuple([M_dict[ind] for ind in bb]) for bb in bond_break]
    N_bond_form  = [tuple([M_dict[ind] for ind in bf]) for bf in bond_form]

    # determine reactant bond-electron mats
    lone,bond,core,Rbond_mat,fc = find_lewis(Felements,Fadj_mat,q_tot=sum(fc),return_pref=False,return_FC=True)
    R_BE = np.diag(lone[0])+Rbond_mat[0]

    # Determine product geometry
    BE_break = break_bonds(R_BE,N_bond_break)
    P_BE     = form_bonds(BE_break,N_bond_form)
    Padj_mat = bond_to_adj(P_BE)
    P_G      = opt_geo(deepcopy(F_geo),Padj_mat,Felements,ff='mmff94',step=1000)
    added_index = range(len(change_inds),len(Felements))

    # Generate canonical sequence
    if canonical:
        Felements,Fadj_mat,hash_list,F_geo,Rbond_mat,Ndup = canon_geo(Felements,Fadj_mat,F_geo,Rbond_mat[0],dup=[range(len(Felements))],change_group_seq=True)
        seq = Ndup[0]
        added_index = [seq.index(ind) for ind in added_index]
        P_G = P_G[seq]
        Padj_mat   = Padj_mat[seq]
        Padj_mat   = Padj_mat[:,seq]

    # write info into dictionary
    model_reaction={}
    model_reaction['E'] = Felements
    model_reaction['original_index'] = sorted(change_inds)
    model_reaction['added_index'] = added_index
    model_reaction['R_geo'] = F_geo
    model_reaction['R_adj'] = Fadj_mat
    model_reaction['P_geo'] = P_G
    model_reaction['P_adj'] = Padj_mat
        
    return model_reaction

# Returns the geometry and relevant property lists corresponding to the
# smaallest fragment that is consistent with the mode being parametrized.
def mode_frag(M,Geometry,Adj_mat,Bond_mat,Elements,gens=1,q_tot=0,fc=[],keep_lone=[],return_M_dict=False,return_remove=True):
    
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    if hasattr(mode_frag,"mass_dict") is False:
        mode_frag.mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                               'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                               'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                               'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                               'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                               'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                               'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                               'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    # Check the consistency of the supplied arguments
    if gens < 1: print("ERROR in mode_frag: gens variable must be an integer >= 1. Exiting..."); quit();

    # Generate the fragment geometry/attribute lists corresponding to the mode
    N_M,N_Geometry,N_Adj_mat,N_Elements,N_Bond_mat,tmp,keep_ind = mode_geo(M,Geometry,Adj_mat,Elements,gens,dup=[fc,range(len(Elements))],bond_mat=Bond_mat,return_remove=return_remove)
    fc,atom_seq = tmp
    keep_lone = [atom_seq.index(j) for j in keep_lone if j in atom_seq]

    # check how many fragments appear
    gs = graph_seps(N_Adj_mat)
    groups,loop_ind  = [],[]
    for i in range(len(gs)):
        if i not in loop_ind:
            new_group =[count_j for count_j,j in enumerate(gs[i,:]) if j >= 0]
            loop_ind += new_group
            groups   +=[new_group] 

    # bond index reference dict
    M_dict = {ind:atom_seq.index(ind) for ind in M}
    
    # loop over each fragment
    for group in groups:

        # determine loop list and fixed_bonds
        loop_list = [ind for ind in N_M if ind in group]
        total_fixed_bonds = []

        for lb,bond_mat in enumerate(N_Bond_mat):

            fixed_bonds = []

            # Include the atoms in the mode and connected atoms within the preserve list.    
            for i in loop_list:     
                fixed_bonds += [(group.index(i),group.index(j),int(k)) for j,k in enumerate(bond_mat[i]) if (k > 1 and j > i and j in loop_list)]

            total_fixed_bonds += [fixed_bonds]
    
        # Only if all fixed_bonds are same, take it as fixed_bonds; else, fixed_bonds=[]
        if len(list(map(list,set(map(tuple,total_fixed_bonds))))) > 1: fixed_bonds = []

        # determine the preserve information
        preserve = []
        for i in loop_list:
            preserve += [ count_j for count_j,j in enumerate(gs[i]) if j > -1 and j < gens and Elements[count_j] not in ['P','S']]
    
        # Perform Hydrogenation
        preserve = set(preserve)
        preserve = [group.index(i) for i in preserve]

        # Add hydrogens to make a close-shell
        frag_adj = N_Adj_mat[group,:][:,group]
        frag_G   = N_Geometry[group]
        frag_E   = [N_Elements[ind] for ind in group]
        frag_fc  = [fc[ind] for ind in group]
        frag_qt  = sum(frag_fc)
        frag_lone= [group.index(ind) for ind in keep_lone if ind in group]
        N_Geo,_,N_E,N_Adj,N_fc,N_added_idx = add_hydrogens(frag_G,frag_adj,frag_E,q_tot=frag_qt,preserve=preserve,return_FC=True,fixed_bonds=fixed_bonds,fc=frag_fc,keep_lone=frag_lone)

        # add new hydrogens into original molecule(s)
        N_Geometry = np.vstack([N_Geometry,N_Geo[N_added_idx]])
        N_Elements = N_Elements + [N_E[ind] for ind in N_added_idx]
        fc += [0]*len(N_added_idx)
        tmp = np.zeros([len(N_Elements),len(N_Elements)])
        N_atoms  = len(N_Elements)-len(N_added_idx)
        tmp[:N_atoms,:N_atoms] = N_Adj_mat
        for lc,idx in enumerate(N_added_idx):
            attach = list(N_Adj[idx]).index(1)
            tmp[N_atoms+lc,group[attach]] = 1
            tmp[group[attach],N_atoms+lc] = 1
        N_Adj_mat = tmp

    # redetermine the added_idx
    added_idx = range(len(gs),len(N_Elements)) 

    # create atom change list
    change_ind = [keep_ind,list(added_idx)]

    if return_remove and return_M_dict: 
        return N_M,N_Geometry,N_Adj_mat,N_Elements,fc,keep_lone,M_dict,change_ind
    elif return_M_dict:
        return N_M,N_Geometry,N_Adj_mat,N_Elements,fc,keep_lone,M_dict
    elif return_remove:
        return N_M,N_Geometry,N_Adj_mat,N_Elements,fc,keep_lone,change_ind
    else:
        return N_M,N_Geometry,N_Adj_mat,N_Elements,fc,keep_lone

# Add hydrogens based upon the supplied atom types. 
# NOTE: Hydrogenation heuristics for geometry assume carbon behavior. This isn't usually a problem when the results are refined with transify, but more specific rules should be implemented in the future
def add_hydrogens(geo,adj_mat,elements,atomtypes=[],q_tot=0,preserve=set([]),saturate=True,retype=False,return_FC=False,fixed_bonds=[],fc=[],keep_lone=[]):
    
    # Initialize the saturation dictionary the first time this function is called
    if not hasattr(add_hydrogens, "sat_dict"):
        add_hydrogens.sat_dict = {  'H':1, 'He':1,\
                                   'Li':1, 'Be':2,                                                                                                                'B':3,     'C':4,     'N':3,     'O':2,     'F':1,    'Ne':1,\
                                   'Na':1, 'Mg':2,                                                                                                               'Al':3,    'Si':4,     'P':3,     'S':2,    'Cl':1,    'Ar':1,\
                                    'K':1, 'Ca':2, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':None, 'As':None, 'Se':None, 'Br':1,    'Kr':None,\
                                   'Rb':1, 'Sr':2,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':1,    'Xe':None,\
                                   'Cs':1, 'Ba':2, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }

        add_hydrogens.lone_e = {    'H':0, 'He':2,\
                                   'Li':0, 'Be':2,                                                                                                                'B':0,     'C':0,     'N':2,     'O':4,     'F':6,    'Ne':8,\
                                   'Na':0, 'Mg':2,                                                                                                               'Al':0,    'Si':0,     'P':2,     'S':4,    'Cl':6,    'Ar':8,\
                                    'K':0, 'Ca':2, 'Sc':None, 'Ti':None,  'V':None, 'Cr':None, 'Mn':None, 'Fe':None, 'Co':None, 'Ni':None, 'Cu':None, 'Zn':None, 'Ga':None, 'Ge':0,    'As':3,    'Se':4,    'Br':6,    'Kr':None,\
                                   'Rb':0, 'Sr':2,  'Y':None, 'Zr':None, 'Nb':None, 'Mo':None, 'Tc':None, 'Ru':None, 'Rh':None, 'Pd':None, 'Ag':None, 'Cd':None, 'In':None, 'Sn':None, 'Sb':None, 'Te':None,  'I':6,    'Xe':None,\
                                   'Cs':0, 'Ba':2, 'La':None, 'Hf':None, 'Ta':None,  'W':None, 'Re':None, 'Os':None, 'Ir':None, 'Pt':None, 'Au':None, 'Hg':None, 'Tl':None, 'Pb':None, 'Bi':None, 'Po':None, 'At':None, 'Rn':None  }

        add_hydrogens.frag = 0

    # set default value for fc
    if len(fc) == 0:
        fc = [0] * len(elements)

    # Intermediate scalars
    H_length = 1.1
    N_atoms  = len(geo)
    init_len = len(geo)

    # If the user specifies a set of atoms to preserve as is, then then bonding_pref entry is set to full saturation.
    if preserve != []: bonding_pref = [ (i,add_hydrogens.sat_dict[elements[i]] + fc[i] ) if i not in keep_lone else (i,add_hydrogens.sat_dict[elements[i]] - 1 ) for i in preserve]
    else: bonding_pref = []

    lone_electrons,bonding_electrons,core_electrons,bonding_pref = frag_find_lewis(elements,adj_mat,q_tot=q_tot,fc_0=fc,keep_lone=keep_lone,fixed_bonds=fixed_bonds,bonding_pref=bonding_pref,return_pref=True,check_lewis_flag=True)
        
    # Update the preserved atoms (check_lewis will extend this list if there are special groups (e.g., nitro moieties) that need to be conserved for the sake of the lewis structure)
    preserve = set([ i[0] for i in bonding_pref ])
    
    # Loop over the atoms in the geometry
    for count_i,i in enumerate(geo):
        
        # ID undercoordinated atoms
        if count_i in preserve:
            continue
        elif add_hydrogens.sat_dict[elements[count_i]] is not None:
            B_expected = add_hydrogens.sat_dict[elements[count_i]] + fc[count_i]
        else:
            print("ERROR in add_hydrogens: could not determine the number of hydrogens to add to {}. Exiting...".format(elements[count_i]))
            quit()

        if count_i in keep_lone:
            B_expected -= 1
        B_current  = bonding_electrons[count_i]

        # Determine the number of nuclei that are attached and expected.
        N_current   = sum(adj_mat[count_i])
        N_expected = N_current + (B_expected - B_current)

        # Add hydrogens to undercoordinated atoms
        if N_expected > N_current:
            
            old_inds = [ count_j for count_j,j in enumerate(adj_mat[count_i]) if j == 1 ]
            
            # Protocols for 1 missing hydrogen
            if N_expected - N_current == 1:
                if N_expected == 1:
                    new = i + np.array([H_length,0.0,0.0])
                elif N_expected == 2:
                    new = -1.0 * normalize(geo[old_inds[0]] - i) * H_length + i + np.array([np.random.random(),np.random.random(),np.random.random()])*0.01 #random factor added for non-carbon types to relax during FF-opt
                elif N_expected == 3:
                    new = -1.0 * normalize( normalize(geo[old_inds[0]] - i) + normalize(geo[old_inds[1]] - i) ) * H_length + i
                elif N_expected == 4:
                    new = -1.0 * normalize( 1.5 * normalize(geo[old_inds[0]] - i) + normalize(geo[old_inds[1]] - i) + normalize(geo[old_inds[2]] - i) ) * H_length + i                

                # Update geometry, adj_mat, elements, and atomtypes with one new atoms
                geo = np.vstack([geo,new])
                if len(atomtypes) > 0:
                    atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]
                elements += ["H"]
                fc += [0]
                tmp = np.zeros([N_atoms+1,N_atoms+1])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[-1,count_i] = 1
                tmp[count_i,-1] = 1
                adj_mat = tmp                
                N_atoms += 1

            # Protocols for 2 missing hydrogens
            # ISSUE, NEW ALGORITHM IS BASED ON BONDED ATOMS NOT BONDED CENTERS
            if N_expected - N_current == 2:
                if N_expected == 2:
                    new_1 = i + np.array([H_length,0.0,0.0])
                    new_2 = i - np.array([H_length,0.0,0.0])
                elif N_expected == 3:
                    rot_vec = normalize(np.cross( geo[old_inds[0]] - i, np.array([np.random.random(),np.random.random(),np.random.random()]) ))
                    new_1 = normalize(axis_rot(geo[old_inds[0]],rot_vec,i,120.0) - i)*H_length + i
                    new_2 = normalize(axis_rot(geo[old_inds[0]],rot_vec,i,240.0) - i)*H_length + i
                elif N_expected == 4:
                    bisector = normalize(geo[old_inds[0]] - i + geo[old_inds[1]] - i) 
                    new_1    = axis_rot(geo[old_inds[0]],bisector,i,90.0)
                    new_2    = axis_rot(geo[old_inds[1]],bisector,i,90.0) 
                    rot_vec  = normalize(np.cross(new_1-i,new_2-i))
                    angle    = ( 109.5 - acos(np.dot(normalize(new_1-i),normalize(new_2-i)))*180.0/np.pi ) / 2.0
                    new_1    = axis_rot(new_1,rot_vec,i,-angle)
                    new_2    = axis_rot(new_2,rot_vec,i,angle)
                    new_1    = -1*H_length*normalize(new_1-i) + i
                    new_2    = -1*H_length*normalize(new_2-i) + i
                    
                # Update geometry, adj_mat, elements, and atomtypes with two new atoms
                geo = np.vstack([geo,new_1])
                geo = np.vstack([geo,new_2])
                if len(atomtypes) > 0:
                    atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]*2
                elements += ["H","H"]
                fc += [0,0]
                tmp = np.zeros([N_atoms+2,N_atoms+2])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[[-1,-2],count_i] = 1
                tmp[count_i,[-1,-2]] = 1
                adj_mat = tmp
                N_atoms += 2

            # Protocols for 3 missing hydrogens
            if N_expected - N_current == 3:
                if N_expected == 3:
                    rot_vec = np.array([0.0,1.0,0.0])
                    new_1 = i + np.array([H_length,0.0,0.0])
                    new_2 = axis_rot(new_1,rot_vec,i,120.0)
                    new_3 = axis_rot(new_1,rot_vec,i,240.0)
                if N_expected == 4:
                    rot_vec = normalize(np.cross( geo[old_inds[0]] - i, np.array([np.random.random(),np.random.random(),np.random.random()]) ))
                    new_1 = H_length*normalize(axis_rot(geo[old_inds[0]],rot_vec,i,109.5)-i) + i
                    new_2 = axis_rot(new_1,normalize(i-geo[old_inds[0]]),i,120.0)
                    new_3 = axis_rot(new_2,normalize(i-geo[old_inds[0]]),i,120.0)

                # Update geometry, adj_mat, elements, and atomtypes with three new atoms
                geo = np.vstack([geo,new_1])
                geo = np.vstack([geo,new_2])
                geo = np.vstack([geo,new_3])
                if len(atomtypes) > 0:
                    atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]*3
                elements += ["H","H","H"]
                fc += [0,0,0]
                tmp = np.zeros([N_atoms+3,N_atoms+3])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[[-1,-2,-3],count_i] = 1
                tmp[count_i,[-1,-2,-3]] = 1
                adj_mat = tmp
                N_atoms += 3

            # Protocols for 4 missing hydrogens
            if N_expected - N_current == 4:
                if N_expected == 4:
                    new_1 = i + np.array([H_length,0.0,0.0])
                    rot_vec = normalize(np.cross( new_1 - i, np.array([np.random.random(),np.random.random(),np.random.random()]) ))
                    new_2 = H_length*normalize(axis_rot(new_1,rot_vec,i,109.5)-i) + i
                    new_3 = axis_rot(new_2,normalize(i-new_1),i,120.0)
                    new_4 = axis_rot(new_3,normalize(i-new_1),i,120.0)
                    
                # Update geometry, adj_mat, elements, and atomtypes with three new atoms
                geo = np.vstack([geo,new_1])
                geo = np.vstack([geo,new_2])
                geo = np.vstack([geo,new_3])
                geo = np.vstack([geo,new_4])
                if len(atomtypes) > 0:
                    atomtypes += ["[1[{}]]".format(atomtypes[count_i].split(']')[0].split('[')[1])]*4
                elements += ["H","H","H","H"]
                fc += [0,0,0,0]
                tmp = np.zeros([N_atoms+4,N_atoms+4])
                tmp[:N_atoms,:N_atoms] = adj_mat
                tmp[[-1,-2,-3,-4],count_i] = 1
                tmp[count_i,[-1,-2,-3,-4]] = 1
                adj_mat = tmp
                N_atoms += 4
    
    if retype is True:
        atom_types=id_types(elements,adj_mat,gens=2,fc=[fc],keep_lone=[keep_lone])
        atomtypes=[atom_type.replace('R','') for atom_type in atom_types]
    
    if return_FC:
        return geo,atomtypes,elements,adj_mat,fc,range(init_len,len(geo))
    else:
        return geo,atomtypes,elements,adj_mat,range(init_len,len(geo))

# Description: Returns the canonical fragment corresponding to the mode defined associated with geo and atoms m_ind. Consider the bond matrix
#
# Inputs:      m_ind:        list of indices involved in the mode
#              geo:          an Nx3 np.array holding the geometry of the molecule
#              adj_mat:      an NxN np.array holding the connectivity of the molecule
#              gens:         an integer specifying the number of generations involved in the geometry search
#                            (Algorithm returns 
#              force_linear: boolean, forces a non-cyclic structure.
#
# Returns:     m_ind:        list of indices involved in the mode (in terms of the new geometry)
#              N_geo:        new geometry for parameterizing the mode.
#              N_adj_mat:    new adjacency matrix
#              N_dup:        user supplied lists indexed to the original geometry, now indexed to the new geometry
def mode_geo(m_ind,geo,adj_mat,Elements,gens=2,dup=[],bond_mat=None,return_remove=False):

    # Seed conditions for...
    # atoms: single atom
    # bonds: both atoms
    # angles: center atom
    # linear dihedral: center atoms
    # improper dihedral: center atom

    m_ind_0 = deepcopy(m_ind)                # A copy is made to assign the mode index at the end
    
    # Graphical separations are used for determining which atoms and bonds to keep
    gs = graph_seps(adj_mat)    

    # all atoms within "gens" of the m_ind atoms are kept
    new_atoms = sorted(list(set([ count_j for i in m_ind for count_j,j in enumerate(gs[i]) if (j <= gens and j >= 0) or (j == gens+1 and Elements[count_j] == 'H') ])))
    
    #if len(m_ind) > 1: new_atoms = list(set([ count_j for i in m_ind for count_j,j in enumerate(gs[i]) if (j <= gens and j > 0)]))
    #else: new_atoms = list(set([ count_j for i in m_ind for count_j,j in enumerate(gs[i]) if j <= gens ]))

    # create a sub-graph for the remaining atoms
    N_adj_mat = adj_mat[new_atoms,:][:,new_atoms]
    N_elements= [Elements[i] for i in new_atoms]

    # generate bond mats
    if len(bond_mat) > 0 and len(bond_mat) != len(geo): # multuple bond_mats 
        N_bond_mats = []
        for lb in range(len(bond_mat)):
            N_bond_mat = bond_mat[lb][new_atoms,:][:,new_atoms] 
            N_bond_mats += [N_bond_mat] 
    else:
        N_bond_mat = bond_mat[new_atoms,:][:,new_atoms]

    # remove the bonds between the "gens" separated atoms    
    edge_ind = list(set([ count_j for i in m_ind for count_j,j in enumerate(gs[i]) if j == gens ]))
    edge_ind = [ new_atoms.index(i) for i in edge_ind if min([ gs[j,i] for j in m_ind ]) == gens ]
    for i in edge_ind:
        for j in edge_ind:
            N_adj_mat[i,j] = 0
            N_adj_mat[j,i] = 0
            if len(bond_mat) > 0 and len(bond_mat) != len(geo): # multuple bond_mats   
                for lb in range(len(bond_mat)):
                    N_bond_mats[lb][i,j] = 0
                    N_bond_mats[lb][j,i] = 0
            else:
                N_bond_mat[i,j] = 0
                N_bond_mat[j,i] = 0

    # Create the new geometry and adj_mat
    N_geo     = np.zeros([len(new_atoms),3])
    for count_i,i in enumerate(new_atoms):
        N_geo[count_i,:] = geo[i,:]

    # Duplicate the respective lists
    N_dup = {}
    for count_i,i in enumerate(dup):
        N_dup[count_i] = []
        for j in new_atoms:
            N_dup[count_i] += [i[j]]
    N_dup = [ N_dup[i] for i in range(len(N_dup.keys())) ]

    # Clean up the geometry
    N_geo = opt_geo(deepcopy(N_geo),N_adj_mat,N_elements,ff='mmff94',step=1000)

    # Assign the mode ind 
    # NOTE: the use of the list.index() method assumes that the mode indices are the first occuring in the geometry
    #       this should be a very good assumption for all conventional modes and seed scenarios (no exceptions have been found).
    m_ind = [ new_atoms.index(i) for i in m_ind_0 ]

    if return_remove:
        if len(bond_mat) > 0 and len(bond_mat) != len(geo): # multuple bond_mats
            return m_ind,N_geo,N_adj_mat,N_elements,N_bond_mats,N_dup,new_atoms
        else:
            return m_ind,N_geo,N_adj_mat,N_elements,N_bond_mat,N_dup,new_atoms
    else:
        if len(bond_mat) > 0 and len(bond_mat) != len(geo): # multuple bond_mats
            return m_ind,N_geo,N_adj_mat,N_elements,N_bond_mats,N_dup
        else:
            return m_ind,N_geo,N_adj_mat,N_elements,N_bond_mat,N_dup

# Function to break given bond list
def break_bonds(BE,bond_break):
    new_BE = deepcopy(BE)
    for bb in bond_break:
        new_BE[bb[0]][bb[1]] -= 1
        new_BE[bb[1]][bb[0]] -= 1
        new_BE[bb[0]][bb[0]] += 1
        new_BE[bb[1]][bb[1]] += 1
    return new_BE

# Function to form given bond list
def form_bonds(BE,bond_form):
    new_BE = deepcopy(BE)
    for bf in bond_form:
        new_BE[bf[0]][bf[1]] += 1
        new_BE[bf[1]][bf[0]] += 1
        new_BE[bf[0]][bf[0]] -= 1
        new_BE[bf[1]][bf[1]] -= 1
    return new_BE

# function to transfer bond_mat to adj_mat
def bond_to_adj(BE):

    adj_mat = deepcopy(BE)

    for i in range(len(BE)):
        for j in range(len(BE)):
            if BE[i][j] > 0: adj_mat[i][j] = 1
            
    for i in range(len(BE)): adj_mat[i][i] = 0

    return adj_mat

# function to get reaction type and model reaction
def return_reaction_types(E,RG,PG,Radj_mat=None,Padj_mat=None):

    # calculate adj_mat if is not provided
    if Radj_mat is None:
        Radj_mat = Table_generator(E,RG)
    if Padj_mat is None:
        Padj_mat = Table_generator(E,PG)
    
    # construct lewis structure for reactant
    lone,bond,core,Rbond_mat,fc = find_lewis(E,Radj_mat,return_pref=False,return_FC=True)

    # find radicals and formal charges
    Rkeep_lone = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone][0]
    Rfc        = fc[0]

    # compute BE matrix
    R_BE   = np.diag(lone[0])+Rbond_mat[0]

    # construct lewis structure for product
    lone,bond,core,Pbond_mat,fc = find_lewis(E,Padj_mat,return_pref=False,return_FC=True)

    # compute BE matrix
    diff_list = []
    for ind in range(len(Pbond_mat)):
        P_BE   = np.diag(lone[ind])+Pbond_mat[ind]
        BE_change = P_BE - R_BE
        diff_list += [np.abs(BE_change).sum()]

    # determine the BE matrix leads to the smallest change
    ind = diff_list.index(min(diff_list))
    P_BE   = np.diag(lone[ind])+Pbond_mat[ind]
    BE_change = P_BE - R_BE

    # determine formal charges and radicals
    Pkeep_lone = [ [ count_i for count_i,i in enumerate(lone_electron) if i%2 != 0] for lone_electron in lone][ind]
    Pfc        = fc[ind]
    
    # determine bonds break and bonds form from Reaction matrix
    bond_break = []
    bond_form  = []
    for i in range(len(E)):
        for j in range(i+1,len(E)):
            if BE_change[i][j] == -1:
                bond_break += [(i,j)]
                
            if BE_change[i][j] == 1:
                bond_form += [(i,j)]
            
    # id reaction type
    try:
        reaction_type,seq,bond_dis = id_reaction_types(E,[Radj_mat,Padj_mat],bond_changes=[bond_break,bond_form],gens=1,algorithm="matrix",return_bond_dis=True)
        return reaction_type,seq,bond_dis
        
    except:
        print("Have trouble getting reaction type, skip...")
        print([bond_break,bond_form])
        return '','',''

def reverse_MR(MR):

    newMR = deepcopy(MR)
    newMR['R_geo'],newMR['R_adj'],newMR['P_geo'],newMR['P_adj'] = MR['P_geo'],MR['P_adj'],MR['R_geo'],MR['R_adj']

    return newMR
    
# Shortcut for normalizing a vector
def normalize(x):
    return x/sum(x**(2.0))**(0.5)

if __name__ == "__main__": 
    main(sys.argv[1:])
    
