#!/bin/env python                                                                                                                                                             
# Author: Aditi Khot (akhot@purdue.edu)

import os, argparse, sys
from pi_stacking import parse_data, quat2mat, calc_hist
sys.path.append('/home/yukselb/bin/taffi/Lib/')
from adjacency import graph_seps, Dijkstra
import numpy as np
from math import pi, acos, sqrt, log
from itertools import combinations
import time
import subprocess as sp
from copy import deepcopy
from scipy.spatial.distance import cdist
from collections import OrderedDict 
import matplotlib
matplotlib.use('Agg')
from pylab import *
import seaborn as sns
from cluster import find_subgraphs
from itertools import combinations_with_replacement
import pipes, json
import random
#import multiprocessing as mp

def main(argv):
    startTime0= time.time()
    parser = argparse.ArgumentParser(description='Reads a lammps trajectory, performs a charge hopping KMC run and calculates the mobility of charge. Includes support for intramolecular transport with clustering to avoid trap regions')
    
    parser.add_argument('-traj', dest="traj", default="", type=str, help='Input lammps trajectory file to be read')

    parser.add_argument('-data', dest="data", default="", type=str, help='Input lammps data file to be read')

    parser.add_argument('-f', dest="first", default=0, type=int, help='First frame of lammps trajectory which should be parsed. Default: 1st frame')

    parser.add_argument('-l', dest="last", default=-1, type=int, help='Last frame of lammps trajectory which should be parsed. Default: Last frame')

    parser.add_argument('-e', dest="every", default=1, type=int, help='Parse every these many frames. Default: every frame')

    parser.add_argument('-lavg', dest="lastavg", default=20, type=int, help='Use these many frames after the timestep, to average rates. Default: 50^th frame')

    parser.add_argument('-eavg', dest="everyavg", default=1, type=int, help='Use frames at this frame frequency after the timestep, to average rates. Default: every 5 frames')

    parser.add_argument('-Nkmc', dest="Nkmc", default=10, type=int, help='Number of KMC runs to be performed for each snapshot. Default: 10')

    parser.add_argument('-excl_type', dest="excl_type", default=1, type=int, help='1: Avoid getting trapped in trap sites (sites from which transition cannot happen) by removing them, 2: Do not avoid them, but if the charge enters this site, assume it will remain there indefinitely')

    parser.add_argument('-rc', dest="rc", default=25.0, type=float, help='If the distance between the centroids of the backbones for two polymer molecules is more than this cut-off, the polymers are considered far apart and are not included in the pi-stacking statistics. Default: 25.0 A')

    #parser.add_argument('-phic', dest="phic", default=20.0, type=float, help='If the torsional angle between neighboring beads is less than equal to this cutoff then the charge is delocalized on this sites. Default: 20.0 degree')

    parser.add_argument('-box_cap', dest="box_cap", default=10, type=int, help='Number of boxes the charge can cross before a site refresh. Default: 10')

    parser.add_argument('-dist_cap', dest="dist_cap", default=500, type=float, help='Distance the box charge can cross before a site refresh. Default: 50 nm')

    parser.add_argument('-kmc_tmax', dest="kmc_tmax", default=2.0, type=float, help='The time for which the KMC calculation has to be performed in ns. Default: 2 ns')

    parser.add_argument('-tmax', dest="tmax", default=1.5, type=float, help='The max time for the MSD calculation in ns. Default: 1.5')

    parser.add_argument('-dt', dest="dt", default=0.01, type=float, help='The time step betweentwo consecutive time bins for MSD calculation in ns. Default: 10 ps') 

    parser.add_argument('-btypes', dest="btypes", default='1', type=str, help='Space delimited string of backbone bead types. Default: 1')

    parser.add_argument('-qtypes', dest="qtypes", default='6', type=str, help='Space delimited string of charge bead types. Default: 6')

    parser.add_argument('-wtypes', dest="wtypes", default='4 5', type=str, help='Space delimited string of charge water bead types. Default: 4 5')

    # rate related inputs
    # parser.add_argument('-intrac', dest="intra_cutoff", default=10.0**14.0, type=float, help='If the value of intramolecular rate is less than this value, the molecule is broken into segments. Default: E14 s-1')

    parser.add_argument('-Etype', dest="Etype", default="constant", type=str, help='Either send a file which contains the energy difference as a function of length or a string "constant", in which case normal DOS is used. Default: constant')

    parser.add_argument('-E', dest="E", default=0, type=float, help='Constant free energy difference in eV. Default: 0 eV')

    parser.add_argument('-Eelec', dest="Eelec", default=0, type=int, help='If the electrostatic energy should be included in the rate calculation. 1: yes, 0: no. Default: No')

    # parser.add_argument('-Eelec_intra_file', dest="Eelec_intra_file", default='', type=str, help='File containing electrostatic energies for monomers generated by calc_Eelec_intra.py')

    parser.add_argument('-dielectric', dest="dielectric", default=3.0, type=float, help='The static dielectric of the surrounding. The dielectric used for calculating external reorganization energy and if the electrostatic energy is included in the rate calculation, the dielectric used for this calculation. Default: 10.0')

    # parser.add_argument('-Ltype', dest="Ltype", default="constant", type=str, help='Either send a file which contains the internal reorganization energy as a function of length or a string "constant", in which case a constant value is used. Default: constant')

    parser.add_argument('-lmbdi', dest="lmbdi", default=0.3, type=float, help='The constant value of reorganization energy to be used in eV. Default: 0.3 eV')

    parser.add_argument('-loffset', dest="loffset", default=3.0, type=float, help='Evaluation of external reoragnization energy is derived by approximating the molecular site as a spherical charge and needs the radius of this sphere. The radius of the molecular site is calculated as an average of the segment radius perpendicular to backbone and the length of the segment along the backbone. This option is for specifying the segment radius perpendicular to backbone in Angstrom. A crude approximation can be sgm_y. Default: 3.0 A')

    parser.add_argument('-cp', dest="cp", default=0, type=float, help='The Pekar factor used in evaluation of external reoragnization energy. It is given by cp=(1/e_opt -1/e_0) where e_opt and e_0 are the relative permittivities at optical and zero frequency respectively. If a nonzero value is supplied it is used, while if set to zero, it is calculated based on the dielectric of the surrounding. Default: 0 ( Calculated based on the dielectric) ')

    parser.add_argument('-Vtype', dest="Vtype", default="constant", type=str, help='Type of intermolecular rate electronic coupling. Either send a space delimited string of V0 (eV), r0 (A) and beta (1/A) in the formula V0*exp(-beta*(r-r0)) for electronic coupling or a string "constant", in which case a constant value is used. Default: constant')

    parser.add_argument('-V', dest="V", default=0.01, type=float, help='The constant value of intermolecular rate electronic coupling to be used in eV. Default:0.02 eV')

    parser.add_argument('-Vintra', dest="Vintra", default=0.5, type=float, help='The prefactor J0 for intramolecular rate coupling J0*cos(phi) to be used in eV. Default:0.5 eV')

    parser.add_argument('-nu', dest="nu", default=10.0**15.0, type=float, help='The prefactor for intramolecular rate/ nuclear frequency to be used in 1/s. Default:10**15.0 1/s')
    
    # parser.add_argument('-gamma', dest="gamma", default=0.02, type=float, help='The  gamma parameter for clustering')

    # Other inputs
    parser.add_argument('-temp', dest='temp', default=300.0, type=float,
                        help = 'Temperature of the simulation in K. Default: 300.0 K')

    # parser.add_argument('-seed', dest='seed', default=1, type=int,
    #                     help = 'Seed value for kmc initialization. Default: 1')

    parser.add_argument('-seed', dest='seed', default=random.randint(1,100000), type=int, help='Seed value for kmc initialization. Default: random')

    parser.add_argument('-o', dest='output', default="calc_mob", type=str,
                        help = 'Name of the output folder. Default: calc_mob')

    parser.add_argument('-plot', dest='plot', default=0, type=int,
                        help = 'If plots have to be generated for distributions/ MSD. 1: yes, 0: no. Default: Yes')

    parser.add_argument('-debug', dest='debug', default=0, type=int,
                        help = 'Print details of the KMC run. 1: yes, 0: no. Default: No')

    parser.add_argument('-wtraj', dest='wtraj', default=0, type=int,
                        help = 'Write the first KMC trajectory for all snapshots in LAMMPS format with the backbone beads and charge. 1: yes, 0: no. Default: No')

    parser.add_argument('-wtraj_unwrap', dest='wtraj_unwrap', default=1, type=int,
                        help = 'Write the unwrapped KMC trajectory. 1: yes, 0: no. Default: Yes')

    parser.add_argument('-atypes', dest="atypes", default=1, type=int, help='Assign specific atom types in the KMC trajectories for visualization. 0: As obtained from lammps, 1: Assign based on electric network which belong to same electronically connected cluster, 2: Assign the site the charge is on, a different type. Default: 1')

    parser.add_argument('-parallel', dest='parallel', default=0, type=int,
                        help = 'If 0:  msd is normalized, diffusivity and mobility are calculated, 1: the unnormalzied msd entries are written to files which are combined for all parallel runs by calc_mob_all_elec_parallel.py later. Default: 0 (Not parallel mode)')

    parser.add_argument('-python', dest="python", default='/apps/spack/bell/apps/anaconda/2019.10-py27-gcc-4.8.5-bdr3zxp/bin', type=str, help='Python path. Default: /home/akhot/anaconda/bin/')

    parser.add_argument('-taffi', dest="taffi", default='/home/yukselb/bin/taffi/', type=str, help='Taffi folder path. Default: /home/akhot/bin/taffi/')

    parser.add_argument('-lammps', dest="lammps", default='/depot/bsavoie/apps/lammps/exe/lmp_mpi_180501', type=str, help='LAMMPS executable to be used. Default: /depot/bsavoie/apps/lammps/exe/lmp_mpi_180501')

    parser.add_argument('-only_dist', dest="only_dist", default=0, type=int, help='Distribution of different quantities (rates, reorganization enegry, electrostatic energy, segment lengths) is generated and KMC simulation is not run. 1: yes, 0: no. Default: 0')

    parser.add_argument('-stypes', dest="stypes", default='1', type=str, help='Space delimited string of hopping site bead types. Default: 1') #BY
    
    parser.add_argument('-lmbdi_const', dest='lmbdi_const', type=float, default=0.3,help='Constant internal reorganization energy (eV)')

    parser.add_argument('-keep_frac', dest='keep_frac', default=1.0, type=float, help='Fraction of KMC sites to KEEP active (0-1). 1.0 keeps all sites, 0.0 disables all sites.')

    parser.add_argument('-network_rate_thresh', dest='network_rate_thresh', default=1.0e10, type=float, help='Minimum rate in 1/s for two sites to be considered connected in network analysis.')

    args = parser.parse_args()
    startTime0=time.time()

    if args.keep_frac < 0.0 or args.keep_frac > 1.0: 
        raise RuntimeError("-keep_frac must be between 0 and 1.")

    print "PROGRAM CALL: python {} {}\n".format(os.path.basename(sys.argv[0]), ' '.join(argv))          
    if args.output[-1]!='/': args.output+='/'
    if os.path.exists(args.output):
        print 'The output folder {} exists. Exiting...\n'.format(args.output)
        #quit()
    else:
        os.makedirs(args.output)

    print 'Fraction of KMC sites to KEEP active (0-1): {}'.format(args.keep_frac)
    # args.intra_cutoff= args.intra_cutoff*(10.0**-9.0) # convert intramolecular rate cutoff to 1/s to 1/ns
    
    # PRELMINARY STEPS TO PARSE ONLY BACKBONE BEADS AND THE MOLECULES THEY BELONG TO
    print 'Parsing the indices of backbone beads, identifying indices corresponding to distinct molecules, and consecutive beads in these molecules along which intramolecular charge transport occurs...\n'
    
    # Parse backbone bead types and charge bead types
    btypes=[int(_) for _ in args.btypes.split()]
    stypes = [int(_) for _ in args.stypes.split()] #BY
    qtypes=[int(_) for _ in args.qtypes.split()]
    wtypes=[int(_) for _ in args.wtypes.split()]

    # Read the data file
    objects = parse_data(args.data)
    # Collect adjacency list (needed for unwrapping coordinates) and adjacency matrix (needed for identifying backbone)
    adj_list = [[] for _ in range(objects['numbers']['atoms'])]
    adj_mat = np.zeros((objects['numbers']['atoms'],objects['numbers']['atoms']), dtype=int)
    for b1, b2 in objects['Bonds']['ids']:   
        adj_list[b1-1].append(b2-1)
        adj_list[b2-1].append(b1-1)
        adj_mat[b1-1,b2-1]=1
        adj_mat[b2-1,b1-1]=1

    # Identify backbone beads, which molecule they belong to, consecutive backbone beads etc.
    # The molecules which belong to backbone bead
    mols= sorted(list(set([m for i,m in enumerate(objects["Molecules"]) if objects["Atom_types"][i] in btypes])))
    b_ids=[]    # Atom ids of backbone beads which belong to each molecule, only those will be unwrapped by lammps trajectory parser
    mol2bidx={}    # And a list of bead indices belonging to each molecule    
    bidx2mol={} # And the other way
    b_idx=[]    # Atom index of backbone beads, Monomer sites are to be used for intramolecular rate, use this to store mapping between monomer site idx and its corresponding bead idx 
    consecutive=[] # Site indices of consecutive beads
    consecutive_bpairs=[] # Pairs of consecutive backbone bead indices (0-based)
    
    # -------------------------
    # -------------------------
    # Define hopping sites as super-sites (all sidechain beads attached to a backbone bead)
    # Each backbone bead that has a sidechain becomes one KMC site consisting of all sidechain beads (types in stypes) connected to it.
    # -------------------------

    # Beads in this molecule
    for m in mols:
        mol2bidx[m]=[objects["Atom_ids"][i]-1 for i, m2 in enumerate(objects["Molecules"]) if m==m2 and objects["Atom_types"][i] in btypes] 
        bidx2mol.update({ _ : m for _ in  mol2bidx[m] })
        b_ids+=[_+1 for _ in mol2bidx[m]]
        b_idx+=mol2bidx[m]
        # Get consecutive beads
        seps=graph_seps(adj_mat[mol2bidx[m],:][:,mol2bidx[m]])
        # s,e=np.where(seps==np.max(seps))[0]
        s, e = np.unravel_index(np.argmax(seps), seps.shape) #BY
        bb=list(Dijkstra(adj_mat[mol2bidx[m],:][:,mol2bidx[m]],s,e))
        sold=len(b_ids)-len(mol2bidx[m])
        consecutive+=[(sold+bb[i],sold+bb[i+1]) for i in range(len(bb)-1)]
        consecutive_bpairs+=[(mol2bidx[m][bb[i]], mol2bidx[m][bb[i+1]]) for i in range(len(bb)-1)]

    # Build super-sites AFTER backbone parsing (b_idx is now known)
    stypes_set=set(stypes)
    msites=[]            # list of sites, each is a list of 0-based bead indices (sidechain beads)
    site2bidx=[]         # site2bidx[s] = 0-based backbone bead index that anchors this super-site
    bidx2site={}         # map: backbone bead index -> site index
    visited_sc=set()     # sidechain beads already assigned to a site

    for b in b_idx:
        # seeds: sidechain beads directly bonded to this backbone bead
        seeds=[n for n in adj_list[b] if objects["Atom_types"][n] in stypes_set]
        if len(seeds)==0:
            continue

        cluster=[]
        stack=list(seeds)
        while len(stack)>0:
            a=stack.pop()
            if a in visited_sc:
                continue
            if objects["Atom_types"][a] not in stypes_set:
                continue
            visited_sc.add(a)
            cluster.append(a)
            for nb in adj_list[a]:
                if nb not in visited_sc and objects["Atom_types"][nb] in stypes_set:
                    stack.append(nb)

        if len(cluster)>0:
            bidx2site[b]=len(msites)
            msites.append(sorted(cluster))
            site2bidx.append(b)

    print 'Total sidechain sites (super-sites) = {}'.format(len(msites))
    if len(msites)>0:
        Ltmp=[len(s) for s in msites]
        print 'The minimum, average and maximum length of super-site is {}, {} and {}.'.format(np.min(Ltmp), np.mean(Ltmp), np.max(Ltmp))

    # If KMC trajectories are to be written, only the backbone beads are retaioned
    # Create a dictionary with keys as original backbone indices and their modified atom id and mol id for this KMC trajectory file as corresponding objects 
    # Obtain bonds, angles and dihedrals which involve only backbone beads
    # Generate these modes in terms of the modified indices of backbone beads 
    if args.wtraj:
        mol2bidx_keys=sorted(mol2bidx.keys())
        mod_idx=OrderedDict()
        c=1
        for m in range(len(mol2bidx_keys)):
            for k in mol2bidx[mol2bidx_keys[m]]:
                mod_idx[k]={'id':c,'mol':m}            
                c+=1
        mod_bonded={}
        for k in ['Bonds', 'Angles', 'Dihedrals']:
            mod_bonded[k]={'ids':[],'types':[]}
            bontype, cbontype={}, 1
            for i, b in enumerate(objects[k]['ids']):   
                if len([1 for _ in b if _ in b_ids])==len(b):
                    mod_bonded[k]['ids'].append([mod_idx[_-1]['id'] for _ in b])
                    if not objects[k]['types'][i] in bontype.keys():
                        bontype[objects[k]['types'][i]]=cbontype
                        cbontype+=1
                    mod_bonded[k]['types'].append(bontype[objects[k]['types'][i]])
            


    # CHECK IF ENERGY DIFFERENCE, REORANIZATION ENERGY AND ELECTRONIC COUPLING ARE FUNCTION OF RADIAL DISTANCE/ SEGMENT LENGTH
    # If yes, read the file and generate dictionaries Etype and Ltype with length of each site as keys and the energy values as values
    if args.Etype!='constant':
        Etype={}
        f=open(args.Etype,'r')
        for line in f:
            line=line.split()
            Etype[int(float(line[0]))]=float(line[1])
        f.close()
        print 'Using length dependent free energy difference for all rates...'
    else: 
        Etype=args.Etype
        print 'Using constant free energy difference for all rates...'

    # if args.Ltype!='constant':
    #     Ltype={}
    #     f=open(args.Ltype,'r')
    #     for line in f:
    #         line=line.split()
    #         Ltype[int(float(line[0]))]=float(line[1])
    #     f.close()
    #     print 'Using length dependent reorganization energy for all rates...'
    # else: 
    #     Ltype=args.Ltype
    #     print 'Using constant reorganization energy for all rates...'

    # And dictionary Vtype with V0, r0, beta as keys for electronic coupling V=V0*exp(-beta*(r-r0))
    if args.Vtype!='constant':
        Vtype=[float(_) for _ in args.Vtype.split()]
        Vtype={'V0':Vtype[0], 'r0':Vtype[1], 'beta':Vtype[2]}
        print 'Using distance dependent electronic coupling for intermolecular rates...'      
    else: 
        Vtype= args.Vtype
        print 'Using constant electronic coupling for intermolecular rates...'
    
    # Collect the indices belonging to charges, if distances wrt charges have to be calculated
    q_idx=[i for i,a in enumerate(objects["Atom_types"]) if a in qtypes]
    w_idx=[i for i,a in enumerate(objects["Atom_types"]) if a in wtypes]
        

    # START THE RUN!
    # Initialize variables
    t_hist= np.arange(0.0, args.tmax, args.dt)  # The entries are binned into these bins of time-stamps in ns
    count_hist, msd_hist= np.zeros(t_hist.shape),  np.zeros(t_hist.shape) # histogram of MSD  
    count_hist_small, msd_hist_small= np.zeros(t_hist.shape), np.zeros(t_hist.shape)  # histogram of MSD  
    count_hist_x, msd_hist_x= np.zeros(t_hist.shape),  np.zeros(t_hist.shape) # histogram of MSD  
    count_hist_x_small, msd_hist_x_small= np.zeros(t_hist.shape), np.zeros(t_hist.shape)  # histogram of MSD  
    count_hist_y, msd_hist_y= np.zeros(t_hist.shape),  np.zeros(t_hist.shape) # histogram of MSD  
    count_hist_y_small, msd_hist_y_small= np.zeros(t_hist.shape), np.zeros(t_hist.shape)  # histogram of MSD  
    count_hist_z, msd_hist_z= np.zeros(t_hist.shape),  np.zeros(t_hist.shape) # hist_zogram of MSD  
    count_hist_z_small, msd_hist_z_small= np.zeros(t_hist.shape), np.zeros(t_hist.shape)  # histogram of MSD  

    
    # For storing distribution of different quantities
    v_intra=[]
    nsites=[]
    Eelecs_intra,Eelecs_inter=[],[] # Store all electrostatic energies
    mean_Eelecs_inter=[] #BY
    distq_intra,distq_inter=[],[] # Distance from nearest charge
    distw_intra,distw_inter=[],[] # Distance from nearest water charged bead
    dEhomo_inter, dEnet_inter=[],[] # Electrostatic energy difference
    dEelec_intra, dEelec_inter=[],[] # Electrostatic energy difference
    lmbds_intra,lmbdnet_inter, lmbdi_inter, lmbde_inter=[],[],[],[] # Reorganization energies
    w12s_intra, w12s_inter=[],[] # Rates
    v_inter=[]
    dts_inter, neighbors_inter=[], []
    w12s_avg_intra, w12s_avg_inter, w12s_avg_all=[],[],[]
    Ls=[] # Length of sites
    Lmax=len(mol2bidx[mol2bidx.keys()[0]])
    mat_dEelecs, mat_dEs, mat_dEsnet=np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax)) # Store all electrostatic energie
    mat_lmbdsi, mat_lmbdso, mat_lmbdsnet=np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax))# Reorganization energies
    mat_vs, mat_w12s_inter, mat_w12s_intra=np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax))
    mat_ns_intra, mat_ns_inter=np.zeros((Lmax,Lmax)), np.zeros((Lmax,Lmax))
    im_ids,im_w12_inter, im_w12_intra=[],[],[]
    dt_ss, dt_s1, ngbr_ss, ngbr_s1, size_ss, N_ss=[], [], [], [], [], []  # Initialize for new distributions for supersites
    trap_sites, trap_sites_norm = [], []

    
    # Read the trajectory one snapshot at a time
    print 'Net time spent until the start of the main loop from the start of the script: {}'.format(time.time() - startTime0)
    startTime=time.time()
    
    # BY: accumulate connectivity over all frames and all KMC trials
    site_connectivity_sum = None
    n_connectivity_trials = 0
    network_count_sum = 0.0
    largest_network_sum = 0.0
    network_analysis_trials = 0
    network_detail_lines = []

    print '\n\nStarting the simulations!!\n'
    for tframe in range(args.first, args.last, args.every):
        print '*'*150
        print '*'*60 + '{:^30s}'.format('FRAME {}'.format(tframe)) + '*'*60
        print '*'*150
        print '\n'+'*'*150

        print '*'*150
        print 'COLLECT FRAMES FOR INTER-SITE (SUPER-SITE) RATES'
        print '*'*150

        md_trajs = [] 
        # We still need md_trajs for electrostatics averaging and inter-site rates below
        # IMPORTANT: must read ALL atoms because calc_elec_energy iterates over all atoms
        for md_traj in gen_lammps_frames(
                args.traj, unwrap=True, adj_list=adj_list, atom_ids=None,
                first=tframe, last=tframe+args.lastavg, every=args.everyavg):
            md_trajs.append(md_traj)
            print 'On timestep {}...'.format(md_traj['timestep'][0])

        print 'Collected {} frames for averaging.'.format(len(md_trajs))
        print 'Time taken to collect frames: {}'.format(time.time() - startTime)
        startTime = time.time()

    
        print '\n'+'*'*150
        print 'PREPROCESSING TO PREPARE ALL PARAMETERS NECESSARY FOR THE KMC RUNS'
        print '*'*150
        

        # For this workflow, we want hopping sites to be the sidechain super-sites (msites) constructed above.
        # Therefore we skip backbone-based delocalization clustering and use these super-sites directly for KMC.
        sites=deepcopy(msites)
        site2bidx_sites=list(site2bidx)
        bidx2site_sites=dict(bidx2site)

        print 'Now, we have {} sites (super-sites) for KMC.'.format(len(sites))
        if args.debug:
            print 'Super-sites: id\t\tno of beads\t\tbead indices'
            for i in range(len(sites)):
                print '{}\t\t{}\t\t{}'.format(i,len(sites[i]),', '.join([str(_) for _ in sites[i]]))
        # Store the length of each segment 
        L= [len(s) for s in sites]                


        if args.only_dist or args.debug: Ls+=L
        print 'The minimum, average and maximum length of site is {}, {} and {}.'.format(np.min(L), np.mean(L), np.max(L))
        nsites+=[len(sites)]
        md_traj=md_trajs[len(md_trajs)/2]
        # Parse geometry from lammps md_traj
        geo = np.array([[ md_traj['atoms']['x'][0][j], md_traj['atoms']['y'][0][j], md_traj['atoms']['z'][0][j] ] for j in range(len(md_traj['atoms']['x'][0])) ])

        print '\nCalculating distances between different sites...'
        # Obtain the Marcus rates for all pair of sites
        # Find the distance between CoM of different sites
        geo_com= np.zeros((len(sites),3))
        for i in range(len(sites)):  
            geo_com[i]=np.mean(geo[sites[i],:],axis=0)                        
        Rcom= cdist_mic(geo_com, md_traj['box'][0])
        # Calculate the minimum distance vectors for all pairs (as this remains unchanged even if the system orientation changes)
        dRcom= get_min_dRcom(geo_com,md_traj['box'][0])


        # Calculate the coupling between all pairs
        # V = coupling(geo,geo_com,sites,md_traj['box'][0],Vtype) 
        V = coupling(geo, geo_com, sites, md_traj['box'][0], Vtype, Vconst=args.V)
        #Vtemp=deepcopy(V)
        #v_all+=list(V[np.triu_indices(len(V),k=1)])
        

        print '\nCalculating intermolecular rates...'


        # Calculate electrostatic energy associated with hole on each site
        if args.Eelec:
            print 'Electrostatics is on. Calculating electrostatic energy required for rate...'
            Eelecs= calc_elec_energy(objects,md_trajs,sites,args.output,args.lammps,dielectric=args.dielectric,debug=args.debug)
            if args.only_dist or args.debug:
                #distq_inter+=list(np.min( cdist_mic_xy(geo_com,geo[q_idx,:],md_traj['box'][0]), axis=1))   # Minimum distances of these sites from charge
                for md_traj_temp in md_trajs:
                    # Parse geometry from lammps md_traj_temp
                    geo_temp = np.array([[ md_traj_temp['atoms']['x'][0][j], md_traj_temp['atoms']['y'][0][j], md_traj_temp['atoms']['z'][0][j] ] for j in range(len(md_traj_temp['atoms']['x'][0])) ])
                    #distq_inter+= [np.min( cdist_mic_xy(geo_temp[_,:],geo_temp[q_idx,:],md_traj_temp['box'][0])) for _ in sites]   # Minimum distances of these sites from charge
                    distq_inter+= [np.mean( np.min(cdist_mic_xy(geo_temp[_,:],geo_temp[q_idx,:],md_traj_temp['box'][0]), axis=1)) for _ in sites]   # Minimum distances of these sites from charge
                    #distw_inter+= [np.min( cdist_mic_xy(geo_temp[_,:],geo_temp[w_idx,:],md_traj_temp['box'][0])) for _ in sites]   # Minimum distances of these sites from charged water beads
                    distw_inter+= [np.mean( np.min(cdist_mic_xy(geo_temp[_,:],geo_temp[w_idx,:],md_traj_temp['box'][0]), axis=1)) for _ in sites]   # Minimum distances of these sites from charged water beads


                #Eelecs_inter+=list(Eelecs[len(Eelecs)/2])
                Eelecs_inter+=list(Eelecs.flatten())
                mean_Eelecs_inter+=list(np.mean(Eelecs, axis=1).flatten()) #BY 
            # dEelec=[np.array([_]).transpose() for _ in Eelecs]
            # dEelec=np.array([cdist(_,_,'minkowski', p=1.0) for _ in dEelec])
            # dEelec_inter+=[ _ for j in dEelec for _ in list(j[j!=0.0].flatten()) ]
            # print 'Minimum, average and maximum electrostatics energy difference {}, {} and {}'.format(np.min(dEelec[dEelec!=0.0]), np.mean(dEelec), np.max(dEelec))
        else:
            Eelecs=[[] for _ in md_trajs]
            
        w12=np.zeros((len(sites),len(sites)))
        w12_inter, lmbd_inter, warning=[],[],''
        lmat_temp_lmbd={}
        Etype_np=np.array([Etype[j] for j in sorted(Etype.keys())])
        # Ltype_np=np.array([Ltype[j] for j in sorted(Ltype.keys())])
        # imsites=[[],[]]
        # # Intramolecular (same-chain) neighboring sites are defined by consecutive backbone bead pairs.
        # # We map backbone bead -> super-site using bidx2site_sites, and only keep pairs where BOTH backbone beads have a sidechain super-site.
        # for (b1,b2) in consecutive_bpairs:
        #     if b1 in bidx2site_sites and b2 in bidx2site_sites:
        #         s1=bidx2site_sites[b1]
        #         s2=bidx2site_sites[b2]
        #         imsites[0].append(s1); imsites[1].append(s2)
        #         imsites[0].append(s2); imsites[1].append(s1)

        for j, Eelec in enumerate(Eelecs):  
            if len(Eelec) or j==0:
                # print j  # debug
                # print Eelec  # debug
                rate, lmbdi, lmbde, lmbdnet, dEhomo, dEelec, dEnet,   warning_temp = calc_rate_inter_parallel( Rcom, np.array(L), args.temp, rc=args.rc, E=args.E, lmbdi=args.lmbdi_const, loffset=args.loffset, dielectric=args.dielectric, cp=args.cp, V=V, Vintra=args.Vintra, Etype=Etype_np, Vtype=Vtype, phi=0.0, Eelec=Eelec )
                w12 += rate
                Vtemp = deepcopy(V)
                np.fill_diagonal(rate,0)
                np.fill_diagonal(lmbdi,np.nan)                
                np.fill_diagonal(lmbde,np.nan)                
                np.fill_diagonal(lmbdnet,np.nan)                
                np.fill_diagonal(dEhomo,np.nan)                
                np.fill_diagonal(dEelec,np.nan)                
                np.fill_diagonal(dEnet,np.nan)                
                np.fill_diagonal(Vtemp,np.nan)                
                
            
                w12_inter+=rate[rate!=0].flatten().tolist()
                lmbdnet_inter+=lmbdnet[~np.isnan(lmbdnet)].flatten().tolist()
                lmbdi_inter+=lmbdi[~np.isnan(lmbdi)].flatten().tolist()
                lmbde_inter+=lmbde[~np.isnan(lmbde)].flatten().tolist()
                dEhomo_inter+=dEhomo[~np.isnan(dEhomo)].flatten().tolist()
                dEelec_inter+=dEelec[~np.isnan(dEelec)].flatten().tolist()
                dEnet_inter+=dEnet[~np.isnan(dEnet)].flatten().tolist()
                v_inter+=Vtemp[~np.isnan(Vtemp)].flatten().tolist()
                if warning_temp: warning=warning_temp  

                # print "RATE DEBUG"
                # for s1 in range(len(sites)):
                #     for s2 in range(len(sites)):
                #         print s1,s2,lmbdnet[s1,s2],dEnet[s1,s2],Vtemp[s1,s2],rate[s1,s2]



        # Average rates if electrostatics is on
        if args.Eelec: 
            w12/=len(md_trajs)
            w12[w12<10]=0.0 # Remove any rates less than E10 s-1 (10 ns-1)

        # print "RATE DEBUG"
        # for s1 in range(len(sites)):
        #     for s2 in range(len(sites)):
        #         print s1,s2,w12[s1,s2]
        # quit()

        w12s_avg_inter+=list(w12[w12!=0.0].flatten())        
        # Calculate rates
        # for s1 in range(len(sites)):
        #     for s2 in range(len(sites)):                    
        #         if bidx2mol[site2bidx_sites[s1]]==bidx2mol[site2bidx_sites[s2]] and abs(s1-s2)==1: # consecutive intramolecular sites
        #             if s1>s2: w12[s1,s2]=w12_intra_strd[b_idx.index(sites[s2][-1]),b_idx.index(sites[s1][0])]
        #             if s1<s2: w12[s1,s2]=w12_intra_strd[b_idx.index(sites[s1][-1]),b_idx.index(sites[s2][0])]


        w12s_avg_all+=list(w12[w12!=0.0].flatten())
        w12s_inter+=w12_inter
        #lmbds_inter+=lmbd_inter

        #print 'The minimum, average and maximum reorganization energies are {:5.2f}, {:5.2f} and {:5.2f} eV'.format(np.min(lmbd_inter), np.mean(lmbdnet_inter), np.max(lmbd_inter))            
        if len (w12s_avg_inter):
            print 'The minimum, average and maximum averaged intermolecular rates are {:.2e}, {:.2e} and {:.2e} 1/s'.format(np.min(w12_inter)*(10.0**9.0), np.mean(w12_inter)*(10.0**9.0), np.max(w12_inter)*(10.0**9.0))
        else:
            print 'All  intermolecular rates are zero :('
        if warning: print warning


        # # XXX MAT ENTRIES XXX
        # for i,s in enumerate(sites[:-1]):
        #     s1, s2 = deepcopy(sites[i]), deepcopy(sites[i+1])
        #     if bidx2mol[s1[0]]==bidx2mol[s2[0]]: # Part of the same molecule
        #         im_ids.append((s1[-1],s2[0]))
        #         im_w12_intra.append(w12_intra_strd[b_idx.index(s1[-1]),b_idx.index(s2[0])])
        #         im_w12_inter.append(w12[i,i+1])
        #         im_ids.append((s2[0],s1[-1]))
        #         im_w12_intra.append(w12_intra_strd[b_idx.index(s1[-1]),b_idx.index(s2[0])])
        #         im_w12_inter.append(w12[i+1,i])
        #         mat_w12s_intra[L[i]-1,L[i+1]-1]+=w12_intra_strd[b_idx.index(s1[-1]),b_idx.index(s2[0])]
        #         mat_w12s_intra[L[i+1]-1,L[i]-1]+=w12_intra_strd[b_idx.index(s1[-1]),b_idx.index(s2[0])]
        #         mat_ns_intra[L[i+1]-1,L[i]-1]+=1
        #         mat_ns_intra[L[i]-1,L[i+1]-1]+=1
        # count1, count2=0, 0
        # cl1, cl2 = 0, 0
        # ids=[]
        # for s1 in range(len(sites)):
        #     for s2 in range(len(sites)):
        #         if s1==s2: continue
        #         if bidx2mol[site2bidx_sites[s1]]==bidx2mol[site2bidx_sites[s2]] and abs(s1-s2)==1: continue # consecutive intramolecular sites                
        #         if args.Eelec:
        #             mat_dEelecs[L[s1]-1,L[s2]-1]+=np.abs(Eelecs[-1][s2]-Eelecs[-1][s1]) # Using only last value
        #             mat_dEs[L[s1]-1,L[s2]-1]+=np.abs(Etype[L[s2]]-Etype[L[s1]])
        #             mat_dEsnet[L[s1]-1,L[s2]-1]+=np.abs(Eelec[s2]-Eelec[s1]+Etype[L[s2]]-Etype[L[s1]] )
        #         else:
        #             mat_dEs[L[s1]-1,L[s2]-1]+=np.abs(Etype[L[s2]]-Etype[L[s1]])
        #             mat_dEsnet[L[s1]-1,L[s2]-1]+=np.abs(Etype[L[s2]]-Etype[L[s1]] )

        #         mat_lmbdsi[L[s1]-1,L[s2]-1]+=(Ltype[L[s2]]+Ltype[L[s1]])/2.0
        #         mat_lmbdso[L[s1]-1,L[s2]-1]+=lmbd[s1,s2]-(Ltype[L[s2]]+Ltype[L[s1]])/2.0 
        #         mat_lmbdsnet[L[s1]-1,L[s2]-1]+=lmbd[s1,s2]             
        #         mat_vs[L[s1]-1,L[s2]-1]+=V[s1,s2]
        #         mat_w12s_inter[L[s1]-1,L[s2]-1]+=w12[s1,s2]
        #         mat_ns_inter[L[s1]-1,L[s2]-1]+=1
             

        # Remove the sites from which transition cannot happen
        transition=list(range(len(sites)))
        notransition=[j for j in range(len(w12)) if np.sum(w12[j,:])==0.0] 
        if len(notransition):
            if args.excl_type==1:
                print 'Removing any possible trap regions (sites from which no transition can occur)...'
                remove=deepcopy(notransition)
                while len(remove):
                    w12[:,remove]=0.0
                    remove=[j for j in range(len(w12)) if np.sum(w12[j,:])==0.0 and j not in notransition]  
                    notransition+=remove
                    if len(sites)<=len(notransition):
                        print 'All sites were removed in this process. Trap regions cannot be avoided for this system. Exiting...'
                        quit()
            elif args.excl_type==2: 
                print 'The trap sites will be included in the KMC run, and the site will be assumed to be trapped till the end of KMC run (time= {}) if it enters one of these sites'.format(args.kmc_tmax)
            # Exclude these sites from sites which are a part of KMC run
            transition=sorted(list(set(transition).difference(set(notransition))))
            print '{} total sites from which no transition can happen {}'.format(len(notransition),' '.join([str(_) for _ in  notransition]))
        else:
            print 'Transition from all sites can happen'


        print 'Indentifying electrically connected clusters and isolated sites...'
        # Check if the entire cluster is connected
        w12_adj= 1*(w12>0)
        #elec_network=find_subgraphs(w12_adj, axis=3)
        #elec_network=cluster_sites(w12_adj)
        w12_adj_n=w12_adj+ np.eye(len(w12_adj))
        for j in range(len(w12_adj)-2):
             w12_adj_n= np.dot(w12_adj_n, w12_adj)
             w12_adj_n=(w12_adj_n>0)*1

        elec_network=find_subgraphs( 1*(w12_adj_n>0) , axis=2)
        elec_network=[[k for k in j if k not in notransition] for j in elec_network]
        elec_network=[list(set(j)) for j in elec_network if len(j)]



        if len(elec_network)>1:
            print 'The entire polymer system is not connected (excluding any sites from which transition cannot happen) but broken into {} clusters with {} sites'.format(len(elec_network),' '.join([str(tuple(j)) for j in elec_network]))
        else:
            print 'The entire polymer system is connected (excluding any sites from which transition cannot happen)'



        print '\nCreate generator and transition rate matrix...'
        # Generator matrix for all sites
        q12=deepcopy(w12)
        for s1 in range(len(sites)): 
            q12[s1,s1]=-1*sum(q12[s1,:])

        # Transition rate matrix for all sites
        jsum=sum(w12,axis=1)
        jsum[jsum==0.0]=1.0
        j12=(w12.transpose()/jsum).transpose()

        # Add the timesteps and number of neighbors to the distributions
        if args.only_dist or args.debug:
            dts_inter+=[log(-1.0/q12[s,s]*(10.0**-9.0))/log(10.0) for s in range(len(sites)) if q12[s,s] ]
            neighbors_inter+=[float(len(np.where(j12[s,:]>0)[0])) for s in range(len(sites)) if q12[s,s] ]
        

        # Cluster the sites into superstates using the aggregation algorithm 5.2
        print "Also calculating the trap sites/ superstates by using clustering algorithm..."
        #supersites=cluster_sites(j12,alpha=0.2,beta=0.2,gamma=args.gamma,toignore=notransition)
        # No clustering: each site is its own supersite
        supersites = [[i] for i in range(j12.shape[0])]

        if args.debug:
            print '\nClustered supersites:'
            print 'id\tsite ids\tatom ids'
            for i in range(len(supersites)):
                print '{}\t{}\t\t\t{}'.format(i,', '.join([str(_) for _ in supersites[i]]), ', '.join([str(_) for j in supersites[i] for _ in sites[j]]) )
                # if len(supersites[i])>1:
                #     print 'max rates:',np.max(w12[supersites[i],:][:,supersites[i]]),
                #     print 'min rates:',np.min(w12[supersites[i],:][:,supersites[i]][w12[supersites[i],:][:,supersites[i]]!=0.0]),
                #     print 'molecule ids::',', '.join([str(bidx2mol[site2bidx_sites[_]]) for _ in supersites[i]] )
            print


        print "Identifying the outer and adjacent sites for all supersites..."
        # For each super states identify its outer states and adjacent states
        # adjacent states: states with one-step nonzero transition probablity into the superstate
        # outer states of superstate: states within superstate with one-step nonzero transition probablity into its adjacent states
        # outersites: all adjacent and outer states, as transfer occurs through these
        # adj={} # Stores adjacent sites of each superstate
        # outer={} # Stores the outer sites of each superstate
        # outer2ss={} # Store the idx of the supersite of which the site is a part
        # outersites_all=[] # Stores all ajacent and outer states through which transfer happens
        # for i in range(len(supersites)):
        #     for s in supersites[i]:  
        #         outer2ss[s]=i  # Store the idx of the supersite of which the site is a part

        #     if len(supersites[i])>1: # It is actually a supersite which contains multiple sites
        #         adj[i]=[]
        #         outer[i]=[]
        #         for s in supersites[i]:
        #             adj[i]+=[_ for _,v in enumerate(j12[s,:]) if v and _ not in supersites[i]] # Store its adjacent sites
        #         adj[i]=list(set(adj[i]))
        #         outersites_all+=adj[i]  # Add it to all outer sites
        #         for s in supersites[i]: 
        #             if len([_ for _,v in enumerate(j12[s,:]) if v and _ in adj[i] ]): 
        #                 if i not in outer.keys(): # Store the outer site 
        #                     outer[i]=[s]
        #                 else:
        #                     outer[i].append(s)
        #                 outersites_all+=[s] # Add it to all outer sites
        #     else:
        #         s=supersites[i][0]
        #         if len([_ for _,v in enumerate(j12[s,:]) if v ]):
        #             outersites_all+=supersites[i] # Add it to all outer sites

        adj = {}
        outer = {}

        # No clustering => identity mapping
        outer2ss = {i: i for i in range(j12.shape[0])}

        # Allow KMC to start from any site that isn't explicitly ignored
        outersites_all = [i for i in range(j12.shape[0]) if i not in notransition]

        if len(outersites_all) == 0:
            raise RuntimeError("outersites_all is empty (all sites ended up in notransition). No KMC start sites exist.")

        # The transfer will happen only through outer states
        outersites_all=list(set(outersites_all))
        # For the set of all outer states, store the transition rate matrix to simulate these trajectories
        js12=j12[outersites_all,:][:,outersites_all] 
        jssum=sum(js12,axis=1)
        jssum[jssum==0.0]=1.0
        js12=(js12.transpose()/jssum).transpose()
        qs12=q12[outersites_all,:][:,outersites_all] 
        np.fill_diagonal(qs12,0.0)
        for s in range(len(qs12)): 
            qs12[s,s]=-1*np.sum(qs12[s,:])


        # If the charge enters a trap/super state, its exit to a neighboring state is modelled as an MJP with the neighboring states acting as absorbing states
        # Also for all super states and their corresponding neighboring states, precomute the transition rate matrix for this MJP, 
        trap_sites+=[0.0]
        N_ss+=[len(supersites)]
        print "Calculate the matrices transition behaviour for MJP associated with them..."
        jmjp, pmjp, qmjp, tmjp = {}, {}, {}, {}
        for i in range(len(supersites)):
            nT = len(supersites[i])
            size_ss+=[nT]
            if nT>1:
                nA = len(adj[i])
                if nA==0:
                    pmjp[i]=np.array([])
                    tmjp[i]=np.array([])
                    trap_sites[-1]+=1
                    continue
                I= eye(nA,nA)
                Z= zeros((nA,nT))
                R= w12[supersites[i],:][:,adj[i]]
                T= w12[supersites[i],:][:,supersites[i]]

                jmjp[i]= concatenate( ( concatenate((I,Z),axis=1), concatenate((R,T),axis=1) ),  axis=0)
                tau=1.0/sum(jmjp[i],axis=1)
                jmjp[i]= (jmjp[i].transpose()*tau).transpose()                
                #print jmjp[i][-1,:]

                # the corresponding probability matrix 
                pmjp[i]=np.linalg.solve(eye(nT,nT) - jmjp[i][-nT:,:][:,-nT:],jmjp[i][-nT:,:][:,0:nA])


                # and the conditional absorption times 
                qmjp[i]= concatenate( ( zeros(( nA, nA+ nT )), concatenate((w12[supersites[i],:][:,adj[i]], w12[supersites[i],:][:,supersites[i]]),axis=1)),  axis=0)
                for j in range(len(qmjp[i])): 
                    qmjp[i][j,j]=-1*sum(qmjp[i][j,:])
                try:
                    tmjp[i]=np.linalg.solve(np.matmul(qmjp[i][-nT:,:][:,-nT:],qmjp[i][-nT:,:][:,-nT:]),qmjp[i][-nT:,:][:,0:nA])
                    tmjp[i]=tmjp[i]/pmjp[i]
                except:
                    tmjp[i]=np.ones((nT,nA))
                    for j in range(len(adj[i])):
                        if np.max(qmjp[i][-nT:,j])<100:
                            tmjp[i][-nT:,j]=-1
                        else:
                            tmjp[i][-nT:,j]=0.01

                    
                while ( ((np.log(tmjp[i]*(10.0**-9.0))/np.log(10))>-10.0).any() or (tmjp[i]<0).any()) and len(adj[i]):
                    if ((np.log(tmjp[i]*(10.0**-9.0))/np.log(10))>-10.0).any():
                        unfeasible_adj=list(set(np.where(((np.log(tmjp[i]*(10.0**-9.0))/np.log(10))>-10.0))[1]))
                        # print 'Warning! Supersite {} with sites {} had a timestep longer than 100ps  to site(s) {}, removing transfer to these sites...'.format(i,supersites[i],unfeasible_adj)
                    else:
                        unfeasible_adj=list(set(np.where(tmjp[i]<0)[1]))
                        # print 'Warning! Supersite {} with sites {} had a negative timestep to site(s) {}, removing transfer to these sites...'.format(i,supersites[i],unfeasible_adj)


                    # print 'Debug report:'
                    # print 'Supersite ids: ', supersites[i]
                    # print 'Old information:'
                    # print 'Adjacent site ids: ', adj[i]
                    # print "Site\tRates from it\tRates to it\tTimesteps from it\tTimestep from it\tProbablity of escape"
                    # for j1 in range(len(tmjp[i])):
                    #     print supersites[i][j1], [(j2,w12[supersites[i][j1],j2]) for j2 in range(len(w12[supersites[i][j1],:])) if w12[supersites[i][j1],j2]],  [(j2,w12[j2,supersites[i][j1]]) for j2 in range(len(w12[:,supersites[i][j1]])) if w12[j2,supersites[i][j1]]], tmjp[i][j1], pmjp[i][j1]
                    # print

                    unfeasible_adj= [adj[i][_] for _ in unfeasible_adj]

                    for j in unfeasible_adj:
                        # print adj[i]
                        adj[i].remove(j)

                    nA = len(adj[i])
                    if nA==0:
                        pmjp[i]=np.array([])
                        tmjp[i]=np.array([])
                        trap_sites[-1]+=1
                        # print '\nNew information:'
                        # print 'No adjacent sites!'
                        # print
                        continue
      
                    I= eye(nA,nA)
                    Z= zeros((nA,nT))
                    R= w12[supersites[i],:][:,adj[i]]
                    T= w12[supersites[i],:][:,supersites[i]]
    
                    jmjp[i]= concatenate( ( concatenate((I,Z),axis=1), concatenate((R,T),axis=1) ),  axis=0)
                    tau=1.0/sum(jmjp[i],axis=1)
                    jmjp[i]= (jmjp[i].transpose()*tau).transpose()                
    
                    # the corresponding probability matrix 
                    pmjp[i]=np.linalg.solve(eye(nT,nT) - jmjp[i][-nT:,:][:,-nT:],jmjp[i][-nT:,:][:,0:nA])
    
    
                    # and the conditional absorption times 
                    qmjp[i]= concatenate( ( zeros(( nA, nA+ nT )), concatenate((w12[supersites[i],:][:,adj[i]], w12[supersites[i],:][:,supersites[i]]),axis=1)),  axis=0)
                    for j in range(len(qmjp[i])): 
                        qmjp[i][j,j]=-1*sum(qmjp[i][j,:])
                    try:
                        tmjp[i]=np.linalg.solve(np.matmul(qmjp[i][-nT:,:][:,-nT:],qmjp[i][-nT:,:][:,-nT:]),qmjp[i][-nT:,:][:,0:nA])
                        tmjp[i]=tmjp[i]/pmjp[i]
                    except:
                        tmjp[i]=np.ones((nT,nA))
                        for j in range(len(adj[i])):
                            if np.max(qmjp[i][-nT:,j])<100:
                                tmjp[i][-nT:,j]=-1
                            else:
                                tmjp[i][-nT:,j]=0.01


                        
                    # print 'New information:'
                    # print 'Adjacent site ids: ', adj[i]
                    # if len(adj[i]):
                    #     print "Site\tRates from it\tRates to it\tTimesteps from it\tTimestep from it\tProbablity of escape"
                    #     for j1 in range(len(tmjp[i])):
                    #         print supersites[i][j1], [(j2,w12[supersites[i][j1],j2]) for j2 in range(len(w12[supersites[i][j1],:])) if w12[supersites[i][j1],j2]],  [(j2,w12[j2,supersites[i][j1]]) for j2 in range(len(w12[:,supersites[i][j1]])) if w12[j2,supersites[i][j1]]], tmjp[i][j1], pmjp[i][j1]
                    #     print
                    

                if (tmjp[i]<0).any():
                    print 'Error! The timestep is still negative...'
                    quit()


                dt_ss+=(log(tmjp[i]*(10.0**-9.0))/log(10.0)).flatten().tolist()
                ngbr_ss+=[len(_[_>0.0]) for _ in pmjp[i] ]
            else:
                s=supersites[i][0]
                if q12[s,s]:
                    dt_s1+=[log(-1.0/q12[s,s]*(10.0**-9.0))/log(10.0)]
                    ngbr_s1+=[float(len(np.where(j12[s,:]>0)[0]))]
                else:
                    trap_sites[-1]+=1

        trap_sites_norm+=[trap_sites[-1]/len(supersites)]
        print 'Time taken to precalculate these quantities: {}\n'.format(time.time()- startTime)
        startTime=time.time()


        # Check if KMC runs are to be performed
        if args.only_dist:
            print "Only distributions are to be calculated, not performing KMC runs..."
            print 'Net time spent until the end of this cycle from the start of the script: {}'.format(time.time() - startTime0)
            print '\n'+'*'*150
            print '*'*50+'{:^50s}'.format('END OF KMC RUN FOR THIS FRAME')+'*'*50
            print '*'*150+'\n'
            continue


        print '\n'+'*'*150
        print "SIMULATING THE ACTUAL KMC TRAJECTORY"
        print '*'*150+'\n'
        #trajs= kmc(sites, j12, q12, geo_com, md_traj['box'][0], N=args.Nkmc, tmax=args.kmc_tmax, debug=args.debug, startTime0=startTime0, output=args.output, timestep=md_traj['timestep'][0], include=transition)
        trajs=[]
        refresh_tabs_all=[]
        refresh_geo_com=[]

        # BY: binary connectivity matrix from already-calculated rates
        # conn[s,j] = 1 if site s can hop to site j, else 0
        conn = (w12 > 0.0).astype(float)
        np.fill_diagonal(conn, 0.0)

        for i in range(args.Nkmc):
            print 'On KMC trial {}...'.format(i)
            np.random.seed((i+1)*1000*args.seed)

            # --- Randomly disable a fraction of KMC sites (outer sites) --- #BY
            n_total = len(outersites_all)
            n_keep  = int(np.floor(args.keep_frac * n_total))

            if n_keep <= 0:
                # No active sites -> KMC can't run
                print "keep_frac={} disables all sites; skipping this KMC trial.".format(args.keep_frac)
                continue

            if n_keep >= n_total:
                active_sites = list(outersites_all)
                inactive_set = set()
            else:
                active_sites = list(np.random.choice(outersites_all, size=n_keep, replace=False)) #replace=False means once a site is selected, it cannot be selected again
                inactive_set = set(outersites_all) - set(active_sites)

            rate_thresh_per_ns = args.network_rate_thresh * 1.0e-9

######################PRINTS##############################
            n_edges = 0

            for s in active_sites:
                for j in active_sites:
                    if s != j and w12[s,j] >= rate_thresh_per_ns:
                        n_edges += 1

            print "Threshold:", args.network_rate_thresh
            print "Edges above threshold:", n_edges

            rates = []

            for s in active_sites:
                for j in active_sites:
                    if s != j:
                        rates.append(w12[s,j])

            rates = np.array(rates)

            print "min rate:", np.min(rates)
            print "max rate:", np.max(rates)

            for p in [50,75,90,95,99,99.9]:
                print p, np.percentile(rates,p)
######################PRINTS##############################
            networks = find_rate_networks(
                w12,
                active_sites,
                sites,
                objects,
                site2bidx_sites,
                rate_thresh_per_ns
            )

            network_count_sum += len(networks)
            largest_network_sum += max([len(n["site_ids"]) for n in networks])
            network_analysis_trials += 1

            network_detail_lines.append(
                "frame {} kmc_trial {} number_of_networks {}\n".format(
                    md_traj['timestep'][0], i, len(networks)
                )
            )

            for n in networks:
                network_detail_lines.append(
                    "network {} n_sites {} sites {} molecule_ids {} atom_ids {} anchor_atom_ids {} anchor_molecule_ids {}\n".format(
                        n["network_id"],
                        len(n["site_ids"]),
                        " ".join([str(x) for x in n["site_ids"]]),
                        " ".join([str(x) for x in n["mol_ids"]]),
                        " ".join([str(x) for x in n["atom_ids"]]),
                        " ".join([str(x) for x in n["anchor_atom_ids"]]),
                        " ".join([str(x) for x in n["anchor_mol_ids"]])
                    )
                )

            # BY: initialize once, using total number of sites
            if site_connectivity_sum is None:
                site_connectivity_sum = np.zeros(len(w12))

            # BY: fast matrix-based connectivity count after keep_frac
            active_mask = np.zeros(len(w12), dtype=float)
            active_mask[active_sites] = 1.0

            # usable_counts[s] = number of active sites j with nonzero rate from s to j
            usable_counts = conn.dot(active_mask)

            # removed keep_frac sites contribute 0
            usable_counts *= active_mask

            site_connectivity_sum += usable_counts
            n_connectivity_trials += 1

            refresh_tabs=[]
            kmc_complete, traj=0,{}
            geo_com_new=deepcopy(geo_com)
            dRcom_new=deepcopy(dRcom)
            while kmc_complete==0:
                # Run KMC
                #traj, kmc_complete= kmc(traj, sites, j12, q12, geo_com_new, dRcom, dRcom_new, md_traj['box'][0], tmax=args.kmc_tmax, debug=args.debug, startTime0=startTime0, output=args.output, timestep=md_traj['timestep'][0], include=transition, box_cap=args.box_cap, dist_cap=args.dist_cap)
                traj, kmc_complete= kmc(traj, sites, supersites, outersites_all, outer2ss, adj, js12, qs12, pmjp, tmjp, geo_com_new, dRcom, dRcom_new, md_traj['box'][0], tmax=args.kmc_tmax, debug=args.debug, startTime0=startTime0, output=args.output, timestep=md_traj['timestep'][0], box_cap=args.box_cap, dist_cap=args.dist_cap,toignore=notransition, active_sites=active_sites, inactive_set=inactive_set)
                # print 'Refresh_tabs: {}'.format(refresh_tabs)
                # Site refresh
                if not kmc_complete: 
                    # Rates should remain same
                    # Should return a new site randomly chosen
                    # Should return a new geometry where the charge is at the position where it left off but the new site is at this postion and the orientation of the box is different
                    # Modifies the trajectory to reflect the new site
                    # Appends this point and the changes in refresh_tab
                    # traj, geo_com_new, dRcom_new, refresh_tabs= site_refresh(traj, geo_com, dRcom, refresh_tabs, outersites_all) # Trajectory, rotated configurations, refresh info, include
                    traj, geo_com_new, dRcom_new, refresh_tabs = site_refresh(traj, geo_com, dRcom, refresh_tabs, active_sites)

                if i==0: refresh_geo_com.append(geo_com_new) # Store all the refreshed geometries only for first KMC simulation 
                
            refresh_tabs_all.append(refresh_tabs)
            trajs.append(traj)    

            #print 'Time taken to perform this KMC run: {}\n'.format(time.time()- startTime)
            #startTime=time.time()
            
        
        if args.Nkmc:
    
            print 'Time taken to perform KMC runs: {}'.format(time.time()- startTime)
            startTime=time.time()
            
            print '\n'+'*'*150
            print 'POSTPROCESSING THE KMC TRAJECTORIES'
            print '*'*150+'\n'
    
            # # Find the transitions with slowest rates
            # kmc_ratest=[w12[trajs[0]['S'][j],trajs[0]['S'][j+1]] for j in range(len(trajs[0]['S'])-1) ]
            # min_ratest=sorted(set(kmc_ratest))[:10]
            # occurences_ratest=[ sum([1 for _ in kmc_ratest if _==r]) for r in min_ratest ]
            # pairs_ratest=[ [ (trajs[0]['S'][j],trajs[0]['S'][j+1]) for j in range(len(kmc_ratest)) if r== kmc_ratest[j] ][0] for r in min_ratest ]
            # for i in range(10):
            #     print min_ratest[i], occurences_ratest[i], pairs_ratest[i]
            # print 
            # max_ratest=sorted(set(kmc_ratest))[-10:]
            # occurences_ratest=[ sum([1 for _ in kmc_ratest if _==r]) for r in max_ratest ]
            # pairs_ratest=[ [ (trajs[0]['S'][j],trajs[0]['S'][j+1]) for j in range(len(kmc_ratest)) if r== kmc_ratest[j] ][0] for r in max_ratest ]
            # for i in range(10):
            #     print max_ratest[i], occurences_ratest[i], pairs_ratest[i]
            # print 
            # print w12[16,44], V[16,44]
            # quit()
    
            # UPDATE MSD TO AVOID MEMORT OVERFLOW
            # Histogram the MSD and Time into time bins
            print 'Updating the MSD distribution...'   
            # for i in range(len(trajs)):  
            #     with open(args.output+'msd_no_avg_t{}_kmc{}.txt'.format(md_traj['timestep'][0],i),'w') as f:
            #         f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
            #         for j in range(len(trajs[i]['t'])):
            #             f.write("{:<25.8f} {:< 25.8f}\n".format(trajs[i]['t'][j], trajs[i]['msd'][j]))
            #     f.close()
            
            for i in range(len(trajs)):
                count_hist_idv,msd_hist_idv=hist_msd(trajs[i]['t'],trajs[i]['msd'],t_hist,[],[])
                # count_hist_x_idv,msd_hist_x_idv=hist_msd(trajs[i]['t'],trajs[i]['msd_x'],t_hist,[],[])
                # count_hist_y_idv,msd_hist_y_idv=hist_msd(trajs[i]['t'],trajs[i]['msd_y'],t_hist,[],[])
                # count_hist_z_idv,msd_hist_z_idv=hist_msd(trajs[i]['t'],trajs[i]['msd_z'],t_hist,[],[])
    
                if len(set(trajs[i]['S'][-int(len(trajs[i]['S'])/2.0):]))>10: #BY changed from 20 to 10 to see if msd.txt will be packekeedddd!
                    print 'For KMC trajectory {}, the number of sites visited (in the last 1/2 of the trajectory) are greater than 20, parsing the MSDin the main MSD...'.format(i)
                    count_hist+=count_hist_idv
                    msd_hist+=msd_hist_idv
                    # count_hist_x+=count_hist_x_idv
                    # msd_hist_x+=msd_hist_x_idv
                    # count_hist_y+=count_hist_y_idv
                    # msd_hist_y+=msd_hist_y_idv
                    # count_hist_z+=count_hist_z_idv
                    # msd_hist_z+=msd_hist_z_idv
    
                else:
                    print 'For KMC trajectory {}, the number of sites visited (in the last 1/2 of the trajectory) are lesser than 20, parsing the MSD in a separate MSD reserved for such small clusters...'.format(i)
                    count_hist_small+=count_hist_idv
                    msd_hist_small+=msd_hist_idv
                    # count_hist_x_small+=count_hist_x_idv
                    # msd_hist_x_small+=msd_hist_x_idv
                    # count_hist_y_small+=count_hist_y_idv
                    # msd_hist_y_small+=msd_hist_y_idv
                    # count_hist_z_small+=count_hist_z_idv
                    # msd_hist_z_small+=msd_hist_z_idv
    
    
                if args.debug:
                    print 'Writing individual MSDs...'
                    count_hist_idv[count_hist_idv==0.0]=1.0
                    msd_hist_idv/=count_hist_idv
                    msd_hist_idv=shave_zeros_msd(msd_hist_idv)
                    with open(args.output+'msd_t{}_kmc{}.txt'.format(md_traj['timestep'][0],i),'w') as f:
                        f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
                        for j in range(len(msd_hist_idv)):
                            f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_idv[j],count_hist_idv[j]))
                    f.close()
                    # count_hist_x_idv[count_hist_x_idv==0.0]=1.0
                    # msd_hist_x_idv/=count_hist_x_idv
                    # msd_hist_x_idv=shave_zeros_msd(msd_hist_x_idv)
                    # with open(args.output+'msd_x_t{}_kmc{}.txt'.format(md_traj['timestep'][0],i),'w') as f:
                    #     f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
                    #     for j in range(len(msd_hist_x_idv)):
                    #         f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_x_idv[j],count_hist_x_idv[j]))
                    # f.close()
                    # count_hist_y_idv[count_hist_y_idv==0.0]=1.0
                    # msd_hist_y_idv/=count_hist_y_idv
                    # msd_hist_y_idv=shave_zeros_msd(msd_hist_y_idv)
                    # with open(args.output+'msd_y_t{}_kmc{}.txt'.format(md_traj['timestep'][0],i),'w') as f:
                    #     f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
                    #     for j in range(len(msd_hist_y_idv)):
                    #         f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_y_idv[j],count_hist_y_idv[j]))
                    # f.close()
                    # count_hist_z_idv[count_hist_z_idv==0.0]=1.0
                    # msd_hist_z_idv/=count_hist_z_idv
                    # msd_hist_z_idv=shave_zeros_msd(msd_hist_z_idv)
                    # with open(args.output+'msd_z_t{}_kmc{}.txt'.format(md_traj['timestep'][0],i),'w') as f:
                    #     f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
                    #     for j in range(len(msd_hist_z_idv)):
                    #         f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_z_idv[j],count_hist_z_idv[j]))
                    # f.close()
    
    
            print 'Time taken to histogram all MSDs : {}\n'.format(time.time()- startTime)
            startTime=time.time()
    
    
            if args.debug:
                # Histogram the sites visited if  debug is on
                print '\nParsing the sites visited...'
                sites_visited_sm=len([ 1 for i in range(len(trajs)) for j in range(len(trajs[i]['S'])-1) if bidx2mol[site2bidx_sites[trajs[i]['S'][j]]]==bidx2mol[site2bidx_sites[trajs[i]['S'][j+1]]]])
                sites_visited_dm=len([ 1 for i in range(len(trajs)) for j in range(len(trajs[i]['S'])-1) if bidx2mol[site2bidx_sites[trajs[i]['S'][j]]]!=bidx2mol[site2bidx_sites[trajs[i]['S'][j+1]]]])
                sites_visited_t=len([ 1 for i in range(len(trajs)) for j in range(len(trajs[i]['S']))])
                print "The absolute number and the relative transitions of intramolecular type are {} and {} respectively".format(sites_visited_sm, sites_visited_sm/float(sites_visited_t))
                print "The absolute number and the relative transitions of intermolecular type are {} and {} respectively".format(sites_visited_dm, sites_visited_dm/float(sites_visited_t))
                # Frequency of sites visited
                # Parse the sites visited during a trajector
                sites_visited=[float(j) for i in range(len(trajs)) for j in trajs[i]['S']]
                sites_visited_bins, sites_visited_hist= calc_hist(sites_visited,h_step=1.0, h_min=-0.5, h_max=len(sites)-0.5)
                f=open(args.output+'dist_sites.txt', 'w')
                for j in range(len(sites_visited_bins)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(sites_visited_bins[j],sites_visited_hist[j]))
                f.close()
                
                # Time vs site id trajectory 
                for i in range(len(trajs)):
                    f=open(args.output+'site_traj_t{}_kmc{}.txt'.format(md_traj['timestep'][0],i), 'w')
                    for j in range(len(trajs[i]['S'])):
                        f.write("{:<25.8f} {:< 25.8f}\n".format(trajs[i]['T'][j], trajs[i]['S'][j]))
                    f.close()
               
                # No of sites visited in last 1/4th of the trajetory
                nsites_tquart=[len(set(trajs[i]['S'][-len(trajs[i]['S'])/4:])) for i in range(len(trajs))]
                nsites_tquart_bins, nsites_tquart_hist= calc_hist(nsites_tquart,h_step=1.0, h_min=-0.5, h_max=len(sites)-0.5)
                f=open(args.output+'dist_nsites_tquart.txt', 'w')
                for j in range(len(nsites_tquart_bins)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(nsites_tquart_bins[j],nsites_tquart_hist[j]))
                f.close()
                
                print 'Time taken to histogram these quantities from KMC trajectories : {}\n'.format(time.time()- startTime)
                startTime=time.time()
    
    
            # WRITE TRAJECTORY AND DATA FILE IF ASKED
            if args.wtraj:
                print '\nWriting the KMC trajectory to LAMMPS format...'
                for tidx in range(args.Nkmc):
                    md_traj['atoms']['type_mod']=[deepcopy(md_traj['atoms']['type'][0])]
                    # just unwrapped charge
                    f=open('{}{}_kmc{}_charge.lammpstrj'.format(args.output,md_traj['timestep'][0],tidx),'w')
                    cap=20000
                    geo_current, geo_com_current= deepcopy(geo), deepcopy(geo_com)
                    for j in range(0,min([len(trajs[tidx]['T']),cap])):                
                        f.write('ITEM: TIMESTEP\n' + str(trajs[tidx]['T'][j]) + '\n')
                        f.write('ITEM: NUMBER OF ATOMS\n1\n')
                        f.write('ITEM: BOX BOUNDS pp pp pp\n' + str(md_traj['box'][0][0][0]) + ' ' + str(md_traj['box'][0][0][1]) + '\n' + str(md_traj['box'][0][1][0]) + ' ' + str(md_traj['box'][0][1][1]) + '\n' + str(md_traj['box'][0][2][0]) + ' ' + str(md_traj['box'][0][2][1]) + '\n')
                        f.write('ITEM: ATOMS id type x y z mol\n')
                        line = '{:<5d} {:<5d} {:<15f} {:<15f} {:<15f} {:<5d}\n'
                        # Write charge
                        f.write(line.format(1,  1, trajs[tidx]['uV'][j][0], trajs[tidx]['uV'][j][1], trajs[tidx]['uV'][j][2],0))
        
                    f.close()
        
        
        
                    # Update the atom types to visualize these cluster
                    if args.atypes==1:
                        print '''Updating the atom types to visualize different clusters and sites which do not participate in kMC simulation (if any). If there are sites which do not participate in the kMC transport, the following atom types will be assigned: 
        Beads in sites which cannot participate: 1
        Beads which can particiate: 2, 3, 4... # clusters+1 depending on the cluster they belong to
        CoM of sites: # clusters+ 3
        Charge: # clusters+ 4  
        Otherwise, the types are:
        Beads which can particiate: 1, 2, 3, 4... # clusters depending on the cluster they belong to
        CoM of sites: # clusters+ 2
        Charge: # clusters+ 3.'''
                        for s in notransition:
                                md_traj['atoms']['type_mod'][0][s]=1
                        for i,clstr in enumerate(elec_network):
                            for s in clstr:
                                for k in sites[s]:
                                    md_traj['atoms']['type_mod'][0][k]=i+2 if len(notransition) else i+1
        
                        for i in b_idx:
                            md_traj['atoms']['type_mod'][0][i]=[md_traj['atoms']['type_mod'][0][i] for j in trajs[tidx]['S'] ] # Convert it to list with the length same as the kMC trajectory
                        matype= len(elec_network) + 1 if len(notransition) else len(elec_network)
        
                    # Assign atom types such that the site the charge is on has a different atom type (actual atom type of back bone beads +1) than the rest
                    elif args.atypes==2:
                        print 'Updating the atom types such that the beads in the site where the charge is on have a different atom type (types of back bone beads +1) than the rest. The CoM of sites and charged have the types types of back bone beads +2 and types of back bone beads +3 respectively...'
                        for i in b_idx:
                            md_traj['atoms']['type_mod'][0][i]=btypes.index(md_traj['atoms']['type'][0][i])+1 # Assign original bead type but modified to account for only backbone bead
                            md_traj['atoms']['type_mod'][0][i]=[md_traj['atoms']['type_mod'][0][i] for j in trajs[tidx]['S'] ] # Convert it to list with the length same as the kMC trajectory
                        # Change the type of the site which contains charge in each kMC frame
                        for j, s in enumerate(trajs[tidx]['S']):
                            for k in sites[s]:
                                md_traj['atoms']['type_mod'][0][k][j]=len(btypes)+1
                        matype=len(btypes)+1
        
                    
                    # Collect all the refresh info for the first trajectory into a dictionary with time as keys and (the old site, new site, rotated geometry of polymers, rotated geomerty of their CoM) as the objects
                    refresh_info={j[0]:(j[1],j[2],rot_geo(geo,j[3],j[4],j[5]),refresh_geo_com[_]) for _,j in enumerate(refresh_tabs_all[0])}            
        
                    # Wrap the position of CoM bt applying PBC
                    
                    if not args.wtraj_unwrap:
                        geo_com= PBC(geo_com, md_traj['box'][0])
                    
                    # Wrap the position of charge by applying PBC
                    if not args.wtraj_unwrap:
                        trajs[tidx]['V']= list(PBC(np.array(trajs[tidx]['V']), md_traj['box'][0]))
        
        
                    f=open('{}{}_kmc{}_all.lammpstrj'.format(args.output,md_traj['timestep'][0],tidx),'w')
                    cap=5000
                    #for j in range(0,len(trajs[tidx]['T']),max([int(len(trajs[tidx]['T'])/cap),1])):
                    for j in range(0,min([len(trajs[tidx]['T']),cap])):
                        clrs=sns.color_palette("hls", matype+2)
                        f.write('ITEM: TIMESTEP\n' + str(trajs[tidx]['T'][j]) + '\n')
                        f.write('ITEM: NUMBER OF ATOMS\n' + str(len(b_ids)+len(sites)+1) + '\n')
                        f.write('ITEM: BOX BOUNDS pp pp pp\n' + str(md_traj['box'][0][0][0]) + ' ' + str(md_traj['box'][0][0][1]) + '\n' + str(md_traj['box'][0][1][0]) + ' ' + str(md_traj['box'][0][1][1]) + '\n' + str(md_traj['box'][0][2][0]) + ' ' + str(md_traj['box'][0][2][1]) + '\n')
                        f.write('ITEM: ATOMS id type x y z c_quaternions[1] c_quaternions[2] c_quaternions[3] c_quaternions[4] mol shapex shapey shapez Color.R Color.G Color.B\n')
                        c=1
                        for m in range(len(mol2bidx_keys)):
                            for k in mol2bidx[mol2bidx_keys[m]]:
                                line = '{:<5d} {:<5d} {:<15f} {:<15f} {:<15f} {:<15f} {:<15f} {:<15f} {:<15f} {:<5d} {} {:<15f} {:<15f} {:<15f}\n'
                                # Write bead positions
                                if not args.wtraj_unwrap:
                                    # write wrapped trajectory
                                    f.write(line.format(c, md_traj['atoms']['type_mod'][0][k][j], md_traj['wrapped']['atoms']['x'][0][k], md_traj['wrapped']['atoms']['y'][0][k], md_traj['wrapped']['atoms']['z'][0][k], md_traj['atoms']['c_quaternions[1]'][0][k], md_traj['atoms']['c_quaternions[2]'][0][k], md_traj['atoms']['c_quaternions[3]'][0][k], md_traj['atoms']['c_quaternions[4]'][0][k], m, '1.8 0.8 1.8',clrs[md_traj['atoms']['type_mod'][0][k][j]-1][0],clrs[md_traj['atoms']['type_mod'][0][k][j]-1][1],clrs[md_traj['atoms']['type_mod'][0][k][j]-1][2]))
                                else:
                                    f.write(line.format(c, md_traj['atoms']['type_mod'][0][k][j], geo[k,0], geo[k,1], geo[k,2], md_traj['atoms']['c_quaternions[1]'][0][k], md_traj['atoms']['c_quaternions[2]'][0][k], md_traj['atoms']['c_quaternions[3]'][0][k], md_traj['atoms']['c_quaternions[4]'][0][k], m, '1.8 0.8 1.8',clrs[md_traj['atoms']['type_mod'][0][k][j]-1][0],clrs[md_traj['atoms']['type_mod'][0][k][j]-1][1],clrs[md_traj['atoms']['type_mod'][0][k][j]-1][2]))
                                #if  md_traj['atoms']['type_mod'][0][k][j]>matype: matype=  md_traj['atoms']['type_mod'][0][k][j]
                                c+=1
        
                        # Write com of all sites
                        for s in range(len(sites)):
                            f.write(line.format(c, matype+1, geo_com[s,0]+1.0, geo_com[s,1]+1.0, geo_com[s,2]+1.0, 0.0, 0.0, 0.0, 0.0, mol2bidx_keys.index(bidx2mol[site2bidx_sites[s]]),'1.2 1.2 1.2',clrs[matype][0],clrs[matype][1],clrs[matype][2])) # offset by 1
                            c+=1
        
                        # Write charge
                        f.write(line.format(c,  matype+2, trajs[tidx]['V'][j][0], trajs[tidx]['V'][j][1], trajs[tidx]['V'][j][2],0.0, 0.0, 0.0, 0.0, m+1,'1.2 1.2 1.2',clrs[matype+1][0],clrs[matype+1][1],clrs[matype+1][2]))
        
                    f.close()
        
        
                    f=open('{}{}_kmc{}_all.data'.format(args.output,md_traj['timestep'][0],tidx),'w')
                    f.write('LAMMPS md_traj file via calc_mob_all_elec.py, on [timestamp]\n\n')
                    f.write(str(len(b_ids)+len(sites)+1) + ' atoms\n'+str( matype+2)+' atom types\n')
                    f.write('{} bonds\n{} bond types\n'.format(len(mod_bonded['Bonds']['ids']), max(mod_bonded['Bonds']['types'])  ))
                    f.write('{} angles\n{} angle types\n'.format(len(mod_bonded['Angles']['ids']), max(mod_bonded['Angles']['types'])  ))
                    f.write('{} dihedrals\n{} dihedral types\n\n'.format(len(mod_bonded['Dihedrals']['ids']), max(mod_bonded['Dihedrals']['types'])  ))
        
                    f.write( '{} {} xlo xhi\n{} {} ylo yhi\n{} {} zlo zhi\n\n'.format(md_traj['box'][0][0][0],md_traj['box'][0][0][1],md_traj['box'][0][1][0],md_traj['box'][0][1][1],md_traj['box'][0][2][0],md_traj['box'][0][2][1]))
        
                    f.write('Masses\n\n'+ ''.join([str(_)+' 1.0\n' for _ in range(1,matype+3)]))
        
                    f.write('\nAtoms\n\n')
                    line = '{:<5d} {:<5d} {:<5d} {:<20f} {:<20f} {:<20f} {:<20f}\n'
                    #    atom number, molecule number, atom type, charge (fixed at 0), x, y, z, ?, ?, ?
                    c=1
                    for m in range(len(mol2bidx_keys)):
                        for k in mol2bidx[mol2bidx_keys[m]]:
                            f.write(line.format(c, m, md_traj['atoms']['type_mod'][0][k][0], 0.0, md_traj['wrapped']['atoms']['x'][0][k], md_traj['wrapped']['atoms']['y'][0][k],  md_traj['wrapped']['atoms']['z'][0][k]))
                            c+=1
                    for s in range(len(sites)):
                        f.write(line.format(c, mol2bidx_keys.index(bidx2mol[site2bidx_sites[s]]),  matype+1, 0.0,  geo_com[s,0], geo_com[s,1], geo_com[s,2]))
                        c+=1
                        
                    f.write(line.format(c, m+1,  matype+2, 0.0, trajs[tidx]['V'][j][0],trajs[tidx]['V'][j][1],trajs[tidx]['V'][j][2]) )
            
                    for k in ['Bonds','Angles','Dihedrals']:
                        f.write('\n{}\n\n'.format(k))
                        for j in range(len(mod_bonded[k]['types'])):
                            f.write(' '.join(['{:<5d}'.format(_) for _ in [j+1, mod_bonded[k]['types'][j] ]+ mod_bonded[k]['ids'][j]])+' \n' )
                    f.close()
        
    
                
    
                print 'Time taken to write trajectory and data file : {}\n'.format(time.time()- startTime)
                startTime=time.time()
    
    
            #print 'Time taken to postprocess the trajectories is: {}'.format(time.time()- startTime)
            #startTime=time.time()
            print 'Net time spent until the end of this cycle from the start of the script: {}'.format(time.time() - startTime0)
    
    
            print '\n'+'*'*150
            print 'END OF KMC RUN FOR THIS FRAME'
            print '*'*150+'\n'

    # Check if KMC run was simulated, if just plot the distributions adn exit
    if network_analysis_trials > 0:

        outfile = args.output + 'rate_networks_after_keepfrac_details.txt'
        with open(outfile, 'w') as f:
            f.write('network_rate_threshold_1_per_s: {:.6e}\n'.format(args.network_rate_thresh))
            f.write('network_rate_threshold_1_per_ns: {:.6e}\n'.format(args.network_rate_thresh * 1.0e-9))
            f.write('keep_frac: {}\n'.format(args.keep_frac))
            f.write('\n')
            for line in network_detail_lines:
                f.write(line)

        print 'Wrote rate-network details to {}'.format(outfile)

        outfile = args.output + 'rate_networks_after_keepfrac_summary.txt'
        with open(outfile, 'w') as f:
            f.write('network_rate_threshold_1_per_s: {:.6e}\n'.format(args.network_rate_thresh))
            f.write('keep_frac: {}\n'.format(args.keep_frac))
            f.write('averaged_over_frame_kmc_trials: {}\n'.format(network_analysis_trials))
            f.write('average_number_of_networks: {:.6f}\n'.format(network_count_sum / float(network_analysis_trials)))
            f.write('average_largest_network_size_sites: {:.6f}\n'.format(largest_network_sum / float(network_analysis_trials)))

        print 'Wrote rate-network summary to {}'.format(outfile)

    # BY: write averaged connectivity over all parsed frames and KMC trials
    if site_connectivity_sum is not None and n_connectivity_trials > 0:

        site_connectivity_avg = site_connectivity_sum / float(n_connectivity_trials)

        outfile = args.output + 'site_connectivity_after_keepfrac_averaged.txt'



        with open(outfile, 'w') as f:
            f.write('total_sites/sites_after_keep_frac: {}/{}\n'.format(
                len(site_connectivity_avg),
                int(np.floor(args.keep_frac * len(site_connectivity_avg)))
            ))

            f.write('averaged_over_frame_kmc_trials: {}\n'.format(n_connectivity_trials))
            f.write('site_id average_number_of_usable_neighbor_sites\n')

            for s in range(len(site_connectivity_avg)):
                f.write('{} {:.6f}\n'.format(s, site_connectivity_avg[s]))

        print 'Wrote averaged per-site connectivity over all frames and KMC trials to {}'.format(outfile)

    if len(w12s_inter):

        w12s_inter=np.array(w12s_inter)*(10.0**9.0)
        print  '\tMinimum, average and maximum individual intermolecular rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_inter), np.mean(w12s_inter), np.sqrt(np.mean((w12s_inter- np.mean(w12s_inter))**2.0)), np.max(w12s_inter))
        w12s_inter=list(np.log(w12s_inter)/log(10.0)) # Use log of rates
        w12s_inter_bins, w12s_inter_hist= calc_hist(w12s_inter,h_step=0.05)
        f=open(args.output+'dist_w12s_inter.txt','w')
        for j in range(len(w12s_inter_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_inter_bins[j],w12s_inter_hist[j]))
        f.close()

        w12s_avg_inter=np.array(w12s_avg_inter)*(10.0**9.0)
        print  '\tMinimum, average and maximum intermolecular averaged rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_avg_inter), np.mean(w12s_avg_inter), np.sqrt(np.mean((w12s_avg_inter- np.mean(w12s_avg_inter))**2.0)), np.max(w12s_avg_inter))
        w12s_avg_inter=list(np.log(w12s_avg_inter)/log(10.0)) # Use log of rates
        w12s_avg_inter_bins, w12s_avg_inter_hist= calc_hist(w12s_avg_inter,h_step=0.05)
        f=open(args.output+'dist_w12s_avg_inter.txt','w')
        for j in range(len(w12s_avg_inter_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_avg_inter_bins[j],w12s_avg_inter_hist[j]))
        f.close()
        
    if args.only_dist or args.debug and not args.parallel:
        print "Calculating distributions as specified by the user..."
        # Joint distribution of electrostatics and distance from charge
        if args.Eelec:
            # print  '\tMinimum, average and maximum intramolecular electrostatics energy {}, {} +/- {} and {}'.format(np.min(Eelecs_intra), np.mean(Eelecs_intra), np.sqrt(np.mean((Eelecs_intra- np.mean(Eelecs_intra))**2.0)), np.max(Eelecs_intra))
            # Eelecsq_intra_bins, distq_intra_bins, histq_intra = calc_joint_hist(Eelecs_intra,distq_intra,xstep=0.008, ystep=0.12)
            # data_write = json.dumps({"xbins": Eelecsq_intra_bins.tolist(),  "ybins": distq_intra_bins.tolist(), "hist": histq_intra.tolist()})
            # temp = pipes.Template()
            # fwrite = temp.open('{}distq_Eelec_intra.pipe'.format(args.output), 'w')
            # fwrite.write(data_write)
            # fwrite.close()

            # Eelecsw_intra_bins, distw_intra_bins, histw_intra = calc_joint_hist(Eelecs_intra,distw_intra,xstep=0.008,ystep=0.12)
            # data_write = json.dumps({"xbins": Eelecsw_intra_bins.tolist(),  "ybins": distw_intra_bins.tolist(), "hist": histw_intra.tolist()})
            # temp = pipes.Template()
            # fwrite = temp.open('{}distw_Eelec_intra.pipe'.format(args.output), 'w')
            # fwrite.write(data_write)
            # fwrite.close()
    
            print  '\tMinimum, average and maximum intermolecular electrostatics energy {}, {} +/- {} and {}'.format(np.min(Eelecs_inter), np.mean(Eelecs_inter), np.sqrt(np.mean((Eelecs_inter- np.mean(Eelecs_inter))**2.0)), np.max(Eelecs_inter))
            Eelecsq_inter_bins, distq_inter_bins, histq_inter = calc_joint_hist(mean_Eelecs_inter,distq_inter,xstep=0.008, ystep=0.08) #BY
            data_write = json.dumps({"xbins": Eelecsq_inter_bins.tolist(),  "ybins": distq_inter_bins.tolist(), "hist": histq_inter.tolist()})
            temp = pipes.Template()
            fwrite = temp.open('{}distq_Eelec_inter.pipe'.format(args.output), 'w')
            fwrite.write(data_write)
            fwrite.close()

            Eelecsw_inter_bins, distw_inter_bins, histw_inter = calc_joint_hist(mean_Eelecs_inter,distw_inter,xstep=0.008, ystep=0.04) #BY
            data_write = json.dumps({"xbins": Eelecsw_inter_bins.tolist(),  "ybins": distw_inter_bins.tolist(), "hist": histw_inter.tolist()})
            temp = pipes.Template()
            fwrite = temp.open('{}distw_Eelec_inter.pipe'.format(args.output), 'w')
            fwrite.write(data_write)
            fwrite.close()
            
            # print  '\tMinimum, average and maximum intramolecular electrostatics energy difference {}, {} +/- {} and {}'.format(np.min(dEelec_intra), np.mean(dEelec_intra), np.sqrt(np.mean((dEelec_intra- np.mean(dEelec_intra))**2.0)), np.max(dEelec_intra))
            # dEelec_intra_bins, dEelec_intra_hist= calc_hist(dEelec_intra,h_step=0.01)
            # f=open(args.output+'dist_dEelec_intra.txt','w')
            # for j in range(len(dEelec_intra_bins)):
            #     f.write("{:<25.8f} {:< 25.8f}\n".format(dEelec_intra_bins[j],dEelec_intra_hist[j]))
            f.close()

            print  '\tMinimum, average and maximum intermolecular electrostatics energy difference {}, {} +/- {} and {}'.format(np.min(dEelec_inter), np.mean(dEelec_inter), np.sqrt(np.mean((dEelec_inter- np.mean(dEelec_inter))**2.0)), np.max(dEelec_inter))
            dEelec_inter_bins, dEelec_inter_hist= calc_hist(dEelec_inter,h_step=0.02)
            f=open(args.output+'dist_dEelec_inter.txt','w')
            for j in range(len(dEelec_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(dEelec_inter_bins[j],dEelec_inter_hist[j]))
            f.close()

            print  '\tMinimum, average and maximum intermolecular HOMO energy difference {}, {} +/- {} and {}'.format(np.min(dEhomo_inter), np.mean(dEhomo_inter), np.sqrt(np.mean((dEhomo_inter- np.mean(dEhomo_inter))**2.0)), np.max(dEhomo_inter))
            dEhomo_inter_bins, dEhomo_inter_hist= calc_hist(dEhomo_inter,h_step=0.02)
            f=open(args.output+'dist_dEhomo_inter.txt','w')
            for j in range(len(dEhomo_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(dEhomo_inter_bins[j],dEhomo_inter_hist[j]))
            f.close()

            print  '\tMinimum, average and maximum intermolecular HOMO energy difference {}, {} +/- {} and {}'.format(np.min(dEnet_inter), np.mean(dEnet_inter), np.sqrt(np.mean((dEnet_inter- np.mean(dEnet_inter))**2.0)), np.max(dEnet_inter))
            dEnet_inter_bins, dEnet_inter_hist= calc_hist(dEnet_inter,h_step=0.02)
            f=open(args.output+'dist_dEnet_inter.txt','w')
            for j in range(len(dEnet_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(dEnet_inter_bins[j],dEnet_inter_hist[j]))
            f.close()


        v_intra=np.array(v_intra)
        print  '\tMinimum, average and maximum distance used in coupling for pairs of sites (with nonzero rate) are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(v_intra), np.mean(v_intra), np.sqrt(np.mean((v_intra- np.mean(v_intra))**2.0)), np.max(v_intra))
        v_intra_bins, v_intra_hist= calc_hist(v_intra,h_step=0.05)
        f=open(args.output+'dist_v_intra.txt','w')
        for j in range(len(v_intra_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(v_intra_bins[j],v_intra_hist[j]))
        f.close()


        # Rate distribution
        # w12s_intra=np.array(w12s_intra)*(10.0**9.0)
        # print  '\tMinimum, average and maximum individual intramolecular rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_intra), np.mean(w12s_intra), np.sqrt(np.mean((w12s_intra- np.mean(w12s_intra))**2.0)), np.max(w12s_intra))
        # w12s_intra=list(np.log(w12s_intra)/log(10.0)) # Use log of rates
        # w12s_intra_bins, w12s_intra_hist= calc_hist(w12s_intra,h_step=0.1)
        # f=open(args.output+'dist_w12s_intra.txt','w')
        # for j in range(len(w12s_intra_bins)):
        #     f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_intra_bins[j],w12s_intra_hist[j]))
        # f.close()

        # w12s_avg_intra=np.array(w12s_avg_intra)*(10.0**9.0)
        # print  '\tMinimum, average and maximum intramolecular averaged rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_avg_intra), np.mean(w12s_avg_intra), np.sqrt(np.mean((w12s_avg_intra- np.mean(w12s_avg_intra))**2.0)), np.max(w12s_avg_intra))
        # w12s_avg_intra=list(np.log(w12s_avg_intra)/log(10.0)) # Use log of rates
        # w12s_avg_intra_bins, w12s_avg_intra_hist= calc_hist(w12s_avg_intra,h_step=0.1)
        # f=open(args.output+'dist_w12s_avg_intra.txt','w')
        # for j in range(len(w12s_avg_intra_bins)):
        #     f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_avg_intra_bins[j],w12s_avg_intra_hist[j]))
        # f.close()


        if len(w12s_inter):

            w12s_inter=np.array(w12s_inter)*(10.0**9.0)
            print  '\tMinimum, average and maximum individual intermolecular rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_inter), np.mean(w12s_inter), np.sqrt(np.mean((w12s_inter- np.mean(w12s_inter))**2.0)), np.max(w12s_inter))
            w12s_inter=list(np.log(w12s_inter)/log(10.0)) # Use log of rates
            w12s_inter_bins, w12s_inter_hist= calc_hist(w12s_inter,h_step=0.05)
            f=open(args.output+'dist_w12s_inter.txt','w')
            for j in range(len(w12s_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_inter_bins[j],w12s_inter_hist[j]))
            f.close()

            w12s_avg_inter=np.array(w12s_avg_inter)*(10.0**9.0)
            print  '\tMinimum, average and maximum intermolecular averaged rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_avg_inter), np.mean(w12s_avg_inter), np.sqrt(np.mean((w12s_avg_inter- np.mean(w12s_avg_inter))**2.0)), np.max(w12s_avg_inter))
            w12s_avg_inter=list(np.log(w12s_avg_inter)/log(10.0)) # Use log of rates
            w12s_avg_inter_bins, w12s_avg_inter_hist= calc_hist(w12s_avg_inter,h_step=0.05)
            f=open(args.output+'dist_w12s_avg_inter.txt','w')
            for j in range(len(w12s_avg_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_avg_inter_bins[j],w12s_avg_inter_hist[j]))
            f.close()

            w12s_avg_all=np.array(w12s_avg_all)*(10.0**9.0)
            print  '\tMinimum, average and maximum intermolecular averaged rates are {:.2e}, {:.2e} +/- {:.2e} and {:.2e}'.format(np.min(w12s_avg_all), np.mean(w12s_avg_all), np.sqrt(np.mean((w12s_avg_all- np.mean(w12s_avg_all))**2.0)), np.max(w12s_avg_all))
            w12s_avg_all=list(np.log(w12s_avg_all)/log(10.0)) # Use log of rates
            w12s_avg_all_bins, w12s_avg_all_hist= calc_hist(w12s_avg_all,h_step=0.05)
            f=open(args.output+'dist_w12s_avg_all.txt','w')
            for j in range(len(w12s_avg_all_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(w12s_avg_all_bins[j],w12s_avg_all_hist[j]))
            f.close()

            if args.Vtype!='constant':
                v_inter=np.array(v_inter)
                print  '\tMinimum, average and maximum distance used in coupling for pairs of sites (with nonzero rate) are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(v_inter), np.mean(v_inter), np.sqrt(np.mean((v_inter- np.mean(v_inter))**2.0)), np.max(v_inter))
                v_inter_bins, v_inter_hist= calc_hist(v_inter,h_step=0.05)
                f=open(args.output+'dist_v_inter.txt','w')
                for j in range(len(v_inter_bins)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(v_inter_bins[j],v_inter_hist[j]))
                f.close()



            #dts_inter=np.array(dts_inter)*(10.0**-9.0)
            print  '\tMinimum, average and maximum timescale for a site (with transfer outside)  are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(dts_inter), np.mean(dts_inter), np.sqrt(np.mean((dts_inter- np.mean(dts_inter))**2.0)), np.max(dts_inter))
            #dts_inter=list(np.log(dts_inter)/log(10.0)) # Use log of times
            dts_inter_bins, dts_inter_hist= calc_hist(dts_inter,h_step=0.1)
            f=open(args.output+'dist_dts_inter.txt','w')
            for j in range(len(dts_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(dts_inter_bins[j],dts_inter_hist[j]))
            f.close()

            neighbors_inter=np.array(neighbors_inter)
            print  '\tMinimum, average and maximum neighbors of a site are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(neighbors_inter), np.mean(neighbors_inter), np.sqrt(np.mean((neighbors_inter- np.mean(neighbors_inter))**2.0)), np.max(neighbors_inter))
            neighbors_inter_bins, neighbors_inter_hist= calc_hist(neighbors_inter, h_min=0.5,h_step=1)
            f=open(args.output+'dist_neighbors_inter.txt','w')
            for j in range(len(neighbors_inter_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(neighbors_inter_bins[j],neighbors_inter_hist[j]))
            f.close()

            # Distributions for super-sites
            print  '\tMinimum, average and maximum timescale for single sites (with transfer outside)  are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(dt_s1), np.mean(dt_s1), np.sqrt(np.mean((dt_s1- np.mean(dt_s1))**2.0)), np.max(dt_s1))
            #dt_s1=list(np.log(dt_s1)/log(10.0)) # Use log of times
            dt_s1_bins, dt_s1_hist= calc_hist(dt_s1,h_step=0.1)
            f=open(args.output+'dist_dt_s1.txt','w')
            for j in range(len(dt_s1_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(dt_s1_bins[j],dt_s1_hist[j]))
            f.close()

            print  '\tMinimum, average and maximum timescale for supersites (with transfer outside)  are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(dt_ss), np.mean(dt_ss), np.sqrt(np.mean((dt_ss- np.mean(dt_ss))**2.0)), np.max(dt_ss))
            #dist_dt_ss=list(np.log(dist_dt_ss)/log(10.0)) # Use log of times
            dist_dt_ss_bins, dist_dt_ss_hist= calc_hist(dt_ss,h_step=0.1)
            f=open(args.output+'dist_dt_ss.txt','w')
            for j in range(len(dist_dt_ss_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(dist_dt_ss_bins[j],dist_dt_ss_hist[j]))
            f.close()

            ngbr_s1=np.array(ngbr_s1)
            print  '\tMinimum, average and maximum neighbors of a single site are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(ngbr_s1), np.mean(ngbr_s1), np.sqrt(np.mean((ngbr_s1- np.mean(ngbr_s1))**2.0)), np.max(ngbr_s1))
            ngbr_s1_bins, ngbr_s1_hist= calc_hist(ngbr_s1, h_min=0.5,h_step=1)
            f=open(args.output+'dist_ngbr_s1.txt','w')
            for j in range(len(ngbr_s1_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(ngbr_s1_bins[j],ngbr_s1_hist[j]))
            f.close()
                
            ngbr_ss=np.array(ngbr_ss)
            print  '\tMinimum, average and maximum neighbors of a supersite are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(ngbr_ss), np.mean(ngbr_ss), np.sqrt(np.mean((ngbr_ss- np.mean(ngbr_ss))**2.0)), np.max(ngbr_ss))
            ngbr_ss_bins, ngbr_ss_hist= calc_hist(ngbr_ss, h_min=0.5,h_step=1)
            f=open(args.output+'dist_ngbr_ss.txt','w')
            for j in range(len(ngbr_ss_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(ngbr_ss_bins[j],ngbr_ss_hist[j]))
            f.close()

            size_ss=np.array(size_ss)
            print  '\tMinimum, average and maximum size of supersite is {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(size_ss), np.mean(size_ss), np.sqrt(np.mean((size_ss- np.mean(size_ss))**2.0)), np.max(size_ss))
            size_ss_bins, size_ss_hist= calc_hist(size_ss, h_min=0.5, h_step=1)
            f=open(args.output+'dist_size_ss.txt','w')
            for j in range(len(size_ss_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(size_ss_bins[j],size_ss_hist[j]))
            f.close()

            N_ss=np.array(N_ss)
            print  '\tMinimum, average and maximum number of supersite are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(N_ss), np.mean(N_ss), np.sqrt(np.mean((N_ss- np.mean(N_ss))**2.0)), np.max(N_ss))
            N_ss_bins, N_ss_hist= calc_hist(N_ss, h_min=10, h_step=1)
            f=open(args.output+'dist_N_ss.txt','w')
            for j in range(len(N_ss_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(N_ss_bins[j],N_ss_hist[j]))
            f.close()

            trap_sites=np.array(trap_sites)
            print  '\tMinimum, average and maximum number of supersite are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(trap_sites), np.mean(trap_sites), np.sqrt(np.mean((trap_sites- np.mean(trap_sites))**2.0)), np.max(trap_sites))
            trap_sites_bins, trap_sites_hist= calc_hist(trap_sites, h_min=-0.5, h_step=1)
            f=open(args.output+'dist_trap_sites.txt','w')
            for j in range(len(trap_sites_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(trap_sites_bins[j],trap_sites_hist[j]))
            f.close()

            trap_sites_norm=np.array(trap_sites_norm)
            print  '\tMinimum, average and maximum number of supersite are {:5.4f}, {:5.4f} +/- {:5.4f} and {:5.4f}'.format(np.min(trap_sites_norm), np.mean(trap_sites_norm), np.sqrt(np.mean((trap_sites_norm- np.mean(trap_sites_norm))**2.0)), np.max(trap_sites_norm))
            trap_sites_norm_bins, trap_sites_norm_hist= calc_hist(trap_sites_norm, h_min=-0.05, h_step=0.01)
            f=open(args.output+'dist_trap_sites_norm.txt','w')
            for j in range(len(trap_sites_norm_bins)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(trap_sites_norm_bins[j],trap_sites_norm_hist[j]))
            f.close()


        else:
            print 'All  intermolecular rates are zero :('

        
        # Reorganization energy distribution
        # print  '\tMinimum, average and maximum intramolecular reorganization energy are {}, {} +/- {} and {}'.format(np.min(lmbds_intra), np.mean(lmbds_intra), np.sqrt(np.mean((lmbds_intra- np.mean(lmbds_intra))**2.0)), np.max(lmbds_intra))
        # lmbds_intra_bins, lmbds_intra_hist= calc_hist(lmbds_intra,h_step=0.01)
        # f=open(args.output+'dist_lmbds_intra.txt','w')
        # for j in range(len(lmbds_intra_bins)):
        #     f.write("{:<25.8f} {:< 25.8f}\n".format(lmbds_intra_bins[j],lmbds_intra_hist[j]))
        # f.close()

        print  '\tMinimum, average and maximum intermolecular reorganization energy are {}, {} +/- {} and {}'.format(np.min(lmbdnet_inter), np.mean(lmbdnet_inter), np.sqrt(np.mean((lmbdnet_inter- np.mean(lmbdnet_inter))**2.0)), np.max(lmbdnet_inter))
        lmbdnet_inter_bins, lmbdnet_inter_hist= calc_hist(lmbdnet_inter,h_step=0.002)
        f=open(args.output+'dist_lmbdnet_inter.txt','w')
        for j in range(len(lmbdnet_inter_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(lmbdnet_inter_bins[j],lmbdnet_inter_hist[j]))
        f.close()

        print  '\tMinimum, average and maximum intermolecular reorganization energy are {}, {} +/- {} and {}'.format(np.min(lmbdi_inter), np.mean(lmbdi_inter), np.sqrt(np.mean((lmbdi_inter- np.mean(lmbdi_inter))**2.0)), np.max(lmbdi_inter))
        lmbdi_inter_bins, lmbdi_inter_hist= calc_hist(lmbdi_inter,h_step=0.002)
        f=open(args.output+'dist_lmbdi_inter.txt','w')
        for j in range(len(lmbdi_inter_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(lmbdi_inter_bins[j],lmbdi_inter_hist[j]))
        f.close()

        print  '\tMinimum, average and maximum intermolecular reorganization energy are {}, {} +/- {} and {}'.format(np.min(lmbde_inter), np.mean(lmbde_inter), np.sqrt(np.mean((lmbde_inter- np.mean(lmbde_inter))**2.0)), np.max(lmbde_inter))
        lmbde_inter_bins, lmbde_inter_hist= calc_hist(lmbde_inter,h_step=0.002)
        f=open(args.output+'dist_lmbde_inter.txt','w')
        for j in range(len(lmbde_inter_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(lmbde_inter_bins[j],lmbde_inter_hist[j]))
        f.close()

        # Site length distribution
        print  '\tMinimum, average and maximum length of the sites are {}, {} +/- {} and {}'.format(np.min(Ls), np.mean(Ls), np.sqrt(np.mean((Ls- np.mean(Ls))**2.0)), np.max(Ls))
        Ls_bins, Ls_hist= calc_hist(Ls,h_step=1.0, h_min=0.5, h_max=len(mol2bidx[mol2bidx.keys()[0]])+0.5)
        f=open(args.output+'dist_Ls.txt','w')
        for j in range(len(Ls_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(Ls_bins[j],Ls_hist[j]))
        f.close()

        print  '\tMinimum, average and maximum number of the sites are {}, {} +/- {} and {}'.format(np.min(nsites), np.mean(nsites), np.sqrt(np.mean((nsites- np.mean(nsites))**2.0)), np.max(nsites))
        nsites_bins, nsites_hist= calc_hist(nsites,h_step=1.0, h_min=0.5, h_max=len(b_ids)+0.5)
        f=open(args.output+'dist_nsites.txt','w')
        for j in range(len(nsites_bins)):
            f.write("{:<25.8f} {:< 25.8f}\n".format(nsites_bins[j],nsites_hist[j]))
        f.close()


        if args.only_dist: 
            print 'Net time spent in the script: {}'.format(time.time() - startTime0)
            print '\nScript calc_mob_all1.py ran successfully!!! Woohoooooooooooo!! \m/'
            print '*'*150+'\n'
            return
        print '*'*150+'\n'
    elif args.only_dist or args.debug and args.parallel:
        print "Parallel mode if on. Writing down all the values to a text file..."
        # Joint distribution of electrostatics and distance from charge
        if args.Eelec:
            # f=open('{}distq_Eelec_intra.txt'.format(args.output),'w')
            # for i in range(len(Eelecs_intra)):
            #     f.write('{:25.8f} {:25.8f}\n'.format(Eelecs_intra[i], distq_intra[i]))
            # f.close()

            # f=open('{}distw_Eelec_intra.txt'.format(args.output),'w')
            # for i in range(len(Eelecs_intra)):
            #     f.write('{:25.8f} {:25.8f}\n'.format(Eelecs_intra[i], distw_intra[i]))
            # f.close()

            f=open('{}distq_Eelec_inter.txt'.format(args.output),'w')
            for i in range(len(mean_Eelecs_inter)): #BY
                f.write('{:25.8f} {:25.8f}\n'.format(mean_Eelecs_inter[i], distq_inter[i]))
            f.close()

            f=open('{}distw_Eelec_inter.txt'.format(args.output),'w')
            for i in range(len(mean_Eelecs_inter)): #BY
                f.write('{:25.8f} {:25.8f}\n'.format(mean_Eelecs_inter[i], distw_inter[i]))

            # f=open('{}dist_dEelec_intra.txt'.format(args.output),'w')
            # for i in range(len(dEelec_intra)):
            #     f.write('{:25.8f}\n'.format(dEelec_intra[i]))

            f=open('{}dist_dEelec_inter.txt'.format(args.output),'w')
            for i in range(len(dEelec_inter)):
                f.write('{:25.8f}\n'.format(dEelec_inter[i]))

        f=open('{}dist_dEhomo_inter.txt'.format(args.output),'w')
        for i in range(len(dEhomo_inter)):
            f.write('{:25.8f}\n'.format(dEhomo_inter[i]))

        f=open('{}dist_dEnet_inter.txt'.format(args.output),'w')
        for i in range(len(dEnet_inter)):
            f.write('{:25.8f}\n'.format(dEnet_inter[i]))


        v_intra=np.array(v_intra)
        f=open(args.output+'dist_v_intra.txt','w')
        for j in range(len(v_intra)):
            f.write("{:<25.8f}\n".format(v_intra[j]))
        f.close()

    
        # Rate distribution
        
        w12s_intra=np.array(w12s_intra)*(10.0**9.0)
        w12s_intra=list(np.log(w12s_intra)/log(10.0)) # Use log of rates
        f=open(args.output+'dist_w12s_intra.txt','w')
        for j in range(len(w12s_intra)):
            f.write("{:<25.8f}\n".format(w12s_intra[j]))
        f.close()
        
        w12s_avg_intra=np.array(w12s_avg_intra)*(10.0**9.0)
        w12s_avg_intra=list(np.log(w12s_avg_intra)/log(10.0)) # Use log of rates
        f=open(args.output+'dist_w12s_avg_intra.txt','w')
        for j in range(len(w12s_avg_intra)):
            f.write("{:<25.8f}\n".format(w12s_avg_intra[j]))
        f.close()
    
    
        if len(w12s_inter):        
            w12s_inter=np.array(w12s_inter)*(10.0**9.0)
            w12s_inter=list(np.log(w12s_inter)/log(10.0)) # Use log of rates
            f=open(args.output+'dist_w12s_inter.txt','w')
            for j in range(len(w12s_inter)):
                f.write("{:<25.8f}\n".format(w12s_inter[j]))
            f.close()
    
            w12s_avg_inter=np.array(w12s_avg_inter)*(10.0**9.0)
            w12s_avg_inter=list(np.log(w12s_avg_inter)/log(10.0)) # Use log of rates
            f=open(args.output+'dist_w12s_avg_inter.txt','w')
            for j in range(len(w12s_avg_inter)):
                f.write("{:<25.8f}\n".format(w12s_avg_inter[j]))
            f.close()
    
            w12s_avg_all=np.array(w12s_avg_all)*(10.0**9.0)
            w12s_avg_all=list(np.log(w12s_avg_all)/log(10.0)) # Use log of rates
            f=open(args.output+'dist_w12s_avg_all.txt','w')
            for j in range(len(w12s_avg_all)):
                f.write("{:<25.8f}\n".format(w12s_avg_all[j]))
            f.close()
    
            if args.Vtype!='constant':
                v_inter=np.array(v_inter)
                f=open(args.output+'dist_v_inter.txt','w')
                for j in range(len(v_inter)):
                    f.write("{:<25.8f}\n".format(v_inter[j]))
                f.close()
    
    
            #dts_inter=np.array(dts_inter)*(10.0**-9.0)
            #dts_inter=list(np.log(dts_inter)/log(10.0)) # Use log of times
            f=open(args.output+'dist_dts_inter.txt','w')
            for j in range(len(dts_inter)):
                f.write("{:<25.8f}\n".format(dts_inter[j]))
            f.close()
    
            neighbors_inter=np.array(neighbors_inter)
            f=open(args.output+'dist_neighbors_inter.txt','w')
            for j in range(len(neighbors_inter)):
                f.write("{:<25.8f}\n".format(neighbors_inter[j]))
            f.close()


            f=open(args.output+'dist_dt_ss.txt','w')
            for j in range(len(dt_ss)):
                f.write("{:<25.8f}\n".format(dt_ss[j]))
            f.close()
    
            ngbr_ss=np.array(ngbr_ss)
            f=open(args.output+'dist_ngbr_ss.txt','w')
            for j in range(len(ngbr_ss)):
                f.write("{:<25.8f}\n".format(ngbr_ss[j]))
            f.close()

            f=open(args.output+'dist_dt_s1.txt','w')
            for j in range(len(dt_s1)):
                f.write("{:<25.8f}\n".format(dt_s1[j]))
            f.close()
    
            ngbr_ss=np.array(ngbr_s1)
            f=open(args.output+'dist_ngbr_s1.txt','w')
            for j in range(len(ngbr_s1)):
                f.write("{:<25.8f}\n".format(ngbr_s1[j]))
            f.close()

            size_ss=np.array(size_ss)
            f=open(args.output+'dist_size_ss.txt','w')
            for j in range(len(size_ss)):
                f.write("{:<25.8f}\n".format(size_ss[j]))
            f.close()

            N_ss=np.array(N_ss)
            f=open(args.output+'dist_N_ss.txt','w')
            for j in range(len(N_ss)):
                f.write("{:<25.8f}\n".format(N_ss[j]))
            f.close()

            trap_sites=np.array(trap_sites)
            f=open(args.output+'dist_trap_sites.txt','w')
            for j in range(len(trap_sites)):
                f.write("{:<25.8f}\n".format(trap_sites[j]))
            f.close()

            trap_sites_norm=np.array(trap_sites_norm)
            f=open(args.output+'dist_trap_sites_norm.txt','w')
            for j in range(len(trap_sites_norm)):
                f.write("{:<25.8f}\n".format(trap_sites_norm[j]))
            f.close()

    
        else:
            print 'All  intermolecular rates are zero :('
    
        
        # Reorganization energy distribution
        f=open(args.output+'dist_lmbds_intra.txt','w')
        for j in range(len(lmbds_intra)):
            f.write("{:<25.8f}\n".format(lmbds_intra[j]))
        f.close()
    
        f=open(args.output+'dist_lmbdnet_inter.txt','w')
        for j in range(len(lmbdnet_inter)):
            f.write("{:<25.8f}\n".format(lmbdnet_inter[j]))
        f.close()
    
        f=open(args.output+'dist_lmbdi_inter.txt','w')
        for j in range(len(lmbdi_inter)):
            f.write("{:<25.8f}\n".format(lmbdi_inter[j]))
        f.close()

    
        f=open(args.output+'dist_lmbde_inter.txt','w')
        for j in range(len(lmbde_inter)):
            f.write("{:<25.8f}\n".format(lmbde_inter[j]))
        f.close()

    
        # Site length distribution
        f=open(args.output+'dist_Ls.txt','w')
        for j in range(len(Ls)):
            f.write("{:<25.8f}\n".format(Ls[j]))
        f.close()

        f=open(args.output+'dist_nsites.txt','w')
        for j in range(len(nsites)):
            f.write("{:<25.8f}\n".format(nsites[j]))
        f.close()

        


    # Normalize the MSD
    if not args.parallel: 
        print 'The script is not run parallely, normalizing the MSD, calculating the derivatives and diffusivity...'
    
        # Total MSD
        msd_hist_all, count_hist_all= msd_hist+msd_hist_small, count_hist+count_hist_small
        print len(msd_hist)
        if not len(msd_hist): 
            print 'Warning !All trajectories fall into small loops....'
        else:
            msd_hist_all, count_hist_all= msd_hist+msd_hist_small, count_hist+count_hist_small            
            count_hist[count_hist==0.0]=1.0
            msd_hist/=count_hist
            msd_hist = shave_zeros_msd(msd_hist)
            with open(args.output+'msd.txt'.format(i),'w') as f:
                f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
                for j in range(len(msd_hist)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist[j]))
            f.close()
        count_hist_all[count_hist_all==0.0]=1.0
        msd_hist_all/=count_hist_all
        msd_hist_all = shave_zeros_msd(msd_hist_all)
        with open(args.output+'msd_all.txt'.format(i),'w') as f:
            f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
            for j in range(len(msd_hist_all)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_all[j]))
        f.close()

        # MSD in x-direction 
        msd_hist_x_all, count_hist_x_all= msd_hist_x+msd_hist_x_small, count_hist_x+count_hist_x_small
        if not len(msd_hist_x): 
            print 'Warning !All trajectories fall into small loops....'
        else:
            msd_hist_x_all, count_hist_x_all= msd_hist_x+msd_hist_x_small, count_hist_x+count_hist_x_small            
            count_hist_x[count_hist_x==0.0]=1.0
            msd_hist_x/=count_hist_x
            msd_hist_x = shave_zeros_msd(msd_hist_x)
            with open(args.output+'msd_x.txt'.format(i),'w') as f:
                f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
                for j in range(len(msd_hist_x)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_x[j]))
            f.close()
        count_hist_x_all[count_hist_x_all==0.0]=1.0
        msd_hist_x_all/=count_hist_x_all
        msd_hist_x_all = shave_zeros_msd(msd_hist_x_all)
        with open(args.output+'msd_x_all.txt'.format(i),'w') as f:
            f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
            for j in range(len(msd_hist_x_all)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_x_all[j]))
        f.close()

        # MSD in y-direction 
        msd_hist_y_all, count_hist_y_all= msd_hist_y+msd_hist_y_small, count_hist_y+count_hist_y_small
        if not len(msd_hist_y): 
            print 'Warning !All trajectories fall into small loops....'
        else:
            msd_hist_y_all, count_hist_y_all= msd_hist_y+msd_hist_y_small, count_hist_y+count_hist_y_small            
            count_hist_y[count_hist_y==0.0]=1.0
            msd_hist_y/=count_hist_y
            msd_hist_y = shave_zeros_msd(msd_hist_y)
            with open(args.output+'msd_y.txt'.format(i),'w') as f:
                f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
                for j in range(len(msd_hist_y)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_y[j]))
            f.close()
        count_hist_y_all[count_hist_y_all==0.0]=1.0
        msd_hist_y_all/=count_hist_y_all
        msd_hist_y_all = shave_zeros_msd(msd_hist_y_all)
        with open(args.output+'msd_y_all.txt'.format(i),'w') as f:
            f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
            for j in range(len(msd_hist_y_all)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_y_all[j]))
        f.close()

        # MSD in z-direction 
        msd_hist_z_all, count_hist_z_all= msd_hist_z+msd_hist_z_small, count_hist_z+count_hist_z_small
        if not len(msd_hist_z): 
            print 'Warning !All trajectories fall into small loops....'
        else:
            msd_hist_z_all, count_hist_z_all= msd_hist_z+msd_hist_z_small, count_hist_z+count_hist_z_small            
            count_hist_z[count_hist_z==0.0]=1.0
            msd_hist_z/=count_hist_z
            msd_hist_z = shave_zeros_msd(msd_hist_z)
            with open(args.output+'msd_z.txt'.format(i),'w') as f:
                f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
                for j in range(len(msd_hist_z)):
                    f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_z[j]))
            f.close()
        count_hist_z_all[count_hist_z_all==0.0]=1.0
        msd_hist_z_all/=count_hist_z_all
        msd_hist_z_all = shave_zeros_msd(msd_hist_z_all)
        with open(args.output+'msd_z_all.txt'.format(i),'w') as f:
            f.write('{:<26s} {:<25s}\n'.format("Time(ns)","MSD(A^2)"))
            for j in range(len(msd_hist_z_all)):
                f.write("{:<25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_z_all[j]))
        f.close()

        # Plot MSD vs time
        if args.plot:
            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams['axes.linewidth'] = 4
            fig=figure()
            a1 = plt.subplot(111)
            a1.plot(t_hist*1000.0,msd_hist)
            plt.xlabel('Time (ps)', fontsize=20)
            plt.ylabel('MSD (A^2)', fontsize=20)
            savefig('{}msd.pdf'.format(args.output),  bbox_inches=0,dpi=300)
            close(fig) 

            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams['axes.linewidth'] = 4
            fig=figure()
            a1 = plt.subplot(111)
            a1.plot(t_hist*1000.0,msd_hist_x)
            plt.xlabel('Time (ps)', fontsize=20)
            plt.ylabel('MSD (A^2)', fontsize=20)
            savefig('{}msd_x.pdf'.format(args.output),  bbox_inches=0,dpi=300)
            close(fig) 


            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams['axes.linewidth'] = 4
            fig=figure()
            a1 = plt.subplot(111)
            a1.plot(t_hist*1000.0,msd_hist_y)
            plt.xlabel('Time (ps)', fontsize=20)
            plt.ylabel('MSD (A^2)', fontsize=20)
            savefig('{}msd_y.pdf'.format(args.output),  bbox_inches=0,dpi=300)
            close(fig) 

            plt.rcParams["font.weight"] = "bold"
            plt.rcParams["axes.labelweight"] = "bold"
            plt.rcParams['axes.linewidth'] = 4
            fig=figure()
            a1 = plt.subplot(111)
            a1.plot(t_hist*1000.0,msd_hist_z)
            plt.xlabel('Time (ps)', fontsize=20)
            plt.ylabel('MSD (A^2)', fontsize=20)
            savefig('{}msd_z.pdf'.format(args.output),  bbox_inches=0,dpi=300)
            close(fig) 

    
        # CALCULATE DIFFUSIVITY AND MOBILITY
        # Call calc_derivative to calculate the log and lin derivates of the msd values
        for k in ['','_x','_y','_z']:
            if k=='': 
                print 'Calculating mobility for net MSD...'
            else:
                print 'Calculating mobility for MSD along {} direction...'.format(k[-1])
            
            sp.call('{}python {}/Parsers/calc_derivative.py -f {}msd{}.txt -x_col 0 -y_col 1 -o {}lin{}.txt -start 1; wait\n'.format(args.python,args.taffi,args.output,k,args.output,k),shell=True)
            sp.call('{}python {}/Parsers/calc_derivative.py -f {}msd{}.txt -x_col 0 -y_col 1 --logx --logy -o {}log{}.txt -start 1; wait\n'.format(args.python,args.taffi,args.output,k, args.output,k),shell=True)
    
            # Call calc_diffusivity to calculate diffusivity
            sp.call("cd {}; {}python {}/Parsers/calc_diffusivity.py -log_file ./log{}.txt -lin_file ./lin{}.txt | awk '{{-F ; print $2 }}' > output{}.txt\nwait".format(args.output, args.python,args.taffi,k,k,k), shell=True)
        
            # Read diffusivity
            if os.path.exists(args.output+'output{}.txt'.format(k)):
                f=open('{}output{}.txt'.format(args.output,k),'r')
                for line in f:
                    try:
                        D=float(line)
                    except:
                        print 'Error! The diffusivity was not calculated correctly. Exiting...'
                        quit()
                f.close()
            else:
                print 'Error! The diffusivity was not calculated correctly. Exiting...'
                quit()
        
            # Calculate Mobility
            # Units:
            # D: cm^2/s, kB: eV/K , T: K, eQ: 1.0 e (charge of electron)  ==> mu: cm^2 / (V.s)
            if D==0.0:
                print 'The MSD trajectory did not reach a linear regime. Sorry! :/ Exiting...'
            #    quit()

            kB= 8.617333262145 * (10.0**-5.0)
            eQ= 1.0
            mu= D*eQ/(kB*args.temp)
        
            # Write to a file and print
            f=open('{}output{}.txt'.format(args.output,k),'w')
            f.write('Diffusivity: {} cm^2/s \nMobility: {} cm^2 /(V.s)\n'.format(D,mu))
            f.close()
            print 'The diffusivity and mobility values of the charge are {} cm^2/s and {} cm^2 /(V.s) respectively.'.format(D,mu)

    else:
        print 'The script is run parallely with other frames. Writing the unnormalized MSD and respective counts for each bins and exiting...'
        # Net MSD
        msd_hist_all, count_hist_all= msd_hist+msd_hist_small, count_hist+count_hist_small
        if not len(msd_hist): 
            print 'Warning! All trajectories fall into small loops....'
        else:
            msd_hist = shave_zeros_msd(msd_hist)
            with open(args.output+'msd.txt'.format(i),'w') as f:
                f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
                for j in range(len(msd_hist)):
                    f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist[j],count_hist[j]))
            f.close()
        msd_hist_all = shave_zeros_msd(msd_hist_all)
        with open(args.output+'msd_all.txt'.format(i),'w') as f:
            f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
            for j in range(len(msd_hist_all)):
                f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_all[j],count_hist_all[j]))
        f.close()

        # # MSD in x-direction
        # msd_hist_x_all, count_hist_x_all= msd_hist_x+msd_hist_x_small, count_hist_x+count_hist_x_small
        # if not len(msd_hist_x): 
        #     print 'Warning! All trajectories fall into small loops....'
        # else:
        #     msd_hist_x = shave_zeros_msd(msd_hist_x)
        #     with open(args.output+'msd_x.txt'.format(i),'w') as f:
        #         f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
        #         for j in range(len(msd_hist_x)):
        #             f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_x[j],count_hist_x[j]))
        #     f.close()
        # msd_hist_x_all = shave_zeros_msd(msd_hist_x_all)
        # with open(args.output+'msd_x_all.txt'.format(i),'w') as f:
        #     f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
        #     for j in range(len(msd_hist_x_all)):
        #         f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_x_all[j],count_hist_x_all[j]))
        # f.close()
        # # MSD in y-direction
        # msd_hist_y_all, count_hist_y_all= msd_hist_y+msd_hist_y_small, count_hist_y+count_hist_y_small
        # if not len(msd_hist_y): 
        #     print 'Warning! All trajectories fall into small loops....'
        # else:
        #     msd_hist_y = shave_zeros_msd(msd_hist_y)
        #     with open(args.output+'msd_y.txt'.format(i),'w') as f:
        #         f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
        #         for j in range(len(msd_hist_y)):
        #             f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_y[j],count_hist_y[j]))
        #     f.close()
        # msd_hist_y_all = shave_zeros_msd(msd_hist_y_all)
        # with open(args.output+'msd_y_all.txt'.format(i),'w') as f:
        #     f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
        #     for j in range(len(msd_hist_y_all)):
        #         f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_y_all[j],count_hist_y_all[j]))
        # f.close()
        # # MSD in z-direction
        # msd_hist_z_all, count_hist_z_all= msd_hist_z+msd_hist_z_small, count_hist_z+count_hist_z_small
        # if not len(msd_hist_z): 
        #     print 'Warning! All trajectories fall into small loops....'
        # else:
        #     msd_hist_z = shave_zeros_msd(msd_hist_z)
        #     with open(args.output+'msd_z.txt'.format(i),'w') as f:
        #         f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
        #         for j in range(len(msd_hist_z)):
        #             f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_z[j],count_hist_z[j]))
        #     f.close()
        # msd_hist_z_all = shave_zeros_msd(msd_hist_z_all)
        # with open(args.output+'msd_z_all.txt'.format(i),'w') as f:
        #     f.write('{:<26s} {:<25s} {:<25s}\n'.format("Time(ns)","MSD(A^2)", "Count"))
        #     for j in range(len(msd_hist_z_all)):
        #         f.write("{:<25.8f} {:< 25.8f} {:< 25.8f}\n".format(t_hist[j],msd_hist_z_all[j],count_hist_z_all[j]))
        # f.close()

    print 'Net time spent in the script: {}'.format(time.time() - startTime0)
    print '\nScript  ran successfully!!! Woohoooooooooooo!! \m/'
    print '*'*150+'\n'
    return

# Get dihedrals
def get_dihedral(x1,x2,u1,u2):
    delta=10**-8.0
    r1,r2,r3=-u1,x2-x1,u2
    c11,c22,c33,c12,c13,c23= np.sum(r1*r1), np.sum(r2*r2), np.sum(r3*r3), np.sum(r1*r2), np.sum(r1*r3), np.sum(r2*r3)
    mc12 = np.sqrt(c11*c22)
    mc23 = np.sqrt(c22*c33)

    if np.abs(c12)-mc12 >=0: c12= np.sign(c12)*np.abs(mc12-delta)
    if np.abs(c23)-mc23 >=0: c23= np.sign(c23)*np.abs(mc23-delta)

    d12 = c11*c22- c12*c12
    d13 = c11*c33- c13*c13
    d23 = c22*c33- c23*c23

    hi= -(c23*c12-c13*c22)
    lo= np.sqrt(d23*d12)
    c = hi/lo
    if c > 1: c=1
    if c < -1: c=-1
    
    theta= acos(np.abs(c))
    
    return theta


# def check_s1(s1,supersites,outer2ss,outersites_all,pmjp,js12):
#     if len(supersites[outer2ss[s1]])>1: 
#         if len(pmjp[outer2ss[s1]])==0: 
#             return 0
#     else:
#         if (js12[ outersites_all.index(s1),:]<=0).all(): 
#             return 0
#     return 1

def check_s1(s1, supersites, outer2ss, outersites_all, pmjp, js12):
    # No clustering: only reject trap sites (no outgoing transitions)
    if (js12[outersites_all.index(s1), :] <= 0).all():
        return 0
    return 1


def find_rate_networks(w12, active_sites, sites, objects, site2bidx_sites, rate_thresh_per_ns):
    active_set = set(active_sites)

    # directed adjacency: s can jump to j if rate is above threshold
    adj = {}
    for s in active_sites:
        adj[s] = []
        for j in active_sites:
            if s == j:
                continue
            if w12[s, j] >= rate_thresh_per_ns:
                adj[s].append(j)

    visited = set()
    networks = []

    for start in active_sites:
        if start in visited:
            continue

        stack = [start]
        network = []

        while stack:
            s = stack.pop()
            if s in visited:
                continue

            visited.add(s)
            network.append(s)

            for j in adj[s]:
                if j not in visited:
                    stack.append(j)

        networks.append(sorted(network))

    network_info = []

    for net_id, net_sites in enumerate(networks):
        atom_ids = []
        mol_ids = []
        anchor_atom_ids = []
        anchor_mol_ids = []

        for s in net_sites:
            for atom_index in sites[s]:
                atom_ids.append(objects["Atom_ids"][atom_index])
                mol_ids.append(objects["Molecules"][atom_index])

            bidx = site2bidx_sites[s]
            anchor_atom_ids.append(objects["Atom_ids"][bidx])
            anchor_mol_ids.append(objects["Molecules"][bidx])

        network_info.append({
            "network_id": net_id,
            "site_ids": net_sites,
            "atom_ids": sorted(list(set(atom_ids))),
            "mol_ids": sorted(list(set(mol_ids))),
            "anchor_atom_ids": sorted(list(set(anchor_atom_ids))),
            "anchor_mol_ids": sorted(list(set(anchor_mol_ids))),
        })

    return network_info

# This function performs the KMC run on a given snapshot N number of times
# It requires the different indices of atoms forming different sites
# Sites within distance rc are considered neighboring sites which can transport charge
# It also needs the Marcus rate paramters and the time tmax till KMC simulation must be performed
# It returns a list of trajectory dictionaries for independent KMC runs
# Each dictionary includes:
# 1. S, V, T: The site S with CoM V where charge is present at time T along the trajectory
# 2. msd: A flattened list of entries from an upper triangular array (- the diagonal elements) of distances between CoMs of these sites specific to each trajectory
# 3. t: And corresponding flattened list of entries from an upper triangular array (- the diagonal elements) of time differences specific to each trajectory
# where the entry Rij/Tij of the upper triangular matrix is the distance/time difference between the charge transition from site i --> site j, where i and j are the indices along the trajectory and not that of sites list.
#def kmc(traj,sites, j12, q12, geo_com, odRcom, dRcom, box, N=10, tmax=2000.0, debug=0, startTime0=0, output='output', timestep=0, include=[],kmc_cap=10**10.0, msd_cap=5000, box_cap=10,dist_cap=500):
def kmc(traj, sites, supersites, outersites_all, outer2ss, adj, js12, qs12,  pmjp, tmjp, geo_com, odRcom, dRcom, box, N=10, tmax=2000.0, debug=0, startTime0=0, output='output', timestep=0, kmc_cap=10**7.0, msd_cap=5000, box_cap=10,dist_cap=500,toignore=[], active_sites=None, inactive_set=None):
    L=np.array([box[_][1] for _ in range(3)])- np.array([box[_][0] for _ in range(3)])
    L2= L/2.0
    net_change=np.zeros(3)
    if traj=={}:
        # Initialize the trajectory
        #s1 =  np.random.choice(include)
        s1_check=0
        while s1_check!=1:
            # s1 = np.random.choice(outersites_all)
            # s1_check= check_s1(s1,supersites,outer2ss,outersites_all,pmjp,js12)
            # s1 = np.random.choice(active_sites)
            if active_sites is None:
                active_sites = outersites_all
            if inactive_set is None:
                inactive_set = set()

            s1 = np.random.choice(active_sites)

            s1_check = check_s1(s1, supersites, outer2ss, outersites_all, pmjp, js12)


        traj={'S':[s1],'V':[geo_com[s1,:]],'T':[0],'dT':[],'dV':[],'uV':[geo_com[s1,:]],'T_refresh':[],'N_refresh':np.array([0,0,0]),'rates':[], 'neighbors':[]}
        t=0
        print 'The starting site is {} with bead indices {}'.format(s1, sites[s1])
    else:
        # Resume at the last point
        t=traj['T'][-1]
        s1=traj['S'][-1]
        traj['T_refresh'].append(t)
        

    # Start/Resume the trajectory
    while t<= tmax and len(traj['T'])<=kmc_cap: 
        #print traj['S']
        ssid = outer2ss[s1]
        # Check if this is a trap/ superstate
        # if len(supersites[ssid])>1:
        #     if len(pmjp[ssid])==0:
        #         Nsteps= int((tmax-traj['T'][-1])/traj['dT'][-1])
        #         if Nsteps> 100: Nsteps=100 #BY changed from 10000 to 100
        #         print 'Warning! It entered a trap site {} with supersite id {}, assuming it stays there till {}. Appending {} steps each of {} time in the trajectory to account for {} time remaining'.format(s1,ssid,tmax,Nsteps,(tmax-traj['T'][-1])/Nsteps,tmax-traj['T'][-1])

        #         traj['S']+=[traj['S'][-1] for _ in range(Nsteps)]
        #         traj['V']+=[traj['V'][-1] for _ in range(Nsteps)]
        #         traj['dV']+=[dRcom[s1, s1 ] for _ in range(Nsteps)]
        #         traj['uV']+=[traj['uV'][-1] for _ in range(Nsteps)]
        #         traj['dT']+=[(tmax-traj['T'][-1])/Nsteps for _ in range(1,Nsteps+1)]
        #         traj['T']+=[traj['T'][-1]+traj['dT'][-1]*_ for _ in range(1,Nsteps+1)]
        #         break

                
        #     # If it is a trap state/ super state, 
        #     s1= supersites[ssid].index(s1) # the id of site s1 is different in MJP
        #     # we modeling the trap state as MJP with its adjacent states as absorbing states
        #     # pick the next outer state for the probabilities calculated for the above MJP
        #     # select the neighbor
        #     r1= np.random.uniform()
        #     s2= np.where(np.cumsum(pmjp[ssid][s1,:])>r1)[0][0] # where s2 is the index in adjacent sites, not the actual site id
        #     # and the times calculated for the above MJP
                
        #     dt=tmjp[ssid][s1,s2]
        #     t+=dt
        #     if dt<0:
        #         print 'Error! The time-step is negative. Exiting...'
        #         quit()
        #     # calculate the actual site id
        #     s2=adj[ssid][s2]
        #     if s2 in toignore:
        #         print 'Error! It visited an isolated site along the KMC trajectory. There is a bug in the algorithm :(. Exiting...'
        #         quit()
        #     traj['rates']+=[qs12[outersites_all.index(traj['S'][-1]),outersites_all.index(s2)]]
        #     traj['neighbors']+=[list(np.where(pmjp[ssid][s1,:]>0.0)[0])] 


        # s1= outersites_all.index(s1) # the id of site s1 is different in outersites_all 

        # if (js12[s1,:]<=0).all():
        #     Nsteps= int((tmax-traj['T'][-1])/traj['dT'][-1])
        #     if Nsteps> 100: Nsteps=100 #BY changed from 10000 to 100
        #     print 'Warning! It entered a trap site {} with supersite id {}, assuming it stays there till {}. Appending {} steps in the trajectory to account for {} time remaining'.format(s1,ssid,tmax,Nsteps,tmax-traj['T'][-1])
        #     traj['S']+=[traj['S'][-1] for _ in range(Nsteps)]
        #     traj['V']+=[traj['V'][-1] for _ in range(Nsteps)]
        #     traj['dV']+=[dRcom[s1, s1 ] for _ in range(Nsteps)]
        #     traj['uV']+=[traj['uV'][-1] for _ in range(Nsteps)]
        #     traj['dT']+=[traj['dT'][-1] for _ in range(1,Nsteps+1)]
        #     traj['T']+=[traj['T'][-1]+traj['dT'][-1]*_ for _ in range(1,Nsteps+1)]
        #     break

        # # select the neighboxr
        # r1= np.random.uniform()
        # # print js12[s1,:]
        # # print js12[s1,:]>0
        # # print len(js12[s1,:]>0)
        # # print np.cumsum(js12[s1,:])
        # # print r1
        # s2= np.where(np.cumsum(js12[s1,:])>r1)[0][0] # where s2 is the index in outer sites, not the actual site id
        # traj['rates']+=[qs12[s1,s2]]
        # traj['neighbors']+=[list(np.where(js12[s1,:]>0.0)[0])]

        # # and the times calculated for the above MJP  
        # r2=np.random.uniform()
        # dt=log(r2)/qs12[s1,s1]
        # if dt<0:
        #     print 'Error! The time-step is negative. Exiting...'
        #     quit()
        # t+=dt
        # # calculate the actual site id
        # s2=outersites_all[s2]
        #------------------------------------------------------------------------
        s1_idx = outersites_all.index(s1)   # index in outer-site list

        # Allowed destination indices: positive probability AND destination site is active
        allowed_dest = []
        for j in np.where(js12[s1_idx, :] > 0.0)[0]:
            site_j = outersites_all[j]
            if site_j in inactive_set:
                continue
            if j == s1_idx:
                continue
            allowed_dest.append(j)

        # If no allowed destinations, treat as trap
        if len(allowed_dest) == 0: #BY
            if len(traj['dT']) == 0: #BY
                # no previous dt to reuse; just end this trajectory
                break #BY
            Nsteps= int((tmax-traj['T'][-1])/traj['dT'][-1])
            if Nsteps> 100: Nsteps=100
            print 'Warning! It entered a trap (or all neighbors disabled). Holding until tmax.'
            traj['S']+=[traj['S'][-1] for _ in range(Nsteps)]
            traj['V']+=[traj['V'][-1] for _ in range(Nsteps)]
            traj['dV']+=[dRcom[s1_idx, s1_idx] for _ in range(Nsteps)]
            traj['uV']+=[traj['uV'][-1] for _ in range(Nsteps)]
            traj['dT']+=[traj['dT'][-1] for _ in range(1,Nsteps+1)]
            traj['T']+=[traj['T'][-1]+traj['dT'][-1]*_ for _ in range(1,Nsteps+1)]
            break

        # Build renormalized jump probabilities over allowed destinations
        p = js12[s1_idx, allowed_dest].astype(float)
        p_sum = p.sum()
        if p_sum <= 0.0:
            # should be redundant with allowed_dest check, but safe
            break
        p /= p_sum

        # Sample destination
        r1 = np.random.uniform()
        cum = np.cumsum(p)
        pick = np.searchsorted(cum, r1, side='right')
        if pick >= len(allowed_dest):
            pick = len(allowed_dest) - 1
        s2_idx = allowed_dest[pick]

        # Record rate to chosen neighbor (use existing qs12 off-diagonal)
        traj['rates'] += [qs12[s1_idx, s2_idx]]
        traj['neighbors'] += [allowed_dest]

        # Recompute total escape rate using only allowed destinations
        k_total = float(np.sum(qs12[s1_idx, allowed_dest]))
        if k_total <= 0.0:
            break

        # Sample waiting time
        r2 = np.random.uniform()
        dt = -np.log(r2) / k_total
        t += dt

        # Convert back to actual site id
        s2 = outersites_all[s2_idx]

        #------------------------------------------------------------------------
        if s2 in toignore:
            print 'Error! It visited an isolated site along the KMC trajectory. There is a bug in the algorithm :(. Exiting...'
            quit()

       
        # # select the neighbor
        # r1= np.random.uniform()
        # #s2= include[np.where(np.cumsum(j12[s1,include])>r1)[0][0]]
        # if (j12[s1,:]==0.0).all():
        #     Nsteps= int((tmax-traj['T'][-1])/steps[-1])
        #     print 'Warning! It entered a trap site {}, assuming it stays there till {}. Appending {} steps in the trajectory to account for {} time remaining'.format(s2,tmax,Nsteps,tmax-traj['T'][-1])
        #     traj['S']+=[s2 for _ in range(Nsteps)]
        #     traj['V']+=[geo_com[s2] for _ in range(Nsteps)]
        #     traj['T']+=[traj['T'][-1]+steps[-1]*_ for _ in range(1,Nsteps+1)]
        #     break
        # else:
        #     s2= np.where(np.cumsum(j12[s1,:])>r1)[0][0]
        # r2=np.random.uniform()            
        # dt=log(r2)/q12[s1,s1]
        # t+=dt
        # Add all the info to the trajectories

        s1= traj['S'][-1] # reassign s1 for future calculations
        traj['S'].append(s2)
        traj['V'].append(geo_com[s2])
        traj['dT'].append(dt)
        traj['T'].append(t)

        #print traj['S']

        # Calculate unwrapped position
        traj['dV'].append(dRcom[s2, s1 ])
        traj['uV'].append(traj['uV'][-1]+traj['dV'][-1])

        #net_change=traj['uV'][-1]-traj['uV'][start_idx]
        net_change+=odRcom[s2, s1] # Even though the box is rotated the interbead distance vector remains same in the sense of original reference frame. So we continue to use that to detect the net change and if the charge has crossed out cutoff distance

        # If it has moved more than 2 boxes in any direction, do a site refresh
        if (np.abs(net_change)>box_cap*L).any():             
            traj['N_refresh']+=1*(np.abs(net_change)>box_cap*L)
            return traj, 0

        # net_change=np.linalg.norm(traj['uV'][-1]-traj['uV'][start_idx])
        # # If it has moved dist_cap distance
        # if net_change>dist_cap:
        #     # print traj['T_refresh']
        #     # print traj['uV'][-1], traj['uV'][start_idx]
        #     # print np.abs(net_change)
        #     return traj, 0

        # update the starting site
        s1=deepcopy(s2)
        
    if t<= tmax:
        print 'Simulation ended bc of kmc_cap reached: {}'.format(len(traj['T']))
    elif len(traj['T'])<=kmc_cap: 
        print 'Simulation ended bc of tmax reached: {}'.format(t)

    # Parse different distributions and print statistics
    #rates=[q12[traj['S'][s],traj['S'][s+1]] for s in range(len(traj['S'])-1)]
    #neighbors=[list(np.where(j12[s,:]>0.0)[0]) for s in traj['S']]
    cnghbrs=[float(len(_)) for _ in traj['neighbors']]

    
    # Parse MSD if the trajectory is over
    if debug:
        print 'Time\t\tSite No'
        for j in range(0,len(traj['T']),max([int(len(traj['T'])/msd_cap),1])):
            if traj['T'][j] in traj['T_refresh']:
                print 'Sites refreshed...'
            print '{:5.5f}\t\t{:2d}'.format(traj['T'][j],traj['S'][j])
        if len(traj['S'])==1:
            print 'The average number of neighbours in each KMC move is 0'
        else:
            print 'The average number of neighbours in each KMC move is {}'.format(np.mean(cnghbrs))
        distinct_sites=sorted(set(traj['S']))
        print 'The number of crossings in each direction leading to refresh are {}'.format( ', '.join([str(int(_)) for _ in traj['N_refresh']]))
        print 'The number of distinct sites in this trajectory is {}'.format(len(distinct_sites))
        print 'The distinct sites in this trajectory are {}'.format(distinct_sites)
        print 'The minimum, maximum and average rates in 1/ns are {}, {} and {} respectively'.format(np.min(traj['rates']), np.max(traj['rates']), np.mean(traj['rates']))

    

    # CONSTRUCT THE TIME AND MSD ARRAYS FOR ALL PAIRS OF FRAME DIFFERENCE ALONG THE TRAJECTORY
    # If the number of charge jumps are above a certain cap, use values at certain intervals to reduce the actual entries in MSD calculation
    # This is deprecated now, as the cap is not applied to the KMC trajectory itself. That is, if the the number of jumps reach this cap, the trajectory is stopped. This makes more sense, as we wouldn't want to exclude consecutive jumps and the number of jumps in a given time can vary with the physics of the system 
    if len(traj['S'])>msd_cap:
        interval=int(ceil(len(traj['S'])/msd_cap))
        print 'The number of charge transfers {} is above {}, using charge transfers at interval {}'.format(len(traj['S']),msd_cap,interval)
    else:
        print 'The number of charge transfers {} is below {}, using charge transfers at every interval '.format(len(traj['S']),msd_cap)
        interval=1

    # Construct an upper triangular matrix traj['msd'], which has relative msd**2.0 between all frames using cdist and flatten it
    # unwrap geo_com along the trajectory: use the unwrap_geo function which was originally written for unwrapping each polymer in a frame using adjacency matrix. Here, we use it to unwrap along time by supplying an adjacency matrix for a linear chain of time
    #adj_list=[[1]]+[ [j-1,j+1] for j in range(1,len(traj['S'])-1)]+[[len(traj['S'])-2]]

    stime=time.time()
    unwrap_geo_com= np.array(traj['uV'])
    # if debug:
    #     traj['dR']=np.linalg.norm(traj['dV'],axis=1)
    #     traj['dX'],traj['dY'],traj['dZ']= [np.abs(np.array(traj['dV'])[:,_]) for _ in range(3)]
    #     print 'The minimum, maximum and average time-steps in ns between charge transfer are {}, {} and {} respectively'.format(np.min(traj['dT']), np.max(traj['dT']), np.mean(traj['dT']))
    #     print 'The minimum, maximum and average distance in Angstroms between charge transfer are {}, {} and {} respectively'.format(np.min(traj['dR']), np.max(traj['dR']), np.mean(traj['dR']))
        
    # obtain a matrix of displacements between frames using cdist
    traj['msd']=cdist(unwrap_geo_com[::interval,:], unwrap_geo_com[::interval,:])**2.0        
    traj['msd_x']=cdist(unwrap_geo_com[::interval,0:1], unwrap_geo_com[::interval,0:1])**2.0        
    traj['msd_y']=cdist(unwrap_geo_com[::interval,1:2], unwrap_geo_com[::interval,1:2])**2.0        
    traj['msd_z']=cdist(unwrap_geo_com[::interval,2:3], unwrap_geo_com[::interval,2:3])**2.0        

    print 'Number of transfers used in constructing msd array is {} '.format(len(traj['msd']))


    # extract only upper triangular elements and flatten the array
    traj['msd']=traj['msd'][np.triu_indices(len(traj['msd']))]
    traj['msd_x']=traj['msd_x'][np.triu_indices(len(traj['msd_x']))]
    traj['msd_y']=traj['msd_y'][np.triu_indices(len(traj['msd_y']))]
    traj['msd_z']=traj['msd_z'][np.triu_indices(len(traj['msd_z']))]

    # Construct the corresponding array for time differeneces, i.e. an upper triangular matrix traj['t'], which has relative time differences between all frames using cdist and flatten it
    traj['t']=cdist(np.array([traj['T']]).transpose()[::interval,:],np.array([traj['T']]).transpose()[::interval,:],'minkowski', p=1.)

    print 'Number of transfers used in constructing time array is {} '.format(len(traj['t']))
    # extract only upvper triangular elements and flatten the array
    traj['t']=traj['t'][np.triu_indices(len(traj['t']))]

    print 'Length of final time and msd arrays: {} and {}'.format(len(traj['t']),len(traj['msd']))
    print
    return traj, 1 # completed

# Peforms the changes associated with site refresh
# Rates should remain same
# Should return a new site randomly chosen
# Should return a new geometry where the charge is at the position where it left off but the new site is at this postion and the orientation of the box is different
# Modifies the trajectory to reflect the new site
# Appends this point and the changes in refresh_tab
def site_refresh(traj,geo_com,dRcom,refresh_tabs,include): # Supplied KMC trajectory, configuration, refresh data, sites are included in the network
    # Pick a random order of axes: x, y, z
    # Rotate along them by a randomly chosen angle
    axes=[0,1,2]
    np.random.shuffle(axes)
    angles=[]
    geo_com_new=deepcopy(geo_com)
    dRcom_new=deepcopy(dRcom)
    Rnet=np.eye(3)
    for i in axes:
        #a= pi*np.random.choice([0,90,180,270])/180.0
        a= np.random.rand()*(2*pi)
        # Rotation matrices for all three directions
        if i==0: 
            R=np.array([[1,0,0],[0,cos(a),-sin(a)],[0, sin(a), cos(a)]])
        elif i==1:
            R=np.array([[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]])
        elif i==2:
            R=np.array([[cos(a),-sin(a),0],[sin(a), cos(a),0],[0, 0, 1]])
        angles.append(a*180/pi) # Stored for debugging
        geo_com_new=np.matmul(R,geo_com_new.transpose()).transpose() # Rotate the geometry
        Rnet=np.matmul(R,Rnet)  # Stored for debugging
        # Obtain the distance vectors between sites for the new geometry
        # The distance vector's magnitude is unchanged. But we need to rotate the vector as per the rotation matrix applied on the new geometry 
        # For nD matrices python treats the last two dimensions to be multiplied and the remaining dimensions are broadcast on each other elementwise. So, they must match.
        # So we have mR which has zeroth dimension N, so essentially a N sized list of 3*3 matrices.
        # And we have dRcom_new which also has zeroth dimension N, and essentialy an N sized list of 3*N matrices

        mR=np.array([R for _ in range(dRcom_new.shape[0])]) # Repeat the rotation matrix #sites times, N*3*3 matrix
        # dRcom_new is N*N*3. So we first swap the last two axes N*3*N. Now we have for each site in axis=0, a matrix of its distance vectors from the other sites of the size 3*N
        # After matrix multiplicatio we have a list of rotated distance matrices (of size 3*N) for each site N
        # We swap back the last two axes to obtain the original shape of dRcom_new, i.e. N*N*3
        dRcom_new=np.swapaxes(np.matmul(mR,np.swapaxes(dRcom_new,1,2)),1,2) 

    # Keep the same site
    #s= traj['S'][-1]
    # Pick a random site
    s=np.random.choice(include)
    # Shift the site position to the current charge location
    # Shift all the polymers by the same amount
    delta=(traj['V'][-1]-geo_com_new[s,:])
    geo_com_new+=delta
    # Append the site refresh info in refresh_tabs
    refresh_tabs.append([traj['T'][-1], traj['S'][-1], s, axes, angles,delta]) # Time of refresh, old site id, new site id, axes order, angles order 
    # Modify the KMC trajectory -- change site id
    traj['S'][-1]=s
    return traj, geo_com_new, dRcom_new, refresh_tabs

# Rotates the geometry for the specified axes order, angles to rotate about these and further displacement to add
def rot_geo(geo,axes,angles,delta):
    for _,i in enumerate(axes):
        a= angles[_]*pi/180.0
        if i==0: 
            R=np.array([[1,0,0],[0,cos(a),-sin(a)],[0, sin(a), cos(a)]])
        elif i==1:
            R=np.array([[cos(a),0,sin(a)],[0,1,0],[-sin(a),0,cos(a)]])
        elif i==2:
            R=np.array([[cos(a),-sin(a),0],[sin(a), cos(a),0],[0, 0, 1]])
        angles.append(a*180/pi)
        geo_new=np.matmul(R,geo.transpose()).transpose()+delta
    return geo_new

# Submit lammps rerun to get electrostatic energy associated with hole on a site
def calc_elec_energy(data,trajs,sites,output,lammps,dielectric=10.0,debug=0):
    startTime=time.time()

    bead_ids=[j for i in sites for j in i]
    N=float(len(bead_ids))
    # Calculate the residual charge to be used on the all beads
    Q=sum([data['Charges'][j] for i in  sites for j in i  ])
    qr=(Q-1)/N
    Eelec=[]
    
    # Write this frame alone to trajectory file, use the same name to avoid unnecessary files
    f=open('{}/input.lammpstrj'.format(output),'w')
    for traj in trajs:
        f.write('ITEM: TIMESTEP\n' + str(traj['timestep'][0]) + '\n')
        f.write('ITEM: NUMBER OF ATOMS\n' + str(data['numbers']['atoms']) + '\n')
        f.write('ITEM: BOX BOUNDS pp pp pp\n' + str(traj['box'][0][0][0]) + ' ' + str(traj['box'][0][0][1]) + '\n' + str(traj['box'][0][1][0]) + ' ' + str(traj['box'][0][1][1]) + '\n' + str(traj['box'][0][2][0]) + ' ' + str(traj['box'][0][2][1]) + '\n')
        f.write('ITEM: ATOMS id type x y z\n')
        for i in range(data['numbers']['atoms']):
            line = '{:<5d} {:<5d} {:<15f} {:<15f} {:<15f} {:<5d}\n'
            # write wrapped trajectory
            if i in bead_ids: # The atom types must have been overwritten to match sites, so we need to check and write 1
                f.write(line.format(i+1, 1, traj['wrapped']['atoms']['x'][0][i], traj['wrapped']['atoms']['y'][0][i], traj['wrapped']['atoms']['z'][0][i],traj['atoms']['mol'][0][i]))
            else:
                f.write(line.format(i+1, traj['atoms']['type'][0][i], traj['wrapped']['atoms']['x'][0][i], traj['wrapped']['atoms']['y'][0][i], traj['wrapped']['atoms']['z'][0][i],traj['atoms']['mol'][0][i]))
    
    f.close()
    
    # Write a new data file
    f=open('{}/input.data'.format(output),'w')
    f.write('LAMMPS data file via calc_mob.py, on [timestamp]\n\n')
    f.write(str( data['numbers']['atoms']) + ' atoms\n'+str( data['numbers']['atom types'])+' atom types\n')
    f.write('{} bonds\n{} bond types\n'.format(data['numbers']['Bonds'],data['numbers']['Bond types']))
    f.write('{} angles\n{} angle types\n'.format(data['numbers']['Angles'],data['numbers']['Angle types']))
    f.write('{} dihedrals\n{} dihedral types\n'.format(data['numbers']['Dihedrals'],data['numbers']['Dihedral types']))        

    f.write( '{} {} xlo xhi\n{} {} ylo yhi\n{} {} zlo zhi\n\n'.format(trajs[0]['box'][0][0][0],trajs[0]['box'][0][0][1],trajs[0]['box'][0][1][0],trajs[0]['box'][0][1][1],trajs[0]['box'][0][2][0],trajs[0]['box'][0][2][1]))


    f.write('Masses\n\n'+ ''.join(['{} {}\n'.format(_, data['Masses'][_]) for _ in sorted(data['Masses'].keys()) ]))

    f.write('\nAtoms\n\n')
    line = '{:<5d} {:<5d} {:<5d} {:<20f} {:<20f} {:<20f} {:<20f}\n'
    #    atom number, molecule number, atom type, charge (fixed at 0), x, y, z, ?, ?, ?
    for i in range(len(data['Atom_ids'])):
        if i in bead_ids:
            f.write(line.format(data['Atom_ids'][i], data['Molecules'][i], data['Atom_types'][i], qr,trajs[0]['wrapped']['atoms']['x'][0][i],trajs[0]['wrapped']['atoms']['y'][0][i],trajs[0]['wrapped']['atoms']['z'][0][i]))
        else:
            f.write(line.format(data['Atom_ids'][i], data['Molecules'][i], data['Atom_types'][i], data['Charges'][i],trajs[0]['wrapped']['atoms']['x'][0][i],trajs[0]['wrapped']['atoms']['y'][0][i],trajs[0]['wrapped']['atoms']['z'][0][i]))

    
    for k in ['Bonds','Angles','Dihedrals']:
        f.write('\n{}\n\n'.format(k))
        for j in range(len(data[k]['types'])):
            f.write(' '.join(['{:<5d}'.format(_) for _ in [j+1, data[k]['types'][j] ]+ data[k]['ids'][j]])+' \n' )
    f.close()

    # For each site,
    submitcommand=['']
    for s, site in enumerate(sites):
        # Calculate the charge on each bead in the site
        m=len(site)
        qs=qr+1.0/m

        # Write a rerun file
        # Retain only coulombic interations, because LAMMPS doesn't have an option to just compute electrostatic energy -_-
        # 
        f=open('{}/{}.in.init'.format(output,s),'w')
        f.write('# LAMMPS input file generated using calc_mob_all1.py\n\n')

        f.write('# VARIABLES\n')
        f.write('variable        data_name       index   input.data\n')
        f.write('variable        log_name        index   {}.log\n'.format(s))

        f.write('# Change the name of the log output #\n')
        f.write('log ${log_name}\n\n')
        
        f.write('#===========================================================\n')
        f.write('# GENERAL PROCEDURES\n')
        f.write('#===========================================================\n')
        f.write('units		real   # g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n')
        f.write('dimension	3      # 3 dimensional simulation\n')
        f.write('boundary	p p p	# periodic boundary conditions\n')
        f.write('atom_style      full	# molecular + charge\n')
        f.write('dielectric      {}\n\n'.format(dielectric))

        f.write('#===========================================================\n')
        f.write('# DEFINE PAIR, BOND, AND ANGLE STYLES\n')
        f.write('#===========================================================\n')
        f.write('special_bonds   coul   0.0 0.0 0.0     # NO 1-4 LJ interactions, reduce electrostatics\n')
        f.write('pair_style      lj/gromacs/coul/gromacs 9.0 12.0 0.0 12.0\n')
        #f.write('kspace_style    ewald 0.0001          # long-range electrostatics sum method\n\n')

        f.write('#===========================================================\n')
        f.write('# SETUP SIMULATIONS\n')
        f.write('#===========================================================\n\n')
            
        f.write('# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n')
        f.write('read_data ${data_name}\n')
        f.write('pair_coeff * * 0.0 0.0\n')
        f.write('neigh_modify every 1 delay 0 check no # More relaxed rebuild criteria can be used\n\n')

        f.write('\n#===========================================================\n')
        f.write('# RERUN THE LAMMPS FILE AND DUMP THE ELECTROSTATIC ENERGY\n')
        f.write('#===========================================================\n\n')
        
        f.write('group site id {}\n'.format(' '.join([str(_+1) for _ in site])))
        f.write('set group site charge {} # Set the charge of new site to +1 value\n'.format(qs))
        f.write('group esite id {}\n'.format(' '.join([str(_+1) for _ in range(data['numbers']['atoms']) if _ not in  site])))
        f.write('compute Eelec site group/group esite\n')
        f.write('variable my_Eelec equal c_Eelec\n')
        f.write('variable my_step equal step\n')
        f.write('thermo_style custom step v_my_Eelec\n')
        f.write('thermo 1\n')
        f.write('fix  Ewrite all ave/time 1 1 1 v_my_Eelec file Eelec_{}.txt\n'.format(s))
        #f.write('fix Ewrite all print 1 "${{v_step}} ${{v_Eelec}}" file "Eelec_{}.txt" screen "no" title "# Step Eelec"\n'.format(s))
        f.write('rerun input.lammpstrj dump x y z\n')
        f.close()
        submitcommand[-1]+='{} -in {}.in.init > /dev/null &\n'.format(lammps,s)
        if (s+1)%20==0: submitcommand+=['']


            
        
    # Run these files
    #for s in submitcommand:
    #    sp.call('cd {}\n'.format(output)+s+'\nwait\n', shell=True)

    Eelec=[]
    for s in range(len(sites)):
        # Read electrostatic energy
        f=open('{}/Eelec_{}.txt'.format('Eelec_intra', s),'r') # I changed output to 'Eelec_intra' BY
        Eelec+=[[]]
        for line in f:
            try:
                Eelec[-1].append(float(line.split()[1]))
            except:
                continue
        if len([Eelec[-1]])!=len(trajs):
            print 'Error! The electrostatic energy calculation for site id {} failed. Please check the LAMMPS input files. Exiting...'.format(s); quit()

        # Delete files if debug is off
        #sp.call('cd {}; rm input.lammpstrj input.data {}.log {}.in.init {}.in.out Eelec_{}.txt > /dev/null ; wait'.format(output,s, s, s, s), shell=True)

    print  'Time taken to calculate electrostatic energies: {}'.format(time.time()- startTime)

    #return np.mean(np.array(Eelec)/23.061,axis=1) # Convert to eV and send averaged over time
    return np.array(Eelec).transpose()/23.061 # Convert to eV and send transposed energies

# Read electrostatic energies
def read_elec_energy(filename, timestep):
    Eelec=[]
    f=open(filename,'r')
    flag='n'
    for line in f:
        if 'TIMESTEP' in line:
            if flag=='y': break
            if int(line.split()[1])==timestep: flag='y'
            continue
        if flag=='y':
            Eelec.append(float(line.split()[1]))
    f.close()
    return Eelec

# Calculate rates
def calc_rate(s1, s2, r, l1, l2, T, rc=10.0, E=0.0, lmbdi=0.3, loffset=3.0, dielectric=3.0, cp=0.0, V=1.0, Vintra=0.1, nu=10**15.0,  Etype="constant", Vtype="constant", phi=0.0, Eelec=[], rtype='inter' ):

    # Constants
    kB = 8.617333262145 * (10.0**-5.0)    # units: eV/K
    h = 6.582119569 *  (10.0**-16.0)  # units: eV s  
    e0 = 55.26349406  # units: e^2 GeV-1 fm-1
    e0 = e0/10000.0 # units: e^2 eV-1 A-1 
    if not cp: cp=(1/1.5**2)-1/dielectric  # If Pekar constant not specified calculate it using dielectric
    
    warning='' # Send a warning if external reorganization energy is negative
    
    
    # Free energy change
    if type(Etype)==dict:
        E=(Etype[l2]-Etype[l1])  

    if len(Eelec):
        E+=Eelec[s2]-Eelec[s1]

    # Reorganization energy
    # if type(Ltype)==dict:
    #     lmbdi= (Ltype[l1]+Ltype[l2])/2.0
    lmbdi = float(lmbdi)
    if loffset: 
        r1, r2 = (5.0*l1+ loffset)/4.0,  (5.0*l2+ loffset)/4.0
    else:
        r1, r2 = 5.0*l1/2.0, 5.0*l2/2.0

    
    lmbde= cp*(1.0/(4*pi*e0))*(1.0/(2.0*r1)+1.0/(2.0*r2)-1.0/r) # External reoganization energy, FOUND CONFLICTING FACTORS

    if lmbde<=0: 
        warning= 'Warning! External reorganization energy was found to be negative, so it was set to be zero!'
        lmbde=0.0
    lmbd=lmbdi+lmbde

    if rtype=='inter':
        if r> rc: return 0.0, 0.0, ''
        rate= (V**2.0/h)*sqrt(pi/(lmbd*kB*T))*exp(-((E+lmbd)**2.0/(4*kB*T*lmbd)))*(10**-9.0) # units : 1/ns
    elif rtype=='intra':
        rate= nu*exp(-((E-lmbd)**2.0/(4*kB*T*lmbd)) + abs(Vintra*cos(phi))/(kB*T) )*(10**-9.0)




    # If the rate is less than 0.1/ps, equate it to zero
    if rate< 10: rate=0

    if [s1,s2] in [[0,15]]:
        print 'rates'
        print s1, s2
        print r1, r2, r
        print l1, l2
        print Etype[l1], Etype[l2]
        # print Ltype[l1], Ltype[l2]
        print V, E, lmbde, lmbdi, lmbd
        #print (V**2.0/h)
        #print sqrt(pi/(lmbd*kB*T))
        #print exp(-((E+lmbd)**2.0/(4*kB*T*lmbd)))
        print (V**2.0/h)*sqrt(pi/(lmbd*kB*T))*exp(-((E+lmbd)**2.0/(4*kB*T*lmbd)))*(10**-9.0)
        print rate
        print    
    

    return rate, lmbd, warning

# Calculate rates
def calc_rate_inter_parallel( r, L, T, rc=10.0, E=0.0, lmbdi=0.3, loffset=3.0, dielectric=3.0, cp=0.0, V=1.0, Vintra=0.1, nu=10**15.0,  Etype="constant", Vtype="constant", phi=0.0, Eelec=[] ):

    # Constants
    kB = 8.617333262145 * (10.0**-5.0)    # units: eV/K
    h = 6.582119569 *  (10.0**-16.0)  # units: eV s  
    e0 = 55.26349406  # units: e^2 GeV-1 fm-1
    e0 = e0/10000.0 # units: e^2 eV-1 A-1 
    # if not cp: cp=1/3.0-1/dielectric  # If Pekar constant not specified calculate it using dielectric
    if not cp: cp=1/(1.5**2)-1/dielectric  # If Pekar constant not specified calculate it using dielectric
    # print("cp:", cp)
    
    warning='' # Send a warning if external reorganization energy is negative
    
    row, col= np.indices((len(L),len(L)))
    E=np.zeros((len(L),len(L)))
    # Free energy change
    if type(Etype)!=str:
        E+=Etype[L[col]-1]-Etype[L[row]-1]
        dEhomo=deepcopy(E)



    if len(Eelec):
        dEelec= deepcopy(Eelec[col]-Eelec[row] )
        E+=Eelec[col]-Eelec[row]
    else:
        dEelec= np.zeros(col.shape)

    # Reorganization energy
    # Use constant internal reorganization energy (do NOT read lmbdi from Ltype file)
    lmbdi_const = float(lmbdi)
    lmbdi = np.full(r.shape, lmbdi_const, dtype=float)  # make it an array for downstream code

    if loffset: 
        r1, r2 = (5.0*L[col]+ loffset)/4.0,  (5.0*L[row]+ loffset)/4.0
    else:
        r1, r2 = 5.0*L[col]/2.0, 5.0*L[row]/2.0


    # lmbde= cp*(1.0/(4*pi*e0))*(1.0/(2.0*r1)+1.0/(2.0*r2)-1.0/r) # External reoganization energy, FOUND CONFLICTING FACTORS
    lmbde= cp*(1.0/(4*pi*e0))*(1.0/(2.0*r1)+1.0/(2.0*r2)-1.0/r) # External reoganization energy, FOUND CONFLICTING FACTORS (cp=1/n^2-1/epsilon)

    if (lmbde<=0).any(): 
        #print 'rij: {}, internal reorganization energy {}, external reorganization energy {} and total reorganization energy {}'.format(r,lmbdi,lmbde,lmbdi+lmbde)
        warning= 'Warning! External reorganization energy was found to be negative, so it was set to be zero!'
    lmbde[lmbde<=0]*=0.0
    lmbd=lmbdi+lmbde

    rate=np.zeros((len(L),len(L))) 
    rate= (V**2.0/h)*np.sqrt(pi/(lmbd*kB*T))*np.exp(-((E+lmbd)**2.0/(4*kB*T*lmbd)))*(10**-9.0) # units : 1/ns #intermolecular rate
    #rate= ((V/1000.0)**2.0/h)*np.sqrt(pi/(lmbd*kB*T))*np.exp(-((lmbd)**2.0/(4*kB*T*lmbd)))*(10**-9.0) # units : 1/ns #intermolecular rate
    #V is constant value H_AB which is Boltzmann-averaged effective coupling.V/1000.0 is for converting meV to eV of H_AB.

    # If the rate is less than 0.1/ps, equate it to zero
    rate[r> rc]=0.0
    #rate[rate< 10]=0.0 #BY COMMENTED OUT
    np.fill_diagonal(rate,0.0)

    # if np.isnan(rate).any():
    #     print np.where(np.isnan(rate))
    #     i1,i2=np.where(np.isnan(rate))[0][0], np.where(np.isnan(rate))[1][0]
    #     print i1,i2
    #     print L[i1], L[i2]
    #     print lmbd[i1,i2], lmbdi[i1,i2], lmbde[i1,i2]
    #     print E[i1,i2],dEhomo[i1,i2], dEelec[i1,i2]
    #     print V[i1,i2]
    #     print rate[i1,i2]
    #     print  (V[i1,i2]**2.0/h)*sqrt(pi/(lmbd[i1,i2]*kB*T))*exp(-((E[i1,i2]+lmbd[i1,i2])**2.0/(4*kB*T*lmbd[i1,i2])))*(10**-9.0)
    # print type(V), type(E), type(lmbde), type(lmbdi), type(lmbd)
    # print "parallel"
    # print r1[0,15], r2[0,15], r[0,15]
    # print L[0], L[15]
    # print Etype[L[0]-1], Etype[L[15]-1]
    # print Ltype[L[0]-1], Ltype[L[15]-1]
    # print V[0,15], E[0,15], lmbde[0,15], lmbdi[0,15], lmbd[0,15]
    # print (V[0,15]**2.0/h)*sqrt(pi/(lmbd[0,15]*kB*T))*exp(-((E[0,15]+lmbd[0,15])**2.0/(4*kB*T*lmbd[0,15])))*(10**-9.0)
    # print rate[0,15]
    # print    



    return rate, lmbdi, lmbde, lmbd, dEhomo, dEelec, E,  warning


# Histograms both Time and MSD into user specified time bins
def hist_msd(t,msd,t_hist,count_hist,msd_hist):
    t_step=t_hist[1]-t_hist[0]
    if not len(count_hist): count_hist=np.zeros(t_hist.shape)
    if not len(msd_hist): msd_hist=np.zeros(t_hist.shape) 
    for i in range(len(t)):
        if t[i] >= t_hist[0] and t[i] < (t_hist[-1]+t_step):
            idx=int(float(t[i]-t_hist[0])/t_step)
            count_hist[idx] += 1
            msd_hist[idx] += msd[i]


    return count_hist, msd_hist

# Removes zeros from the end of MSD array. This can happen if the charge jump is very fast and the KMC trajectory is stopped before it reaches the time specified by the user due to sufficient jumps
def shave_zeros_msd(msd):
    idx=np.where(msd==0)[0]
    idx=[_ for _ in idx if (msd[_:min([_+5,len(msd)])]==0).all()]   # Check the next five values are also zero
    idx=idx[0] if len(idx) else len(msd)
    return msd[:idx]

# Apply cdist to given geometric arrays but with min image convention distance 
# geo: numpy array of shape (N,3)
# box: [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
def cdist_mic(geo,box):
    Rs= np.zeros((geo.shape[0],geo.shape[0]))
    for i in range(3):
        R = cdist(geo[:,i:i+1],geo[:,i:i+1],'minkowski', p=1.0) 
        l,l2= box[i][1]-box[i][0],(box[i][1]-box[i][0])/2.0
        while not (R<l2).all():
            R-=l*(R>l2)
            R=np.abs(R)
        Rs+=R**2.0
    Rs=np.sqrt(Rs)
    
    return Rs

# Apply cdist to given two different geometric arrays but with min image convention distance 
# geo1, geo2: numpy array of shape (N1,3), (N2,3)
# box: [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
# returns R:  numpy array of shape (N1,N2)
def cdist_mic_xy(geo1,geo2,box):
    Rs= np.zeros((geo1.shape[0],geo2.shape[0]))
    for i in range(3):
        R = cdist(geo1[:,i:i+1],geo2[:,i:i+1],'minkowski', p=1.0) 
        l,l2= box[i][1]-box[i][0],(box[i][1]-box[i][0])/2.0
        while not (R<l2).all():
            R-=l*(R>l2)
            R=np.abs(R)
        Rs+=R**2.0
    Rs=np.sqrt(Rs)
    
    return Rs

# Find the nearest CoM image for a pair of sites
# Calculates the average minimum bead distance using these images
def rcoupling(geo,geo_com,sites,box):
    Rmin= np.zeros((geo_com.shape[0],geo_com.shape[0]))
    temp_Rmin= np.zeros((geo_com.shape[0],geo_com.shape[0]))
    L=np.array([box[_][1] for _ in range(3)])- np.array([box[_][0] for _ in range(3)])
    L2=L/2.0
    for s1 in range(geo_com.shape[0]):
        for s2 in range(s1+1,geo_com.shape[0]):
            dgeo=geo_com[s1,:]-geo_com[s2,:]
            # Image corresponding to min possible COM distance
            img= -np.sign(dgeo)*((np.abs(dgeo)+L2)//L)*(L) 
            temp=cdist(geo[sites[s1],:]+img, geo[sites[s2],:])
            Rmin[s1,s2]= np.mean(np.min(temp,axis=1)) 
            Rmin[s2,s1]= np.mean(np.min(temp,axis=0)) 
            Rmin[s1,s2]=(Rmin[s1,s2]+Rmin[s2,s1])/2.0
            Rmin[s2,s1]=Rmin[s1,s2]
            temp_Rmin[s1,s2]=np.min(temp)
            temp_Rmin[s2,s1]=np.min(temp)
            img_min=img

    return Rmin, temp_Rmin

# Find the nearest CoM image for a pair of sites
# Calculates the average minimum bead distance using these images
def coupling(geo,geo_com,sites,box,Vtype,Vconst=None): 

    
    # If constant coupling requested, return a constant coupling matrix (no distance dependence)
    if (isinstance(Vtype, str) and Vtype == 'constant'):
        if Vconst is None:
            raise ValueError("Vconst must be provided when Vtype is 'constant'")
        n = len(sites)
        V = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                if i != j:
                    V[i, j] = float(Vconst)
        return V
    
    V= np.zeros((geo_com.shape[0],geo_com.shape[0]))
    L=np.array([box[_][1] for _ in range(3)])- np.array([box[_][0] for _ in range(3)])
    L2=L/2.0
    for s1 in range(geo_com.shape[0]):
        for s2 in range(s1+1,geo_com.shape[0]):
            dgeo=geo_com[s1,:]-geo_com[s2,:]
            # Image corresponding to min possible COM distance
            img= -np.sign(dgeo)*((np.abs(dgeo)+L2)//L)*(L) 
            temp=cdist(geo[sites[s1],:]+img, geo[sites[s2],:])

            V[s1,s2]= np.mean(np.exp(-Vtype['beta']*(np.min(temp,axis=1)- Vtype['r0']))) 
            V[s2,s1]= np.mean(np.exp(-Vtype['beta']*(np.min(temp,axis=0)- Vtype['r0']))) 
            V[s1,s2]=(V[s1,s2]+V[s2,s1])/2.0
            V[s2,s1]=V[s1,s2]
            
            img_min=img

    return Vtype['V0']*V

# Takes position of N sites
# Returns a matrix Rmin of (N,N,3) containing the closest vectors according to minimum image convention separation for all pairs
# By convention Rmin[i,j]= geo[i,:]-geo[j,:]
def get_min_dRcom(geo,box):
    N=geo.shape[0]
    Rmin=np.zeros((N,N,3))
    L=np.array([box[_][1] for _ in range(3)])- np.array([box[_][0] for _ in range(3)])
    L2=L/2.0
    idx=np.triu_indices(N)
    dgeo=geo[idx[0]]-geo[idx[1]]
    dgeo=dgeo-np.sign(dgeo)*((np.abs(dgeo)+L2)//L)*(L) 
    Rmin[idx]=dgeo
    Rmin=Rmin-np.swapaxes(Rmin,0,1)
    return Rmin


# Wrap the array to satisfy PBC
# geo: numpy array of shape (N,3)
# box: [[xlo,xhi],[ylo,yhi],[zlo,zhi]]
def PBC(geo, box):
    L=np.array([box[_][1] for _ in range(3)])- np.array([box[_][0] for _ in range(3)])
    L2=L/2.0
    geo= geo-np.sign(geo)*((np.abs(geo)+L2)//L)*(L)
    #while (geo>L2).any(): geo[geo>L2]-=L
    #while (geo<-L2).any(): geo[geo<-L2]+=L

    return geo

# # Unwrap trajectory, works only for a  cubic box
# def unwrap_geo_msd(geo, L):
#     L2=L/2.0
#     dgeo=np.zeros((geo.shape[0],3))
#     dgeo[1:,:]= geo[1:,:]-geo[:-1,:]
#     dgeo[1:,:]= dgeo[1:,:] -np.sign(dgeo[1:,:])*((np.abs(dgeo[1:,:])+L2)//L)*(L) 
#     ugeo=deepcopy(geo)
#     for i in range(geo.shape[0]-1):
#         ugeo[i+1,:]=ugeo[i,:]+dgeo[i+1,:]

#     f,b=np.zeros(3),np.zeros(3)
#     Nold=(ugeo[0,:]+L2)//L 
#     change=[]
#     for i in range(1,geo.shape[0]):
#         Nnew=(ugeo[i,:]+L2)//L
#         f+=(Nnew>Nold)
#         b+=(Nnew<Nold)
#         if i==1: change.append([i,f-b])
#         if not np.array_equal(f-b, change[-1][1]): change.append([i,f-b])
#         Nold=deepcopy(Nnew)
#     print 'The final number of boxes moved: {}'.format(change[-1])
#     return ugeo

# Calculates joint histogram
def calc_joint_hist(X, Y, xbins=50, ybins=50, xstep=None, ystep=None):
    if xstep!=None:
        x=np.arange(np.min(X),np.max(X),xstep)
    else:
        x=np.linspace(np.min(X),np.max(X),50)
    if ystep!=None:
        y=np.arange(np.min(Y),np.max(Y),ystep)
    else:
        y=np.linspace(np.min(Y),np.max(Y),50)
    
    dx, dy=x[1]-x[0], y[1]-y[0]
    Xs,Ys=np.meshgrid(x,y)
    Hist=np.zeros(Xs.shape)

    # Add these entries to the join histogram
    for i in range(len(X)):
        i1=int((X[i]-np.min(x))/dx)
        i2=int((Y[i]-np.min(y))/dy)
        Hist[i2,i1]+=1.0
    
    return Xs, Ys, Hist


# Read lammps file step by step
# Returns a dictionary with keys timestep, box, atoms and objects as list of timesteps, list of box-dim for each timestep ( each box-dim is a list [[xlo,xhi],[ylo,yhi],[zlo,zhi]]) and a dictionaries for objects under ATOMS section for each timestep ( each dictionary has keys which are columns of ATOMS section with objects as lists [[entry for each atom]... for each timestep]) respectively
# Additional arguments:
# Set 'unwrap' to true to make the function unwrap the trajectory according to the adj_list supplied with 'adj_list'
# Use 'atom_ids' argument to apply unwrapping only for specific atom-ids to save time. This is useful when you want only specific molecule types/ atom  types
# IMPORTANT: If you use this, the final lists will have entries for every atom, but unwrapped trajectory only for the atom-ids supplied. This is different from the function in cluster.py
# Arguments 'first', 'last' and 'every' are the self-explonatory arguments to tell the function which timesteps to parse
# Set 'sort' to true to make the function sort all the entries with respect to the Atoms ids
def gen_lammps_frames(filename,unwrap=False, adj_list=None, atom_ids=None, sort= False, first=0, last=-1, every=1, bc="p p p"):
    print 'Reading lammps file {}'.format(filename)
    # Gather the timestep indices
    timestep_idx,flag_t=0,'n'

    f=open(filename,'r')
    for line in f:
        if 'TIMESTEP' in line:
            data={'timestep':[],'box':[],'atoms':{}, 'wrapped':{'atoms':{}}}
            idx={}
            flag='t'
            continue
        if 'ITEM: NUMBER OF ATOMS' in line:
            flag='n'
            continue
        if 'ITEM: BOX' in line:
            if flag_t=='y':
                data['box'].append([])
            flag='b'
            continue
        if 'ITEM: ATOMS ' in line:
            if flag_t=='y':
                line=line.replace('ITEM: ATOMS ','').split()
                for i,j in enumerate(line):
                    idx[j]=i
                    if j not in data['atoms'].keys():
                        data['atoms'][j]=[[]]
                    else:
                        data['atoms'][j]+=[[]]
            flag='a'
            continue
        if flag=='t':
            if timestep_idx ==  last and last!=first:
                break
            elif timestep_idx >last and last==first:
                break
            elif timestep_idx >= first and timestep_idx % every == 0:
                flag_t='y'
            else:
                flag_t='n'
            
            timestep_idx+=1
            if flag_t=='y':
                data['timestep'].append(int(line))
        if flag=='n':
            Natoms=int(line)
        if flag=='b' and flag_t=='y':
            data['box'][-1].append([float(_) for _ in line.split()])
        if flag=='a' and flag_t=='y':
            line=line.split()
            for i in data['atoms'].keys():
                if i=='id' or i=='type' or i=='mol':
                    data['atoms'][i][-1].append(int(line[idx[i]]))
                else:
                    data['atoms'][i][-1].append(float(line[idx[i]]))
    
            if flag_t=='y' and len( data['atoms']['x'][0])==Natoms:
                # Sort the indices if asked by the user
                if sort:
                    keys=data['atoms'].keys()
                    keys.remove('id')
                    keys.insert(0,'id')
                    for i in len(data['timestep']):
                        atoms = [data['atoms'][k][i] for k in keys ]
                        for k,l in enumerate(sorted(zip(atoms))):
                            data['atoms'][keys[k]][i]= l
                
    
                if unwrap==True:
                    if atom_ids==None: atom_ids=range(1,Natoms+1)
                    for i,k in enumerate(['x','y','z']):
                        if k in data['atoms'].keys() and bc.split()[i]=="p":
                            data['wrapped']['atoms'][k]=deepcopy(data['atoms'][k])  
                            data['atoms'][k]= unwrap_geo(data['atoms'][k],adj_list,[b[i] for b in data['box'] ], atom_ids)
                yield data
    f.close()

# Description: Performed the periodic boundary unwrap of the geometry
def unwrap_geo(geo,adj_list,box,atom_ids):
    # Unwrap the molecules using the adjacency matrix
    # Loops over the individual atoms and if they haven't been unwrapped yet, performs a walk
    # of the molecular graphs unwrapping based on the bonds. 
    for t in range(len(geo)):
        b2 = ( box[t][1] - box[t][0] ) / 2.0        
        unwrapped = []
        for count_i,i in enumerate(geo[t]):
            if count_i+1 in atom_ids:        
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
                            while (geo[t][k] - geo[t][j])   >  b2: geo[t][k] -= (b2*2.0) 
                            while (geo[t][k] - geo[t][j]) < -b2: geo[t][k] += (b2*2.0) 
        
                        # append the just unwrapped atoms to the molecular graph so that their connections can be looped over and unwrapped. 
                        unwrap += new
    
    return geo


if __name__ == '__main__':
    main(sys.argv[1:])