#!/bin/env python
"""

This script will build a TAFFI-style database from a given .xyz file and master database file containing all force field parameters.

This can be supplemented by the -supp_param. This will convert one set of atomtypes inside the database file to TAFFI-style atomtypes
for use in builder scripts that use TAFFI-style atomtypes. 

Atomtypes to read from the database file can be specified in one of two ways: automatically detected using Amber's ANTECHAMBER (for use
when the master database file is Amber-style, like GAFF (perovskites) or Amber (radical polymers/TEMPO); or can be used-specified
by including them in the fifth column of the xyz file (to override or change a subset of ANTECHAMBER types).

The "_reference_chart.out" file contains detailed information on the logic behind the choices. It provides (1) the mapping dictionary
between TAFFI and other atom types, (2) the xyz file completely mapped, (3) masses, (4) bonds, (5) angles, (6) dihedrals, (7) VDW terms.
In general, it will supply the internally-determined TAFFI-style atom type, followed by the matching parameter from the master db, if it is
found; if not, it will throw an error. For the dihedrals, after the atom type identification, it will show all the terms that go into that
diehdral, since there may be up to 4 terms; each term is separated by <>, while after each term (with the final entry in the list being the 
OPLS-style value), the OPLS-style 4-term updated values list is shown (separated by |).

                                                                                                 
@author: Stephen Shiring
"""

import argparse
import subprocess
import sys
import os
import numpy
from copy import deepcopy

# Add TAFFI Lib to path
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-2])+'/lib')
import adjacency
import id_types

def main(argv):
    parser = argparse.ArgumentParser(description='Generate a TAFFI-style Forcefield (with TAFFI-style atomtypes) .db file from a '\
    'molecular structure and Amber (GAFF or Amber o any Amber FF file) parameters file. If atomtypes '\
    'are not specified in the .xzy file, this script will call Amber antechamber to generate the Amber atomtypes (must be installed).')

    parser.add_argument('xyz', type=str,
                        help='Name of coordinate xyz file. User-defined atom types can be specified in the 5th column of geometry section, in which case running antechamber to obtain the atomtypes will be skipped.')
    
    parser.add_argument('parameters_file', type=str,
                        help='Name of parameter file containing all parameters for a given force field.')
    
    parser.add_argument('-supp_param', dest='supplemental_params', type=str, default=None,
                        help='Name of parameter file containing any supplemental parameters for a given force field. Format expected to be identical to parameters_file.')
    
    parser.add_argument('-charge_file', dest='charge_file', type=str, default=None,
                        help='If a TAFFI-style charge .db file is specified (with atomtypes matching final atomtypes), merge found charges into final db file. Default: None')
    
    parser.add_argument('-FF', dest='FF', type=str, default='gaff',
                        help='Force field type, for antechamber atomtype identification. Accepts: gaff, amber. Default: gaff')
    
    parser.add_argument('-q', dest='charge', type=int, default=0,
                        help='Molecular charge, for antechamber atomtype identification. Default: 0')
    
    parser.add_argument('-m', dest='multiplicity', type=int, default=1,
                        help='Molecular multiplicity, for antechamber atomtype identification. Default: 1')
    
    parser.add_argument('-vdw', dest='vdw_type', type=str, default='lj',
                        help='Set VDW type to write to db. Default: lj')
    
    parser.add_argument('-o', dest='output', type=str, default=None,
                        help='Output filename. Default: xyz filename + file')
    
    parser.add_argument('-comments', dest='comments', type=str, default='',
                        help='Comments to place in the header of the db file.')
    
    parser.add_argument('--perovskite', dest='perovskite_flag', action='store_const', const=True, default=False,
                        help = 'When invoked, add all terms for a Pb-I/Br perovskite simulation.')
    
    parser.add_argument('-water', dest='water_model', type=str, default='TIP3P',
                        help='Water model type to use, if --perovskite is requested. Accepts: SPCE, TIP3P, TIP4P. Default: TIP3P')
    
    parser.add_argument('-amber', dest='antechmaber_path', type=str, default='/depot/bsavoie/apps/amber20/bin/',
                        help='Path to antechmaber executable. Default: /depot/bsavoie/apps/amber20/bin/')
    
    parser.add_argument('--no_TAFFI', dest='TAFFI_flag', action='store_const', const=True, default=False,
                        help = 'When invoked, do not use TAFFI atom types. Use the antechamber ones.')
    
    parser.add_argument('--readme', dest='print_readme', action='store_const', const=True, default=False,
                        help = 'When invoked, print out the readme / notes and exit.')
    
    parser.add_argument('--impropers', dest='impropers_flag', action='store_const', const=True, default=False,
                        help = 'When invoked, find and use imprompers.')
    
    global args
    args = parser.parse_args()
    
    if args.output == None:
        f = args.xyz.split('/')[-1]
        args.output = f[:f.rfind('.')]
        
    if args.print_readme:
        print_readme()
        exit()
    
    if not os.path.isfile(args.xyz):
        print('\nERROR: Specified xyz file "{}" not found. Aborting....\n'.format(args.xyz))
        exit()
    
    if not os.path.isfile(args.parameters_file):
        print('\nERROR: Specified parameter_file file "{}" not found. Aborting....\n'.format(args.parameters_file))
        exit()
    
    if args.FF.lower() not in ['gaff','amber']:
        print('\nERROR: -FF argument ({}) not recognized. Accepted values: gaff, amber. Exiting...\n'.format(args.FF))
        exit()
    
    if args.water_model.upper() not in ['SPCE','TIP3P','TIP4P']:
        print('\nERROR: -water argument ({}) not recognized. Accepted values: SPCE, TIP3P, TIP4P. Exiting...\n'.format(args.water_model))
        
    if not os.path.isdir(args.output):
        os.mkdir(args.output)
    
    sys.stdout = Logger(args.output+'/'+args.output)
    
    print("{}\n\nPROGRAM CALL: python {}\n".format('-'*150, ' '.join([ i for i in sys.argv])))
    
    print('Reading xyz file...')
    molecule = read_xyz(args.xyz)
    
    if args.charge_file != None:
        print('Charge file specified, reading charge file...')
        if not os.path.isfile(args.charge_file):
            print('\nERROR: Specified charge file "{}" not found. Aborting....\n'.format(args.charge_file))
            exit()
        
        print('Reading charges db...')
        molecule['charges'] = read_charges_db(args.charge_file)
        
    
    print('Reading parameter file...')
    parameter_data = read_parameters_file(args.parameters_file)
    if args.supplemental_params != None:
        supplemental_parameter_data = read_parameters_file(args.supplemental_params)
        
        # Combine them...
        for key in list(parameter_data.keys()):
            parameter_data[key].update(supplemental_parameter_data[key])
    
    if len(molecule['read_atom_types']) == 0:
        print('\nRunning AMBER antechamber to determine atomtypes...')
        
        if not os.path.isdir(args.output+'/antechamber'):
            os.mkdir(args.output+'/antechamber')
        print('    antechamber -i {}.com -fi 8 -o antechamber -fo 1 -nc {} -m {} -at {}'.format(args.output, args.charge, args.multiplicity, args.FF.lower()))
        with cd(args.output+'/antechamber/'):
            write_Gaussian_com(args.output, molecule['geometry'], molecule['elements'])
            subprocess.call('{}antechamber -i {}.com -fi 8 -o antechamber -fo 1 -nc {} -m {} -at {}'.format(args.antechmaber_path, args.output, args.charge, args.multiplicity, args.FF.lower()), shell=True)
            
            if not os.path.isfile('ANTECHAMBER_AC.AC'):
                print('\nERROR: antechamber failed. Could not find "ANTECHAMBER_AC.AC". Aborting....\n')
                exit()
            
            print('Reading antechamber file...')
            antechamber_atom_types, atom_types_mapping = process_antechamber_file('ANTECHAMBER_AC.AC', molecule['atom_types'])
            
            if args.TAFFI_flag:
                molecule['atom_types'] = antechamber_atom_types
        
        mapped_filename = args.output+'/'+args.output+'_antechamber.xyz'
        mapped_comment = 'ANTECHAMBER-style'
        
    else:
        print('Using read atom types...')
        antechamber_atom_types = molecule['atom_types']
        atom_types_mapping = {}
        
        for i in range(len(molecule['atom_types'])):
            if molecule['atom_types'][i] not in list(atom_types_mapping.keys()):
                atom_types_mapping[molecule['atom_types'][i]] = molecule['read_atom_types'][i]
                
        mapped_filename = args.output+'/'+args.output+'_read_atomtypes.xyz'
        mapped_comment = 'Read atom types'
    
    print('Writing atomtypes mapping ({})...'.format(args.output+'_atomtypes_mapping.out'))
    with open(args.output+'/'+args.output+'_atomtypes_mapping.out', 'w') as o:
        o.write('{:75s} {:75s}\n'.format('TAFFI-style atomtype', 'other-style atomtype'))
        for key in list(atom_types_mapping.keys()):
            o.write('{:75s} {:75s}\n'.format(key, atom_types_mapping[key]))

    print('Finding modes...')
    molecule["bond_types"], molecule["angle_types"],  molecule["dihedral_types"], molecule["improper_types"] = get_modes(molecule["adj_mat"], molecule["atom_types"], args.impropers_flag)  
    
    print('Writing out .xyz files with atomtypes as fifth column...')
    write_xyz(mapped_filename, molecule['elements'], molecule['geometry'], molecule['atom_types'], comment='TAFFI-style')
    write_xyz(mapped_filename, molecule['elements'], molecule['geometry'], [ atom_types_mapping[_] for _ in molecule['atom_types'] ], comment=mapped_comment)
    
    print('\nWriting reference chart ({})...'.format(args.output+'_reference_chart.out'))
    with open(args.output+'/'+args.output+'_reference_chart.out', 'w') as o:
        o.write('# Reference structure: {}\n')
        o.write('# Determined atom type. Parameters file atom type. Parameter type. Parameters (converted for use in LAMMPS).\n')
        o.write('# bond, angle: harmonic (Ang, kcal/mol).\n')
        o.write('# dihedral: OPLS.\n#')
        o.write('# VDW: LJ (sigma, eps)\n')
        o.write('# Ordered according to order in {}.\n\n'.format(args.xyz))
        
        o.write('\n#\n# Mapping dictionary:\n#\n')
        o.write('{:75s} {:75s}\n'.format('TAFFI-style atomtype', 'other-style atomtype'))
        for key in list(atom_types_mapping.keys()):
            o.write('{:75s} {:75s}\n'.format(key, atom_types_mapping[key]))
        
        o.write('\n#\n# Mapped molecule:\n#\n')
        o.write('{:<5s} {:<5s} {:<50s} {:<50s} ({:<20s}, {:<20s}, {:<20s})\n'.format('#', 'Element', 'Determined atomtype', 'Parameter file atomtype', 'x-coordinate', 'y-coordinate', 'z-coordinate'))
        for i in range(len(molecule['elements'])):
            o.write('{:<5d} {:<5s} {:<50s} {:<50s} ({:< 20.6f}, {:< 20.6f}, {:< 20.6f})\n'.format(i+1, molecule['elements'][i], molecule['atom_types'][i], atom_types_mapping[molecule['atom_types'][i]], molecule['geometry'][i][0],  molecule['geometry'][i][1], molecule['geometry'][i][2]))
        
        o.write('\n#\n# Masses:\n#\n')
        for d in molecule['atom_types']:
            o.write('{:<50}'.format(d))
            o.write('{:<2}'.format(atom_types_mapping[d]))
            
            if atom_types_mapping[d] not in list(parameter_data['masses'].keys()):
                o.write('  --> !! NOT FOUND !! <--\n')
            else:
                o.write('  {:<20.6f}\n'.format(parameter_data['masses'][atom_types_mapping[d]]))
        
        o.write('\n#\n# Bonds:\n#\n')
        for d in molecule['bond_types']:
            o.write('{:<50} {:<50}  '.format(d[0], d[1]))
            o.write('{:<2}-{:<2}  '.format(atom_types_mapping[d[0]], atom_types_mapping[d[1]]))
            
            if atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]] in list(parameter_data['bonds'].keys()):
                o.write('  {:<20.6f} {:<20.6f} \n'.format(parameter_data['bonds'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]][0], parameter_data['bonds'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]][1]))
            elif atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]] in list(parameter_data['bonds'].keys()):
                o.write('  {:<20.6f} {:<20.6f} \n'.format(parameter_data['bonds'][atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][0], parameter_data['bonds'][atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][1]))
            else:
                o.write('            {:^40s}\n'.format('--> !! NOT FOUND !! <--'))
            
        o.write('\n#\n# Angles:\n#\n')
        for d in molecule['angle_types']:
            o.write('{:<50} {:<50} {:<50}  '.format(d[0], d[1], d[2]))
            o.write('{:<2}-{:<2}-{:<2}  '.format(atom_types_mapping[d[0]], atom_types_mapping[d[1]], atom_types_mapping[d[2]]))
            
            if atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]] in list(parameter_data['angles'].keys()):
                o.write( '            {:<20.6f} {:<20.6f}\n'.format( parameter_data['angles'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]][0], parameter_data['angles'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]][1] ) )
            elif atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]] in list(parameter_data['angles'].keys()):
                o.write( '            {:<20.6f} {:<20.6f}\n'.format( parameter_data['angles'][atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][0], parameter_data['angles'][atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][1] ) )
            else:
                o.write('            {:^40s}\n'.format('--> !! NOT FOUND !! <--'))
                
        o.write('\n#\n# Dihedrals:\n#\n')
        for d in molecule['dihedral_types']:
            
            o.write('{:<50} {:<50} {:<50} {:<50}  '.format(d[0], d[1], d[2], d[3]))
            o.write('{:<2}-{:<2}-{:<2}-{:<2}  '.format(atom_types_mapping[d[0]], atom_types_mapping[d[1]], atom_types_mapping[d[2]], atom_types_mapping[d[3]]))
            
            dihedral_params = [0.0, 0.0, 0.0, 0.0]
            
            if atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[3]] in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[3]]]:
                    dihedral_params[int(r[3])-1] = r[4]
                    o.write('{} | {} <> '.format(r, dihedral_params))
                o.write('\n')
            elif atom_types_mapping[d[3]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]] in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals'][atom_types_mapping[d[3]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]]:
                    dihedral_params[int(r[3])-1] = r[4]
                    o.write('{} | {} <> '.format(r, dihedral_params))
                o.write('\n')
            elif 'X'+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+'X' in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals']['X'+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+'X']:
                    dihedral_params[int(r[3])-1] = r[4]
                    o.write('{} | {} <> '.format(r, dihedral_params))
                o.write('\n')
            elif 'X'+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+'X' in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals']['X'+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+'X']:
                    dihedral_params[int(r[3])-1] = r[4]
                    o.write('{} | {} <> '.format(r, dihedral_params))
                o.write('\n')
            else:
                o.write('            {:^40s}\n'.format('--> !! NOT FOUND !! <--'))
                
        if args.impropers_flag:
            o.write('\n#\n# Impropers:\n#\n')
            for d in molecule['improper_types']:
                o.write('{:<50} {:<50} {:<50} {:<50} \n'.format(atom_types_mapping[d[0]], atom_types_mapping[d[1]], atom_types_mapping[d[2]], atom_types_mapping[d[3]]))
                o.write('{:<50} {:<50} {:<50} {:<50} \n'.format(d[0], d[1], d[2], d[3]))
                
                o.write('\n')
        
        o.write('\n#\n# VDW (self-terms):\n#\n')
        for d in molecule['atom_types']:
            o.write('{:<50} {:<50}'.format(d, d))
            if atom_types_mapping[d] not in list(parameter_data['VDW'].keys()):
                o.write('{:<2} {:<2}            {:^40s}\n'.format(atom_types_mapping[d], atom_types_mapping[d], '--> !! NOT FOUND !! <--'))
            else:
                o.write('{:<2} {:<2}'.format(atom_types_mapping[d], atom_types_mapping[d]))
                o.write('  {:<20.6f} {:<20.6f}\n'.format(parameter_data['VDW'][atom_types_mapping[d]][1], parameter_data['VDW'][atom_types_mapping[d]][0]))
    
    
    
    #
    # Write out db file
    #
    
    missing_modes               = {}
    missing_modes['atoms']      = 0
    missing_modes['vdw']        = 0
    missing_modes['bonds']      = 0
    missing_modes['angles']     = 0
    missing_modes['dihedrals']  = 0
    missing_modes['impropers']  = 0
    missing_modes['charges']    = 0
    
    if args.perovskite_flag:
        print('\nWriting db file ({}) with perovskite parameters...'.format(args.output+'.db'))
    else:
        print('\nWriting db file ({})...'.format(args.output+'.db'))
    
    with open(args.output+'/'+args.output+'.db', 'w') as f:
        f.write("# Database file for FF parameters\n")
        f.write("# formatting notes:\n")
        f.write("#     vdw-lj units:          kcal/mol angstrom (eps,sigma respectively)\n")
        f.write("#     harmonic-bond units:   kcal/mol angstroms (k,r_0 respectively)\n")
        f.write("#     harmonic-angle units:  kcal/mol degrees (k,theta_0 respectively)\n")
        f.write("#     opls-torsion units:    kcal/mol for all (fourier coefficients)\n")
        f.write("#     charge units:          fraction of elementary charge\n\n")
        
        f.write("# {}\n".format(args.comments))
        
        # Write atom type definitions
        f.write("\n\n# Atom type definitions\n#\n{:<10s} {:<60s} {:<59s} {:<20s} {:<6s}\n"\
                .format("#","Atom_type","Label","Mass", "Mol_ID"))        
        done = []
        for i in sorted(molecule['atom_types']):
            if i not in done:
                if atom_types_mapping[i] not in list(parameter_data['masses'].keys()):
                    f.write("{:<10s} {:<60s} {:<59s} {:^40s}\n".format("atom",i,atom_types_mapping[i],'--> !! NOT FOUND !! <--'))                
                    missing_modes['atoms'] += 1
                else:
                    f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n".format("atom",i,atom_types_mapping[i],parameter_data['masses'][atom_types_mapping[i]]))                
                done.append(i)

        # Write VDW definitions
        f.write("\n# VDW definitions\n#\n{:<10s} {:<60s} {:<60s} {:<15s} {:<41s} {:<6s}\n"\
                .format("#","Atom_type","Atom_type","Potential","params (style determines #args)","Mol_ID"))
        done = []
        for i in sorted(molecule['atom_types']):
            if i not in done:
                if atom_types_mapping[i] not in list(parameter_data['VDW'].keys()):
                    f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:^40s}\n".format("vdw",i,i,args.vdw_type, '--> !! NOT FOUND !! <--')) 
                    missing_modes['vdw'] += 1
                else:
                    f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format("vdw",i,i,args.vdw_type, parameter_data['VDW'][atom_types_mapping[i]][1], parameter_data['VDW'][atom_types_mapping[i]][0])) 
                done.append(i)

        # Write bond definitions
        f.write("\n# Bond type definitions\n#\n{:<10s} {:<40s} {:<40s} {:<15s} {:<41s} {:<6s}\n".format("#","Atom_type","Atom_type","style","params (style determines #args)","Mol_ID"))
        for d in molecule['bond_types']:
            f.write("{:<10s} {:<40s} {:<40s} {:<15s} ".format("bond",d[0],d[1],'harmonic'))
            
            if atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]] in list(parameter_data['bonds'].keys()):
                f.write('{:<20.6f} {:<20.6f} \n'.format(parameter_data['bonds'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]][0], parameter_data['bonds'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]][1]))
            elif atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]] in list(parameter_data['bonds'].keys()):
                f.write('{:<20.6f} {:<20.6f} \n'.format(parameter_data['bonds'][atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][0], parameter_data['bonds'][atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][1]))
            else:
                f.write('{:^40s}\n'.format('--> !! NOT FOUND !! <--'))
                missing_modes['bonds'] += 1
            
        # Write angle definitions
        f.write("\n# Angle type definitions\n#\n{:<10s} {:<40s} {:<40s} {:<40s} {:<15s} {:<41s} {:<6s}\n".format("#","Atom_type","Atom_type","Atom_type","style","params (style determines #args)","Mol_ID"))
        for d in molecule['angle_types']:
            f.write("{:<10s} {:<40s} {:<40s} {:<40s} {:<15s} ".format("angle", d[0], d[1], d[2], 'harmonic'))
            
            if atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]] in list(parameter_data['angles'].keys()):
                f.write( '{:<20.6f} {:<20.6f}\n'.format( parameter_data['angles'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]][0], parameter_data['angles'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]][1] ) )
            elif atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]] in list(parameter_data['angles'].keys()):
                f.write( '{:<20.6f} {:<20.6f}\n'.format( parameter_data['angles'][atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][0], parameter_data['angles'][atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]][1] ) )
            else:
                f.write('{:^40s}\n'.format('--> !! NOT FOUND !! <--'))
                missing_modes['angles'] += 1

        # Write dihedral definitions
        f.write("\n# Dihedral/Torsional type definitions\n#\n{:<10s} {:<40s} {:<40s} {:<40s} {:<40s} {:<15s} {:<83s} {:<6s}\n".format("#","Atom_type","Atom_type","Atom_type","Atom_type","style","params (style determines #args)","Mol_ID"))
        for d in molecule['dihedral_types']:
            f.write("{:<10s} {:<40s} {:<40s} {:<40s} {:<40s} {:<15s} ".format("torsion",d[0], d[1], d[2], d[3],'opls'))
            
            # for OPLS style
            dihedral_params = [0.0, 0.0, 0.0, 0.0]
            
            if atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[3]] in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals'][atom_types_mapping[d[0]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[3]]]:
                    dihedral_params[int(r[3])-1] = r[4]
                f.write('{:<20.6f} {:<20.6f} {:<20.6f} {:<20.6f} \n'.format(dihedral_params[0], dihedral_params[1], dihedral_params[2], dihedral_params[3]))
            elif atom_types_mapping[d[3]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]] in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals'][atom_types_mapping[d[3]]+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[0]]]:
                    dihedral_params[int(r[3])-1] = r[4]
                f.write('{:<20.6f} {:<20.6f} {:<20.6f} {:<20.6f} \n'.format(dihedral_params[0], dihedral_params[1], dihedral_params[2], dihedral_params[3]))
            elif 'X'+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+'X' in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals']['X'+','+atom_types_mapping[d[1]]+','+atom_types_mapping[d[2]]+','+'X']:
                    dihedral_params[int(r[3])-1] = r[4]
                f.write('{:<20.6f} {:<20.6f} {:<20.6f} {:<20.6f} # {} \n'.format(dihedral_params[0], dihedral_params[1], dihedral_params[2], dihedral_params[3], 'X-'+atom_types_mapping[d[1]]+'-'+atom_types_mapping[d[2]]+'-X'))
            elif 'X'+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+'X' in list(parameter_data['dihedrals'].keys()):
                for r in parameter_data['dihedrals']['X'+','+atom_types_mapping[d[2]]+','+atom_types_mapping[d[1]]+','+'X']:
                    dihedral_params[int(r[3])-1] = r[4]
                f.write('{:<20.6f} {:<20.6f} {:<20.6f} {:<20.6f} # {} \n'.format(dihedral_params[0], dihedral_params[1], dihedral_params[2], dihedral_params[3], 'X-'+atom_types_mapping[d[2]]+'-'+atom_types_mapping[d[1]]+'-X'))
            else:
                f.write('{:^40s}\n'.format('--> !! NOT FOUND !! <--'))
                missing_modes['dihedrals'] += 1

        # Write charge definitions
        f.write('\n# Charge definitions\n#\n{:<10s} {:<61s} {:<6s}\n'.format('#','Atom_type','Charge','Mol_ID'))
        
        if args.charge_file != None:
            done = []
            for i in sorted(molecule['atom_types']):
                if i not in done:
                    if i not in list(molecule['charges'].keys()):
                        f.write('{:<10s} {:<60s} {:^40s}\n'.format('charge', i, '--> !! NOT FOUND !! <--'))
                        missing_modes['charges'] += 1
                    else:
                        f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', i, molecule['charges'][i]))
                    done.append(i)
        else:
            f.write('#--> !! NEED CHARGES !! <--\n')
        
        if args.perovskite_flag:
            f.write('\n#Perovskite parameters, including water\n')
            
            if args.water_model.upper() == 'SPCE':
                f.write('# SPC/E water\n')
                f.write('# water spc/e. from AmberTools17/amber16/dat/leap/parm/frcmod.spce\n')
                f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n".format('atom','[1[8[1]]]','H_spce',1.008))
                f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n\n".format('atom','[8[1][1]]','O_spce',15.9994))
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format('vdw','[1[8[1]]]','[1[8[1]]]','lj', 0.0000, 0.0000)) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format('vdw','[8[1][1]]','[8[1][1]]','lj', 0.1553, 3.16572)) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[1[8[1]]]','[8[1][1]]','lj', 0.0000,0.0000,'# H_spce-O_spce' )) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[53]','[1[8[1]]]','lj',0.0000,2.511443,'# H_spce-I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.370273             4.094443 {} \n".format('vdw','[53]','[8[1][1]]','lj','# O_spce-I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.9f} {} \n".format('vdw','[35]','[1[8[1]]]','lj',0.0000,2.245065,'# H_spce-Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.345882             3.828065 {} \n".format('vdw','[35]','[8[1][1]]','lj','# O_spce-Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.9f} {} \n".format('vdw','[82[35]]','[1[8[1]]]','lj',0.0000,1.5662,'# H_spce-Pb_Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.167185             3.1492 {} \n".format('vdw','[82[35]]','[8[1][1]]','lj','# O_spce-Pb_Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.9f} {} \n".format('vdw','[82[53]]','[1[8[1]]]','lj',0.0000,1.5662,'# H_spce-Pb_I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.167185             3.1492 {} \n\n".format('vdw','[82[53]]','[8[1][1]]','lj','# O_spce-Pb_I')) 
                f.write("{:<10s} {:<40s} {:<40s} {:<15s} {:<20.6f} {:<20.6f} \n\n".format('bond','[1[8[1]]]','[8[1][1]]','harmonic', 1000.00, 1.000))
                f.write("{:<10s} {:<40s} {:<40s} {:<40s} {:<15s} {:<20.6f} {:<20.6f}\n\n".format('angle', '[1[8[1]]]', '[8[1][1]]', '[1[8[1]]]', 'harmonic', 100.00, 109.47))
                f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', '[1[8[1]]]', 0.4238))
                f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', '[8[1][1]]', -0.8476))

            elif args.water_model.upper() == 'TIP3P':
                f.write('# TIP3P water\n')
                f.write('# water TIP3P. from AmberTools17/amber16/dat/leap/parm/. monovalent: frcmod.ions1lm_126_tip3p; divalent: frcmod.ions234lm_126_tip3p\n')
                f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n".format('atom','[1[8[1]]]','H_tip3p',1.008))
                f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n\n".format('atom','[8[1][1]]','O_tip3p',15.9994))
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format('vdw','[1[8[1]]]','[1[8[1]]]','lj', 0.0000, 0.0000)) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format('vdw','[8[1][1]]','[8[1][1]]','lj', 0.102, 3.188)) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[1[8[1]]]','[8[1][1]]','lj', 0.0000, 0.0000, '# H_tip3p-O_tip3p' )) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             2.426808108 {} \n".format('vdw','[53]','[1[8[1]]]','lj',0.0000,'# H_tip3p-I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.29517207             4.020808108 {} \n".format('vdw','[53]','[8[1][1]]','lj', '# O_tip3p-I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             2.163102088 {} \n".format('vdw','[35]','[1[8[1]]]','lj',0.0000, '# H_tip3p-Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.271131626             3.757102088 {} \n".format('vdw','[35]','[8[1][1]]','lj', '# O_tip3p-Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             1.554618263 {}\n".format('vdw','[82[35]]','[1[8[1]]]','lj',0.0000, '# H_tip3p-Pb_Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.131751415             3.148618263 {} \n".format('vdw','[82[35]]','[8[1][1]]','lj', '# O_tip3p-Pb_Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             1.554618263 {}\n".format('vdw','[82[53]]','[1[8[1]]]','lj',0.0000, '# H_tip3p-Pb_I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.131751415             3.148618263 {} \n\n".format('vdw','[82[53]]','[8[1][1]]','lj', '# O_tip3p-Pb_I')) 
                f.write("{:<10s} {:<40s} {:<40s} {:<15s} {:<20.6f} {:<20.6f} \n\n".format('bond','[1[8[1]]]','[8[1][1]]','harmonic', 450.00, 0.9572))
                f.write("{:<10s} {:<40s} {:<40s} {:<40s} {:<15s} {:<20.6f} {:<20.6f}\n\n".format('angle', '[1[8[1]]]', '[8[1][1]]', '[1[8[1]]]', 'harmonic', 55.00, 104.52))
                f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', '[1[8[1]]]', 0.415))
                f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', '[8[1][1]]', -0.830))
            
            elif args.water_model.upper() == 'TIP4P':
                f.write('# TIP4P water\n')
                f.write('# water TIP4P. from AmberTools17/amber16/dat/leap/parm/. monovalent: frcmod.ions1lm_126_tip4pew; divalent: frcmod.ions234lm_126_tip4pew\n')
                f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n".format('atom','[1[8[1]]]','H_tip4p',1.008))
                f.write("{:<10s} {:<60s} {:<59s} {:<20.6f}\n\n".format('atom','[8[1][1]]','O_tip4p',15.9994))
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format('vdw','[1[8[1]]]','[1[8[1]]]','tip4p', 0.0000, 0.0000)) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f}\n".format('vdw','[8[1][1]]','[8[1][1]]','tip4p', 0.16275, 3.16435)) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[1[8[1]]]','[8[1][1]]','tip4p', 0.0000, 0.0000, '# H_tip4p-O_tip4p' )) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             2.511443486 {} \n".format('vdw','[53]','[1[8[1]]]','lj',0.0000,'# H_tip4p-I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.379049953             4.093618486 {} \n".format('vdw','[53]','[8[1][1]]','lj', '# O_tip4p-I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             2.24506477 {} \n".format('vdw','[35]','[1[8[1]]]','lj',0.0000, '# H_tip4p-Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.354080802             3.82723977 {} \n".format('vdw','[35]','[8[1][1]]','lj', '# O_tip4p-Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             1.566199946 {}\n".format('vdw','[82[35]]','[1[8[1]]]','lj',0.0000, '# H_tip4p-Pb_Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.171148123             3.148374946 {} \n".format('vdw','[82[35]]','[8[1][1]]','lj', '# O_tip4p-Pb_Br')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f}             1.566199946 {}\n".format('vdw','[82[53]]','[1[8[1]]]','lj',0.0000, '# H_tip4p-Pb_I')) 
                f.write("{:<10s} {:<60s} {:<60s} {:<15s} 0.171148123             3.148374946 {} \n\n".format('vdw','[82[53]]','[8[1][1]]','lj', '# O_tip4p-Pb_I')) 
                f.write("{:<10s} {:<40s} {:<40s} {:<15s} {:<20.6f} {:<20.6f} \n\n".format('bond','[1[8[1]]]','[8[1][1]]','harmonic', 450.00, 0.9572))
                f.write("{:<10s} {:<40s} {:<40s} {:<40s} {:<15s} {:<20.6f} {:<20.6f}\n\n".format('angle', '[1[8[1]]]', '[8[1][1]]', '[1[8[1]]]', 'harmonic', 55.00, 104.52))
                f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', '[1[8[1]]]', 0.5242))
                f.write('{:<10s} {:<60s} {:<20.6f}\n'.format('charge', '[8[1][1]]', -1.0484))
            
            f.write('\n# Pb-Pb\n# [82[35]]: heterojunction, associated with Br model\n# [82[53]]: heterojunction, associated with I model\n')
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 74933300.5606326   0.123246948356808   0.000000  {} \n".format('vdw','[82[35]]','[82[35]]','buck','# Pb_Br-Pb_Br')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 70359906.629702    0.131258            0.000000  {} \n".format('vdw','[82[53]]','[82[53]]','buck','# Pb_I-Pb_I')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 72610605.4987941   0.127252474         0.000000  {} \n".format('vdw','[82[35]]','[82[53]]','buck','# Pb_Br-Pb_I\n')) 
            f.write('\n# Halides\n')
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 24274.90558983   0.45286103286385   654.4127155 {} \n".format('vdw','[35]','[35]','buck','# Br-Br')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 23522.46038      0.467539016        675.6811288 {} \n".format('vdw','[35]','[53]','buck','# Br-I')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 22793.33858      0.482217           696.949542 {} \n".format('vdw','[53]','[53]','buck','# I-I\n')) 
            f.write('\n# Pb-Halide crossterms\n')
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 110223.38165565   0.302100469483568   0.000000 {} \n".format('vdw','[82[35]]','[35]','buck','# Pb_Br-Br')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 1306897.1232447   0.292059516431925   0.000000 {} \n".format('vdw','[82[53]]','[35]','buck','# Pb_I-Br')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 1306897.1232447   0.302731974178404   0.000000 {} \n".format('vdw','[82[35]]','[53]','buck','# Pb_Br-I')) 
            f.write("{:<10s} {:<60s} {:<60s} {:<15s} 103496.133010     0.321737            0.000000 {} \n".format('vdw','[82[53]]','[53]','buck','# Pb_I-I')) 

            f.write('\n# Halide - C, N, S, O, F\n')
            done = []
            for i in range(len(molecule['elements'])):
                if molecule['elements'][i] in ['C', 'N', 'S', 'O', 'F']:
                    if molecule['atom_types'][i] not in done:
                        f.write("{:<10s} {:<60s} {:<60s} {:<15s} 94836.351975893   0.3352375   0.000000 # Br-{} \n".format('vdw','[35]',molecule['atom_types'][i],'buck',molecule['elements'][i]))
                        f.write("{:<10s} {:<60s} {:<60s} {:<15s} 112936.714213     0.342426    0.000000 # I-{} \n".format('vdw','[53]',molecule['atom_types'][i],'buck',molecule['elements'][i]))
                        done.append(molecule['atom_types'][i])
                        
                        
            f.write('\n# Halide - H\n')
            done = []
            
            for i in range(len(molecule['elements'])):
                if molecule['elements'][i] == 'H':
                    if molecule['atom_types'][i] not in done:
                        if molecule['atom_types'][i] == '[1[7[6][1][1]]]':
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} 2.45535714285714 {} \n".format('vdw','[35]','[1[7[6][1][1]]]','lj',2.1853,'# Br-H, HN, nitrogen'))
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[53]','[1[7[6][1][1]]]','lj',2.447536,2.75,'# I-H, HN, nitrogen'))
                        else:
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} 2.767857143 {} \n".format('vdw','[35]',molecule['atom_types'][i],'lj',2.1853,'# Br-H'))
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[53]',molecule['atom_types'][i],'lj',2.447536,3.1,'# I-H'))
                        done.append(molecule['atom_types'][i])
            
            f.write('\n# Pb - C, N, S, O, F\n')
            done = []
            for i in range(len(molecule['elements'])):
                if molecule['elements'][i] in ['C', 'N', 'S', 'O', 'F']:
                    if molecule['atom_types'][i] not in done:
                        f.write("{:<10s} {:<60s} {:<60s} {:<15s} 32690390.937995 0.150947 0.000000 # Pb_Br-{} \n".format('vdw','[82[35]]',molecule['atom_types'][i],'buck',molecule['elements'][i]))
                        f.write("{:<10s} {:<60s} {:<60s} {:<15s} 32690390.937995 0.150947 0.000000 # Pb_I-{} \n".format('vdw','[82[53]]',molecule['atom_types'][i],'buck',molecule['elements'][i]))
                        done.append(molecule['atom_types'][i])

            f.write('\n# Pb-H\n')
            done = []
            for i in range(len(molecule['elements'])):
                if molecule['elements'][i] == 'H':
                    if molecule['atom_types'][i] not in done:
                        if molecule['atom_types'][i] == '[1[7[6][1][1]]]':
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[82[35]]','[1[7[6][1][1]]]','lj', 0.59696,2.26454,'# # Pb-H, HN, nitrogen'))
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[82[53]]','[1[7[6][1][1]]]','lj', 0.59696,2.26454,'# # Pb-H, HN, nitrogen'))
                        else:
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[82[35]]',molecule['atom_types'][i],'lj',0.59696,2.70999,'# Pb-H, HC, carbon'))
                            f.write("{:<10s} {:<60s} {:<60s} {:<15s} {:<20.6f} {:<20.6f} {} \n".format('vdw','[82[53]]',molecule['atom_types'][i],'lj',0.59696,2.70999,'# Pb-H, HC, carbon'))
                        done.append(molecule['atom_types'][i])

    if missing_modes['atoms'] + missing_modes['vdw'] + missing_modes['bonds'] + missing_modes['angles'] + missing_modes['dihedrals'] > 0:
        print('\nWARNING: Missing parameters for some modes...')
        for key in list(missing_modes.keys()):
            if missing_modes[key] > 0:
                print('    - Did not find parameters for {} {}.'.format(missing_modes[key], key))
    
    print('\nFinished!\n')
    return

# Read in the antechamber file that contains the identified atoms and then build a mapping dictonary
# between the TAFFI-generated atom types and the antechamber atom types
def process_antechamber_file(name, TAFFI_atom_types):

    other_atom_types = []
    with open(name, 'r') as m:
        for line in m:
            line = line.split()
            if line[0] == 'ATOM':
                other_atom_types.append(line[-1])
            elif line[0] == 'BOND':
                break
    
    atom_types_mapping = {}
    
    if not args.TAFFI_flag:
        if len(other_atom_types) != len(TAFFI_atom_types):
            print('ERROR: Number of TAFFI atom types does not match the number of other atom types. Check .xyz file. Aborting...')
            exit()
        
        for i in range(len(TAFFI_atom_types)):
            if TAFFI_atom_types[i] not in list(atom_types_mapping.keys()):
                atom_types_mapping[TAFFI_atom_types[i]] = other_atom_types[i]
    else:
        for i in range(len(other_atom_types)):
            if other_atom_types[i] not in list(atom_types_mapping.keys()):
                atom_types_mapping[other_atom_types[i]] = other_atom_types[i]
    
    return other_atom_types, atom_types_mapping
    
def read_charges_db(db_name):
    charges = {}
    with open(db_name, 'r') as f:
        for line in f:
            fields = line.split()
            
            if len(fields) == 0:
                continue
            elif line.split()[0] == '#':
                continue
            elif line[0] == '#':
                continue
            elif '-UA' in line:
                continue
            else:
                charges[str(fields[1])] = float(fields[2])
    return charges

def read_parameters_file(parameters_file):
    
    parameter_data               = {}
    parameter_data['masses']     = {}       # sigma (A), epsilon
    parameter_data['bonds']      = {}
    parameter_data['angles']     = {}
    parameter_data['dihedrals']  = {}
    parameter_data['impropers']  = {}
    parameter_data['VDW']        = {}       # sigma (A), epsilon
    vdw_flag                     = False
    break_counter                = 0
    lc                           = 0
    params_holder                = []
    
    with open(parameters_file, 'r') as f:
        for line in f:
            fields = line.split()
            
            if len(fields) == 0:
                break_counter += 1
                lc = 0
                continue
            
            # Masses
            if break_counter == 0:
                lc += 1
                if lc > 1:
                    parameter_data['masses'][str(fields[0])] =  float(fields[1])
            
            # Bond parameters
            elif break_counter == 1:
                lc += 1
                if lc > 1:
                    # Want to format as (atom1,atom2,atom3) and remove any whitespace between the dashes (for single-character names, a space is used for padding)
                    # e.g. 'i -c2-i'
                    name = [_.strip() for _ in line[:6].split('-')]
                    parameter_data['bonds']['{},{}'.format(name[0],name[1])] =  [float(line[7:14].strip()), float(line[16:22].strip())]
            
            # Angle parmaeters
            elif break_counter == 2:
                name = [_.strip() for _ in line[:9].split('-')]
                parameter_data['angles']['{},{},{}'.format(name[0],name[1],name[2])] =  [float(line[10:18].strip()), float(line[21:29].strip())]
                
            
            # Dihedral parameters
            # IDIVF, PK, phase, PN, (2*PK)/IDIVF
            # NB: there are four terms for each dihedral angle. they are 0 unless specified. if __ has a negative value, this flags that there are additional
            #     terms, so keep reading until the next positive value is encountered.

            elif break_counter == 3:
                name = [_.strip() for _ in line[:11].split('-')]
                parameters = [int(line[13:17].strip()), float(line[17:28].strip()), float(line[28:43].strip()), abs( int(float(line[43:56].strip())) ), (2*float(line[17:28].strip()))/float(line[13:17].strip())]
                
                if int(float(line[43:56].strip())) < 0.0:
                    params_holder.append(deepcopy(parameters))
                else:
                    if len(params_holder) > 0:
                        params_holder.append(deepcopy(parameters))
                        parameter_data['dihedrals']['{},{},{},{}'.format(name[0],name[1],name[2],name[3])] = deepcopy(params_holder)
                        params_holder = []
                    else:
                        parameter_data['dihedrals']['{},{},{},{}'.format(name[0],name[1],name[2],name[3])] = [deepcopy(parameters)]
            
            # Improper parameters
            elif break_counter == 4:
                name = [_.strip() for _ in line[:11].split('-')]
                parameters = [float(line[17:30].strip()), float(line[30:43].strip()), abs( int(float(line[43:55].strip())) )]
                
                if int(float(line[43:55].strip())) < 0.0:
                    params_holder.append(deepcopy(parameters))
                else:
                    if len(params_holder) > 0:
                        params_holder.append(deepcopy(parameters))
                        parameter_data['impropers']['{},{},{},{}'.format(name[0],name[1],name[2],name[3])] = deepcopy(params_holder)
                        params_holder = []
                    else:
                        parameter_data['impropers']['{},{},{},{}'.format(name[0],name[1],name[2],name[3])] = deepcopy(parameters)
            
    
            # VDW section
            if line.strip() == 'MOD4      RE':
                vdw_flag = True
                continue
            
            if line.strip() == 'END':
                vdw_flag = False
                break
            
            if vdw_flag and len(fields) > 0:
                # second field is R_min; converting to sigma, sigma = (2*R_min)/(2^(1/6))
                # 2^(1/6) = 1.12246204830937
                parameter_data['VDW'][str(fields[0])] =  [((2*float(fields[1]))/1.12246204830937), float(fields[2])]
    
    return parameter_data

def read_xyz(xyz_file):
    
    # Check to make sure the files exist.
    if not os.path.isfile(xyz_file):
        print('\nERROR: Specified .yxz file "{}" not found. Aborting....\n'.format(xyz_file))
        exit()
    
    molecule = {}
    molecule['elements']          = []
    molecule['read_atom_types']   = []
    flag_read_atoms               = False
    
    with open(xyz_file, 'r') as f:
        for count, line in enumerate(f):
            if count == 0:
                molecule['atom_count']  = int(line)
                molecule['geometry']    = numpy.zeros([molecule['atom_count'], 3])
                molecule['atom_types']  = []
                atoms_read = 0
            
            elif count == 1:
                # skip the blank line / comment line
                continue
            
            else:
                if atoms_read > molecule['atom_count']:
                    print('ERROR: Discovered more atoms in .xyz file than expected. Aborting....')
                    exit()
                
                fields = line.split()
                if len(fields) > 0:
                    molecule['elements'].append(str(fields[0]))
                    molecule['geometry'][count-2] = float(fields[1]), float(fields[2]), float(fields[3])
                    if len(fields) == 5: 
                        molecule['read_atom_types'].append(str(fields[4]))
                        flag_read_atoms = True
                    atoms_read += 1
    
    if len(molecule['read_atom_types']) != len(molecule['elements']) and flag_read_atoms:
        print('ERROR: The number of read atom types does not match the number of elements. If you are specifying atomtypes, ensure that each element has an atomtype. Exiting...')
    
    molecule['adj_mat'] = adjacency.Table_generator(molecule['elements'], molecule['geometry'])
    molecule['atom_types'] = id_types.id_types(molecule['elements'], molecule['adj_mat'])
    
    return molecule

def get_modes(Adj_mat, Atom_types, Improper_flag):
    
        # List comprehension to determine bonds from a loop over the adjacency matrix. Iterates over rows (i) and individual elements
    # ( elements A[count_i,count_j] = j ) and stores the bond if the element is "1". The count_i < count_j condition avoids
    # redudant bonds (e.g., (i,j) vs (j,i) ). By convention only the i < j definition is stored.
    print("\t\tParsing bonds...")
    Bonds          = [ (count_i,count_j) for count_i,i in enumerate(Adj_mat) for count_j,j in enumerate(i) if j == 1 ]
    Bond_types     = [ (Atom_types[i[0]],Atom_types[i[1]]) for i in Bonds ]

    # List comprehension to determine angles from a loop over the bonds. Note, since there are two bonds in every angle, there will be
    # redundant angles stored (e.g., (i,j,k) vs (k,j,i) ). By convention only the i < k definition is stored.
    print("\t\tParsing angles...")
    Angles          = [ (count_j,i[0],i[1]) for i in Bonds for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Angle_types     = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]]) for i in Angles ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("\t\tParsing dihedrals...")
    Dihedrals      = [ (count_j,i[0],i[1],i[2]) for i in Angles for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Dihedral_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Dihedrals ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    if Improper_flag:    print("\t\tParsing impropers...")
    Impropers      = [ (i[1],i[0],i[2],count_j) for i in Angles for count_j,j in enumerate(Adj_mat[i[1]]) if j == 1 and count_j not in i ]
    Improper_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Impropers ]

    # Canonicalize the modes
    for i in range(len(Bonds)):
        Bond_types[i],Bonds[i] = adjacency.canon_bond(Bond_types[i],ind=Bonds[i])
    for i in range(len(Angles)):
        Angle_types[i],Angles[i] = adjacency.canon_angle(Angle_types[i],ind=Angles[i])
    for i in range(len(Dihedrals)):
        Dihedral_types[i],Dihedrals[i] = adjacency.canon_dihedral(Dihedral_types[i],ind=Dihedrals[i])
    for i in range(len(Impropers)):
        Improper_types[i],Impropers[i] = adjacency.canon_improper(Improper_types[i],ind=Impropers[i])        

    # Remove redundancies
    if len(Bonds) > 0:
        Bonds,Bond_types = list(zip(*[ (i,Bond_types[count_i]) for count_i,i in enumerate(Bonds) if count_i == [ count_j for count_j,j in enumerate(Bonds) if j == i or j[::-1] == i ][0]  ]))
    
    if len(Angles) > 0:
        Angles,Angle_types = list(zip(*[ (i,Angle_types[count_i]) for count_i,i in enumerate(Angles) if count_i == [ count_j for count_j,j in enumerate(Angles) if j == i or j[::-1] == i ][0]  ]))
    
    if len(Dihedrals) > 0:
        Dihedrals,Dihedral_types = list(zip(*[ (i,Dihedral_types[count_i]) for count_i,i in enumerate(Dihedrals) if count_i == [ count_j for count_j,j in enumerate(Dihedrals) if j == i or j[::-1] == i ][0]  ]))
    
    if len(Impropers) > 0:
        Impropers,Improper_types = list(zip(*[ (i,Improper_types[count_i]) for count_i,i in enumerate(Impropers) if count_i == [ count_j for count_j,j in enumerate(Impropers) if j[0] == i[0] and len(set(i[1:]).intersection(set(j[1:]))) ][0] ]))
    
    unique_Bond_types = []
    for d in Bond_types:
        if d not in unique_Bond_types:
            unique_Bond_types.append(d)
            
    unique_Angle_types = [] 
    for d in Angle_types:
        if d not in unique_Angle_types:
            unique_Angle_types.append(d)
    
    unique_Dihedral_types = [] 
    for d in Dihedral_types:
        if d not in unique_Dihedral_types:
            unique_Dihedral_types.append(d)
    
    unique_Improper_types = []
    for d in Improper_types:
        if d not in unique_Improper_types:
            unique_Improper_types.append(d)
    
    #return Bond_types, Angle_types, Dihedral_types, Improper_types
    return unique_Bond_types, unique_Angle_types, unique_Dihedral_types, unique_Improper_types

def get_masses(elements):
    # Returns the mass of an element
    # Initialize mass_dict (used for identifying the dihedral among a coincident set that will be explicitly scanned)
    mass_dict = {'H':1.00794,'He':4.002602,'Li':6.941,'Be':9.012182,'B':10.811,'C':12.011,'N':14.00674,'O':15.9994,'F':18.9984032,'Ne':20.1797,\
                 'Na':22.989768,'Mg':24.3050,'Al':26.981539,'Si':28.0855,'P':30.973762,'S':32.066,'Cl':35.4527,'Ar':39.948,\
                 'K':39.0983,'Ca':40.078,'Sc':44.955910,'Ti':47.867,'V':50.9415,'Cr':51.9961,'Mn':54.938049,'Fe':55.845,'Co':58.933200,'Ni':58.6934,'Cu':63.546,'Zn':65.39,\
                 'Ga':69.723,'Ge':72.61,'As':74.92159,'Se':78.96,'Br':79.904,'Kr':83.80,\
                 'Rb':85.4678,'Sr':87.62,'Y':88.90585,'Zr':91.224,'Nb':92.90638,'Mo':95.94,'Tc':98.0,'Ru':101.07,'Rh':102.90550,'Pd':106.42,'Ag':107.8682,'Cd':112.411,\
                 'In':114.818,'Sn':118.710,'Sb':121.760,'Te':127.60,'I':126.90447,'Xe':131.29,\
                 'Cs':132.90545,'Ba':137.327,'La':138.9055,'Hf':178.49,'Ta':180.9479,'W':183.84,'Re':186.207,'Os':190.23,'Ir':192.217,'Pt':195.078,'Au':196.96655,'Hg':200.59,\
                 'Tl':204.3833,'Pb':207.2,'Bi':208.98038,'Po':209.0,'At':210.0,'Rn':222.0}

    #masses = [ mass_dict[i] for i in elements ]
    masses = [ mass_dict[i] for i in elements ]
    #return mass_dict[elements]
    return masses

# A wrapper for the commands to parse the bonds, angles, and dihedrals from the adjacency matrix.
# Returns:   list of atomtypes, bond_types, bond instances, angle_types, angle instances, dihedral_types,
#            diehdral instances, charges, and VDW parameters.
def Find_parameters(Adj_mat,Geometry,Atom_types,FF_db="FF_file",Improper_flag = False):

    # List comprehension to determine bonds from a loop over the adjacency matrix. Iterates over rows (i) and individual elements
    # ( elements A[count_i,count_j] = j ) and stores the bond if the element is "1". The count_i < count_j condition avoids
    # redudant bonds (e.g., (i,j) vs (j,i) ). By convention only the i < j definition is stored.
    print("\t\tParsing bonds...")
    Bonds          = [ (count_i,count_j) for count_i,i in enumerate(Adj_mat) for count_j,j in enumerate(i) if j == 1 ]
    Bond_types     = [ (Atom_types[i[0]],Atom_types[i[1]]) for i in Bonds ]

    # List comprehension to determine angles from a loop over the bonds. Note, since there are two bonds in every angle, there will be
    # redundant angles stored (e.g., (i,j,k) vs (k,j,i) ). By convention only the i < k definition is stored.
    print("\t\tParsing angles...")
    Angles          = [ (count_j,i[0],i[1]) for i in Bonds for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Angle_types     = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]]) for i in Angles ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    print("\t\tParsing dihedrals...")
    Dihedrals      = [ (count_j,i[0],i[1],i[2]) for i in Angles for count_j,j in enumerate(Adj_mat[i[0]]) if j == 1 and count_j not in i ]
    Dihedral_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Dihedrals ]

    # List comprehension to determine dihedrals from a loop over the angles. Note, since there are two angles in every dihedral, there will be
    # redundant dihedrals stored (e.g., (i,j,k,m) vs (m,k,j,i) ). By convention only the i < m definition is stored.
    if Improper_flag:    print("\t\tParsing impropers...")
    Impropers      = [ (i[1],i[0],i[2],count_j) for i in Angles for count_j,j in enumerate(Adj_mat[i[1]]) if j == 1 and count_j not in i ]
    Improper_types = [ (Atom_types[i[0]],Atom_types[i[1]],Atom_types[i[2]],Atom_types[i[3]]) for i in Impropers ]

    # Canonicalize the modes
    for i in range(len(Bonds)):
        Bond_types[i],Bonds[i] = adjacency.canon_bond(Bond_types[i],ind=Bonds[i])
    for i in range(len(Angles)):
        Angle_types[i],Angles[i] = adjacency.canon_angle(Angle_types[i],ind=Angles[i])
    for i in range(len(Dihedrals)):
        Dihedral_types[i],Dihedrals[i] = adjacency.canon_dihedral(Dihedral_types[i],ind=Dihedrals[i])
    for i in range(len(Impropers)):
        Improper_types[i],Impropers[i] = adjacency.canon_improper(Improper_types[i],ind=Impropers[i])        

    # Remove redundancies
    if len(Bonds) > 0:
        Bonds,Bond_types = list(zip(*[ (i,Bond_types[count_i]) for count_i,i in enumerate(Bonds) if count_i == [ count_j for count_j,j in enumerate(Bonds) if j == i or j[::-1] == i ][0]  ]))
    
    if len(Angles) > 0:
        Angles,Angle_types = list(zip(*[ (i,Angle_types[count_i]) for count_i,i in enumerate(Angles) if count_i == [ count_j for count_j,j in enumerate(Angles) if j == i or j[::-1] == i ][0]  ]))
    
    if len(Dihedrals) > 0:
        Dihedrals,Dihedral_types = list(zip(*[ (i,Dihedral_types[count_i]) for count_i,i in enumerate(Dihedrals) if count_i == [ count_j for count_j,j in enumerate(Dihedrals) if j == i or j[::-1] == i ][0]  ]))
    
    if len(Impropers) > 0:
        Impropers,Improper_types = list(zip(*[ (i,Improper_types[count_i]) for count_i,i in enumerate(Impropers) if count_i == [ count_j for count_j,j in enumerate(Impropers) if j[0] == i[0] and len(set(i[1:]).intersection(set(j[1:]))) ][0] ]))

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
    with open(FF_db,'r') as f:
        content=f.readlines()
        
    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'atom':
            Masses[fields[1]] = float(fields[3]) 
             

    # Read in bond parameters
    Bond_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0] == 'bond':
            if fields[3] == "harmonic":
                Bond_params[(fields[1],fields[2])] = [float(fields[4]),float(fields[5])]
            elif fields[3] == "fixed":
                Bond_params[(fields[1],fields[2])] = ["fixed",0.0,float(fields[4])]
            else:
                print("ERROR: only harmonic and fixed bond definitions are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Read in angle parameters
    Angle_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() == 'angle':
            if fields[4] == "harmonic":
                Angle_params[(fields[1],fields[2],fields[3])] = [float(fields[5]),float(fields[6])]
            elif fields[4] == "fixed":
                Angle_params[(fields[1],fields[2],fields[3])] = ["fixed",0.0,float(fields[5])]
            else:
                print("ERROR: only harmonic angle definitions are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Read in dihedral parameters
    Dihedral_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['dihedral','torsion']:            
            if fields[5] == "opls":
                Dihedral_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(i) for i in fields[6:10] ]
            elif fields[5] == "harmonic":
                Dihedral_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),int(float(fields[7])),int(float(fields[8])) ]
            else:
                print("ERROR: Only opls and harmonic dihedral types are currently supported by perovskite_builder.py. Exiting...")
                quit()

    # Read in improper parameters
    Improper_params = {}
    with open(FF_db,'r') as f:
        content=f.readlines()

    for lines in content:
        fields=lines.split()

        # Skip empty lines
        if len(fields) == 0:
            continue

        if fields[0].lower() in ['improper']:
            if fields[5] == "harmonic":
                Improper_params[(fields[1],fields[2],fields[3],fields[4])] = [fields[5]] + [ float(fields[6]),float(fields[7])]
            else:
                print("ERROR: Only opls type dihedral definitions are currently supported by perovskite_builder.py. Exiting...")
                quit()
                
    # Search for charges based on atom type
    with open(FF_db,'r') as f:
        content=f.readlines()

    Charges = numpy.zeros(len(Atom_types))
    for i in range(len(Atom_types)):
        for lines in content:
            fields=lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue
                    
            if fields[0].lower() in ['charge'] and Atom_types[i] == fields[1]:
                Charges[i] = float(fields[2])

    # Search for VDW parameters
    VDW_params = {}
    VDW_comments = {}
    with open(FF_db,'r') as f:
        for lines in f:
            fields = lines.split()

            # Skip empty lines
            if len(fields) == 0:
                continue
                    
            if fields[0].lower() in ['vdw']:

                # Only two parameters are required for lj types
                if fields[3] == "lj":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5])]
                        if len(fields) > 6:
                            VDW_comments[(fields[1],fields[2])] = ' '.join(fields[6:])
                        else:
                            VDW_comments[(fields[1],fields[2])] = ''
                    else:
                        VDW_params[(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5])]
                        if len(fields) > 6:
                            VDW_comments[(fields[2],fields[1])] = ' '.join(fields[6:])
                        else:
                            VDW_comments[(fields[2],fields[1])] = ''
                
                elif fields[3] == "buck":
                    if fields[1] > fields[2]:
                        VDW_params[(fields[1],fields[2])] = [fields[3],float(fields[4]),float(fields[5]),float(fields[6])]
                        if len(fields) > 7:
                            VDW_comments[(fields[1],fields[2])] = ' '.join(fields[7:])
                        else:
                            VDW_comments[(fields[1],fields[2])] = ''
                    else:
                        VDW_params[(fields[2],fields[1])] = [fields[3],float(fields[4]),float(fields[5]),float(fields[6])]
                        if len(fields) > 7:
                            VDW_comments[(fields[2],fields[1])] = ' '.join(fields[7:])
                        else:
                            VDW_comments[(fields[2],fields[1])] = ''

    # Check for missing parameters
    Missing_masses = [ i for i in Atom_types if str(i) not in list(Masses.keys()) ] 
    Missing_charges = [ count_i for count_i,i in enumerate(Charges) if i == -100.0 ]; Missing_charges = [ Atom_types[i] for i in Missing_charges ]
    Missing_bonds = [ i for i in Bond_types if (i[0],i[1]) not in list(Bond_params.keys()) ]
    Missing_angles = [ i for i in Angle_types if (i[0],i[1],i[2]) not in list(Angle_params.keys()) ]
    Missing_dihedrals = [ i for i in Dihedral_types if (i[0],i[1],i[2],i[3]) not in list(Dihedral_params.keys()) ]
    Missing_impropers = []
    if Improper_flag is True: Missing_impropers = [ i for i in Improper_types if (i[0],i[1],i[2],i[3]) not in list(Improper_params.keys()) ]

    # Print diagnostics on missing parameters and quit if the prerequisites are missing.
    if ( len(Missing_masses) + len(Missing_charges) + len(Missing_bonds) + len(Missing_angles) + len(Missing_dihedrals) + len(Missing_impropers) ) > 0:
        print("\nUh Oh! There are missing FF parameters...\n")

        if Missing_masses:
            print("Missing masses for the following atom types: {}".format([ i for i in set(Missing_masses) ]))
        if Missing_charges:
            print("Missing charges for the following atom types: {}".format([ i for i in set(Missing_charges) ]))
        if Missing_bonds:
            print("Missing bond parameters for the following bond types: {}".format([ i for i in set(Missing_bonds) ]))
        if Missing_angles:
            print("Missing angle parameters for the following angle types: {}".format([ i for i in set(Missing_angles) ]))
        if Missing_dihedrals:
            print("Missing dihedral parameters for the following dihedral types: {}".format([ i for i in set(Missing_dihedrals) ]))
        if Improper_flag and Missing_impropers:
            print("Missing improper parameters for the following improper types: {}".format([ i for i in set(Missing_impropers) ]))
        
        print("\nEnsure the specification of the missing parameters. Exiting...")
        quit()

    return list(Bonds),list(Bond_types),Bond_params,list(Angles),list(Angle_types),Angle_params,list(Dihedrals),list(Dihedral_types),Dihedral_params,list(Impropers),list(Improper_types),Improper_params,Charges.tolist(),Masses,VDW_params,VDW_comments

# Wrapper function for writing the lammps data file which is read by the .in.init file during run initialization
def Write_data(Filename,Atom_types,Sim_Box,Elements,Geometry,Bonds,Bond_types,Bond_params,Angles,Angle_types,Angle_params,Dihedrals,Dihedral_types,Dihedral_params,\
               Impropers,Improper_types,Improper_params,Charges,VDW_params,Masses,Molecule,VDW_comments,Improper_flag=False):
    
    # Write an xyz for easy viewing
    #with open(Filename+'/'+Filename.split('/')[-1]+'.xyz','w') as f:
    #with open(Filename.split('/')[-1]+'.xyz','w') as f:
    with open(Filename+'.xyz','w') as f:
        f.write('{}\n\n'.format(len(Geometry)))
        for count_i,i in enumerate(Geometry):
            f.write('{:20s} {:< 20.6f} {:< 20.6f} {:< 20.6f}\n'.format(Elements[count_i],i[0],i[1],i[2]))

    # Create type dictionaries (needed to convert each atom,bond,angle, and dihedral type to consecutive numbers as per LAMMPS convention)
    # Note: LAMMPS orders atomtypes, bonds, angles, dihedrals, etc as integer types. Each of the following dictionaries holds the mapping between
    #       the true type (held in the various types lists) and the lammps type_id, obtained by enumerated iteration over the respective set(types).
    Atom_type_dict = {}
    for count_i,i in enumerate(sorted(set(Atom_types))):
        for j in Atom_types:
            if i == j:
                Atom_type_dict[i]=count_i+1
            if i in list(Atom_type_dict.keys()):
                break
    Bond_type_dict = {}
    for count_i,i in enumerate(sorted(set(Bond_types))):
        for j in Bond_types:
            if i == j:
                Bond_type_dict[i]=count_i+1
            if i in list(Bond_type_dict.keys()):
                break
    Angle_type_dict = {}
    for count_i,i in enumerate(sorted(set(Angle_types))):
        for j in Angle_types:
            if i == j:
                Angle_type_dict[i]=count_i+1
            if i in list(Angle_type_dict.keys()):
                break
    Dihedral_type_dict = {}
    for count_i,i in enumerate(sorted(set(Dihedral_types))):
        for j in Dihedral_types:
            if i == j:
                Dihedral_type_dict[i]=count_i+1
            if i in list(Dihedral_type_dict.keys()):
                break
    Improper_type_dict = {}
    for count_i,i in enumerate(sorted(set(Improper_types))):
        for j in Improper_types:
            if i == j:
                Improper_type_dict[i]=count_i+1
            if i in list(Improper_type_dict.keys()):
                break

    # Write the data file
    #with open(Filename+'/'+Filename.split('/')[-1]+'.data','w') as f:
    #with open(Filename.split('/')[-1]+'.data','w') as f:
    with open(Filename+'.data','w') as f:
        
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
        f.write("{:< 20.16f} {:< 20.16f} xlo xhi\n".format(Sim_Box[0],Sim_Box[1]))
        f.write("{:< 20.16f} {:< 20.16f} ylo yhi\n".format(Sim_Box[2],Sim_Box[3]))
        f.write("{:< 20.16f} {:< 20.16f} zlo zhi\n".format(Sim_Box[4],Sim_Box[5]))

        # Write Masses
        f.write("\nMasses\n\n")
        for count_i,i in enumerate(sorted(set(Atom_types))):
            for j in set(Atom_types):
                if Atom_type_dict[j] == count_i+1:
                    f.write("{} {:< 8.6f}\n".format(count_i+1,Masses[str(j)])) # count_i+1 bc of LAMMPS 1-indexing
        f.write("\n")

        # Write Atoms
        f.write("Atoms\n\n")
        for count_i,i in enumerate(Atom_types):
            f.write("{:<8d} {:< 4d} {:< 4d} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:< 20.16f} {:d} {:d} {:d}\n"\
            .format(count_i+1,Molecule[count_i],Atom_type_dict[i],Charges[count_i],Geometry[count_i,0],Geometry[count_i,1],Geometry[count_i,2],0,0,0))

        # Write Bonds
        if len(Bonds) > 0:
            f.write("\nBonds\n\n")
            for count_i,i in enumerate(Bonds):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Bond_type_dict[Bond_types[count_i]],i[0]+1,i[1]+1))
                #f.write("{:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Bond_type_dict[Bond_types[count_i]],i[0]+5,i[1]+5))

        # Write Angles
        if len(Angles) > 0:
            f.write("\nAngles\n\n")
            for count_i,i in enumerate(Angles):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Angle_type_dict[Angle_types[count_i]],i[0]+1,i[1]+1,i[2]+1))
                #f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Angle_type_dict[Angle_types[count_i]],i[0]+5,i[1]+5,i[2]+5))

        # Write Dihedrals
        if len(Dihedrals) > 0: 
            f.write("\nDihedrals\n\n")
            for count_i,i in enumerate(Dihedrals):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Dihedral_type_dict[Dihedral_types[count_i]],i[0]+1,i[1]+1,i[2]+1,i[3]+1))  # original line
                #f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Dihedral_type_dict[Dihedral_types[count_i]],i[0]+5,i[1]+5,i[2]+5,i[3]+5))  #+1 already there. added 4 to get 5

        # Write Impropers
        if Improper_flag and len(Impropers) > 0: 
            f.write("\nImpropers\n\n")
            for count_i,i in enumerate(Impropers):
                f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Improper_type_dict[Improper_types[count_i]],i[0]+1,i[1]+1,i[2]+1,i[3]+1))
                #f.write("{:<8d} {:<8d} {:<8d} {:<8d} {:<8d} {:<8d}\n".format(count_i+1,Improper_type_dict[Improper_types[count_i]],i[0]+5,i[1]+5,i[2]+5,i[3]+5))

    # Write the settings file
    fixed_modes = {'bonds':[], 'angles':[]}
    #with open(Filename+'/'+Filename.split('/')[-1]+'.in.settings','w') as f:
    #with open(Filename.split('/')[-1]+'.in.settings','w') as f:
    with open(Filename+'.in.settings','w') as f:

        # Write non-bonded interactions (the complicated form of this loop is owed
        # to desire to form a nicely sorted list in terms of the lammps atom types
        # Note: Atom_type_dict was initialize according to sorted(set(Atom_types) 
        #       so iterating over this list (twice) orders the pairs, too.
        f.write("     {}\n".format("# Non-bonded interactions (pair-wise)"))
        for count_i,i in enumerate(sorted(set(Atom_types))):     
            for count_j,j in enumerate(sorted(set(Atom_types))): 
                lj_note = ''

                # Skip duplicates
                if count_j < count_i:
                    continue

                # Conform to LAMMPS i <= j formatting
                if Atom_type_dict[i] <= Atom_type_dict[j]:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[i],Atom_type_dict[j]))
                else:
                    f.write("     {:20s} {:<10d} {:<10d} ".format("pair_coeff",Atom_type_dict[j],Atom_type_dict[i]))

                # Determine key (ordered by initialize_VDW function such that i > j)
                if i > j:
                    key = (i,j)
                else:
                    key = (j,i)
                
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
            f.write("     {:20s} {:<10d} ".format("bond_coeff",Bond_type_dict[i]))
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
            f.write("     {:20s} {:<10d} ".format("angle_coeff",Angle_type_dict[i]))
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
            f.write("     {:20s} {:<10d} ".format("dihedral_coeff",Dihedral_type_dict[i]))
            for j in Dihedral_params[i][1:]:               # Starting from [1:] surpresses writing out the dihedral type, since they all should be OPLS for the perovskites
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
                f.write("     {:20s} {:<10d} ".format("improper_coeff",Improper_type_dict[i]))
                for j in Improper_params[i][1:]:
                    if type(j) is str:
                        f.write("{:20s} ".format(j))
                    if type(j) is float:
                        f.write("{:< 20.6f} ".format(j))
                    if type(j) is int:
                        f.write("{:< 20d} ".format(j))
                f.write("\n")

    return Atom_type_dict,Bond_type_dict,Angle_type_dict,fixed_modes

# Description: Initialize VDW_dict based on UFF parameters for the initial guess of the fit.
def initialize_VDW(atomtypes,sigma_scale=1.0,eps_scale=1.0,VDW_FF={},Force_UFF=0,mixing_rule='lb'):

    # Initialize UFF parameters (tuple corresponds to eps,sigma pairs for each element)
    # Taken from UFF (Rappe et al. JACS 1992)
    # Note: LJ parameters in the table are specificed in the eps,r_min form rather than eps,sigma
    #       the conversion between r_min and sigma is sigma = r_min/2^(1/6)
    # Note: Units for sigma = angstroms and eps = kcal/mol 
    UFF_dict = { 1:(0.044,2.5711337005530193),  2:(0.056,2.1043027722474816),  3:(0.025,2.183592758161972),  4:(0.085,2.4455169812952313),\
                 5:(0.180,3.6375394661670053),  6:(0.105,3.4308509635584463),  7:(0.069,3.260689308393642),  8:(0.060,3.1181455134911875),\
                 9:(0.050,2.996983287824101),  10:(0.042,2.88918454292912),   11:(0.030,2.657550876212632), 12:(0.111,2.6914050275019648),\
                13:(0.505,4.008153332913386),  14:(0.402,3.82640999441276),   15:(0.305,3.694556984127987), 16:(0.274,3.594776327696269),\
                17:(0.227,3.5163772404999194), 18:(0.185,3.4459962417668324), 19:(0.035,3.396105913550973), 20:(0.238,3.0281647429590133),\
                21:(0.019,2.935511276272418),  22:(0.017,2.828603430095577),  23:(0.016,2.800985569833227), 24:(0.015,2.6931868249382456),\
                25:(0.013,2.6379511044135446), 26:(0.013,2.5942970672246677), 27:(0.014,2.558661118499054), 28:(0.015,2.5248069672097215),\
                29:(0.005,3.113691019900486),  30:(0.124,2.4615531582217574), 31:(0.415,3.904809081609107), 32:(0.379,3.813046513640652),\
                33:(0.309,3.7685015777336357), 34:(0.291,3.746229109780127),  35:(0.251,3.731974730289881), 36:(0.220,3.689211591819145),\
                37:(0.040,3.6651573264293558), 38:(0.235,3.2437622327489755), 39:(0.072,2.980056212179435), 40:(0.069,2.78316759547042),\
                41:(0.059,2.819694442914174),  42:(0.056,2.7190228877643157), 43:(0.048,2.670914356984738), 44:(0.056,2.6397329018498255),\
                45:(0.053,2.6094423454330538), 46:(0.048,2.5827153838888437), 47:(0.036,2.804549164705788), 48:(0.228,2.537279549263686),\
                49:(0.599,3.976080979060334),  50:(0.567,3.9128271700723705), 51:(0.449,3.937772334180300), 52:(0.398,3.982317270087316),\
                53:(0.339,4.009044231631527),  54:(0.332,3.923517954690054),  55:(0.045,4.024189509839913), 56:(0.364,3.2989979532736764),\
                72:(0.072,2.798312873678806),  73:(0.081,2.8241489365048755), 74:(0.067,2.734168165972701), 75:(0.066,2.631714813386562),\
                76:(0.037,2.7796040005978586), 77:(0.073,2.5301523595185635), 78:(0.080,2.453535069758495), 79:(0.039,2.9337294788361374),\
                80:(0.385,2.4098810325696176), 81:(0.680,3.872736727756055),  82:(0.663,3.828191791849038), 83:(0.518,3.893227398273283),\
                84:(0.325,4.195242063722858),  85:(0.284,4.231768911166611),  86:(0.248,4.245132391938716) }

    # NEW: Initialize VDW_dict first guess based on element types and Lorentz-Berthelot mixing rules
    # Order of operations: (1) If the parameters are in the supplied FF database then they are used as is
    # (2) If the self-terms are in the supplied FF datebase then they are used to generate the mixed
    # interactions (3) UFF parameters are used. 
    VDW_dict = {}
    origin = {}
    VDW_styles = []
    for count_i,i in enumerate(atomtypes):
        for count_j,j in enumerate(atomtypes):
            if count_i < count_j:
                continue

            # Check for parameters in the database
            if (i,j) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(i,j)][0] == "lj":
                    VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(i,j)][0] == "buck":
                    VDW_type = "buck/coul/long"
                else:
                    print("ERROR in initialize_VDW: only lj and buck pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type] + VDW_FF[(i,j)][1:]
                else:
                    VDW_dict[(j,i)] = [VDW_type] + VDW_FF[(i,j)][1:]
                origin[(i,j)] = origin[(j,i)] = "read"

            # Check for reverse combination
            elif (j,i) in VDW_FF and Force_UFF != 1:

                # Determine appropriate lammps style
                if VDW_FF[(j,i)][0] == "lj":
                    VDW_type = "lj/cut/coul/long"
                elif VDW_FF[(j,i)][0] == "buck":
                    VDW_type = "buck/coul/long"
                else:
                    print("ERROR in initialize_VDW: only lj and buck pair types are supported. Exiting...")
                    quit()

                # Assign style
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type] + VDW_FF[(j,i)][1:]
                else:
                    VDW_dict[(j,i)] = [VDW_type] + VDW_FF[(j,i)][1:]
                origin[(i,j)] = origin[(j,i)] = "read"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i,i) in VDW_FF and (j,j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'lb':

                # Check compatibility with mixing rules
                if VDW_FF[(i,i)][0] != "lj" or VDW_FF[(j,j)][0] != "lj":
                    print("ERROR in initialize_VDW: only lj styles support mixing rules. Exiting...")
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                eps    = (VDW_FF[(i,i)][1]*VDW_FF[(j,j)][1])**(0.5) * eps_scale
                sigma  = (VDW_FF[(i,i)][2]+VDW_FF[(j,j)][2])/2.0 * sigma_scale
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "lb"

            # Check if the database has the self-terms necessary for applying mixing rules
            elif (i,i) in VDW_FF and (j,j) in VDW_FF and Force_UFF != 1 and mixing_rule == 'wh':

                # Check compatibility with mixing rules
                if VDW_FF[(i,i)][0] != "lj" or VDW_FF[(j,j)][0] != "lj":
                    print("ERROR in initialize_VDW: only lj styles support mixing rules. Exiting...")
                    quit()

                # Apply mixing rules and assign
                VDW_type = "lj/cut/coul/long"
                sigma  = ((VDW_FF[(i,i)][2]**(6.0)+VDW_FF[(j,j)][2]**(6.0))/2.0)**(1.0/6.0)
                eps    = (VDW_FF[(i,i)][1]*VDW_FF[(i,i)][2]**(6.0) * VDW_FF[(j,j)][1]*VDW_FF[(j,j)][2]**(6.0) )**(0.5) / sigma**(6.0)
                if i > j: 
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "wh"

            # Last resort: Use UFF parameters.
            else:
                VDW_type = "lj/cut/coul/long"
                type_1 = int(i.split('[')[1].split(']')[0])
                type_2 = int(j.split('[')[1].split(']')[0])
                eps    = (UFF_dict[type_1][0]*UFF_dict[type_2][0])**(0.5) * eps_scale
                sigma  = (UFF_dict[type_1][1]+UFF_dict[type_2][1])/2.0 * sigma_scale
                if i > j:
                    VDW_dict[(i,j)] = [VDW_type,eps,sigma]
                else:
                    VDW_dict[(j,i)] = [VDW_type,eps,sigma]
                origin[(i,j)] = origin[(j,i)] = "UFF"
                
            # Collect a list of the LAMMPS styles used in the simulation
            VDW_styles += [VDW_type]

    # Print summary
    #print "\n{}".format("*"*177)
    #print "* {:^173s} *".format("Initializing VDW parameters for the simulation (those with * were read from the FF file(s))")
    #print "*{}*".format("-"*175)
    #print "* {:<50s} {:<50s} {:<20s}  {:<18s} {:<18s} {:<8s}    *".format("Type","Type","VDW_type","eps (kcal/mol)","sigma (angstroms)","origin")
    #print "{}".format("*"*177)
    #for j in VDW_dict.keys():
    #    print "  {:<50s} {:<50s} {:<20s} {:< 18.4f} {:< 18.4f}  {:<18s}".format(j[0],j[1],VDW_dict[j][0],VDW_dict[j][1],VDW_dict[j][2],origin[j])
    #print ""

    return VDW_dict

def write_xyz(name, elements, geometry, atomtypes, comment=''):
    with open(name, 'w') as f:
        f.write("{}\n{}\n".format(len(geometry), comment))
        
        for count_i,i in enumerate(geometry):
            if len(atomtypes) > 0:
                f.write("{:<10s} {:<20.6f} {:<20.6f} {:<20.6f} {:<10s}\n".format(elements[count_i], i[0], i[1], i[2], atomtypes[count_i]))
            else:
                f.write("{:<10s} {:<20.6f} {:<20.6f} {:<20.6f}\n".format(elements[count_i], i[0], i[1], i[2]))

def print_readme():
    print('README!')
    
def write_Gaussian_com(name, geo, elements):
    with open(name+'.com', 'w') as f:
        f.write('%chk=dummy.chk\n')
        f.write('# hf/3-21g geom=connectivity\n')
        f.write('\nDummy\n\n')
        f.write('{} {}\n'.format(args.charge, args.multiplicity))
        for count_i,i in enumerate(geo):
            f.write("{:<40s} {:<20.6f} {:<20.6f} {:<20.6f}\n".format(elements[count_i], i[0], i[1], i[2]))

class Logger(object):
    def __init__(self,folder):
        self.terminal = sys.stdout
        self.log = open(folder+".log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass
    
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

        
if  __name__ == '__main__': 
    main(sys.argv[1:])
