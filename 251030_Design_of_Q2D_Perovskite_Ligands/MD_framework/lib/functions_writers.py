#!/bin/env python
"""
This contains functions for writing input and submission scripts for the various calculations performed
in preparing a ligand and evaluating its stability.

@author: Stephen Shiring
"""

import random
import os

def write_LAMMPS_init(name, data_name=None, temperature=298.0, pressure=1.0, pressure_axis='y', timesteps_npt=1000000, \
                      timesteps_nvt=1000000, nve_cycles=0, nve_dist=0.01, extension_number=0, tether=None, tether_types=None, \
                      avg_freq=1000, coords_freq=1000, thermo_freq=1000, dump4avg=1, fixes=None, commands=None, tip4p=None, \
                      kspace='pppm', submission_manager='SLURM', output='nvt.in.init', flag_velocities=False, \
                      flag_ecoh=False, neighbor_build=0, submit_file='submit.sh'):

    if tether != None:
        tether = tether.split()
        if tether[0] not in ['xyz', 'xy', 'xz', 'yz', 'x', 'y', 'z']:
            print('ERROR: Direction for -tether not recognized. Accepts only LAMMPS values:  xyz, xy, xz, yz, x, y, or z. Exiting...')
            return False
        if '.' not in tether[-1]:
            tether.append('100.0')
    
    if tether_types != None:
        tether_types = tether_types.split()
        if tether_types[0] not in ['xyz', 'xy', 'xz', 'yz', 'x', 'y', 'z']:
            print('ERROR: Direction for -tether not recognized. Accepts only LAMMPS values:  xyz, xy, xz, yz, x, y, or z. Exiting...')
            return False
        if '.' not in tether_types[-1]:
            tether_types.append('5.0')
    
    if fixes != None:
        fixes = fixes.split('|')
    
    if commands != None:
        commands = commands.split('|')

    if data_name == None:
        data_name = name + '_nve.end.data'
    
    pressure_axis = pressure_axis.split()
    exit_flag = False
    for axis in pressure_axis:
        if axis not in ['iso', 'x', 'y', 'z']:
            print('ERROR: Specified pressure axis ({}) not recognized. Exiting....'.format(axis))
            exit_flag = True
    if exit_flag: return False
    
    if kspace.lower() not in ['ewald', 'pppm']:
        print('ERROR: kspace style ("{}") not recognized. Exiting...'.format(kspace))
        return False
    
    if tip4p != None and kspace.lower() == 'ewald':
        print('ERROR: Cannot use the ewald kspace style with the TIP4P water model. Exiting...')
        return False
    
    if submission_manager.upper() not in ['SLURM', 'PBS']:
        print('ERROR: Submission manager not recognized. Accepts SLURM, PBS. Exiting...')
        return False

    with open(output, 'w') as o:
        o.write('# LAMMPS input file for perovskite\n')
        o.write('\n')
        o.write('# VARIABLES\n')
        o.write('variable        data_name       index   {}\n'.format(data_name))
        o.write('variable        restart_name    index   extend.end.restart\n')
        o.write('variable        extend_name     index   extend.end.data\n')
        o.write('variable        settings_name   index   {}.in.settings\n'.format(name))
        o.write('variable        nSteps_NPT      index   {}\n'.format(timesteps_npt))
        o.write('variable        nSteps_NVT      index   {}\n'.format(timesteps_nvt))
        o.write('variable        avg_freq        index   {}\n'.format(avg_freq))
        o.write('variable        coords_freq     index   {}\n'.format(coords_freq))
        o.write('variable        thermo_freq     index   {}\n'.format(thermo_freq))
        o.write('variable        dump4avg        index   {}\n'.format(dump4avg))
        o.write('variable        Temp            index   {}\n'.format(temperature))
        o.write('variable        pressure        index   {}\n'.format(pressure))
        for i in range(nve_cycles):
            o.write('variable        vseed{}          index   {}\n'.format(i+1, int(random.random()*100000)))
        if flag_velocities:
            o.write('variable        vseed           index   {}\n'.format(int(random.random()*100000)))
        o.write('variable        run             index   0\n\n')

        o.write('# Change the name of the log output #\n')
        o.write('log ${run}.eval.log\n\n')

        o.write('#===========================================================\n')
        o.write('# GENERAL PROCEDURES\n')
        o.write('#===========================================================\n')
        o.write('units		real   # g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n')
        o.write('dimension	3      # 3 dimensional simulation\n')
        if tip4p != None:
            o.write('newton		on  	# use Newton\'s 3rd law\n')
        else:
            o.write('newton		off 	# use Newton\'s 3rd law\n')
        o.write('boundary	p p p	# periodic boundary conditions\n')
        o.write('atom_style	full	# molecular + charge\n\n')

        o.write('#===========================================================\n')
        o.write('# DEFINE PAIR, BOND, AND ANGLE STYLES\n')
        o.write('#===========================================================\n')
        o.write('special_bonds   amber\n')
        if tip4p != None:
            o.write('pair_style      hybrid buck/coul/long 15.0 15.0 lj/cut/coul/long 15.0 15.0 {}\n'.format(tip4p))
        else:
            o.write('pair_style      hybrid buck/coul/long 15.0 15.0 lj/cut/coul/long 15.0 15.0\n')
        o.write('pair_modify     shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n')
        o.write('bond_style      harmonic\n')
        o.write('angle_style     harmonic\n')
        o.write('dihedral_style  opls\n')
        if tip4p != None:
            o.write('kspace_style    pppm/tip4p 0.0001\n\n')
        else:
            o.write('kspace_style    {} 0.0001\n\n'.format(kspace.lower()))
        
        if flag_ecoh:
            o.write('# Compute cohesive energy\n')
            o.write('compute eng all pe/atom \n')
            o.write('compute eatoms all reduce sum c_eng\n\n')

        o.write('#===========================================================\n')
        o.write('# SETUP SIMULATIONS\n')
        o.write('#===========================================================\n\n')

        o.write('# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n')
        o.write('if "${run} == 0" then &\n')
        o.write('   "read_data ${data_name}" &\n')
        o.write('else &\n')
        o.write('   "read_restart ${restart_name}"\n')
        o.write('include ${settings_name}\n\n')
        
        o.write('# SET RUN PARAMETERS\n')
        o.write('timestep	1.0		# fs\n')
        o.write('run_style	verlet 		# Velocity-Verlet integrator\n')

        if neighbor_build == 1:
            o.write('neigh_modify every 1 delay 0 check yes one 10000\n')
        elif neighbor_build == 2:
            o.write('neigh_modify every 1 delay 0 check yes page 500000 one 50000\n')
        else:
            o.write('neigh_modify every 1 delay 10 check yes one 10000\n')
            
        o.write('\n# SET OUTPUTS\n')
        if flag_ecoh:
            o.write('thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press c_eatoms\n')
        else:
            o.write('thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n')
        o.write('thermo_modify   format float %14.6f\n')
        o.write('thermo ${thermo_freq}\n\n')

        o.write('# DECLARE RELEVANT OUTPUT VARIABLES\n')
        o.write('variable        my_step equal   step\n')
        o.write('variable        my_temp equal   temp\n')
        o.write('variable        my_rho  equal   density\n')
        o.write('variable        my_pe   equal   pe\n')
        o.write('variable        my_ebon equal   ebond\n')
        o.write('variable        my_eang equal   eangle\n')
        o.write('variable        my_edih equal   edihed\n')
        o.write('variable        my_evdw equal   evdwl\n')
        o.write('variable        my_eel  equal   (ecoul+elong)\n')
        o.write('variable        my_ent  equal   enthalpy\n')
        o.write('variable        my_P    equal   press\n')
        o.write('variable        my_vol  equal   vol\n\n')
        
        if flag_ecoh:
            o.write('variable 	natoms 	equal "count(all)" \n')
            o.write('variable 	teng 	equal "c_eatoms"\n')
            o.write('variable 	ecoh 	equal "v_teng/v_natoms"\n\n')
            o.write('fix averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_ecoh file ${run}.thermo.avg\n\n')
        else:
            o.write('fix averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P file ${run}.thermo.avg\n\n')

        if tether != None:
            o.write('# Set tether fix\n')
            o.write('group tethered_atoms id {}\n'.format(' '.join(tether[1:-1]) ))
            o.write('fix tether_atoms tethered_atoms spring/self {} {}\n'.format(tether[-1], tether[0]))
            o.write('\n')
        
        if tether_types != None:
            o.write('# Set tether fix for all atoms of types\n')
            o.write('group tethered_types type {}\n'.format(' '.join(tether_types[1:-1])))
            o.write('fix tether_types tethered_types spring/self {} {}\n'.format(tether_types[-1], tether_types[0]))
            o.write('\n')
            
        for i in range(nve_cycles):
            o.write('\n# CREATE COORDINATE DUMPS FOR NVE RELAXATION\n')
            o.write('dump relax all custom {} {}.nve.{}.lammpstrj id type x y z \n'.format('${coords_freq}', '${run}', str(i+1) ) )
            o.write('dump_modify relax sort  id\n\n')
            o.write('# INITIALIZE VELOCITIES AND CREATE THE CONSTRAINED RELAXATION FIX\n')
            o.write('if "${run} == 0" then &\n')
            o.write('"velocity        all create {} {} mom yes rot yes     # DRAW VELOCITIES" &\n'.format('${Temp}','${vseed'+ str(i+1) +'}'))
            o.write('"fix relax all nve/limit {}" &\n'.format(nve_dist))
            o.write('"run             10000" &\n')
            o.write('"unfix relax" &\n')
            o.write('"undump relax" \n\n')
            
        if fixes != None:
            o.write('# User-specified fixes\n')
            for fix in fixes:
                o.write('{}\n'.format(fix))
            o.write('\n')
            
        o.write('#===========================================================\n')
        o.write('# RUN PRODUCTION\n')
        o.write('#===========================================================\n\n')
        
        o.write('# UPDATE RUN PARAMETERS AND CREATE FIX\n')
        if flag_velocities:
            o.write('velocity        all create ${Temp} ${vseed} mom yes rot yes     # DRAW VELOCITIES\n')
        o.write('fix mom all momentum 1000 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n')
        
        o.write('\n# NPT equilibration, x and z direction to accomodate any lattice changes\n')
        axis_fixes = ''
        for axis in pressure_axis:
            axis_fixes += '{} {} {} 1000.0 '.format(axis, '${pressure}', '${pressure}')
        o.write('fix equil all npt temp {} {} 100.0 {} # NPT, nose-hoover 100 fs T relaxation\n'.format('${Temp}', '${Temp}', axis_fixes))
        o.write('dump equil all custom ${coords_freq} ${run}.npt.lammpstrj id type x y z \n')
        o.write('dump_modify equil sort  id\n\n')

        o.write('# RUN NPT\n')
        o.write('run		${nSteps_NPT}\n')
        o.write('unfix equil\n')
        o.write('undump equil\n\n')
        
        o.write('\n# NVT run\n')
        o.write('fix equil all nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')

        o.write('# CREATE COORDINATE DUMPS FOR EQUILIBRIUM\n')
        o.write('dump equil all custom ${coords_freq} ${run}.nvt.lammpstrj id type x y z \n')
        o.write('dump_modify equil sort  id\n\n')

        o.write('# RUN NVT\n')
        o.write('run		${nSteps_NVT}\n')
        o.write('unfix equil\n')
        o.write('undump equil\n\n')

        o.write('# WRITE RESTART FILES, CLEANUP, AND EXIT\n')
        o.write('write_restart   extend.end.restart\n')
        o.write('write_data      extend.end.data pair ii\n')
        o.write('unfix           averages\n')
        if tether != None: o.write('unfix           tether_atoms\n')
        if tether_types != None: o.write('unfix           tether_types\n\n')
        
        if fixes != None:
            for fix in fixes:
                fix = fix.split()
                o.write('unfix           {}\n'.format(fix[1]))
            o.write('\n')
        
        o.write('\n# Update run number\n')
        o.write('variable sub equal (v_run+1)\n')
        o.write('shell sed -i /variable.*run.*index/s/{}/{}/g {}\n'.format('${run}', '${sub}', output))
        o.write('shell echo "Run ${run} finished." >> ${run}.success\n')
        
        if extension_number > 0:
            if submission_manager.upper() == 'PBS':
                o.write("if '{} < {}' then 'shell qsub {}'\n".format('${run}', extension_number, submit_file))
            elif submission_manager.upper() == 'SLURM':
                o.write("if '{} < {}' then 'shell sbatch {}'\n".format('${run}', extension_number, submit_file))
            
        # User-specified shell commands
        if commands != None:
            o.write('\n# User-specified shell commands\n')
            for cmd in commands:
                o.write('{}\n'.format(cmd))
            o.write('\n')

    print('File written to {}.'.format(output))
    print('Success!')
    
    return True
    
def write_LAMMPS_init_gilbreth(name, data_name=None, temperature=298.0, pressure=1.0, \
                      timesteps_nvtr=1000000, timesteps_nvtp=5000000,nve_cycles=0, nve_dist=0.01, tether=None, tether_types=None, \
                      avg_freq=1000, coords_freq=1000, thermo_freq=1000, dump4avg=1, fixes=None, commands=None, tip4p=None, \
                      kspace='pppm', submission_manager='SLURM', output='nvt.in.init', flag_velocities=False, \
                      flag_ecoh=False, neighbor_build=0,):

    if tether != None:
        tether = tether.split()
        if tether[0] not in ['xyz', 'xy', 'xz', 'yz', 'x', 'y', 'z']:
            print('ERROR: Direction for -tether not recognized. Accepts only LAMMPS values:  xyz, xy, xz, yz, x, y, or z. Exiting...')
            return False
        if '.' not in tether[-1]:
            tether.append('100.0')
    
    if tether_types != None:
        tether_types = tether_types.split()
        if tether_types[0] not in ['xyz', 'xy', 'xz', 'yz', 'x', 'y', 'z']:
            print('ERROR: Direction for -tether not recognized. Accepts only LAMMPS values:  xyz, xy, xz, yz, x, y, or z. Exiting...')
            return False
        if '.' not in tether_types[-1]:
            tether_types.append('5.0')
    
    if fixes != None:
        fixes = fixes.split('|')
    
    if commands != None:
        commands = commands.split('|')

    if data_name == None:
        data_name = name + '_nve.end.data'
    
    
    if kspace.lower() not in ['ewald', 'pppm']:
        print('ERROR: kspace style ("{}") not recognized. Exiting...'.format(kspace))
        return False
    
    if tip4p != None and kspace.lower() == 'ewald':
        print('ERROR: Cannot use the ewald kspace style with the TIP4P water model. Exiting...')
        return False
    
    if submission_manager.upper() not in ['SLURM', 'PBS']:
        print('ERROR: Submission manager not recognized. Accepts SLURM, PBS. Exiting...')
        return False

    with open(output, 'w') as o:
        o.write('# LAMMPS input file for perovskite\n')
        o.write('\n')
        o.write('# VARIABLES\n')
        o.write('variable        data_name       index   {}\n'.format(data_name))
        o.write('variable        restart_name    index   extend.end.restart\n')
        o.write('variable        extend_name     index   extend.end.data\n')
        o.write('variable        settings_name   index   {}.in.settings\n'.format(name))
        o.write('variable        nSteps_NVTR     index   {}\n'.format(timesteps_nvtr))
        o.write('variable        nSteps_NVTP     index   {}\n'.format(timesteps_nvtp))
        o.write('variable        avg_freq        index   {}\n'.format(avg_freq))
        o.write('variable        coords_freq     index   {}\n'.format(coords_freq))
        o.write('variable        thermo_freq     index   {}\n'.format(thermo_freq))
        o.write('variable        dump4avg        index   {}\n'.format(dump4avg))
        o.write('variable        Temp            index   {}\n'.format(temperature))
        o.write('variable        pressure        index   {}\n'.format(pressure))
        for i in range(nve_cycles):
            o.write('variable        vseed{}          index   {}\n'.format(i+1, int(random.random()*100000)))
        if flag_velocities:
            o.write('variable        vseed           index   {}\n'.format(int(random.random()*100000)))
        o.write('variable        run             index   0\n\n')

        o.write('# Change the name of the log output #\n')
        o.write('log ${run}.eval.log\n\n')

        o.write('#===========================================================\n')
        o.write('# GENERAL PROCEDURES\n')
        o.write('#===========================================================\n')
        o.write("suffix gpu\n")
        o.write("package gpu 2 neigh off pair/only on\n")
        o.write('units		real   # g/mol, angstroms, fs, kcal/mol, K, atm, charge*angstrom\n')
        o.write('dimension	3      # 3 dimensional simulation\n')
        if tip4p != None:
            o.write('newton		on  	# use Newton\'s 3rd law\n')
        else:
            o.write('newton		off 	# use Newton\'s 3rd law\n')
        o.write('boundary	p p p	# periodic boundary conditions\n')
        o.write('atom_style	full	# molecular + charge\n\n')

        o.write('#===========================================================\n')
        o.write('# DEFINE PAIR, BOND, AND ANGLE STYLES\n')
        o.write('#===========================================================\n')
        o.write('special_bonds   amber\n')
        if tip4p != None:
            o.write('pair_style      hybrid buck/coul/long 15.0 15.0 lj/cut/coul/long 15.0 15.0 {}\n'.format(tip4p))
        else:
            o.write('pair_style      hybrid buck/coul/long 15.0 15.0 lj/cut/coul/long 15.0 15.0\n')
        o.write('pair_modify     shift yes mix arithmetic       # using Lorenz-Berthelot mixing rules\n')
        o.write('bond_style      harmonic\n')
        o.write('angle_style     harmonic\n')
        o.write('dihedral_style  opls\n')
        if tip4p != None:
            o.write('kspace_style    pppm/tip4p 0.0001\n\n')
        else:
            o.write('kspace_style    {} 0.0001\n\n'.format(kspace.lower()))
        
        if flag_ecoh:
            o.write('# Compute cohesive energy\n')
            o.write('compute eng all pe/atom \n')
            o.write('compute eatoms all reduce sum c_eng\n\n')

        o.write('#===========================================================\n')
        o.write('# SETUP SIMULATIONS\n')
        o.write('#===========================================================\n\n')

        o.write('# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY\n')
        o.write('if "${run} == 0" then &\n')
        o.write('   "read_data ${data_name}" &\n')
        o.write('else &\n')
        o.write('   "read_restart ${restart_name}"\n')
        o.write('include ${settings_name}\n\n')
        
        o.write('# SET RUN PARAMETERS\n')
        o.write('timestep	1.0		# fs\n')
        o.write('run_style	verlet 		# Velocity-Verlet integrator\n')

        if neighbor_build == 1:
            o.write('neigh_modify every 1 delay 0 check yes one 10000\n')
        elif neighbor_build == 2:
            o.write('neigh_modify every 1 delay 0 check yes page 500000 one 50000\n')
        else:
            o.write('neigh_modify every 1 delay 10 check yes one 10000\n')
            
        o.write('\n# SET OUTPUTS\n')
        if flag_ecoh:
            o.write('thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press c_eatoms\n')
        else:
            o.write('thermo_style    custom step temp vol density etotal pe ebond eangle edihed ecoul elong evdwl enthalpy press\n')
        o.write('thermo_modify   format float %14.6f\n')
        o.write('thermo ${thermo_freq}\n\n')

        o.write('# DECLARE RELEVANT OUTPUT VARIABLES\n')
        o.write('variable        my_step equal   step\n')
        o.write('variable        my_temp equal   temp\n')
        o.write('variable        my_rho  equal   density\n')
        o.write('variable        my_pe   equal   pe\n')
        o.write('variable        my_ebon equal   ebond\n')
        o.write('variable        my_eang equal   eangle\n')
        o.write('variable        my_edih equal   edihed\n')
        o.write('variable        my_evdw equal   evdwl\n')
        o.write('variable        my_eel  equal   (ecoul+elong)\n')
        o.write('variable        my_ent  equal   enthalpy\n')
        o.write('variable        my_P    equal   press\n')
        o.write('variable        my_vol  equal   vol\n\n')
        
        if flag_ecoh:
            o.write('variable 	natoms 	equal "count(all)" \n')
            o.write('variable 	teng 	equal "c_eatoms"\n')
            o.write('variable 	ecoh 	equal "v_teng/v_natoms"\n\n')
            o.write('fix averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P v_ecoh file ${run}.thermo.avg\n\n')
        else:
            o.write('fix averages all ave/time ${dump4avg} $(v_avg_freq/v_dump4avg) ${avg_freq} v_my_temp v_my_rho v_my_vol v_my_pe v_my_edih v_my_evdw v_my_eel v_my_ent v_my_P file ${run}.thermo.avg\n\n')

        if tether != None:
            o.write('# Set tether fix\n')
            o.write('group tethered_atoms id {}\n'.format(' '.join(tether[1:-1]) ))
            o.write('fix tether_atoms tethered_atoms spring/self {} {}\n'.format(tether[-1], tether[0]))
            o.write('\n')
        
        if tether_types != None:
            o.write('# Set tether fix for all atoms of types\n')
            o.write('group tethered_types type {}\n'.format(' '.join(tether_types[1:-1])))
            o.write('fix tether_types tethered_types spring/self {} {}\n'.format(tether_types[-1], tether_types[0]))
            o.write('\n')
            
        for i in range(nve_cycles):
            o.write('\n# CREATE COORDINATE DUMPS FOR NVE RELAXATION\n')
            o.write('dump relax all custom {} {}.nve.{}.lammpstrj id type x y z \n'.format('${coords_freq}', '${run}', str(i+1) ) )
            o.write('dump_modify relax sort  id\n\n')
            o.write('# INITIALIZE VELOCITIES AND CREATE THE CONSTRAINED RELAXATION FIX\n')
            o.write('if "${run} == 0" then &\n')
            o.write('"velocity        all create {} {} mom yes rot yes     # DRAW VELOCITIES" &\n'.format('${Temp}','${vseed'+ str(i+1) +'}'))
            o.write('"fix mom all momentum 10 linear 1 1 1 angular # Zero out system linear and angular momentum every ps" &\n')
            o.write('"fix relax all nve/limit {}" &\n'.format(nve_dist))
            o.write('"run            50000" &\n')
            o.write('"unfix relax" &\n')
            o.write('"undump relax" \n\n')
            
        if fixes != None:
            o.write('# User-specified fixes\n')
            for fix in fixes:
                o.write('{}\n'.format(fix))
            o.write('\n')
            
        o.write('#===========================================================\n')
        o.write('# RUN PRODUCTION\n')
        o.write('#===========================================================\n\n')
        
        o.write('# UPDATE RUN PARAMETERS AND CREATE FIX\n')
        if flag_velocities:
            o.write('velocity        all create ${Temp} ${vseed} mom yes rot yes     # DRAW VELOCITIES\n')
        o.write('fix mom all momentum 10 linear 1 1 1 angular # Zero out system linear and angular momentum every ps \n')
        
        o.write('\n# NVTR equilibration, x and z direction to accomodate any lattice changes\n')
        o.write('fix equil all nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')

        o.write('# CREATE COORDINATE DUMPS FOR EQUILIBRIUM\n')
        o.write('dump equil all custom ${coords_freq} r${run}.nvt.lammpstrj id type x y z \n')
        o.write('dump_modify equil sort  id\n\n')

        o.write('# RUN NVT\n')
        o.write('run		${nSteps_NVTR}\n')
        o.write('unfix equil\n')
        o.write('undump equil\n\n')
        
          
        o.write('\n# NVT production run\n')
        o.write('fix equil all nvt temp ${Temp} ${Temp} 100.0 # NVT, nose-hoover 100 fs T relaxation\n\n')

        o.write('# CREATE COORDINATE DUMPS FOR EQUILIBRIUM\n')
        o.write('dump equil all custom ${coords_freq} p${run}.nvt.lammpstrj id type x y z \n')
        o.write('dump_modify equil sort  id\n\n')

        o.write('# RUN NVT\n')
        o.write('run		${nSteps_NVTP}\n')
        o.write('unfix equil\n')
        o.write('undump equil\n\n')

        o.write('# WRITE RESTART FILES, CLEANUP, AND EXIT\n')
        o.write('write_restart   extend.end.restart\n')
        o.write('write_data      extend.end.data pair ii\n')
        o.write('unfix           averages\n')
        if tether != None: o.write('unfix           tether_atoms\n')
        if tether_types != None: o.write('unfix           tether_types\n\n')
        
        if fixes != None:
            for fix in fixes:
                fix = fix.split()
                o.write('unfix           {}\n'.format(fix[1]))
            o.write('\n')
        
        o.write('\n# Update run number\n')
        o.write('variable sub equal (v_run+1)\n')
        o.write('shell sed -i /variable.*run.*index/s/{}/{}/g {}\n'.format('${run}', '${sub}', output))
        o.write('shell echo "Run ${run} finished." >> ${run}.success\n')
        

    print('File written to {}.'.format(output))
    print('Success!')
    
    return True
    


def make_LAMMPS_submission_bell(lammps_init, job_name, nodes, ppn, queue, walltime, \
                           lammps_exe=None, \
                           resub=0, repeat=0, output='submit.sh', log_name='LAMMPS_run.out'):
    
    with open(output, 'w') as o:
    
        o.write('#!/bin/sh\n')
        o.write('#SBATCH --job-name={}\n'.format(job_name))
         
        o.write('#SBATCH -N {}\n'.format(nodes))
        o.write('#SBATCH -n {}\n'.format(ppn))
       
        o.write('#SBATCH -t {}:00:00\n'.format(walltime))
        o.write('#SBATCH -A {}\n'.format(queue))
         
         
        o.write('#SBATCH -o {}.out\n'.format(job_name))
        o.write('#SBATCH -e {}.err\n'.format(job_name))
         
        
        
        o.write('\n# Load default LAMMPS, compiler, and openmpi packages\n')
        o.write("module load gcc/9.3.0 \n")  
        o.write("module load openmpi/3.1.4 \n") 
        o.write("module load ffmpeg/4.2.2  \n")
        o.write("module load  openblas/0.3.8  \n")
        o.write("module load  gsl/2.4  \n\n")
        
        o.write('\n# cd into the submission directory\n')
        o.write('cd {}\n'.format(os.getcwd()))
        o.write('echo Working directory is {}\n'.format(os.getcwd()))
        o.write('echo Running on host `hostname`\necho Time is `date`\n')
        o.write('t_start=$SECONDS\n\n')
        
        o.write('# Submiting LAMMPS job for {}\n'.format(lammps_init))
        o.write('cd .\n')
        
        
        for i in range(repeat+1):
             o.write('mpirun -np {} {} -in {} >> {} &\n'.format(ppn, lammps_exe, lammps_init, log_name))
             o.write('wait\n')

        
        o.write('cd {} &\n'.format(os.getcwd()))
        o.write('wait\n\n')
        
        
        o.write('\nt_end=$SECONDS\n')
        o.write('t_diff=$(( ${t_end} - ${t_start} ))\n')
        o.write('eval "echo $(date -ud "@$t_diff" +\'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.\')"\n')
        o.write('echo Completion time is `date`\n')
        
        if resub > 0:
            o.write('\nrun=0\n')
            o.write('if [[ ${run} -le {} ]]; then\n'.format(resub))
            o.write('    sbatch submit.sh\n')
            o.write('fi\n')
            o.write('sub=$(( ${run} + 1 ))\n')
            o.write('sed -i s/run=${run}/run=${sub}/g submit.sh\n\n')
        
    print("\nsubmit_LAMMPS.py: Success!")
    return True

def make_LAMMPS_submission_gilbreth(lammps_init, job_name, nodes, ppn, gpu, queue, walltime, \
                           lammps_exe=None, \
                           resub=0, repeat=0, output='submit.sh', log_name='LAMMPS_run.out'):
    
    with open(output, 'w') as o:
    
        o.write('#!/bin/sh\n')
        o.write('#SBATCH --job-name={}\n'.format(job_name))
         
        o.write('#SBATCH --nodes={}\n'.format(nodes))
        o.write('#SBATCH --ntasks={}\n'.format(ppn))
        o.write('#SBATCH --ntasks-per-node={} --gpus-per-node={}\n'.format(ppn, gpu))
        if queue == "standby":
            o.write('#SBATCH --constraint=H\n')
        o.write('#SBATCH --cpus-per-task=1\n')
        o.write('#SBATCH -t {}:00:00\n'.format(walltime))
        o.write('#SBATCH -A {}\n'.format(queue))
         
         
        o.write('#SBATCH -o {}.out\n'.format(job_name))
        o.write('#SBATCH -e {}.err\n'.format(job_name))
         
        
        
        o.write('\n# Load default LAMMPS, compiler, and openmpi packages\n')
        o.write("module --force purge\n")
        o.write("module load gcc/12.3.0\n")
        o.write("module load openmpi/3.1.6-gpu-cuda11\n")
        o.write("module load openblas/0.3.7\n")
        o.write("module load gsl/2.4\n")
        o.write("module load ffmpeg/4.2.1\n")
        o.write("module load fftw/3.3.7\n")


        o.write('echo Running on host `hostname`\necho Time is `date`\n')
        o.write('t_start=$SECONDS\n\n')
        
        o.write('# Submiting LAMMPS job for {}\n'.format(lammps_init))
        
        o.write("module load utilities\n")
        o.write("module load monitor\n")
        
        o.write("# track GPU load\n")
        o.write("monitor gpu percent --sample-rate 30 > gpu-percent.log &\n")
        o.write("GPU_PID=$!\n")
        
        o.write("# track cpu load\n")
        o.write("monitor cpu percent --sample-rate 30 > cpu-percent.log &\n")
        o.write("CPU_PID=$!\n")
        
        # track GPU load
        # monitor gpu percent --sample-rate 10 > gpu-percent.log &
        # GPU_PID=$!
        
        for i in range(repeat+1):
             o.write('mpirun -np $SLURM_NTASKS --display-map --map-by core:PE=1  {} -in {} >> {} \n'.format(lammps_exe, lammps_init, log_name))
             

        o.write('kill -s INT $GPU_PID $CPU_PID\n')
        
        o.write('\nt_end=$SECONDS\n')
        o.write('t_diff=$(( ${t_end} - ${t_start} ))\n')
        o.write('eval "echo $(date -ud "@$t_diff" +\'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.\')"\n')
        o.write('echo Completion time is `date`\n')
        
        if resub > 0:
            o.write('\nrun=0\n')
            o.write('if [[ ${run} -le {} ]]; then\n'.format(resub))
            o.write('    sbatch submit.sh\n')
            o.write('fi\n')
            o.write('sub=$(( ${run} + 1 ))\n')
            o.write('sed -i s/run=${run}/run=${sub}/g submit.sh\n\n')
        
    print("\nsubmit_LAMMPS.py: Success!")
    return True

def make_LAMMPS_submission_negishi(lammps_init, job_name, nodes, ppn, queue, walltime, \
                           lammps_exe=None, \
                           resub=0, repeat=0, output='submit.sh', log_name='LAMMPS_run.out'):
    
    with open(output, 'w') as o:
    
        o.write('#!/bin/sh\n')
        o.write('#SBATCH --job-name={}\n'.format(job_name))
         
        o.write('#SBATCH --nodes={}\n'.format(nodes))
        o.write('#SBATCH --ntasks={}\n'.format(ppn))
        o.write('#SBATCH --ntasks-per-node={}\n'.format(ppn))
        o.write('#SBATCH --cpus-per-task=1\n')
        o.write('#SBATCH -t {}:00:00\n'.format(walltime))
        o.write('#SBATCH -A {}\n'.format(queue))
         
         
        o.write('#SBATCH -o {}.out\n'.format(job_name))
        o.write('#SBATCH -e {}.err\n'.format(job_name))
         
        
        
        o.write('\n# Load default LAMMPS, compiler, and openmpi packages\n')
        o.write("module --force purge\n")
        o.write("module load gcc/12.2.0\n")
        o.write("module load openmpi/4.1.4\n")
        o.write("module load lammps/20220623\n")


        o.write('echo Running on host `hostname`\necho Time is `date`\n')
        o.write('t_start=$SECONDS\n\n')
        
        o.write('# Submiting LAMMPS job for {}\n'.format(lammps_init))
        
        
        # track GPU load
        # monitor gpu percent --sample-rate 10 > gpu-percent.log &
        # GPU_PID=$!
        
        for i in range(repeat+1):
             o.write('mpirun -np $SLURM_NTASKS lmp -in {} >> {} \n'.format(lammps_init, log_name))
             

        o.write('kill -s INT $GPU_PID $CPU_PID\n')
        
        o.write('\nt_end=$SECONDS\n')
        o.write('t_diff=$(( ${t_end} - ${t_start} ))\n')
        o.write('eval "echo $(date -ud "@$t_diff" +\'This job took $((%s/3600/24)) days %H hours %M minutes %S seconds to complete.\')"\n')
        o.write('echo Completion time is `date`\n')
        
        if resub > 0:
            o.write('\nrun=0\n')
            o.write('if [[ ${run} -le {} ]]; then\n'.format(resub))
            o.write('    sbatch submit.sh\n')
            o.write('fi\n')
            o.write('sub=$(( ${run} + 1 ))\n')
            o.write('sed -i s/run=${run}/run=${sub}/g submit.sh\n\n')
        
    print("\nsubmit_LAMMPS.py: Success!")
    return True


def main():
    print('input_writers::main()')
    return

if __name__ == '__main__':
    main()
