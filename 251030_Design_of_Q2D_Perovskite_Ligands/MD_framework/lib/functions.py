#!/bin/env python
"""
Created on Mon Jan 18 11:08:31 2021

@author: Stephen Shiring

Holds shared functions
"""

import sys
import os
import datetime
import subprocess

# Context manager for changing the current working directory
class cd:
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

# General logger
class Logger(object):
    def __init__(self,logname):
        self.terminal = sys.stdout
        now = datetime.datetime.now() 
        d = '{}-{}-{}_{}{}{}'.format(now.year, now.month, now.day, now.hour, now.minute, now.second)
        self.log = open(logname+'.'+d+'.log', "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass
        
    #def close(self):
    #    self.log.close()

# Reads and processes a config file
def read_config(config_file):

    # Check to make sure the trajectory xyz file exists.
    if not os.path.isfile(config_file):
        print('\nERROR: Specified configuration file file "{}" not found. Aborting....\n'.format(config_file))
        exit()
    
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            
            # Skip comment lines and omit any inline comments
            if line[0] == '#':
                continue
            if '#' in line:
                line = line.split('#')[0]
            
            fields = line.split()
            if len(fields) > 0 :
                if fields[0].upper() not in list(config.keys()):
                    config[fields[0].upper()] = ' '.join(fields[1:])
    
    settings = ['JOB_MD_PPN', 'JOB_MD_NODES', 'JOB_MD_WALLTIME', 'JOB_MD_QUEUE','LAMMPS_PATH']
    
    missing = False
    for s in settings:
        if s not in list(config.keys()):
            print('ERROR: Missing setting {} in configuration file.'.format(s))
            missing = True
    if missing:
        print('Aborting...')
        exit()
        
    settings = ['JOB_MD_PPN', 'JOB_MD_NODES', 'JOB_MD_WALLTIME']
    for s in settings:
        config[s] = int(config[s])
    
    
    return config

# Returns the pending and running jobids for the user as a list
# From taffi / TCIT taffi-driver.py
def check_queue():

    # The first time this function is executed, find the user name and scheduler being used. 
    if not hasattr(check_queue, "user"):

        # Get user name
        check_queue.user = subprocess.check_output("echo ${USER}", shell=True).decode('utf-8').strip("\r\n")
        
        # Get batch system being used
        squeue_tmp = subprocess.Popen(['which', 'squeue'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        qstat_tmp  = subprocess.Popen(['which', 'qstat'], stdout=subprocess.PIPE,stderr=subprocess.STDOUT).communicate()[0].decode('utf-8').strip("\r\n")
        check_queue.sched =  None
        if "no squeue in" not in squeue_tmp:
            check_queue.sched = "slurm"
        elif "no qstat in" not in qstat_tmp:
            check_queue.sched = "pbs"
        else:
            print("ERROR in check_queue: neither slurm or pbs schedulers are being used.")
            quit()

    # Get running and pending jobs using the slurm scheduler
    if check_queue.sched == "slurm":

        # redirect a squeue call into output
        output = subprocess.check_output("squeue -l", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        id_ind = None
        for count_i,i in enumerate(output.split('\n')):            
            fields = i.split()
            if len(fields) == 0: continue                
            if id_ind is None and "JOBID" in fields:
                id_ind = fields.index("JOBID")
                if "STATE" not in fields:
                    print("ERROR in check_queue: Could not identify STATE column in squeue -l output.")
                    quit()
                else:
                    state_ind = fields.index("STATE")
                if "USER" not in fields:
                    print("ERROR in check_queue: Could not identify USER column in squeue -l output.")
                    quit()
                else:
                    user_ind = fields.index("USER")
                continue

            # If this job belongs to the user and it is pending or running, then add it to the list of active jobs
            if id_ind is not None and fields[user_ind] == check_queue.user and fields[state_ind] in ["PENDING","RUNNING"]:
                jobs += [fields[id_ind]]

    # Get running and pending jobs using the pbs scheduler
    elif check_queue.sched == "pbs":

        # redirect a qstat call into output
        output = subprocess.check_output("qstat -f", shell=True).decode('utf-8')

        # Initialize job information dictionary
        jobs = []
        job_dict = {}
        current_key = None
        for count_i,i in enumerate(output.split('\n')):
            fields = i.split()
            if len(fields) == 0: continue
            if "Job Id" in i:

                # Check if the previous job belongs to the user and needs to be added to the pending or running list. 
                if current_key is not None:
                    if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                        jobs += [current_key]
                current_key = i.split()[2]
                job_dict[current_key] = { "State":"NA" , "Name":"NA", "Walltime":"NA", "Queue":"NA", "User":"NA"}
                continue
            if "Job_Name" == fields[0]:
                job_dict[current_key]["Name"] = fields[2]
            if "job_state" == fields[0]:
                job_dict[current_key]["State"] = fields[2]
            if "queue" == fields[0]:
                job_dict[current_key]["Queue"] = fields[2]
            if "Resource_List.walltime" == fields[0]:
                job_dict[current_key]["Walltime"] = fields[2]        
            if "Job_Owner" == fields[0]:
                job_dict[current_key]["User"] = fields[2].split("@")[0]

        # Check if the last job belongs to the user and needs to be added to the pending or running list. 
        if current_key is not None:
            if job_dict[current_key]["State"] in ["R","Q"] and job_dict[current_key]["User"] == check_queue.user:
                jobs += [current_key]

    return jobs

def main():
    print('Functions::main()')
    return

if __name__ == '__main__':
    main()
