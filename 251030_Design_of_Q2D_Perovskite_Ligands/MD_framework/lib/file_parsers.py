#!/bin/env python
# Author: Brett Savoie (brettsavoie@gmail.com)

from numpy import *



# Description: Simple wrapper function for grabbing the coordinates and
#              elements from an xyz file
#
# Inputs      input: string holding the filename of the xyz
# Returns     Elements: list of element types (list of strings)
#             Geometry: Nx3 array holding the cartesian coordinates of the
#                       geometry (atoms are indexed to the elements in Elements)
#
def xyz_parse(input,read_types=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry,Atom_types


# Description: Parses keywords and geometry block from an orca input file
#
# Inputs        input: string holding the filename of the orca input file
# Returns       orca_dict: dictionary holding the run information for each job in the input file
#                          the first key in the dictionary corresponds to the job number (i.e.,
#                          orca_dict["0"] references the information in the first job. The job info
#                          can be accessed with content specific keys ("header_commands", "geo", 
#                          "elements", "constraints", "N_proc", "job_name", "charge", "multiplicity",
#                          "content", "geom_block" )
def orca_in_parse(input):

    # Iterate over the contents and return a dictionary of input components indexed to each job in the input file
    job_num = 0
    orca_dict = {str(job_num):{"header_commands": "","content": "","elements": None, "geo": None, "constraints": None, "geo_opts_block": None, "job_name": None}}
    geo_opts_flag = 0
    geo_block_flag = 0
    con_flag  = 0
    
    # Open the file and begin the parse
    with open(input,'r') as f:
        for lc,lines in enumerate(f):

            # Grab fields 
            fields = lines.split()            
            
            # Update the "content" block, which contains everything
            orca_dict[str(job_num)]["content"] += lines

            # If a new job is encountered reset all flags and update the job_num counter            
            if len(fields) > 0 and fields[0] == "$new_job":
                job_num += 1
                con_flag = 0
                geo_opts_flag = 0
                geo_block_flag = 0
                orca_dict[str(job_num)] = {"header_commands": "","content": "","elements": None, "geo": None, "constraints": None, "geo_opts_block": None, "N_proc": orca_dict[str(job_num-1)]["N_proc"]}

            # Component based parse commands
            if len(fields) > 0 and fields[0] == "!":
                orca_dict[str(job_num)]["header_commands"] += " ".join(fields[1:]) + " "
                if "PAL" in lines:
                    orca_dict[str(job_num)]["N_proc"] = int([ i.split("PAL")[1] for i in fields if "PAL" in i ][0])
                elif job_num != 0:
                    orca_dict[str(job_num)]["N_proc"] = orca_dict[str(job_num-1)]["N_proc"]
                else:
                    orca_dict[str(job_num)]["N_proc"] = 1                    
            if len(fields) > 0 and fields[0] == "%base":
                orca_dict[str(job_num)]["job_name"] = fields[1]
                
            # Check for turning on flags
            if len(fields) > 0 and fields[0] == "%geom":
                geo_opts_flag = 1
                orca_dict[str(job_num)]["geo_opts_block"] = ""                
                continue
            if len(fields) > 0 and fields[0] == "Constraints":
                if geo_opts_flag == 1:
                    orca_dict[str(job_num)]["geo_opts_block"] += lines                
                con_flag = 1
                orca_dict[str(job_num)]["constraints"] = ""
                continue
            if len(fields) >= 2 and fields[0] == "*" and fields[1] == "xyz":
                geo_block_flag = 1
                orca_dict[str(job_num)]["charge"] = float(fields[2])
                orca_dict[str(job_num)]["multiplicity"] = int(fields[3])
                orca_dict[str(job_num)]["geo"] = []
                orca_dict[str(job_num)]["elements"] = []
                continue
            if len(fields) >= 2 and fields[0] == "*" and fields[1] == "xyzfile":
                orca_dict[str(job_num)]["charge"] = float(fields[2])
                orca_dict[str(job_num)]["multiplicity"] = int(fields[3])                
                orca_dict[str(job_num)]["geo"] = None
                orca_dict[str(job_num)]["elements"] = None
                continue

            # Checks for turning off flags
            if con_flag == 1 and len(fields) > 0 and fields[0] == "end":
                con_flag = 0
                continue
            if geo_opts_flag == 1 and len(fields) > 0 and fields[0] == "end":
                geo_opts_flag = 0            
                continue
            if geo_block_flag == 1 and len(fields) > 0 and fields[0] == "*":
                geo_block_flag = 0            
                continue
            
            # Flag based parse commands
            if geo_opts_flag == 1:
                orca_dict[str(job_num)]["geo_opts_block"] += lines
            if con_flag == 1:
                orca_dict[str(job_num)]["constraints"] += lines
            if geo_block_flag == 1:
                orca_dict[str(job_num)]["geo"] += [ [ float(i) for i in fields[1:] ] ]
                orca_dict[str(job_num)]["elements"] += [ str(fields[0]) ]
            
    return orca_dict

