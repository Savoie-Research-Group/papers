#!/bin/env python                                                                                                                                                             
# Author: Zih-Yu Lin (lin1209@purdue.edu)
import sys,argparse,subprocess,os,time,math,shutil

# Add path
sys.path.append('/'.join(os.path.abspath(__file__).split('/')[:-1])+'/util')
from monitor_jobs import *
from matplotlib import pyplot as plt


def main(argv):

   # Get username and path for this py file
   global username, repo_path
   username = os.getlogin()
   repo_path = os.path.dirname(os.path.abspath(__file__))

   
   parser = argparse.ArgumentParser(description='parametrize xyz files in the executed folder in batch')
   #optional arguments                                                                                                                                                        
   parser.add_argument('-opt_method',dest='opt_method',type=str,default='dft',
                        help = 'method to optimize geometry, default: dft, options: xtb, dft')
   parser.add_argument('-orca_exe',dest='orca_exe',type=str,default='/depot/bsavoie/apps/orca_4_1_2/orca',
                        help = 'orca executable path')
   parser.add_argument('-xtb_exe',dest='xtb_exe',type=str,default='/depot/bsavoie/apps/xTB/bin/xtb',
                        help = 'xtb executable path')
   parser.add_argument('-amber_exe',dest='amber_exe',type=str,default='/depot/bsavoie/apps/amber20/bin/',
                        help = 'Amber Tools executable bin "folder"')
   parser.add_argument('-cpu',dest='cpu',type=int,default=128,
                        help = 'number of cpu for ORCA jobs, default:128')
   parser.add_argument('-queue',dest='queue',type=str,default='standby',
                        help = 'submission queue, default: standby')
   parser.add_argument('-wall_time',dest='wall_time',type=int,default=4,
                        help = 'wall time for submission in hour, default: 4 hr')
   parser.add_argument('--collect', dest='collect_flag', default=False, action='store_const', const=True,
                        help = 'When this flag is supplied will only collect files instead of runiing the whole parametrization') 
   parser.add_argument('--geo', dest='geo_flag', default=False, action='store_const', const=True,
                        help = 'When this flag is supplied will run xtb geometry optimization, default: no geo optimization')

   # Parse the inputs
   global args, exe_dir
   args=parser.parse_args(argv)
   exe_dir = os.getcwd()   

   sys.stdout = Logger('step0.prepare_ligand')
   print("PROGRAM CALL:  python {}\n".format(' '.join([i for i in sys.argv])))
   if args.geo_flag:
      print("You're using {} as your geometry optimization method".format(args.opt_method))
   else:
      print("WARNING: You're not doing geometry optimization") 

   # collect xyz files in the execution folder
   xyzs = [ _ for _ in os.listdir(exe_dir) if _.split('.')[-1] == 'xyz']


   # collect existing files and check parametrization status only 
   if args.collect_flag:
      # collect files and print stats:
      if os.path.isdir('Ready') is False:
         os.makedirs('Ready')
      print("\n\nERROR LIST:")
      for xyz in xyzs:
         ligand_name = xyz.split('.')[-2]
         if args.opt_method == 'xtb':
            geo_dir = '{}/{}/geo_xtb'.format(exe_dir,ligand_name)
         elif args.opt_method == 'dft':
            geo_dir = '{}/{}/geoopt'.format(exe_dir,ligand_name)
         charge_dir = '{}/{}/charge'.format(exe_dir,ligand_name)
         makedb_dir = '{}/{}/make_db'.format(exe_dir,ligand_name)
         db_file = '{}/{}/{}.db'.format(makedb_dir,ligand_name,ligand_name)
         if os.path.isfile(db_file):
            if check_makedb_success(makedb_dir) is False:
               print('{} has missing parameters'.format(xyz))
               continue
            if args.geo_flag:
               if args.opt_method == 'xtb':
                  shutil.copy('{}/xtbopt.xyz'.format(geo_dir),'Ready/{}_optimized.xyz'.format(ligand_name))
               elif args.opt_method == 'dft':
                  shutil.copy('{}/geo_opt.xyz'.format(geo_dir),'Ready/{}_optimized.xyz'.format(ligand_name))
               shutil.copy(db_file,'Ready/{}_optimized.db'.format(ligand_name))
            else:
               shutil.copy('{}'.format(xyz),'Ready/{}'.format(xyz))
               shutil.copy(db_file,'Ready/{}.db'.format(ligand_name))
         else:
            if args.geo_flag:
               if args.opt_method == 'xtb':
                  if check_xtb_complete('{}/xtb.err'.format(geo_dir)) is False:     
                     print('{} xtb optimization failed'.format(xyz))
                     continue
               elif args.opt_method == 'dft':
                  if check_orca_complete('{}/geoopt.out'.format(geo_dir)) is False:     
                     print('{} DFT optimization failed'.format(xyz))
                     continue
            if check_orca_complete('{}/charge.out'.format(charge_dir)) is False:
               print('{} charge ORCA job failed'.format(xyz))
               continue
            if os.path.isfile('{}/charge_parse/fit_charges.db'.format(charge_dir)) is False:
               print('{} charge fitting job failed'.format(xyz))
               continue
            print('{} makedb failed'.format(xyz))
      quit()
      

   # check for executable
   if args.geo_flag is True:
      if args.opt_method == 'dft':
         if os.path.isfile(args.orca_exe) is False:
            print('ERROR: ORCA executable: {} does not exist'.format(args.orca_exe))
            quit()
      elif args.opt_method == 'xtb':
         if os.path.isfile(args.orca_exe) is False:
            print('ERROR: xtb executable: {} does not exist'.format(args.xtb_exe))
            quit()
      else:
         print('ERROR: unrecognized optmization method {}, only xtb and dft are supported'.format(args.opt_method))
         quit()
   if os.path.isdir(args.amber_exe) is False:
      print('ERROR: Amber Tools executable folder: {} does not exist'.format(args.amber_exe))
      quit()
   

   # create folders
   for xyz in xyzs:
      # create Ligand folder
      ligand_name = xyz.split('.')[-2]
      if os.path.isdir(ligand_name) is False:
         os.makedirs(ligand_name)
      if args.opt_method == 'xtb':
         geo_dir = '{}/{}/geo_xtb'.format(exe_dir,ligand_name)
      elif args.opt_method == 'dft':
         geo_dir = '{}/{}/geoopt'.format(exe_dir,ligand_name)
      else:
         print("ERROR: opt method: {} not supported".format(args.opt_method))
         quit()
      if args.geo_flag:
         if os.path.isdir(geo_dir) is False:
            os.makedirs(geo_dir)
      charge_dir = '{}/{}/charge'.format(exe_dir,ligand_name)
      if os.path.isdir(charge_dir) is False:
         os.makedirs(charge_dir)
      makedb_dir = '{}/{}/make_db'.format(exe_dir,ligand_name)
      if os.path.isdir(makedb_dir) is False:
         os.makedirs(makedb_dir)

   if args.geo_flag: 
      # run DFT geo parametrization
      print('submitting DFT geometry optimization')
      jobs = []
      submit_xyzs = []
      for xyz in xyzs:
         if args.opt_method == 'xtb':
            tmp_job = submit_xtbopt(xyz)
         else:
            tmp_job = submit_dftopt(xyz)

         if tmp_job != []:
            submit_xyzs.append(xyz)
         jobs +=  tmp_job
      print('the following xyzs are submitted for geometry optimization: {}'.format(' '.join(submit_xyzs)))
      monitor_jobs(jobs,username)
      

   # run charge parametrization
   print('submitting charge singlepoint jobs')
   jobs = []
   for xyz in xyzs:
      tmp_job = submit_charge_singlepoint(xyz,args.geo_flag)
      jobs += tmp_job
   monitor_jobs(jobs,username)

   # run extract charges
   print('submitting extract charge')
   jobs = []
   for xyz in xyzs:
      tmp_job = submit_charge_extract(xyz,args.geo_flag)
      jobs += tmp_job
   monitor_jobs(jobs,username)

   # run makedb 
   print('run makedb')
   for xyz in xyzs:
      run_makedb(xyz,args.geo_flag)

   # collect files and print stats:
   if os.path.isdir('Ready') is False:
      os.makedirs('Ready')
   print("\n\nERROR LIST:")
   for xyz in xyzs:
      ligand_name = xyz.split('.')[-2]
      if args.opt_method == 'xtb':
         geo_dir = '{}/{}/geo_xtb'.format(exe_dir,ligand_name)
      elif args.opt_method == 'dft':
         geo_dir = '{}/{}/geoopt'.format(exe_dir,ligand_name)
      charge_dir = '{}/{}/charge'.format(exe_dir,ligand_name)
      makedb_dir = '{}/{}/make_db'.format(exe_dir,ligand_name)
      db_file = '{}/{}/{}.db'.format(makedb_dir,ligand_name,ligand_name)
      if os.path.isfile(db_file):
         if check_makedb_success(makedb_dir) is False:
            print('{} has missing parameters'.format(xyz))
            continue
         if args.geo_flag:
            if args.opt_method == 'xtb':
               shutil.copy('{}/xtbopt.xyz'.format(geo_dir),'Ready/{}_optimized.xyz'.format(ligand_name))
            elif args.opt_method == 'dft':
               shutil.copy('{}/geo_opt.xyz'.format(geo_dir),'Ready/{}_optimized.xyz'.format(ligand_name))
            shutil.copy(db_file,'Ready/{}_optimized.db'.format(ligand_name))
         else:
            shutil.copy('{}'.format(xyz),'Ready/{}'.format(xyz))
            shutil.copy(db_file,'Ready/{}.db'.format(ligand_name))
      else:
         if args.geo_flag:
            if args.opt_method == 'xtb':
               if check_xtb_complete('{}/xtb.err'.format(geo_dir)) is False:     
                  print('{} xtb optimization failed'.format(xyz))
                  continue
            elif args.opt_method == 'dft':
               if check_orca_complete('{}/geoopt.out'.format(geo_dir)) is False:     
                  print('{} DFT optimization failed'.format(xyz))
                  continue
         if check_orca_complete('{}/charge.out'.format(charge_dir)) is False:
            print('{} charge ORCA job failed'.format(xyz))
            continue
         if os.path.isfile('{}/charge_parse/fit_charges.db'.format(charge_dir)) is False:
            print('{} charge fitting job failed'.format(xyz))
            continue
         print('{} makedb failed'.format(xyz))
      
   quit()

def check_makedb_success(makedb_dir):
   with open('{}/makedb.out'.format(makedb_dir),'r') as f:
      for lc,lines in enumerate(f):
         fields = lines.split()
         if len(fields)>3 and fields[1] == 'Missing' and fields[2] == 'parameters':
            name = makedb_dir.split('/')[-2]
            db_file = '{}/{}/{}.db'.format(makedb_dir,name,name)
            if os.path.isfile(db_file):
               with open(db_file,'r') as g:
                  for lg,lines_g in enumerate(g):
                     fields = lines_g.split()
                     if len(fields) > 4 and fields[-3] == 'FOUND' and fields[-4] == 'NOT':
                        return False
      
   return True

def submit_xtbopt(xyz):
   ligand_name = xyz.split('.')[-2]
   ori_dir = os.getcwd() 
   geo_dir = '{}/{}/geo_xtb'.format(exe_dir,ligand_name)
   os.chdir(geo_dir)
   job = []
   if check_xtb_complete('xtb.err') is False:
      # write submit file
      write_xtbsh('bsavoie',1,ligand_name+'_geo',xyz)
      # submit job and wait 
      job = submit_job_nomonitor('xtb.sh')
   os.chdir(ori_dir)

   return job

def submit_dftopt(xyz):
   ligand_name = xyz.split('.')[-2]
   ori_dir = os.getcwd() 
   geo_dir = '{}/{}/geoopt'.format(exe_dir,ligand_name)
   os.chdir(geo_dir)
   job = []
   in_xyz = '{}/{}'.format(exe_dir,xyz)
   if check_orca_complete('geoopt.out') is False and os.path.isfile('geo_opt.xyz') is False:
      # if there are files from previous unfinished job, delete them
      files = [ _ for _ in os.listdir() if _ not in ['geoopt.out','geoopt.in'] ]
      for _ in files:
         os.remove(_)
      # write orca input file
      write_optin(args.cpu,in_xyz)
      # write submit file
      write_orcash(args.queue,args.cpu,ligand_name+'_geo','geoopt.in','geoopt')
      # submit job and wait 
      job = submit_job_nomonitor('orca.sh')
   os.chdir(ori_dir)
   return job
         
def submit_charge_singlepoint(xyz,geo_flag):
   ligand_name = xyz.split('.')[-2]
   ori_dir = os.getcwd() 
   c_dir = '{}/{}/charge'.format(exe_dir,ligand_name)
   if args.opt_method == 'xtb':
      geo_dir = '{}/{}/geo_xtb'.format(exe_dir,ligand_name)
   elif args.opt_method == 'dft':
      geo_dir = '{}/{}/geoopt'.format(exe_dir,ligand_name)
   os.chdir(c_dir)
   job = []
   in_xyz = '{}/{}'.format(exe_dir,xyz)
   if geo_flag:
      # if previous geometry optimization failed, don't run singlepoint
      if args.opt_method == 'xtb': 
         if check_xtb_complete('{}/xtb.err'.format(geo_dir)) is False:  
            os.chdir(ori_dir)
            return job
         else:
            in_xyz = '{}/xtbopt.xyz'.format(geo_dir) 
      elif args.opt_method == 'dft':
         if check_orca_complete('{}/geoopt.out'.format(geo_dir)) is False:
            os.chdir(ori_dir)
            return job
         else:
            in_xyz = '{}/geo_opt.xyz'.format(geo_dir) 

   if check_orca_complete('charge.out') is False:
      # write orca input file
      write_chargein(args.cpu,in_xyz)
      # write submit file
      write_orcash(args.queue,args.cpu,ligand_name+'_Q','charge.in','charge')
      # submit job and wait 
      job = submit_job_nomonitor('orca.sh')
   os.chdir(ori_dir)

   return job

def submit_charge_extract(xyz,geo_flag):
   ligand_name = xyz.split('.')[-2]
   ori_dir = os.getcwd() 
   c_dir = '{}/{}/charge'.format(exe_dir,ligand_name)
   os.chdir(c_dir)
   job = []
   # if previous ORCA singlepoint failed, don't run extract
   if check_orca_complete('charge.out') is False:
      os.chdir(ori_dir)
      return job

   if os.path.isfile('{}/charge_parse/fit_charges.db'.format(c_dir)) is False:
      if os.path.isdir('{}/charge_parse'.format(c_dir)):
         shutil.rmtree('{}/charge_parse'.format(c_dir))
      # extract charge
      write_extractQsh(args.queue,ligand_name+'_ext_Q',xyz,geo_flag)
      job = submit_job_nomonitor('extract_Q.sh')
   os.chdir(ori_dir)
   return job

def run_makedb(xyz,geo_flag):
   ori_dir = os.getcwd() 
   ligand_name = xyz.split('.')[-2]
   if args.opt_method == 'xtb':
      geo_dir = '{}/{}/geo_xtb'.format(exe_dir,ligand_name)
   elif args.opt_method == 'dft':
      geo_dir = '{}/{}/geoopt'.format(exe_dir,ligand_name)
   c_dir = '{}/{}/charge'.format(exe_dir,ligand_name)
   makedb_dir = '{}/{}/make_db'.format(exe_dir,ligand_name)
   if os.path.isfile('{}/charge_parse/fit_charges.db'.format(c_dir)) is False:
      return

   os.chdir(makedb_dir)
   if os.path.isfile('{}/{}/{}.db'.format(makedb_dir,ligand_name,ligand_name)) is False:
       with open('makedb.sh','w') as f:
            f.write("#!/bin/bash\n")
            if geo_flag:
               if args.opt_method == 'xtb':
                  f.write('python {}/util/make_db.py {}/xtbopt.xyz {}/gaff2_perov.dat -charge_file ../charge/charge_parse/fit_charges.db -m 1 -q 1 -amber {} --perovskite -o {} > makedb.out\n'.format(repo_path,geo_dir,repo_path,args.amber_exe,ligand_name))
               elif args.opt_method == 'dft':
                  f.write('python {}/util/make_db.py {}/geo_opt.xyz {}/gaff2_perov.dat -charge_file ../charge/charge_parse/fit_charges.db -m 1 -q 1 -amber {} --perovskite -o {} > makedb.out\n'.format(repo_path,geo_dir,repo_path,args.amber_exe,ligand_name))
            else:
               f.write('python {}/util/make_db.py {}/{} {}/gaff2_perov.dat -charge_file ../charge/charge_parse/fit_charges.db -m 1 -q 1 -amber {} --perovskite -o {} > makedb.out\n'.format(repo_path,exe_dir,xyz,repo_path,args.amber_exe,ligand_name))
       command = 'sh makedb.sh'.split()
       process = subprocess.Popen(command)
       status = process.poll()                                                                                                                                                                        
       while status == None:
          status = process.poll()
          time.sleep(3)
   os.chdir(ori_dir)
   return


def write_chargein(cpu,xyz):

   with open('charge.in','w') as f:
     f.write('# CHELPG calculation for  {}\n'.format(xyz)) 
     f.write('! wB97X-D3 def2-TZVP TightSCF CHELPG Grid4 \n\n')
     f.write('%base \"charges\"\n\n') 
     f.write('%pal nprocs {}\n'.format(cpu))
     f.write('end\n')
     f.write('#     charge multiplicity\n')
     f.write('* xyz 1 1\n')
     with open(xyz,'r') as g:
         for lc,lines in enumerate(g):
            fields = lines.split()
            if lc >= 2 and len(fields) != 0:
               f.write('  ')
               f.write(lines)
     f.write('*')

   return

def write_optin(cpu,xyz):

   with open('geoopt.in','w') as f:
     f.write('# Geometry optimization {}\n'.format(xyz)) 
     f.write('! wB97X-D3 def2-TZVP Opt TightSCF Grid4 xyzfile\n\n')
     f.write('%base \"geo_opt\"\n\n') 
     f.write('%pal nprocs {}\n'.format(cpu))
     f.write('end\n')
     f.write('#     charge multiplicity\n')
     f.write('* xyz 1 1\n')
     with open(xyz,'r') as g:
         for lc,lines in enumerate(g):
            fields = lines.split()
            if lc >= 2 and len(fields) != 0:
               f.write('  ')
               f.write(lines)
     f.write('*')

   return

def write_orcash(queue,cpu,jobname,infile,outname):

   with open('orca.sh','w') as f:
      f.write('#!/bin/bash\n')
      f.write('#\n')
      f.write('#SBATCH --job-name {}\n'.format(jobname))
      f.write('#SBATCH -o orca.out\n')
      f.write('#SBATCH -e orca.err\n')
      f.write('#SBATCH -A {}\n'.format(queue))
      f.write('#SBATCH -N 1\n')
      f.write('#SBATCH -n {}\n'.format(cpu)) 
      f.write('#SBATCH -t {}:00:00\n'.format(args.wall_time))
      f.write('\n#load necessary module\n\n')
      f.write('module load openmpi/3.1.4\n\n')
      f.write('{} {} > {}.out\n'.format(args.orca_exe,infile,outname))

   return

def write_xtbsh(queue,cpu,jobname,xyz):

   with open('xtb.sh','w') as f:
      f.write('#!/bin/bash\n')
      f.write('#\n')
      f.write('#SBATCH --job-name {}\n'.format(jobname))
      f.write('#SBATCH -o xtb.out\n')
      f.write('#SBATCH -e xtb.err\n')
      f.write('#SBATCH -A {}\n'.format(queue))
      f.write('#SBATCH -N 1\n')
      f.write('#SBATCH -n {}\n'.format(cpu)) 
      f.write('#SBATCH -t {}:00:00\n'.format(args.wall_time))
      f.write('{} {}/{} --opt --chrg 1\n'.format(args.xtb_exe,exe_dir,xyz))

   return


def check_xtb_complete(xtberr):
   success = False
   if os.path.isfile(xtberr):
      with open(xtberr,'r') as f:
         for lc,lines in enumerate(f):
            fields = lines.split()
            if lc == 0 and fields[0] == 'normal':
               success = True
   return success

def check_orca_complete(outname):
   success = False
   if os.path.isfile(outname):
      with open(outname,'r') as f:
         for lc,lines in enumerate(f):
            if "****ORCA TERMINATED NORMALLY****" in lines: 
               success = True
   return success
            

def write_extractQsh(queue,jobname,xyz,geo_flag):

   with open('extract_Q.sh','w') as f:
      f.write('#!/bin/bash\n')
      f.write('#\n')
      f.write('#SBATCH --job-name {}\n'.format(jobname))
      f.write('#SBATCH -o orca.out\n')
      f.write('#SBATCH -e orca.err\n')
      f.write('#SBATCH -A {}\n'.format(queue))
      f.write('#SBATCH -N 1\n')
      f.write('#SBATCH -n 1\n') 
      f.write('#SBATCH -t {}:00:00\n'.format(args.wall_time))
      f.write('\n')
      if geo_flag:
         if args.opt_method == 'xtb':
            f.write('python {}/util/extract_charges.py charges.vpot -xyz ../geo_xtb/xtbopt.xyz -out charge.out -q 1 --two_step\n'.format(repo_path,exe_dir,xyz))
         elif args.opt_method == 'dft':
            f.write('python {}/util/extract_charges.py charges.vpot -xyz ../geoopt/geo_opt.xyz -out charge.out -q 1 --two_step\n'.format(repo_path,exe_dir,xyz))
      else:
         f.write('python {}//util/extract_charges.py charges.vpot -xyz {}/{} -out charge.out -q 1 --two_step\n'.format(repo_path,exe_dir,xyz))

   return
      
      
      
def submit_job_nomonitor(shname):
    command  = 'sbatch {}'.format(shname).split()
    output = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8').communicate()[0]
    jobid = [ output.split("\n")[-2].split()[-1]]
   
    return jobid 

def submit_job(shname):
    command  = 'sbatch {}'.format(shname).split()
    output = subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8').communicate()[0]
    jobid = [ output.split("\n")[-2].split()[-1]]
    monitor_jobs(jobid,username)
   
    return 
               

# Create logger to save stdout to logfile
class Logger(object):

    def __init__(self,logname):
        self.terminal = sys.stdout
        self.log = open(logname+'.log', "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  
    def flush(self):
        pass

   

if __name__ == "__main__":
   main(sys.argv[1:])
