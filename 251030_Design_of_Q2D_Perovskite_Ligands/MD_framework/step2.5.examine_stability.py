import numpy as np
import sys, os, time, argparse
# sys.path.append('/home/nianz/code_repo/high-throughput-2D-Perovskite-ligand-screening-tool/MD/lib')
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import functions

def main(*argv):
    print('-'*50)
    print('python step2.5.examine_stability.py' + ' '.join([str(x) for x in argv]))
    print('-'*50)

    parser = argparse.ArgumentParser()
    # positional arguments
    parser.add_argument("-stab_file", type=str, help="stability file", default='stability.out')
    parser.add_argument('-d', '--dir', type=str, help='directory', default='')
    parser.add_argument('-o', '--out', type=str, help='output file', default='summary.txt')
    
    args = parser.parse_args()
    path = args.dir
    if path == '':
        directories = sorted([x for x in os.listdir() if os.path.isdir(x) and x.startswith('NH3')])
    else:
        directories = sorted([x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x)) and x.startswith('NH3')])
    
    out_file = args.out
    stability_file = args.stab_file
    Data = {}
    for d in directories:
        if not os.path.isdir(os.path.join(path, d)):
            print('No directory found in {}'.format(d))
            continue
        with functions.cd(os.path.join(path, d)):
            if not os.path.isfile(stability_file):
                print('No stability file found in {}'.format(d))
                continue
            bond_flag = False
            angle_flag = False
            d_h_flag = False
            bond_lqes, angle_vars, d_hs, m_x_ms = [], [], [], []
            with open(stability_file, 'r') as f:
                lines = f.readlines()
                for l in lines:
                    temp = l.split()
                    if 'bond_lqe' in temp and 'angle_var' in temp:
                        bond_flag = True
                        angle_flag = True
                    
                    if bond_flag and angle_flag:
                        try:
                            float(temp[1]), float(temp[2])
                        except:
                            if len(temp) == 0:
                                bond_flag = False
                                angle_flag = False
                        else:
                            bond_lqes.append(float(temp[1]))
                            angle_vars.append(float(temp[2]))
                    
                    if 'drifting_halides' in temp:
                        d_h_flag = True
                    
                    if d_h_flag:
                        try:
                            float(temp[1])
                        except:
                            if len(temp) == 0:
                                d_h_flag = False
                        else:
                            d_hs.append(float(temp[1]))
                            m_x_ms.append(float(temp[2]))
                            
                print('current directory: {}'.format(d))
                print('bond_lqe: {}'.format(np.mean(bond_lqes)))
                print('angle_var: {}'.format(np.mean(angle_vars)))
                print('drifting_halides: {}'.format(np.mean(d_hs)))
                print('m_x_m: {}'.format(np.mean(m_x_ms)))
                
                Data[d] = (bond_lqes, angle_vars, d_hs, m_x_ms)
    
    with open(out_file, 'w') as f:
        f.write("summary of the stability of the system\n")
        f.write("{:<100} {:<20} {:<20} {:<20} {:<20}\n".format('directory', 'bond_lqe', 'angle_var', 'drifting_halides', 'm_x_m'))
        for d in Data:
            f.write("{:<100} {:<20.5f} {:<20.5f} {:<20.5f} {:<20.5f}".format(d, np.mean(Data[d][0]), np.mean(Data[d][1]), np.mean(Data[d][2]), abs(180-np.mean(Data[d][3]))))
            if np.mean(Data[d][2]) > 5:
                f.write('Warning: drifting_halides > 5\n')
            else:
                f.write('\n')

    return

if __name__ == '__main__':
    main(*sys.argv[1:])