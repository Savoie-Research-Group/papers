import pandas as pd
import matplotlib.pyplot as plt

plotting = {
    'uDFT': {'width': 3.5, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#648FFF', 'line': '-'},
    'rDFT': {'width': 2.2, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#FE6100', 'line': '--'},
    'AIMNET2': {'width': 1.5, 'marker': 'x', 'freq': None, 'size': 6.0, 'color': '#785EF0', 'line': ':'},
    'AIMNET2-NSE': {'width': 1.5, 'marker': 'v', 'freq': None, 'size': 4.0, 'color': "#39A24E", 'line': ':'},
    'UMA-OMOL': {'width': 1.5, 'marker': 's', 'freq': None, 'size': 2.5, 'color': '#DC267F', 'line': ':'},
    'MACE-OMOL': {'width': 1.5, 'marker': 'o', 'freq': None, 'size': 3.0, 'color': '#FFB000', 'line': ':'},
    'MACE-POLAR': {'width': 1.5, 'marker': 'd', 'freq': None, 'size': 3.5, 'color': "#DF82F9FF", 'line': ':'},
    'ORB-OMOL': {'width': 1.5, 'marker': '<', 'freq': None, 'size': 2.5, 'color': '#7B5C73', 'line': ':'}
}

def setup_fonts():
    """
    Configures matplotlib global font settings.
    Adjust sizes and family here.
    """
    plt.rcParams.update({
        'font.size': 12,              # Global text size
        'axes.titlesize': 16,         # Title of subplots
        'axes.labelsize': 10,         # X and Y labels
        'xtick.labelsize': 9,        # Tick numbers
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',  # Options: 'serif', 'sans-serif', 'monospace'
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Tries to find these fonts in order
        'lines.linewidth': 2          # Default line width
    })

hartree2ev = 27.2113834

clrad_wb97m_d3bj_tzvpp = -460.198150092473
clrad_wb97mv_tzvpd = -460.126133284408

clrad_h2o_wb97m_d3bj_tzvpp = -536.677190857420
clrad_h2o_wb97mv_tzvpd = -536.561549856428

ch3rad_wb97m_d3bj_tzvpp = -39.858431131849
ch3rad_wb97mv_tzvpd = -39.817860984076

clan_wb97m_d3bj_tzvpp = -460.321344191436
clan_wb97mv_tzvpd = -460.259990695393

ch3cat_wb97m_d3bj_tzvpp = -39.498897076068
ch3cat_wb97mv_tzvpd = -39.461370990600

hcl_h2o_wb97m_d3bj_tzvpp = -537.353734212187
hcl_h2o_wb97mv_tzvpd = -537.233044621148

ch3cat_wb97m_d3bj_tzvpp = -39.498897076068
ch3cat_wb97mv_tzvpd = -39.461370990600

hcl_h2o_wb97m_d3bj_tzvpp = -537.353734212187
hcl_h2o_wb97mv_tzvpd = -537.233044621148

ch3cl_reference = {'ks': (clrad_wb97mv_tzvpd + ch3rad_wb97mv_tzvpd) * hartree2ev,
                        'omol': (clrad_wb97mv_tzvpd + ch3rad_wb97mv_tzvpd) * hartree2ev,
                        'aimnet2': (clrad_wb97m_d3bj_tzvpp + ch3rad_wb97m_d3bj_tzvpp) * hartree2ev}

ch3cl_h2o_reference = {'ks': (clrad_h2o_wb97mv_tzvpd + ch3rad_wb97mv_tzvpd) * hartree2ev,
                        'omol': (clrad_h2o_wb97mv_tzvpd + ch3rad_wb97mv_tzvpd) * hartree2ev,
                        'aimnet2': (clrad_h2o_wb97m_d3bj_tzvpp + ch3rad_wb97m_d3bj_tzvpp) * hartree2ev}

ch3cl_h3o_reference = {'ks': (hcl_h2o_wb97mv_tzvpd + ch3cat_wb97mv_tzvpd) * hartree2ev,
                        'omol': (hcl_h2o_wb97mv_tzvpd + ch3cat_wb97mv_tzvpd) * hartree2ev,
                        'aimnet2': (hcl_h2o_wb97m_d3bj_tzvpp + ch3cat_wb97m_d3bj_tzvpp) * hartree2ev}

def get_relative_energies(raw_df, ref_dict):

    rel_df = pd.DataFrame()
    rel_df['R'] = raw_df['dist_angs']

    for col in raw_df.columns:
        if 'uks' in col:
            rel_df['uDFT'] = raw_df[col] * hartree2ev - ref_dict['ks']
        elif 'rks' in col:
            rel_df['rDFT'] = raw_df[col] * hartree2ev - ref_dict['ks']
        elif 'fairchem' in col:
            rel_df['UMA-OMOL'] = raw_df[col] - ref_dict['omol']
        # elif 'orb' in col:
        #     rel_df['ORB-OMOL'] = raw_df[col] - ref_dict['omol']
        elif 'mace' in col:
            rel_df['MACE-OMOL'] = raw_df[col] - ref_dict['omol']
        elif 'polar' in col:
            rel_df['MACE-POLAR'] = raw_df[col] - ref_dict['omol']
        elif 'aimnet2' in col:
            rel_df['AIMNET2'] = raw_df[col] - ref_dict['aimnet2']
        elif 'aim2nse' in col:
            rel_df['AIMNET2-NSE'] = raw_df[col] - ref_dict['aimnet2']

    return rel_df


def main():
    setup_fonts()

    ch3cl_dft = pd.read_csv('../../data/sn1/ch3cl/ch3cl_wb97mv_tzvpd.csv')
    ch3cl_h2o_dft = pd.read_csv('../../data/sn1/ch3cl_h2o/ch3clh2o_wb97mv_tzvpd.csv')
    ch3cl_h3o_dft = pd.read_csv('../../data/sn1/ch3cl_h3o/ch3clh3o_wb97mv_tzvpd.csv')

    ch3cl_mlip = pd.read_csv('../../data/sn1/ch3cl/MLIP_ch3cl.csv')
    ch3cl_h2o_mlip = pd.read_csv('../../data/sn1/ch3cl_h2o/MLIP_ch3clh2o.csv')
    ch3cl_h3o_mlip = pd.read_csv('../../data/sn1/ch3cl_h3o/MLIP_ch3clh3o.csv')

    ch3cl_df = pd.merge(ch3cl_dft, ch3cl_mlip, on='file', how='inner')
    ch3cl_rel = get_relative_energies(ch3cl_df, ch3cl_reference)
    ch3cl_h2o_df = pd.merge(ch3cl_h2o_dft, ch3cl_h2o_mlip, on='file', how='inner')
    ch3cl_h2o_rel = get_relative_energies(ch3cl_h2o_df, ch3cl_h2o_reference)
    ch3cl_h3o_df = pd.merge(ch3cl_h3o_dft, ch3cl_h3o_mlip, on='file', how='inner')
    ch3cl_h3o_rel = get_relative_energies(ch3cl_h3o_df, ch3cl_h3o_reference)

    plot_order = [
        'uDFT',
        'rDFT',
        'AIMNET2-NSE',
        'AIMNET2',
        'MACE-POLAR',
        'MACE-OMOL',
        'UMA-OMOL'
    ]

    fig, ax = plt.subplots(1, 3, figsize=(6.4, 4.0), sharey=True)

    ax[0].set_ylabel("Relative Energy (eV)")
    ax[0].set_ylim(-5.0, 4.0)

    ax[0].set_xlabel("C -- Cl Distance (Å)")
    ax[0].set_xlim(1.0, 6.0)
    ax[1].set_xlabel("C -- Cl Distance (Å)")
    ax[1].set_xlim(1.0, 6.0)
    ax[2].set_xlabel("C -- Cl Distance (Å)")
    ax[2].set_xlim(1.0, 6.0)

    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[2].grid(True, linestyle='--', alpha=0.7)

    for col in plot_order:
        if col in ch3cl_rel.columns:
            if col != 'R':
                ps = plotting.get(col)
                ax[0].plot(ch3cl_rel['R'], ch3cl_rel[col],
                            label=col,
                            linewidth=ps['width'], 
                            marker=ps['marker'], 
                            markersize=ps['size'], 
                            markevery=ps['freq'], 
                            color=ps['color'], 
                            linestyle=ps['line'])

    for col in plot_order:
        if col in ch3cl_h2o_rel.columns:
            if col != 'R':
                ps = plotting.get(col)
                ax[1].plot(ch3cl_h2o_rel['R'], ch3cl_h2o_rel[col],
                            label=col,
                            linewidth=ps['width'], 
                            marker=ps['marker'], 
                            markersize=ps['size'], 
                            markevery=ps['freq'], 
                            color=ps['color'], 
                            linestyle=ps['line'])

    for col in plot_order:
        if col in ch3cl_h3o_rel.columns:
            if col != 'R':
                ps = plotting.get(col)
                ax[2].plot(ch3cl_h3o_rel['R'], ch3cl_h3o_rel[col],
                            label=col,
                            linewidth=ps['width'], 
                            marker=ps['marker'], 
                            markersize=ps['size'], 
                            markevery=ps['freq'], 
                            color=ps['color'], 
                            linestyle=ps['line'])

    handles, labels = ax[0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    fig.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.15), # Place above the figure
               ncol=4,
               frameon=True) 

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    plt.savefig('sn1_3panel.png', dpi=900)
    plt.close()




if __name__ == '__main__':
    main()
