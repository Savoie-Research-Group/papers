import pandas as pd
import matplotlib.pyplot as plt

hartree2ev = 27.2113834

plotting = {
    'uDFT': {'width': 3.5, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#648FFF', 'line': '-'},
    'rDFT': {'width': 2.2, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#FE6100', 'line': '--'},
    'CASPT2': {'width': 1.5, 'marker': 's', 'freq': 2, 'size': 2.0, 'color': '#FFB000', 'line': ':'},
}

def setup_fonts():
    """
    Configures matplotlib global font settings.
    Adjust sizes and family here.
    """
    plt.rcParams.update({
        'axes.labelsize': 11,         # X and Y labels
        'xtick.labelsize': 9,        # Tick numbers
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',  # Options: 'serif', 'sans-serif', 'monospace'
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Tries to find these fonts in order
        'lines.linewidth': 2          # Default line width
    })

def load_cas_data(filepath):
    """
    Loads dissociation data from a text file, converts energy to eV if needed,
    and returns the data as a Pandas DataFrame.
    """
    # Read the first line to check the header for units
    with open(filepath, 'r') as f:
        header = f.readline()

    # Read the data using pandas, skipping the header row.
    # We use a regular expression for the separator to handle any amount of whitespace.
    df = pd.read_csv(filepath, sep=r"\s+", skiprows=1, names=['R', 'E'])

    # Check if conversion from Hartrees to eV is needed
    if 'hartree' in header.lower():
        df['E'] = df['E'] * hartree2ev

    return df


def main():
    setup_fonts()

    # Get CASPT2 data
    cas_f2_data = load_cas_data('../../data/diatomics/caspt2/f2_2e2o_tzvpd.txt')
    f2_cas_ref = -199.23449429 * hartree2ev # from 12.0 AA distance
    cas_f2_data['E_rel'] = cas_f2_data['E'] - f2_cas_ref

    # Get DFT reference data
    dft_reference_data = pd.read_csv('../../data/diatomics/dft/atom_reference_spe.csv')

    f_atm_df = dft_reference_data[dft_reference_data['system'].str.contains('F_0_2_wB97M-V-def2-TZVPD')]
    na_atm_df = dft_reference_data[dft_reference_data['system'].str.contains('Na_0_2_wB97M-V-def2-TZVPD')]
    cl_atm_df = dft_reference_data[dft_reference_data['system'].str.contains('Cl_0_2_wB97M-V-def2-TZVPD')]

    dissoc_dft_ref = {
        'F2': f_atm_df['energy_hartree'].item() * hartree2ev * 2.0,
        'NaCl': na_atm_df['energy_hartree'].item() * hartree2ev + cl_atm_df['energy_hartree'].item() * hartree2ev
    }

    print(dissoc_dft_ref)

    # Get DFT scan data
    dft_scan_data = pd.read_csv('../../data/diatomics/dft/dft_scan_energies.csv')

    f2_udft_scan = dft_scan_data.filter(regex='bond_distance|(F2.*_0_.*UKS)')
    f2_udft_scan['E_rel'] = f2_udft_scan.iloc[:, 1] * hartree2ev - dissoc_dft_ref['F2']
    f2_rdft_scan = dft_scan_data.filter(regex='bond_distance|(F2.*_0_.*RKS)')
    f2_rdft_scan['E_rel'] = f2_rdft_scan.iloc[:, 1] * hartree2ev - dissoc_dft_ref['F2']

    nacl_udft_scan = dft_scan_data.filter(regex='bond_distance|(Na_Cl.*_0_.*UKS)')
    nacl_udft_scan['E_rel'] = nacl_udft_scan.iloc[:, 1] * hartree2ev - dissoc_dft_ref['NaCl']
    nacl_rdft_scan = dft_scan_data.filter(regex='bond_distance|(Na_Cl.*_0_.*RKS)')
    nacl_rdft_scan['E_rel'] = nacl_rdft_scan.iloc[:, 1] * hartree2ev - dissoc_dft_ref['NaCl']

    # Let's get to plotting!
    fig, ax = plt.subplots(1, 2, figsize=(4.5, 3), sharey=True)

    ax[0].set_ylabel("Relative Energy (eV)")
    ax[0].set_xlabel("Internuclear Distance (Å)")
    ax[1].set_xlabel("Internuclear Distance (Å)")

    ax[0].set_ylim(-5.0, 7.0)
    ax[0].set_xlim(0.5, 6.0)
    ax[1].set_xlim(1.0, 6.0)

    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[1].grid(True, linestyle='--', alpha=0.7)

    # First, F2 data on left plot
    label = 'uDFT'
    ps = plotting.get(label)
    ax[0].plot(
        f2_udft_scan['bond_distance'], f2_udft_scan['E_rel'],
        label=label, 
        linewidth=ps['width'], 
        marker=ps['marker'], 
        markersize=ps['size'], 
        markevery=ps['freq'], 
        color=ps['color'], 
        linestyle=ps['line']
    )

    label = 'rDFT'
    ps = plotting.get(label)
    ax[0].plot(
        f2_rdft_scan['bond_distance'], f2_rdft_scan['E_rel'],
        label=label, 
        linewidth=ps['width'], 
        marker=ps['marker'], 
        markersize=ps['size'], 
        markevery=ps['freq'], 
        color=ps['color'], 
        linestyle=ps['line']
    )

    label = 'CASPT2'
    ps = plotting.get(label)
    ax[0].plot(
        cas_f2_data['R'], cas_f2_data['E_rel'],
        label=label, 
        linewidth=ps['width'], 
        marker=ps['marker'], 
        markersize=ps['size'], 
        markevery=ps['freq'], 
        color=ps['color'], 
        linestyle=ps['line']
    )

    # Then NaCl on right plot
    label = 'uDFT'
    ps = plotting.get(label)
    ax[1].plot(
        nacl_udft_scan['bond_distance'], nacl_udft_scan['E_rel'],
        label=label, 
        linewidth=ps['width'], 
        marker=ps['marker'], 
        markersize=ps['size'], 
        markevery=ps['freq'], 
        color=ps['color'], 
        linestyle=ps['line']
    )

    label = 'rDFT'
    ps = plotting.get(label)
    ax[1].plot(
        nacl_rdft_scan['bond_distance'], nacl_rdft_scan['E_rel'],
        label=label, 
        linewidth=ps['width'], 
        marker=ps['marker'], 
        markersize=ps['size'], 
        markevery=ps['freq'], 
        color=ps['color'], 
        linestyle=ps['line']
    )


    # --- Global Legend ---
    # Extract handles/labels from one subplot (assuming all have same models)
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Filter duplicates just in case
    unique_labels = dict(zip(labels, handles))
    
    fig.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', 
               bbox_to_anchor=(0.55, 1.0), # Place above the figure
               ncol=len(unique_labels),
               frameon=True) 

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig('f2_v_nacl.png', dpi=900)


if __name__ == '__main__':
    main()
