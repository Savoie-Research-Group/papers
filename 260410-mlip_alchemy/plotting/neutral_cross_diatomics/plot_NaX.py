import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
hartree2ev = 27.2113834

plotting = {
    'uDFT': {'width': 3.5, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#648FFF', 'line': '-'},
    'rDFT': {'width': 2.2, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#FE6100', 'line': '--'},
    'AIMNET2': {'width': 1.5, 'marker': 'x', 'freq': None, 'size': 5.5, 'color': '#785EF0', 'line': ':'},
    'AIMNET2-NSE': {'width': 1.5, 'marker': 'v', 'freq': None, 'size': 3.5, 'color': "#39A24E", 'line': ':'},
    'UMA-OMOL': {'width': 1.5, 'marker': 's', 'freq': None, 'size': 3.0, 'color': '#DC267F', 'line': ':'},
    'MACE-OMOL': {'width': 1.5, 'marker': 'o', 'freq': None, 'size': 2.5, 'color': '#FFB000', 'line': ':'},
    'ORB-OMOL': {'width': 1.5, 'marker': '<', 'freq': None, 'size': 2.5, 'color': '#7B5C73', 'line': ':'}
}

def setup_fonts():
    """
    Configures matplotlib global font settings.
    """
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'lines.linewidth': 2
    })

def get_formatted_data(label, raw_data, reference_data):
    """
    Helper function to normalize data format.
    Calculates E_rel = E_total - (E_atom1 + E_atom2)
    """
    data = pd.DataFrame()
    
    if 'DFT' in label:
        data['R'] = raw_data['bond_distance']
        # Reference data is already sum of component atoms (H + X)
        data['E_rel'] = raw_data.iloc[:, 1] * hartree2ev - reference_data['DFT']
    elif 'OMOL' in label:
        data['R'] = raw_data['Distance_angs']
        data['E_rel'] = raw_data.iloc[:, 1] - reference_data['OMOL']
    elif 'AIMNET2' in label:
        data['R'] = raw_data['Distance_angs']
        data['E_rel'] = raw_data.iloc[:, 1] - reference_data['AIMNET2']
    
    return data

def plot_on_axis(ax, data_dict, ref_data, subplot_label=None):
    """
    Plots a dictionary of datasets onto a specific axis object.
    """
    for label, raw_data in data_dict.items():
        try:
            df = get_formatted_data(label, raw_data, ref_data)
            ps = plotting.get(label, {'width': 1, 'marker': '', 'size': 1, 'color': 'k', 'line': '-'})
            
            ax.plot(df['R'], df['E_rel'], 
                    label=label, 
                    linewidth=ps['width'], 
                    marker=ps['marker'], 
                    markersize=ps['size'], 
                    markevery=ps['freq'], 
                    color=ps['color'], 
                    linestyle=ps['line'])
        except Exception as e:
            # Silent fail for cleaner output, or print e for debugging
            pass

    # Add text label (e.g. HF)
    if subplot_label:
        ax.text(0.95, 0.95, subplot_label, transform=ax.transAxes, 
                fontsize=16, fontweight='bold', verticalalignment='top', horizontalalignment='right')
    
    ax.grid(True, linestyle='--', alpha=0.7)

def get_single_atom_energy(atom_symbol, ref_df, method_key):
    """
    Fetches the raw Hartree energy for a single atom from the reference CSV.
    method_key: 'mv' for wB97M-V (DFT/OMOL) or 'd3bj' for wB97M-D3BJ (AIMNET2)
    """
    subset = ref_df[ref_df['system'].str.contains(f"^{atom_symbol}_", regex=True)]
    
    if method_key == 'mv':
        row = subset[subset['system'].str.contains('wB97M-V-def2-TZVPD')]
    elif method_key == 'd3bj':
        row = subset[subset['system'].str.contains('wB97M-D3BJ-def2-TZVPP')]
    
    if not row.empty:
        return row['energy_hartree'].item()
    return 0.0

def make_hx_grid_plot(all_systems_data, plot_name):
    """
    Creates the 2x2 Figure for HF, HCl, HBr, HI.
    """
    
    fig, ax = plt.subplots(2, 2, figsize=(6.4, 5), sharex=True)
    
    # Mapping molecules to grid positions
    grid_map = {
        'NaF':  (0, 0),
        'NaCl': (0, 1),
        'NaBr': (1, 0),
        'NaI':  (1, 1)
    }

    print("Generating subplots...")
    
    for molecule, (row, col) in grid_map.items():
        if molecule in all_systems_data:
            print(f"  - Plotting {molecule} at [{row}, {col}]")
            bundle = all_systems_data[molecule]
            plot_on_axis(ax[row, col], bundle['data'], bundle['ref'], subplot_label=molecule)
            
            # Optional: Dynamic Limiting based on molecule size if needed
            # ax[row, col].set_xlim(0.5, 5.0) 
            ax[row, col].set_ylim(-7.0, 30.0) # Set reasonable energy limits
        else:
            ax[row, col].text(0.5, 0.5, 'Data Missing', ha='center')

    # --- Axes formatting ---
    
    # Y Labels (Left column)
    ax[0, 0].set_ylabel("Relative Energy (eV)")
    ax[1, 0].set_ylabel("Relative Energy (eV)")

    # X Labels (Bottom row)
    ax[1, 0].set_xlabel("Internuclear Distance (Å)")
    ax[1, 1].set_xlabel("Internuclear Distance (Å)")

    ax[1, 0].set_xlim(0.0, 8.0)

    # --- Global Legend ---
    handles, labels = ax[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    fig.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', 
               bbox_to_anchor=(0.55, 0.995), 
               ncol=3,
               frameon=True,
               markerscale=2.0) 

    plt.tight_layout()
    plt.subplots_adjust(top=0.88) 
    
    print(f"Saving {plot_name}...")
    plt.savefig(plot_name, dpi=900)
    plt.close()

def main():
    setup_fonts()

    # Load Data
    try:
        dft_scan_data = pd.read_csv('../../data/diatomics/dft/dft_scan_energies.csv')
        mlip_scan_data = pd.read_csv('../../data/diatomics/mlip/data_all.csv')
        atomic_reference_data = pd.read_csv('../../data/diatomics/dft/atom_reference_spe.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    targets = ['NaF', 'NaCl', 'NaBr', 'NaI'] 
    all_systems_data = {}

    # Pre-fetch Hydrogen Energy (constant for all)
    na_mv = get_single_atom_energy('Na', atomic_reference_data, 'mv')
    na_d3bj = get_single_atom_energy('Na', atomic_reference_data, 'd3bj')

    for sys_name in targets:
        print(f"Processing data for {sys_name}...")
        
        # Identify Halogen part (e.g. 'F' from 'HF')
        halogen = sys_name.replace('Na', '') 
        
        # Get Halogen Energies
        x_mv = get_single_atom_energy(halogen, atomic_reference_data, 'mv')
        x_d3bj = get_single_atom_energy(halogen, atomic_reference_data, 'd3bj')

        # Calculate Heteronuclear Reference (E_H + E_X)
        dissoc_ref = {
            'DFT': (na_mv + x_mv) * hartree2ev,
            'OMOL': (na_mv + x_mv) * hartree2ev,
            'AIMNET2': (na_d3bj + x_d3bj) * hartree2ev
        }

        # Filter for Neutral Charge (using _0_ for DFT and charge0 for MLIPs)
        # Note: Regex allows for strict matching of system name (e.g., HF) to avoid matching "HF2-" if that existed
        dft_pattern = 'Na_' + halogen
        mlip_pattern = "".join(sorted(['Na', halogen]))

        neutral_data = {
            'uDFT': dft_scan_data.filter(regex=f'bond_distance|({dft_pattern}.*_0_.*UKS)'),
            'rDFT': dft_scan_data.filter(regex=f'bond_distance|({dft_pattern}.*_0_.*RKS)'),
            'UMA-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({mlip_pattern}.*_charge0_.*fairchem-omol)'),
            'MACE-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({mlip_pattern}.*_charge0_.*mace-omol)'),
            'ORB-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({mlip_pattern}.*_charge0_.*orb-omol)')
        }

        # Clean empty columns
        neutral_data = {k: v for k, v in neutral_data.items() if v.shape[1] > 1}

        if neutral_data:
            all_systems_data[sys_name] = {
                'data': neutral_data,
                'ref': dissoc_ref
            }
        else:
            print(f"Warning: No valid data found for {sys_name}")

    # Generate Plot
    if len(all_systems_data) > 0:
        make_hx_grid_plot(all_systems_data, "NaX_Neutral_Grid.png")
    else:
        print("No systems processed successfully.")

if __name__ == '__main__':
    main()