import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
hartree2ev = 27.2113834

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
        'axes.labelsize': 12,         # X and Y labels
        'xtick.labelsize': 10,        # Tick numbers
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'font.family': 'sans-serif',  # Options: 'serif', 'sans-serif', 'monospace'
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Tries to find these fonts in order
        'lines.linewidth': 2          # Default line width
    })

def get_formatted_data(label, raw_data, reference_data):
    """
    Helper function to normalize data format (converting Hartree to eV, etc)
    Returns: DataFrame with 'R' and 'E_rel' columns
    """
    data = pd.DataFrame()
    
    # Handle real data scenario
    if 'DFT' in label:
        data['R'] = raw_data['bond_distance']
        # Note: Ensure referencing the correct column index or name
        data['E_rel'] = raw_data.iloc[:, 1] * hartree2ev - 2 * reference_data['DFT']
    elif 'OMOL' in label:
        data['R'] = raw_data['Distance_angs']
        data['E_rel'] = raw_data.iloc[:, 1] - 2 * reference_data['OMOL']
    elif 'POLAR' in label:
        data['R'] = raw_data['Distance_angs']
        data['E_rel'] = raw_data.iloc[:, 1] - 2 * reference_data['OMOL']
    elif 'AIMNET2' in label:
        data['R'] = raw_data['Distance_angs']
        data['E_rel'] = raw_data.iloc[:, 1] - 2 * reference_data['AIMNET2']
    
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
            print(f"Skipping {label} due to missing data or error: {e}")

    # Add text label inside plot (e.g., "K2 Neutral") if desired
    if subplot_label:
        ax.text(0.05, 0.9, subplot_label, transform=ax.transAxes, 
                fontsize=14, fontweight='bold', verticalalignment='top')
    
    ax.grid(True, linestyle='--', alpha=0.7)

def make_combined_plot(k_data, cl_data, plot_name):
    """
    Creates the 2x2 Figure.
    k_data: dict containing {'neutral': {}, 'charged': {}, 'ref': {}}
    cl_data: dict containing {'neutral': {}, 'charged': {}, 'ref': {}}
    """
    
    # Create 2x2 grid, sharing axes to clean up internal labels
    fig, ax = plt.subplots(2, 2, figsize=(7, 5), sharex=True, sharey='row')
    
    # --- Top Row: K2 ---
    print("Plotting K2 (Top Row)...")
    plot_on_axis(ax[0, 0], k_data['neutral'], k_data['ref'])
    plot_on_axis(ax[0, 1], k_data['charged'], k_data['ref'])

    # --- Bottom Row: Cl2 ---
    print("Plotting Cl2 (Bottom Row)...")
    plot_on_axis(ax[1, 0], cl_data['neutral'], cl_data['ref'])
    plot_on_axis(ax[1, 1], cl_data['charged'], cl_data['ref'])

    # --- Axes formatting ---
    
    # Set limits (adjust these as necessary for your specific data range)
    ax[0, 0].set_ylim(-5.0, 20.0)
    ax[1, 0].set_ylim(-8.0, 10.0)
    ax[0, 0].set_xlim(0.5, 6.5)

    # Shared Labels
    # Set Y-label only on the left column (row 0 and 1, col 0)
    ax[0, 0].set_ylabel("Relative Energy (eV)")
    ax[1, 0].set_ylabel("Relative Energy (eV)")

    # Set X-label only on the bottom row (row 1, col 0 and 1)
    ax[1, 0].set_xlabel("Internuclear Distance (Å)")
    ax[1, 1].set_xlabel("Internuclear Distance (Å)")

    # --- Global Legend ---
    # Extract handles/labels from one subplot (assuming all have same models)
    handles, labels = ax[1, 0].get_legend_handles_labels()
    
    # Filter duplicates just in case
    unique_labels = dict(zip(labels, handles))
    
    fig.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center', 
               bbox_to_anchor=(0.54, 1.0), # Place above the figure
               ncol=4,
               frameon=True,
               markerscale=2.0) 

    plt.tight_layout()
    # Adjust top to make room for legend (rect=[left, bottom, right, top])
    plt.subplots_adjust(top=0.88) 
    
    print(f"Saving {plot_name}...")
    plt.savefig(plot_name, dpi=600)
    plt.close()


def main():
    setup_fonts()

    dft_scan_data = pd.read_csv('../../data/diatomics/dft/dft_scan_energies.csv')
    most_mlip_data = pd.read_csv('../../data/diatomics/mlip/data_all.csv')
    aimnet2nse_data = pd.read_csv('../../data/diatomics/mlip/combined_energy_aimnet2nse.csv')
    macepol_data = pd.read_csv('../../data/diatomics/mlip/data_macepol.csv')
    mlip_scan_data = pd.merge(most_mlip_data, aimnet2nse_data, on="Distance_angs", how="inner")
    mlip_scan_data = pd.merge(mlip_scan_data, macepol_data, on="Distance_angs", how="inner")
    atomic_reference_data = pd.read_csv('../../data/diatomics/dft/atom_reference_spe.csv')

    targets = ['K', 'Cl'] # Defined targets for top and bottom rows
    data_bundles = {}

    for atm in targets:
        print(f"Processing data for {atm}...")
        
        # Get Reference Energies
        atm_df = atomic_reference_data[atomic_reference_data['system'].str.contains(atm)]
        # Add error checking if dataframe is empty
        if atm_df.empty:
            print(f"Warning: No reference data found for {atm}")
            continue

        mv_df = atm_df[atm_df['system'].str.contains('wB97M-V-def2-TZVPD')]
        d3bj_df = atm_df[atm_df['system'].str.contains('wB97M-D3BJ-def2-TZVPP')]
        
        dissoc_ref = {
            'DFT': mv_df['energy_hartree'].item() * hartree2ev,
            'OMOL': mv_df['energy_hartree'].item() * hartree2ev,
            'AIMNET2': d3bj_df['energy_hartree'].item() * hartree2ev
        }

        # Regex filters for Neutral and Charged
        if atm == 'Cl':
            charged = {
                'uDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_-2_.*UKS)'),
                'rDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_-2_.*RKS)'),
                'AIMNET2-NSE': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*aim2nse)'),
                'AIMNET2': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*aimnet2)'),
                'MACE-POLAR': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*mace-omol-polar)'),
                'MACE-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*mace-omol)'),
                'UMA-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*fairchem-omol)'),
                'ORB-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*orb-omol)')
            }
        elif atm == 'K':
            charged = {
                'uDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_2_.*UKS)'),
                'rDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_2_.*RKS)'),
                'AIMNET2-NSE': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge2_.*aim2nse)'),
                'AIMNET2': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge2_.*aimnet2)'),
                'MACE-POLAR': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge2_.*mace-omol-polar)'),
                'MACE-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge2_.*mace-omol)'),
                'UMA-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge2_.*fairchem-omol)'),
                'ORB-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge2_.*orb-omol)')
            }

        neutral = {
            'uDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_0_.*UKS)'),
            'rDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_0_.*RKS)'),
            'AIMNET2-NSE': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*aim2nse)'),
            'AIMNET2': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*aimnet2)'),
            'MACE-POLAR': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*mace-omol-polar)'),
            'MACE-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*mace-omol)'),
            'UMA-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*fairchem-omol)'),
            'ORB-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*orb-omol)')
        }

        # Clean up empty dataframes from the dict (if columns were not found)
        neutral = {k: v for k, v in neutral.items() if v.shape[1] > 1}
        charged = {k: v for k, v in charged.items() if v.shape[1] > 1}

        data_bundles[atm] = {
            'neutral': neutral,
            'charged': charged,
            'ref': dissoc_ref
        }

    # Verify we have both atoms before plotting
    if 'K' in data_bundles and 'Cl' in data_bundles:
        make_combined_plot(data_bundles['K'], data_bundles['Cl'], "K2_Cl2_combined_figure.png")
    else:
        print("Error: Could not find data for both K and Cl to generate the combined plot.")

if __name__ == '__main__':
    main()