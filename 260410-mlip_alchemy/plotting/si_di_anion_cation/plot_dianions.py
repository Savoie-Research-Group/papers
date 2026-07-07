import pandas as pd
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
hartree2ev = 27.2113834

plotting = {
    'uDFT': {'width': 3.5, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#648FFF', 'line': '-'},
    'rDFT': {'width': 2.2, 'marker': '', 'freq': None, 'size': 1.0, 'color': '#FE6100', 'line': '--'},
    'AIMNET2': {'width': 1.5, 'marker': 'x', 'freq': 2, 'size': 5.0, 'color': '#785EF0', 'line': ':'},
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
        'axes.titlesize': 12,         # Title of subplots
        'axes.labelsize': 10,         # X and Y labels
        'xtick.labelsize': 9,        # Tick numbers
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        # 'font.family': 'sans-serif',  # Options: 'serif', 'sans-serif', 'monospace'
        # 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], # Tries to find these fonts in order
        'lines.linewidth': 2          # Default line width
    })

def plot_dataset(ax, data_dict, reference_data, subplot_label=None):
    """
    Helper to normalize and plot data on a given axis.
    """
    for label, raw_data in data_dict.items():
        data = pd.DataFrame()

        # Normalize Data
        if 'DFT' in label:
            data['R'] = raw_data['bond_distance']
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

        # Plot
        ps = plotting.get(label, {'width': 1, 'color': 'k'}) # Fallback styling
        ax.plot(data['R'], data['E_rel'], 
                label=label, 
                linewidth=ps['width'], 
                marker=ps['marker'], 
                markersize=ps['size'], 
                markevery=ps['freq'], 
                color=ps['color'], 
                linestyle=ps['line'])
    
    ax.grid(True, linestyle='--', alpha=0.5)
    if subplot_label:
        ax.text(0.95, 0.95, subplot_label, transform=ax.transAxes, 
                fontsize=12, fontweight='bold', verticalalignment='top', horizontalalignment='right')

def main():

    setup_fonts()
    
    # Load Data
    try:
        dft_scan_data = pd.read_csv('../../data/diatomics/dft/dft_scan_energies.csv')
        most_mlip_data = pd.read_csv('../../data/diatomics/mlip/data_all.csv')
        aimnet2nse_data = pd.read_csv('../../data/diatomics/mlip/combined_energy_aimnet2nse.csv')
        macepol_data = pd.read_csv('../../data/diatomics/mlip/data_macepol.csv')
        mlip_scan_data = pd.merge(most_mlip_data, aimnet2nse_data, on="Distance_angs", how="inner")
        mlip_scan_data = pd.merge(mlip_scan_data, macepol_data, on="Distance_angs", how="inner")
        atomic_reference_data = pd.read_csv('../../data/diatomics/dft/atom_reference_spe.csv')
    except Exception as e:
        print(f"Error loading data files: {e}")
        return

    atoms = ['F', 'Cl', 'Br', 'I']
    
    # Setup Figure: 4 Rows (Atoms), 2 Columns (Neutral/Charged)
    # sharex=True aligns all X axes. sharey='row' ensures Neutral/Charged F2 share the same energy scale.
    fig, axes = plt.subplots(4, 2, sharex=True, sharey='row', figsize=(6.4, 8.5))
    
    print("Generating 4x2 Grid...")

    for idx, atm in enumerate(atoms):
        print(f" - Processing Row {idx}: {atm}2")

        # 1. Get Reference Energy for this atom
        atm_df = atomic_reference_data[atomic_reference_data['system'].str.contains(atm)]
        mv_df = atm_df[atm_df['system'].str.contains('wB97M-V-def2-TZVPD')]
        d3bj_df = atm_df[atm_df['system'].str.contains('wB97M-D3BJ-def2-TZVPP')]
        
        dissoc_ref = {
            'DFT': mv_df['energy_hartree'].item() * hartree2ev,
            'OMOL': mv_df['energy_hartree'].item() * hartree2ev,
            'AIMNET2': d3bj_df['energy_hartree'].item() * hartree2ev
        }

        # 2. Filter Data
        charged_data = {
            'uDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_-2_.*UKS)'),
            'rDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_-2_.*RKS)'),
            'AIMNET2-NSE': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*aim2nse)'),
            'AIMNET2': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*aimnet2)'),
            'MACE-POLAR': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*mace-omol-polar)'),
            'MACE-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*mace-omol)'),
            'UMA-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*fairchem-omol)'),
            'ORB-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge-2_.*orb-omol)')
        }

        neutral_data = {
            'uDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_0_.*UKS)'),
            'rDFT': dft_scan_data.filter(regex=f'bond_distance|({atm}2.*_0_.*RKS)'),
            'AIMNET2-NSE': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*aim2nse)'),
            'AIMNET2': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*aimnet2)'),
            'MACE-POLAR': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*mace-omol-polar)'),
            'MACE-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*mace-omol)'),
            'UMA-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*fairchem-omol)'),
            'ORB-OMOL': mlip_scan_data.filter(regex=f'Distance_angs|({atm}2.*_charge0_.*orb-omol)')
        }
        
        # Clean empty columns
        charged_data = {k: v for k, v in charged_data.items() if v.shape[1] > 1}
        neutral_data = {k: v for k, v in neutral_data.items() if v.shape[1] > 1}

        # 3. Plotting
        # Left Column (Neutral) -> axes[idx, 0]
        plot_dataset(axes[idx, 0], neutral_data, dissoc_ref, subplot_label=f"{atm}2")
        
        # Right Column (Charged) -> axes[idx, 1]
        plot_dataset(axes[idx, 1], charged_data, dissoc_ref, subplot_label=f"{atm}2")

        # 4. Row Styling
        # Add Molecule Label to the left of the left-most plot
        axes[idx, 0].set_ylabel("Relative Energy (eV)")
        
        # Set Row Limits (Optional: Adjust per atom if needed, or keep uniform)
        axes[idx, 0].set_ylim(-8.0, 10.0) 
        axes[idx, 0].set_xlim(0.0, 8.0)

    # --- Final Grid Formatting ---

    # Set Column Titles (only on top row)
    axes[0, 0].set_title("Neutral", fontsize=14, fontweight='bold')
    axes[0, 1].set_title("Charged [-2]", fontsize=14, fontweight='bold')

    # Set X Labels (only on bottom row)
    axes[3, 0].set_xlabel("Internuclear Distance (Å)")
    axes[3, 1].set_xlabel("Internuclear Distance (Å)")

    # Global Legend
    # Extract handles from the very first plot (assuming all plots share same models)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    fig.legend(unique_labels.values(), unique_labels.keys(),
               loc='upper center',
               bbox_to_anchor=(0.54, 0.995), # Place just below the top edge
               ncol=4,
               frameon=True,
               markerscale=2.0)

    plt.tight_layout()
    # Adjust layout to make room for legend at top
    plt.subplots_adjust(top=0.90)
    
    print("Saving combo_dianion_grid.png...")
    plt.savefig("combo_dianion_grid.png", dpi=900)
    print("Done.")

if __name__ == '__main__':
    main()