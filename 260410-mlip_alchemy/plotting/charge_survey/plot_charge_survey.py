import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

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

def process_and_plot_bond_data(file_path):
    # --- Constants ---
    hartree2ev = 27.2113834
    
    # Updated Reference energies (converted to eV)
    # Ref for non-aimnet2 data
    wb97mv_ref = (-460.126133284408 + -39.817860984076) * hartree2ev
    
    # Ref for aimnet2 data
    wb97m_d3bj_ref = (-460.198150092473 + -39.858431131849) * hartree2ev

    # --- Load Data ---
    df = pd.read_csv(file_path)

    # --- PART A: Data Processing ---
    
    # Identify columns
    all_cols = df.columns
    uks_cols = [c for c in all_cols if 'uks' in c]
    aimnet2_cols = [c for c in all_cols if 'aimnet2' in c]
    # non-aimnet2 includes uks, fairchem, mace, omol
    non_aimnet2_cols = [c for c in all_cols if 'aimnet2' not in c and c != 'r']

    # 1. Convert uks to eV
    for col in uks_cols:
        df[col] = df[col] * hartree2ev

    # 2. Subtract wb97mv_ref for non-aimnet2
    for col in non_aimnet2_cols:
        df[col] = df[col] - wb97mv_ref

    # 3. Subtract wb97m_d3bj_ref for aimnet2
    for col in aimnet2_cols:
        df[col] = df[col] - wb97m_d3bj_ref
    
    # Save processed data (optional)
    df.to_csv('processed_E_data.csv', index=False)

    # --- PART B: Data Plotting ---
    
    charges = {
        'charge1': 'q = +1',
        'charge-1': 'q = -1',
        'charge2': 'q = +2',
        'charge-2': 'q = -2',
        'charge0': 'q = 0'
    }

    # Style Configurations
    # Colors for each charge state
    charge_colors = {
        'q = 0': '#7B5C73',
        'q = +1': '#648FFF',
        'q = -1': '#DC267F',
        'q = +2': "#39A24E",
        'q = -2': '#FFB000' 
    }

    # Markers and line styles for each method
    # markevery=5 prevents the markers from overcrowding the line
    method_styles = {
        'UKS':      {'marker': '', 'linestyle': '-',  'markevery': 5},
        'AIMNET2':  {'marker': 'x', 'linestyle': ':', 'markevery': 5},
        'UMA': {'marker': 's', 'markersize': 4.0, 'linestyle': ':',  'markevery': 5},
        'MACE':     {'marker': 'o', 'markersize': 4.0, 'linestyle': ':', 'markevery': 5}
        # 'MACE':     {'marker': 'o', 'linestyle': '', 'markevery': 5},
        # 'ORB':     {'marker': '<', 'linestyle': '',  'markevery': 5}
    }

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 4), sharey=True)

    # --- Plot 1: UKS vs AIMNet2 ---
    ax1 = axes[0]
    for charge in charges.keys():
        label = charges[charge]
        color = charge_colors.get(label, 'gray')
        
        # Plot UKS
        uks_matches = [c for c in uks_cols if charge in c]
        if uks_matches:
            col_name = uks_matches[0]
            label = f'DFT {label}'
            style = method_styles['UKS']
            ax1.plot(df['r'], df[col_name], label=label, color=color, **style)

        # Plot AIMNet2
        aimnet_matches = [c for c in aimnet2_cols if charge in c]
        if aimnet_matches:
            col_name = aimnet_matches[0]
            label = f'AIMNET2 {label}'
            style = method_styles['AIMNET2']
            ax1.plot(df['r'], df[col_name], label=label, color=color, **style)

    ax1.set_xlabel('C -- Cl Distance (Å)')
    ax1.set_ylabel('Relative Energy (eV)')
    ax1.set_ylim(-10.0, 40.0)
    ax1.set_xlim(0.4, 7.0)

    # ax1.text(0.1, 0.99, f'A', transform=ax1.transAxes,
    #             fontsize=16, fontweight='bold', va='top', ha='right')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 2: UKS vs UMA ---
    ax2 = axes[1]
    for charge in charges.keys():
        label = charges[charge]
        color = charge_colors.get(label, 'gray')

        # Plot UKS
        uks_matches = [c for c in uks_cols if charge in c]
        if uks_matches:
            col_name = uks_matches[0]
            label = f'DFT {label}'
            style = method_styles['UKS']
            ax2.plot(df['r'], df[col_name], label=label, color=color, **style)

        # Fairchem (looks for 'fairchem')
        fc_matches = [c for c in df.columns if 'fairchem' in c and charge in c]
        if fc_matches:
            style = method_styles['UMA']
            ax2.plot(df['r'], df[fc_matches[0]], label=f'UMA {label}', color=color, **style)

    ax2.set_xlabel('C -- Cl Distance (Å)')
    ax2.set_xlim(0.4, 7.0)

    # ax2.text(0.1, 0.99, f'B', transform=ax2.transAxes,
    #             fontsize=16, fontweight='bold', va='top', ha='right')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # --- Plot 3: UKS vs MACE ---
    ax3 = axes[2]
    for charge in charges.keys():
        label = charges[charge]
        color = charge_colors.get(label, 'gray')

        # Plot UKS
        uks_matches = [c for c in uks_cols if charge in c]
        if uks_matches:
            col_name = uks_matches[0]
            label = f'DFT {label}'
            style = method_styles['UKS']
            ax3.plot(df['r'], df[col_name], label=label, color=color, **style)

        # Mace (looks for 'mace')
        mace_matches = [c for c in df.columns if 'mace' in c and charge in c]
        if mace_matches:
            style = method_styles['MACE']
            ax3.plot(df['r'], df[mace_matches[0]], label=f'MACE {label}', color=color, **style)

    ax3.set_xlabel('C -- Cl Distance (Å)')
    ax3.set_xlim(0.4, 7.0)

    # ax3.text(0.1, 0.99, f'C', transform=ax3.transAxes,
    #             fontsize=16, fontweight='bold', va='top', ha='right')
    ax3.grid(True, linestyle='--', alpha=0.7)

    # --- Create Custom Legends ---
    
    # Legend 1: Methods (Black color, varies by marker/linestyle)
    method_handles = []
    for method, style in method_styles.items():
        if method == "UKS":
            h = Line2D([0], [0], color='black', label="uDFT", **style)
            method_handles.append(h)
        else:
            h = Line2D([0], [0], color='black', label=method, **style)
            method_handles.append(h)
    
    # Legend 2: Charges (Solid line, varies by color)
    charge_handles = []
    for charge, color in charge_colors.items():
        h = Line2D([0], [0], color=color, label=charge, linestyle='-')
        charge_handles.append(h)

    # Place legends at the bottom
    # Adjust layout first to create space at the bottom
    plt.subplots_adjust(bottom=0.2)
    
    # Add Method Legend
    fig.legend(handles=method_handles, loc='lower center', bbox_to_anchor=(0.5, 0.08), 
               ncol=len(method_handles), frameon=False)
    
    # Add Charge Legend below Method Legend
    fig.legend(handles=charge_handles, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
               ncol=len(charge_handles), frameon=False)


    plt.tight_layout()
    plt.subplots_adjust(bottom=0.30)
    plt.savefig('charge_survey.png', dpi=900)
    plt.close()

# Run the function
if __name__ == "__main__":
    setup_fonts()
    process_and_plot_bond_data('../../data/sn1/ch3cl/charge_survey/combined_E_data.csv')