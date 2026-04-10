import pandas as pd
import matplotlib.pyplot as plt
import re
import os

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

def plot_multiple_molecular_charges(file_list):
    num_files = len(file_list)
    # Create a figure with subplots (1 column, N rows)
    fig, axes = plt.subplots(1, num_files, figsize=(6.5, 4), sharey=True)
    
    # Ensure axes is an array even if there is only 1 file
    if num_files == 1:
        axes = [axes]

    panel_labels = ['A', 'B', 'C']

    # Standard colors for consistency
    color_ch3 = '#648FFF'      # Blue
    color_cl = '#FFB000'       # Gold/Yellow
    color_solvent = '#DC267F'  # Magenta/Pink
    color_total = '#7B5C73'    # Muted Purple

    for i, file_path in enumerate(file_list):
        fname = os.path.basename(file_path)
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.split()
                # Parse frame number
                frame_num = int(re.search(r'\d+', parts[0]).group())
                
                # Parse all elements and charges in order
                # Structure is expected to be: Frame C Cl H H H [O H...]
                elements = []
                charges = []
                
                total_q = 0.0
                
                for item in parts[1:]:
                    if ':' in item:
                        el, chg = item.split(':')
                        chg = float(chg)
                        elements.append(el)
                        charges.append(chg)
                        total_q += chg

                # Calculate CH3 Sum (C + first 3 H atoms)
                # Assuming index 0 is C, index 1 is Cl, indices 2,3,4 are H
                ch3_q = 0.0
                if len(charges) >= 5:
                    ch3_q = charges[0] + charges[2] + charges[3] + charges[4] # C + 3H
                
                # Calculate Solvent Sum (O + remaining H atoms)
                solvent_q = None
                if 'O' in elements:
                    o_index = elements.index('O')
                    # Sum O and everything after it
                    solvent_q = sum(charges[o_index:])
                
                # Get Cl charge (usually at index 1)
                cl_q = charges[1] if len(charges) > 1 and elements[1] == 'Cl' else 0.0

                row = {
                    'Frame': frame_num,
                    'CH3_sum': ch3_q,
                    'Cl': cl_q,
                    'Solvent_sum': solvent_q,
                    'Total': total_q
                }
                data.append(row)

        df = pd.DataFrame(data)
        
        # Select the current axis
        ax = axes[i]
        # ax.text(0.1, 0.99, f'{panel_labels[i]}', transform=ax.transAxes,
        #                 fontsize=14, fontweight='bold', va='top', ha='right')

        # --- PLOTTING ---
        
        # 1. Plot CH3 Group
        ax.plot(df['Frame']/10, df['CH3_sum'], label='CH$_3$', 
                color=color_ch3, marker='o', markevery=5)
        
        # 2. Plot Cl
        ax.plot(df['Frame']/10, df['Cl'], label='Cl', 
                color=color_cl, marker='>', markevery=5)
        
        # 3. Plot Solvent (H2O or H3O) - only if data exists
        if df['Solvent_sum'].notna().any():
            # Determine label based on file content/name logic or generic 'Solvent'
            solvent_label = 'H$_2$O / H$_3$O'
            ax.plot(df['Frame']/10, df['Solvent_sum'], label=solvent_label, 
                    color=color_solvent, linestyle='-', linewidth=2)
        
        # 4. Plot Total Charge (sanity check, usually close to 0 or +/-1)
        ax.plot(df['Frame']/10, df['Total'], label='Total System', 
                color=color_total, linestyle='--')        

        ax.set_xlabel('C -- Cl Distance (Å)')
        ax.grid(True, linestyle=':', alpha=0.6)

    # Create a unified legend based on the last subplot (which likely has all traces)
    # Or combine handles from all plots if needed. 
    # Since specific labels change (H2O vs H3O), we might want per-plot legends 
    # or one combined legend. Here is a combined approach:
    
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    
    fig.legend(handles, labels,
               loc='upper center', 
               bbox_to_anchor=(0.5, 0.1),
               ncol=4,
               frameon=False)
    axes[0].set_ylim(-0.75, 1.25)
    axes[0].set_ylabel('Charge (e)')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig('sn1_charges.png', dpi=900)
    plt.close()

def main():
    setup_fonts()

    ch3cl = '../../data/partial_charges/ch3cl.txt'
    ch3cl_h2o = '../../data/partial_charges/ch3cl_h2o.txt'
    ch3cl_h3o = '../../data/partial_charges/ch3cl_h3o.txt'
    files = [ch3cl, ch3cl_h2o, ch3cl_h3o]
    plot_multiple_molecular_charges(files)


if __name__ == '__main__':
    main()