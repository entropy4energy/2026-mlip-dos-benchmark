#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "pandas",
#     "loguru",
#     "numpy",
# ]
# ///

import pandas as pd
import numpy as np
from loguru import logger
import sys
import os
import re

MLIPS = ['chgnet', 'esen', 'mace', 'uma']
FAMILIES = ['alloy', 'iodide', 'oxide']  # Matched to manuscript/FIGURE order
MLIP_NAMES = {
    'chgnet': 'CHGNet',
    'esen': 'eSEN',
    'mace': 'MACE',
    'uma': 'UMA'
}

def find_data_csv():
    """Return the released CSV path from common repository locations."""
    candidates = [
        'data/data.csv',
        'data.csv',
        '../data/data.csv',
        '../data.csv',
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def determine_family(species):
    """Classify a species name (e.g. Ba_svI) into iodide, oxide, or alloy family."""
    if not isinstance(species, str) or not species:
        return 'unknown'
    system = species
    system = re.sub(r'(_pv|_sv|_h|_d|_\d|_s)$', '', system)
    elements = re.findall(r'[A-Z][a-z]?', system)
    if 'O' in elements:
        return 'oxide'
    elif 'I' in elements:
        return 'iodide'
    else:
        return 'alloy'

def load_and_process_data():
    """
    Load data.csv and construct fig3_df and fig4_df in the format expected by the table generator.
    """
    csv_path = find_data_csv()
    if csv_path is None:
        logger.error("Could not find data/data.csv or data.csv.")
        return pd.DataFrame(), pd.DataFrame()
        
    df = pd.read_csv(csv_path)
    
    # Filter out unphysical energy outliers (between -20 and 5 eV/atom)
    energy_cols = [c for c in df.columns if c == 'gt_energy' or c.endswith('_energy') or c.endswith('_pred_energy')]
    for col in energy_cols:
        df = df[df[col].isna() | ((df[col] >= -20.0) & (df[col] <= 5.0))]
        
    df['family'] = df['species'].apply(determine_family)
    df = df[df['family'] != 'unknown']
    
    # Construct fig3_plot_df
    fig3_rows = []
    metric_e_fermi_label = r'$E_{\text{F,pred}} - E_\text{F}$ (eV)'
    metric_cpu_hours_label = 'CPU hours'
    
    # Ground truth CPU hours
    for _, row in df.iterrows():
        if pd.notna(row['gt_core_hours_relax']):
            fig3_rows.append({
                'metric': metric_cpu_hours_label,
                'family': row['family'],
                'mlip': 'ground_truth',
                'value': row['gt_core_hours_relax']
            })
            
    # MLIP models
    for mlip in MLIPS:
        for _, row in df.iterrows():
            # Fermi energy residual
            fermi_val = row.get(f'{mlip}_e_fermi')
            gt_fermi = row.get('gt_e_fermi')
            if pd.notna(fermi_val) and pd.notna(gt_fermi):
                fig3_rows.append({
                    'metric': metric_e_fermi_label,
                    'family': row['family'],
                    'mlip': mlip,
                    'value': fermi_val - gt_fermi
                })
            # CPU hours static
            cpu_val = row.get(f'{mlip}_core_hours_static')
            if pd.notna(cpu_val):
                fig3_rows.append({
                    'metric': metric_cpu_hours_label,
                    'family': row['family'],
                    'mlip': mlip,
                    'value': cpu_val
                })
                
    fig3_plot_df = pd.DataFrame(fig3_rows)
    
    # Construct fig4_plot_df
    fig4_rows = []
    for mlip in MLIPS:
        for _, row in df.iterrows():
            # Cosine Distance
            cos_val = row.get(f'{mlip}_dos_cosine_dist')
            if pd.notna(cos_val):
                fig4_rows.append({
                    'metric': 'DOS Cosine Distance',
                    'family': row['family'],
                    'mlip': mlip,
                    'value': cos_val
                })
            # Wasserstein Distance
            wass_val = row.get(f'{mlip}_dos_wasserstein_dist')
            if pd.notna(wass_val):
                fig4_rows.append({
                    'metric': 'DOS Wasserstein Distance',
                    'family': row['family'],
                    'mlip': mlip,
                    'value': wass_val
                })
                
    fig4_plot_df = pd.DataFrame(fig4_rows)
    
    return fig3_plot_df, fig4_plot_df

def generate_figure3_latex_tables(fig3_plot_df, fig4_plot_df, output_tex_file="figure3_statistics.tex"):
    """
    Generates a single LaTeX table for Figure 3 statistics and writes it to a file.
    Includes Fermi energy residual, CPU hours speedup, DOS metrics, RMSD, predicted energy residual,
    and DFT-calculated energy residual, with internal separations for each family.
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(output_tex_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    metric_e_fermi_label = r'$E_{\text{F,pred}} - E_\text{F}$ (eV)'
    metric_cpu_hours_label = 'CPU hours'
    metric_dos_cosine_label = r'$1 - \cos \theta$'
    metric_dos_wasserstein_label = r'$d_\text{w}$'
    metric_rmsd_label = 'RMSD (Å)'
    metric_energy_pred_residual_label = r'$E_{\text{pred}} - E$ (eV)'
    metric_energy_dft_residual_label = r'$E_{\text{DFT}} - E$ (eV)'

    # Prepare header for the combined table
    header = [
        "MLIP",
        metric_e_fermi_label,
        metric_energy_pred_residual_label,
        metric_energy_dft_residual_label,
        metric_rmsd_label,
        f"{metric_cpu_hours_label} (Speedup)",
        metric_dos_cosine_label,
        metric_dos_wasserstein_label
    ]
    num_cols = len(header)

    all_table_rows = []

    for idx, family in enumerate(FAMILIES):
        # Add a family header row
        if idx == 0:
            all_table_rows.append(f"\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{family.capitalize()}}}}} \\\\\n\\midrule\n")
        else:
            all_table_rows.append(f"\\midrule\n\\multicolumn{{{num_cols}}}{{l}}{{\\textbf{{{family.capitalize()}}}}} \\\\\n\\midrule\n")

        # Calculate average ground_truth CPU hours for speedup calculation
        gt_cpu_hours_data = fig3_plot_df[(fig3_plot_df['metric'] == metric_cpu_hours_label) & 
                                    (fig3_plot_df['family'] == family) & 
                                    (fig3_plot_df['mlip'] == 'ground_truth')]
        avg_gt_cpu_hours = gt_cpu_hours_data['value'].mean() if not gt_cpu_hours_data.empty else np.nan

        # Prepare data for current family
        family_table_data = []
        
        for mlip in MLIPS:
            formatted_mlip = MLIP_NAMES.get(mlip, mlip.replace('_', '-').upper())
            row_entry = {"MLIP": formatted_mlip}
            
            # E_Fermi Residual
            subset_e_fermi = fig3_plot_df[(fig3_plot_df['metric'] == metric_e_fermi_label) & 
                                     (fig3_plot_df['family'] == family) & 
                                     (fig3_plot_df['mlip'] == mlip)]
            
            if not subset_e_fermi.empty:
                mean_val = subset_e_fermi['value'].mean()
                std_val = subset_e_fermi['value'].std()
                row_entry[metric_e_fermi_label] = f"${mean_val:.2f} \\pm {std_val:.2f}$"
            else:
                row_entry[metric_e_fermi_label] = "N/A"

            try:
                csv_path = find_data_csv()
                if csv_path is None:
                    raise FileNotFoundError("Could not find data/data.csv or data.csv.")
                raw_df = pd.read_csv(csv_path)
                
                # Filter out unphysical energy outliers (between -20 and 5 eV/atom)
                energy_cols = [c for c in raw_df.columns if c == 'gt_energy' or c.endswith('_energy') or c.endswith('_pred_energy')]
                for col in energy_cols:
                    raw_df = raw_df[raw_df[col].isna() | ((raw_df[col] >= -20.0) & (raw_df[col] <= 5.0))]
                    
                raw_df['family'] = raw_df['species'].apply(determine_family)
                raw_df = raw_df[raw_df['family'] != 'unknown']
            except Exception as e:
                logger.error(f"Error loading raw data for table generation: {e}")
                row_entry[metric_energy_pred_residual_label] = "N/A"
                row_entry[metric_energy_dft_residual_label] = "N/A"
                row_entry[metric_rmsd_label] = "N/A"
            else:
                mlip_family_raw_df = raw_df[(raw_df['family'] == family)]

                # Predicted Energy Residual
                if f'{mlip}_pred_energy' in mlip_family_raw_df.columns and 'gt_energy' in mlip_family_raw_df.columns:
                    energy_pred_residual_data = (mlip_family_raw_df['gt_energy'] - mlip_family_raw_df[f'{mlip}_pred_energy']).dropna()
                    if not energy_pred_residual_data.empty:
                        mean_val = energy_pred_residual_data.mean()
                        std_val = energy_pred_residual_data.std()
                        row_entry[metric_energy_pred_residual_label] = f"${mean_val:.2f} \\pm {std_val:.2f}$"
                    else:
                        row_entry[metric_energy_pred_residual_label] = "N/A"
                else:
                     row_entry[metric_energy_pred_residual_label] = "N/A"

                # DFT-calculated Energy Residual
                if f'{mlip}_energy' in mlip_family_raw_df.columns and 'gt_energy' in mlip_family_raw_df.columns:
                    energy_dft_residual_data = (mlip_family_raw_df['gt_energy'] - mlip_family_raw_df[f'{mlip}_energy']).dropna()
                    if not energy_dft_residual_data.empty:
                        mean_val = energy_dft_residual_data.mean()
                        std_val = energy_dft_residual_data.std()
                        row_entry[metric_energy_dft_residual_label] = f"${mean_val:.2f} \\pm {std_val:.2f}$"
                    else:
                        row_entry[metric_energy_dft_residual_label] = "N/A"
                else:
                    row_entry[metric_energy_dft_residual_label] = "N/A"

                # RMSD
                if f'{mlip}_rmsd' in mlip_family_raw_df.columns:
                    rmsd_data = mlip_family_raw_df[f'{mlip}_rmsd'].dropna()
                    if not rmsd_data.empty:
                        mean_val = rmsd_data.mean()
                        std_val = rmsd_data.std()
                        row_entry[metric_rmsd_label] = f"${mean_val:.2f} \\pm {std_val:.2f}$"
                    else:
                        row_entry[metric_rmsd_label] = "N/A"
                else:
                    row_entry[metric_rmsd_label] = "N/A"
            
            # CPU hours (Speedup)
            subset_cpu_hours = fig3_plot_df[(fig3_plot_df['metric'] == metric_cpu_hours_label) & 
                                       (fig3_plot_df['family'] == family) & 
                                       (fig3_plot_df['mlip'] == mlip)]
            
            if not subset_cpu_hours.empty and not np.isnan(avg_gt_cpu_hours) and avg_gt_cpu_hours != 0:
                mean_mlip_cpu_hours = subset_cpu_hours['value'].mean()
                std_mlip_cpu_hours = subset_cpu_hours['value'].std()
                speedup = avg_gt_cpu_hours / mean_mlip_cpu_hours
                row_entry[f"{metric_cpu_hours_label} (Speedup)"] = f"${mean_mlip_cpu_hours:.1f} \\pm {std_mlip_cpu_hours:.1f}$ ($\\times${speedup:.1f})"
            else:
                row_entry[f"{metric_cpu_hours_label} (Speedup)"] = "N/A"
            
            # DOS Cosine Distance
            subset_dos_cosine = fig4_plot_df[(fig4_plot_df['metric'] == 'DOS Cosine Distance') &
                                              (fig4_plot_df['family'] == family) &
                                              (fig4_plot_df['mlip'] == mlip)]
            if not subset_dos_cosine.empty:
                mean_val = subset_dos_cosine['value'].mean()
                std_val = subset_dos_cosine['value'].std()
                row_entry[metric_dos_cosine_label] = f"${mean_val:.2f} \\pm {std_val:.2f}$"
            else:
                row_entry[metric_dos_cosine_label] = "N/A"

            # DOS Wasserstein Distance
            subset_dos_wasserstein = fig4_plot_df[(fig4_plot_df['metric'] == 'DOS Wasserstein Distance') &
                                                   (fig4_plot_df['family'] == family) &
                                                   (fig4_plot_df['mlip'] == mlip)]
            if not subset_dos_wasserstein.empty:
                mean_val = subset_dos_wasserstein['value'].mean()
                std_val = subset_dos_wasserstein['value'].std()
                row_entry[metric_dos_wasserstein_label] = f"${mean_val:.2f} \\pm {std_val:.2f}$"
            else:
                row_entry[metric_dos_wasserstein_label] = "N/A"

            family_table_data.append(row_entry)
        
        # Add rows for the current family to the overall list
        for row in family_table_data:
            all_table_rows.append(f"{row['MLIP']} & {row[metric_e_fermi_label]} & {row[metric_energy_pred_residual_label]} & {row[metric_energy_dft_residual_label]} & {row[metric_rmsd_label]} & {row[f'{metric_cpu_hours_label} (Speedup)']} & {row[metric_dos_cosine_label]} & {row[metric_dos_wasserstein_label]} \\\\\n")

    col_format = "l c c c c c c c"

    # Build the exact table block requested by the user
    table_block = (
        "\\begin{table*}[ht]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{\\small\n"
        "\\textbf{Summary of structural and electronic\n"
        "metrics across material classes.}\n"
        "Values reported as mean $\\pm$ standard\n"
        "deviation.\n"
        "Speedup factors (in parentheses) are\n"
        "relative to full \\DFT\\ relaxation.\n"
        "}\n"
        "%\\vspace{-0.25cm}\n"
        "\\label{tab:figure3_combined}\n"
        "\\resizebox{\\textwidth}{!}{\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\t\\toprule\n"
        "\t" + " & ".join(header) + " \\\\\n"
        "\t\\midrule\n"
    )
    for row_str in all_table_rows:
        # Prepend tab to each line in row_str
        lines = row_str.strip('\n').split('\n')
        for line in lines:
            table_block += "\t" + line + "\n"
    table_block += (
        "\t\\bottomrule\n"
        "\\end{tabular}\n"
        "  }\n"
        "\\end{table*}\n"
    )

    # Write output_tex_file (Figure 3 statistics standalone)
    with open(output_tex_file, 'w', encoding='utf-8') as f_tex:
        f_tex.write(table_block)
    logger.info(f"Figure 3 statistics saved to {output_tex_file}")

    # Write paper/FIGURE/table_metrics.tex if paper directory exists
    paper_dir = "../paper" if os.path.exists("../paper") else "paper"
    if os.path.exists(paper_dir):
        table_metrics_path = os.path.join(paper_dir, "FIGURE", "table_metrics.tex")
        try:
            os.makedirs(os.path.dirname(table_metrics_path), exist_ok=True)
            with open(table_metrics_path, 'w', encoding='utf-8') as f:
                f.write(table_block)
            logger.info(f"Successfully generated {table_metrics_path}")
        except Exception as e:
            logger.error(f"Failed to generate {table_metrics_path}: {e}")

    # Write root FIGURE/table_metrics.tex if FIGURE directory exists
    root_figure_dir = "../FIGURE" if os.path.exists("../FIGURE") else "FIGURE"
    if os.path.exists(root_figure_dir):
        root_table_metrics_path = os.path.join(root_figure_dir, "table_metrics.tex")
        try:
            with open(root_table_metrics_path, 'w', encoding='utf-8') as f_out:
                f_out.write(table_block)
            logger.info(f"Successfully generated root {root_table_metrics_path}")
        except Exception as e:
            logger.error(f"Failed to generate root {root_table_metrics_path}: {e}")

    # Update paper/manuscript_agenerated.tex directly if it exists
    manuscript_path = os.path.join(paper_dir, "manuscript_agenerated.tex")
    if os.path.exists(manuscript_path):
        try:
            with open(manuscript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Format the new tabular block
            new_tab_content = ""
            for row_str in all_table_rows:
                new_tab_content += "\t" + row_str
                
            # Replace inside manuscript_agenerated.tex
            pattern = r'(\\label\{tab:figure3_combined\}.*?\\begin\{tabular\}\{l c c c c c c c\}).*?(\\end\{tabular\})'
            
            def repl(match):
                prefix = match.group(1)
                suffix = match.group(2)
                header_line = "\t" + " & ".join(header) + " \\\\\n\t\\midrule\n"
                return f"{prefix}\n{header_line}{new_tab_content}\t{suffix}"
                
            new_content, count = re.subn(pattern, repl, content, flags=re.DOTALL)
            if count > 0:
                with open(manuscript_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                logger.info(f"Successfully updated inline metrics table in {manuscript_path} ({count} replacement).")
            else:
                logger.warning(f"Could not find matching table block in {manuscript_path} to update.")
        except Exception as e:
            logger.error(f"Failed to update manuscript file {manuscript_path}: {e}")

if __name__ == '__main__':
    logger.info("Generating Figure 3 statistics tables.")
    fig3_df, fig4_df = load_and_process_data()
    if not fig3_df.empty or not fig4_df.empty:
        generate_figure3_latex_tables(fig3_df, fig4_df)
    else:
        logger.error("No data available to generate Figure 3 tables.")
