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
from loguru import logger
import sys
import os
import re

FAMILIES = ['oxide', 'iodide', 'alloy']

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
    raise FileNotFoundError("Could not find data/data.csv or data.csv.")

def determine_family(species):
    """Classify a species name (e.g. Ba_svI) into iodide, oxide, or alloy family."""
    if not isinstance(species, str) or not species:
        return 'unknown'
    system = re.sub(r'(_pv|_sv|_h|_d|_\d|_s)$', '', species)
    elements = re.findall(r'[A-Z][a-z]?', system)
    if 'O' in elements:
        return 'oxide'
    elif 'I' in elements:
        return 'iodide'
    else:
        return 'alloy'

def generate_composition_latex_table(output_tex_file="composition_table.tex"):
    """
    Generates a LaTeX table showing Composition and a comma-separated list of Prototype IDs.
    Families are used as bolded separator rows.
    Uses small monospace font and horizontal lines between rows.
    """
    try:
        csv_path = find_data_csv()
        raw_df = pd.read_csv(csv_path)
        
        # Filter out unphysical energy outliers (between -20 and 5 eV/atom)
        energy_cols = [c for c in raw_df.columns if c == 'gt_energy' or c.endswith('_energy') or c.endswith('_pred_energy')]
        for col in energy_cols:
            raw_df = raw_df[raw_df[col].isna() | ((raw_df[col] >= -20.0) & (raw_df[col] <= 5.0))]
            
        raw_df['family'] = raw_df['species'].apply(determine_family)
        
        # Filter out unknown families and duplicates to get unique systems
        unique_systems = raw_df[raw_df['family'] != 'unknown'][['catalog', 'species', 'prototype', 'family']].drop_duplicates()
        
        # The composition column of the table is the species name
        unique_systems['composition'] = unique_systems['species']
        
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        return

    # Ensure output directory exists
    out_dir = os.path.dirname(output_tex_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    with open(output_tex_file, 'w') as f_tex:
        f_tex.write("\\documentclass[11pt]{article}\n")
        f_tex.write("\\usepackage[margin=1in]{geometry}\n")
        f_tex.write("\\usepackage{booktabs}\n")
        f_tex.write("\\usepackage{longtable}\n")
        f_tex.write("\\begin{document}\n\n")
        f_tex.write("\\section*{System Compositions and Prototypes}\n\n")

        f_tex.write("{\\small\n")
        # Use p{width} for both to allow wrapping and monospace formatting
        f_tex.write("\\begin{longtable}{p{3cm} p{11cm}}\n")
        f_tex.write("\\caption{List of chemical compositions and prototype IDs by family.} \\\\\n")
        f_tex.write("\\toprule\n")
        f_tex.write("Composition & Prototype IDs \\\\\n")
        f_tex.write("\\midrule\n")
        f_tex.write("\\endfirsthead\n")
        f_tex.write("\\toprule\n")
        f_tex.write("Composition & Prototype IDs \\\\\n")
        f_tex.write("\\midrule\n")
        f_tex.write("\\endhead\n")
        f_tex.write("\\bottomrule\n")
        f_tex.write("\\endfoot\n")
        f_tex.write("\\bottomrule\n")
        f_tex.write("\\endlastfoot\n")

        for family in FAMILIES:
            family_df = unique_systems[unique_systems['family'] == family]
            
            if family_df.empty:
                continue
            
            # Group by composition and join prototypes with commas
            grouped = family_df.groupby('composition')['prototype'].apply(lambda x: ', '.join(sorted(list(x)))).reset_index()
            grouped = grouped.sort_values('composition')
            
            # Add a spanning row for the family name as a separator
            f_tex.write(f"\\multicolumn{{2}}{{l}}{{\\textbf{{{family.capitalize()}}}}} \\\\\n")
            f_tex.write("\\midrule\n")
            
            for _, row in grouped.iterrows():
                comp_esc = row['composition'].replace('_', '\\_')
                # Escape underscores in the joined prototype string
                protos_esc = row['prototype'].replace('_', '\\_')
                
                # Use texttt for monospace and add hline for separation
                f_tex.write(f"\\texttt{{{comp_esc}}} & \\texttt{{{protos_esc}}} \\\\\n")
                f_tex.write("\\hline\n")
            
            f_tex.write("\\midrule\n")

        f_tex.write("\\end{longtable}\n")
        f_tex.write("}\n")
        f_tex.write("\\end{document}\n")

    logger.info(f"Composition table saved to {output_tex_file}")

    # Write root FIGURE/composition_table.tex if FIGURE directory exists
    root_figure_dir = "../FIGURE" if os.path.exists("../FIGURE") else "FIGURE"
    if os.path.exists(root_figure_dir):
        root_composition_table_path = os.path.join(root_figure_dir, "composition_table.tex")
        try:
            import shutil
            shutil.copy(output_tex_file, root_composition_table_path)
            logger.info(f"Successfully copied composition table to root {root_composition_table_path}")
        except Exception as e:
            logger.error(f"Failed to copy composition table to root {root_composition_table_path}: {e}")

if __name__ == '__main__':
    generate_composition_latex_table()
