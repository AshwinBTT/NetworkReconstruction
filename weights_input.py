#!/usr/bin/env python3
import os
import re
import pandas as pd

# ==========================================
# Configuration
# ==========================================

# Input filenames
EDGE_FILE = 'edges_irreducible.txt'
FIRM_FILE = 'FIRM_STRENGTH.xlsx'
MAP_FILE = 'MAPPING.xlsx'

# Output configuration
OUTPUT_DIR = 'processed_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_NETWORK = os.path.join(OUTPUT_DIR, 'network.txt')
OUTPUT_MONEY = os.path.join(OUTPUT_DIR, 'money.txt')
OUTPUT_SECTOR = os.path.join(OUTPUT_DIR, 'firmSector.txt')

# ==========================================
# 1. Network Processing
# ==========================================

buyer_to_suppliers = {}

try:
    with open(EDGE_FILE, 'r') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            
            i = int(parts[0]) 
            j = int(parts[1]) 

            s = buyer_to_suppliers.get(i)
            if s is None:
                buyer_to_suppliers[i] = {j}
            else:
                s.add(j)

    with open(OUTPUT_NETWORK, 'w') as fout:
        for buyer in sorted(buyer_to_suppliers.keys()):
            suppliers = sorted(buyer_to_suppliers[buyer])
            fout.write(",".join([str(buyer)] + [str(s) for s in suppliers]) + "\n")

    print(f"network.txt has been written to: {OUTPUT_NETWORK}")

except FileNotFoundError:
    print(f"Skipping network generation: {EDGE_FILE} not found.")

# ==========================================
# 2. Money Processing
# ==========================================

def extract_numeric(val):
    match = re.search(r'(\d+)', str(val))
    return int(match.group(1)) if match else None

if os.path.exists(FIRM_FILE):
    df_money = pd.read_excel(FIRM_FILE)
    
    df_money['ID'] = df_money['Firm_ID'].apply(extract_numeric)
    
    money_out = pd.DataFrame({
        'ID': df_money['ID'],
        'Money': df_money['Size']
    })
    
    money_out.to_csv(OUTPUT_MONEY, index=False, header=False)
    print(f"Written {len(money_out)} firm lines to: {OUTPUT_MONEY}")
else:
    print(f"Skipping money generation: {FIRM_FILE} not found.")

# ==========================================
# 3. Sector Processing
# ==========================================

if os.path.exists(MAP_FILE):
    df_map = pd.read_excel(MAP_FILE, dtype=str)
    df_map.columns = df_map.columns.str.strip()

    lower_map = {c.lower().strip(): c for c in df_map.columns}

    def pick_col(*candidates):
        for c in candidates:
            if c in lower_map:
                return lower_map[c]
        return None

    firm_col = pick_col("firm_id", "firm id", "firmid", "id")
    sic_col = pick_col("sic", "sector", "sector_code", "sector code")

    if firm_col and sic_col:
        firm_extracted = df_map[firm_col].astype(str).str.extract(r'(\d+)', expand=False)
        sic_extracted = df_map[sic_col].astype(str).str.extract(r'(\d+)', expand=False)

        clean_map = pd.DataFrame({
            "Firm_ID": firm_extracted,
            "SIC": sic_extracted,
        })

        n_total = len(clean_map)
        clean_map = clean_map.dropna(subset=["Firm_ID", "SIC"])
        
        clean_map["Firm_ID"] = clean_map["Firm_ID"].astype(int)
        clean_map["SIC"] = clean_map["SIC"].astype(int)
        
        clean_map = clean_map.sort_values(["Firm_ID", "SIC"], kind="stable")
        
        clean_map[["Firm_ID", "SIC"]].to_csv(OUTPUT_SECTOR, header=False, index=False)

        print(f"[done] Read: {MAP_FILE}")
        print(f"[done] Rows total: {n_total}")
        print(f"[done] Kept: {len(clean_map)}")
        print(f"[done] Dropped: {n_total - len(clean_map)}")
        print(f"[done] Saved: {OUTPUT_SECTOR}")
    else:
        print("Could not find required columns in Mapping file.")
else:
    print(f"Skipping sector generation: {MAP_FILE} not found.")