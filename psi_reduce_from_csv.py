#!/usr/bin/env python3
# psi_reduce_from_csv.py
# Reduce one or more per-shell CSVs into headline ψ-Cone stats.
#
# Usage:
#   python psi_reduce_from_csv.py --csv psi_shell_stats_1.csv psi_shell_stats_2.csv --json out_summary.json
#
# The CSVs must have columns:
#   shell_z,sum_cos,sum_sin,N_kept,N_total,decimate_k
#
import argparse
import csv
import json
import math
from collections import defaultdict

def rayleigh_stats(N: int, S_mag: float):
    if N <= 0:
        return float('nan'), float('nan')
    Rbar = S_mag / max(1, N)
    Z = N * (Rbar ** 2)
    # same approximation as in proof script
    term1 = math.exp(-Z)
    term2 = 1.0 + (2.0*Z - Z*Z) / (4.0*N)
    term3 = (24.0*Z - 132.0*Z*Z + 76.0*(Z**3) - 9.0*(Z**4)) / (288.0*(N**2))
    p = term1 * (term2 - term3)
    if p < 0.0: p = 0.0
    return Z, p

def reduce_csvs(paths, out_json=None):
    # Aggregate by shell_z
    agg = defaultdict(lambda: [0.0, 0.0, 0, 0])  # sumr, sumi, kept, total
    for path in paths:
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            needed = {"shell_z","sum_cos","sum_sin","N_kept","N_total"}
            if not needed.issubset(reader.fieldnames):
                raise ValueError(f"{path} is missing required columns; has {reader.fieldnames}")
            for row in reader:
                z = int(row["shell_z"])
                sumr = float(row["sum_cos"])
                sumi = float(row["sum_sin"])
                kept = int(float(row["N_kept"]))  # tolerate "1.0" etc.
                total = int(float(row["N_total"]))
                a = agg[z]
                a[0] += sumr; a[1] += sumi; a[2] += kept; a[3] += total

    # Global vector
    S_real = sum(a[0] for a in agg.values())
    S_imag = sum(a[1] for a in agg.values())
    N = sum(a[2] for a in agg.values())
    S_mag = math.hypot(S_real, S_imag)
    mu_global_deg = (math.degrees(math.atan2(S_imag, S_real)) + 360.0) % 360.0
    PLV_global = S_mag / N if N > 0 else float('nan')
    Z, pval = rayleigh_stats(N, S_mag)

    # Shell-weighted and mean shell PLV
    U_real, U_imag, Zcount = 0.0, 0.0, 0
    plv_sum, plv_count = 0.0, 0
    for z, (sumr, sumi, kept, total) in agg.items():
        if kept <= 0:
            continue
        mag = math.hypot(sumr, sumi)
        plv = mag / kept if kept > 0 else float('nan')
        if not math.isnan(plv):
            plv_sum += plv; plv_count += 1
        if mag > 0.0:
            U_real += sumr / mag
            U_imag += sumi / mag
            Zcount += 1

    PLV_shell_mean = (plv_sum / plv_count) if plv_count > 0 else float('nan')
    U_mag = math.hypot(U_real, U_imag)
    PLV_shell_weighted = U_mag / Zcount if Zcount > 0 else float('nan')
    mu_sw_deg = (math.degrees(math.atan2(U_imag, U_real)) + 360.0) % 360.0

    summary = {
        "csv_inputs": paths,
        "shells_nonempty": Zcount,
        "N_kept": N,
        "mu_global_deg": mu_global_deg,
        "PLV_global": PLV_global,
        "rayleigh_Z": Z,
        "rayleigh_p": pval,
        "mu_shell_weighted_deg": mu_sw_deg,
        "PLV_shell_weighted": PLV_shell_weighted,
        "PLV_shell_mean": PLV_shell_mean,
    }

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Print as a simple table
    print("=== ψ-Cone Reduce Summary ===")
    for k in ["N_kept","shells_nonempty","mu_global_deg","PLV_global","rayleigh_Z","rayleigh_p","mu_shell_weighted_deg","PLV_shell_weighted","PLV_shell_mean"]:
        print(f"{k}: {summary[k]}")

    return summary

def main():
    ap = argparse.ArgumentParser(description="Reduce ψ-Cone per-shell CSVs into headline stats.")
    ap.add_argument("--csv", nargs="+", required=True, help="One or more per-shell CSV paths.")
    ap.add_argument("--json", default=None, help="Optional output JSON summary path.")
    args = ap.parse_args()
    reduce_csvs(args.csv, args.json)

if __name__ == "__main__":
    main()
