import argparse
import math
import os
import random

import pandas as pd


# These columns must appear in FIRM.csv
SIZE_COLUMNS = [
    '$0-99,000', '$100,000-499,000', '$500,000-999,000',
    '$1,000,000-2,499,000', '$2,500,000-4,999,000', '$5,000,000-7,499,000',
    '$7,500,000-9,999,000', '$10,000,000-14,999,000', '$15,000,000-19,999,000',
    '$20,000,000-24,999,000', '$25,000,000-29,999,000', '$30,000,000-34,999,000',
    '$35,000,000-39,999,000', '$40,000,000-44,999,000', '$45,000,000-49,999,000',
    '50,000,000-74,999,000', '$75,000,000-99,999,000', '$100,000,000+'
]

# Corresponding numeric ranges for sampling Sizes
SIZE_RANGES = [
    (0, 99000), (100000, 499000), (500000, 999000),
    (1000000, 2499000), (2500000, 4999000), (5000000, 7499000),
    (7500000, 9999000), (10000000, 14999000), (15000000, 19999000),
    (20000000, 24999000), (25000000, 29999000), (30000000, 34999000),
    (35000000, 39999000), (40000000, 44999000), (45000000, 49999000),
    (50000000, 74990000), (75000000, 99990000), (100000000, 10000000000)
]


def custom_round(x: float):
    """
    Rounds a number so that if the first decimal digit is 0,
    it rounds to an integer; otherwise, to one decimal place.
    """
    first_decimal = int((x - int(x)) * 10)
    if first_decimal == 0:
        return round(x)
    else:
        return round(x, 1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample synthetic firms by SIC and size band from FIRM.csv."
    )
    parser.add_argument(
        "--input-file",
        default="FIRM.csv",
        help="Path to input aggregated firm file (default: FIRM.csv)."
    )
    parser.add_argument(
        "--mapping-out",
        default="MAPPING.xlsx",
        help="Output Excel path for Firm_ID–SIC mapping (default: MAPPING.xlsx)."
    )
    parser.add_argument(
        "--strength-out",
        default="FIRM_STRENGTH.xlsx",
        help="Output Excel path for Firm_ID–Size mapping (default: FIRM_STRENGTH.xlsx)."
    )
    parser.add_argument(
        "--target-firms",
        required=True,
        help=(
            "Desired total number of sampled synthetic firms. "
            "Either a positive integer (e.g. 50000) or the keyword ALL "
            "to sample all firms implied by FIRM.csv."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (optional)."
    )
    return parser.parse_args()


def parse_target_firms(target_str: str):
    """
    Parse the --target-firms argument, which can be an integer string
    or the keyword ALL (case-insensitive).

    Returns:
        (mode, value)
        - If mode == 'all', value is None (we decide the actual number after reading the data).
        - If mode == 'int', value is an integer > 0.
    """
    if target_str.strip().upper() == "ALL":
        return "all", None

    try:
        val = int(target_str)
    except ValueError:
        raise ValueError(
            f"Invalid --target-firms value: {target_str!r}. "
            "Must be a positive integer or the keyword ALL."
        )

    if val <= 0:
        raise ValueError(
            f"Invalid --target-firms value: {target_str!r}. "
            "Must be a positive integer or the keyword ALL."
        )

    return "int", val


def compute_sample_counts(counts, target, rng):
    """
    Given a list of non-negative integer counts and a desired total sample size
    `target`, compute how many samples to draw from each cell so that:

    - sum(sample_counts) == target
    - sample_counts[i] ≈ target * counts[i] / sum(counts) in expectation

    Uses floor of expected values plus a randomized allocation of the remaining
    units based on fractional parts.
    """
    if target <= 0:
        raise ValueError("target must be a positive integer.")

    total = sum(counts)
    if total <= 0:
        raise ValueError("Total count across all cells is zero; nothing to sample.")

    # Expected values
    fraction = target / total
    expected = [c * fraction for c in counts]

    # Base = floor(expected), remainder = fractional part
    base = [int(math.floor(e)) for e in expected]
    remainders = [e - b for e, b in zip(expected, base)]

    base_sum = sum(base)
    deficit = target - base_sum

    # In exact math, deficit >= 0 because sum(floor(e)) <= sum(e)=target.
    # Floating point can cause off-by-1; we guard for negative just in case.
    if deficit < 0:
        # Drop some units uniformly at random among non-empty base cells
        indices = [i for i, b in enumerate(base) if b > 0]
        if not indices:
            raise RuntimeError("Numerical issue: negative deficit but no positive base counts.")
        for _ in range(-deficit):
            i = rng.choice(indices)
            base[i] -= 1
        deficit = 0

    if deficit == 0:
        return base

    # If there is a positive deficit, distribute these extra units based on remainders
    rem_sum = sum(remainders)
    if rem_sum <= 0:
        # Fallback: uniform among cells with positive original counts
        positive_indices = [i for i, c in enumerate(counts) if c > 0]
        if not positive_indices:
            # Should not happen, but just in case
            return base
        for _ in range(deficit):
            i = rng.choice(positive_indices)
            base[i] += 1
        return base

    # Weighted random allocation according to remainders
    probs = [r / rem_sum for r in remainders]

    # Draw `deficit` indices with replacement, each time adding 1 to that base.
    for _ in range(deficit):
        u = rng.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if u <= cumulative:
                base[i] += 1
                break

    return base


def main():
    args = parse_args()

    # Global seed for any library calls that use random
    if args.seed is not None:
        random.seed(args.seed)

    mode, target_val = parse_target_firms(args.target_firms)

    print(f"[INFO] Reading input file: {args.input_file}")
    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    firm_df = pd.read_csv(args.input_file)
    firm_df.columns = firm_df.columns.str.strip()

    # Ensure required columns exist
    required_cols = ["SIC"] + SIZE_COLUMNS
    missing_cols = [col for col in required_cols if col not in firm_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in the CSV file: {missing_cols}")

    # Build a flattened list of (sic, lower_bound, upper_bound, count)
    cells = []
    counts = []

    print("[INFO] Parsing SIC and size-band counts ...")
    for _, row in firm_df.iterrows():
        sic_code = str(row["SIC"]).strip()
        for col, (lower_bound, upper_bound) in zip(SIZE_COLUMNS, SIZE_RANGES):
            val = row[col]
            if pd.isna(val):
                continue
            count = int(val)
            if count <= 0:
                continue
            cells.append((sic_code, lower_bound, upper_bound, count))
            counts.append(count)

    if not cells:
        raise ValueError("No positive counts found in FIRM.csv after parsing; nothing to sample.")

    total_original = sum(counts)
    print(f"[INFO] Total original firm count (sum of all cells) = {total_original}")

    rng = random.Random()
    if args.seed is not None:
        rng.seed(args.seed)

    # Decide sampling counts per cell
    if mode == "all":
        # Sample ALL firms: exactly expand each cell to its count
        print("[INFO] --target-firms ALL detected: sampling all firms implied by FIRM.csv.")
        sample_counts = list(counts)
        total_sampled = total_original
    else:
        # Proportional sampling to target_val
        target = target_val
        print(f"[INFO] Target sample size (proportional sampling) = {target}")
        sample_counts = compute_sample_counts(counts, target, rng)
        total_sampled = sum(sample_counts)

    print(f"[INFO] Total samples assigned across cells = {total_sampled}")

    # Generate synthetic firms
    sampled_firms = []
    firm_counter = 1

    print("[INFO] Generating synthetic firms ...")
    for (sic_code, lower_bound, upper_bound, _count), n_samp in zip(cells, sample_counts):
        for _ in range(n_samp):
            firm_id = f"Firm_{firm_counter}"
            size = rng.uniform(lower_bound, upper_bound)
            size = custom_round(size)
            sampled_firms.append([sic_code, firm_id, size])
            firm_counter += 1

    print(f"[INFO] Sampling complete. Total sampled firms = {len(sampled_firms)}")

    sampled_df = pd.DataFrame(sampled_firms, columns=["SIC", "Firm_ID", "Size"])

    # Split into mapping and strength tables
    mapping_df = sampled_df[["Firm_ID", "SIC"]].copy()
    strength_df = sampled_df[["Firm_ID", "Size"]].copy()

    # Write outputs
    print(f"[INFO] Writing outputs:\n  Mapping -> {args.mapping_out}\n  Strength -> {args.strength_out}")
    with pd.ExcelWriter(args.mapping_out) as w1:
        mapping_df.to_excel(w1, index=False)
    with pd.ExcelWriter(args.strength_out) as w2:
        strength_df.to_excel(w2, index=False)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()



