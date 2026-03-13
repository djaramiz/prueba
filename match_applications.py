#!/usr/bin/env python3
"""
Match installed applications (Flexera/SNOW) against purchased licenses (CMDB).
Uses fuzzy matching with manufacturer boosting.
"""

import csv
import re
import unicodedata
from rapidfuzz import fuzz


# --- Configuration ---
INPUT_INSTALLED = "LICvsSWinstalado_EPM_11032026.csv"
INPUT_LICENSES = "CMDB_ContratosVsLicencias.csv"
OUTPUT_FILE = "match_applications_result.csv"
THRESHOLD = 50  # Minimum match percentage to include
MANUFACTURER_BOOST = 10  # Bonus points when manufacturers match
DELIMITER = ";"


def normalize(text):
    """Normalize text for comparison: lowercase, remove accents, trim."""
    if not text or text.strip() == "":
        return ""
    text = text.strip().lower()
    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_manufacturer(text):
    """Normalize manufacturer name more aggressively for comparison."""
    text = normalize(text)
    if not text or text == "desconocido":
        return ""
    # Remove common corporate suffixes
    suffixes = [
        r"\b(inc\.?|incorporated|corp\.?|corporation|ltd\.?|ltda\.?|llc|"
        r"s\.?a\.?s\.?|s\.?a\.?|s\.?r\.?l\.?|gmbh|co\.?|pty|"
        r"technologies|software|systems|solutions)\b",
    ]
    for pattern in suffixes:
        text = re.sub(pattern, "", text)
    text = re.sub(r"[.,;:()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def manufacturer_similarity(mfr_installed, mfr_license):
    """Check if manufacturers are similar enough to boost score."""
    norm_inst = normalize_manufacturer(mfr_installed)
    norm_lic = normalize_manufacturer(mfr_license)
    if not norm_inst or not norm_lic:
        return 0  # Can't compare, no boost or penalty
    # Check token set ratio for manufacturer (handles reordering, partial)
    score = fuzz.token_set_ratio(norm_inst, norm_lic)
    if score >= 70:
        return MANUFACTURER_BOOST
    return 0


def compute_match_score(app_name, product_name, mfr_installed, mfr_license):
    """
    Compute match score between an installed app and a licensed product.
    Uses multiple fuzzy matching techniques and picks the best.
    """
    norm_app = normalize(app_name)
    norm_prod = normalize(product_name)

    if not norm_app or not norm_prod:
        return 0

    # 1. Standard ratio (Levenshtein-based)
    score_ratio = fuzz.ratio(norm_app, norm_prod)

    # 2. Partial ratio (best substring match) - heavily discounted to avoid
    #    false positives on short overlaps
    score_partial = fuzz.partial_ratio(norm_app, norm_prod)

    # 3. Token sort ratio (order-independent)
    score_token_sort = fuzz.token_sort_ratio(norm_app, norm_prod)

    # 4. Token set ratio (handles subset/superset tokens)
    score_token_set = fuzz.token_set_ratio(norm_app, norm_prod)

    # 5. WRatio - rapidfuzz's weighted ratio that auto-picks best strategy
    score_wratio = fuzz.WRatio(norm_app, norm_prod)

    # Penalize partial and token_set when lengths differ a lot
    len_ratio = min(len(norm_app), len(norm_prod)) / max(len(norm_app), len(norm_prod))
    partial_weight = 0.6 + 0.3 * len_ratio  # 0.6-0.9 depending on length similarity
    token_set_weight = 0.7 + 0.3 * len_ratio  # 0.7-1.0

    base_score = max(
        score_ratio,
        score_partial * partial_weight,
        score_token_sort,
        score_token_set * token_set_weight,
        score_wratio,
    )

    # Manufacturer boost
    mfr_boost = manufacturer_similarity(mfr_installed, mfr_license)
    final_score = min(100, base_score + mfr_boost)

    return round(final_score, 1)


def read_csv_semicolon(filepath):
    """Read a semicolon-delimited CSV file."""
    rows = []
    with open(filepath, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=DELIMITER)
        for row in reader:
            rows.append(row)
    return rows


def main():
    print("Reading installed applications (Flexera)...")
    installed = read_csv_semicolon(INPUT_INSTALLED)
    print(f"  -> {len(installed)} records")

    print("Reading purchased licenses (CMDB)...")
    licenses_raw = read_csv_semicolon(INPUT_LICENSES)
    print(f"  -> {len(licenses_raw)} records")

    # Deduplicate licenses by product name (take first occurrence)
    seen = set()
    licenses = []
    for row in licenses_raw:
        product = row.get("Nombre del producto", "").strip()
        if product and product not in seen:
            seen.add(product)
            licenses.append(row)
    print(f"  -> {len(licenses)} unique products after deduplication")

    # Build results
    print("Matching applications...")
    results = []
    total = len(installed)

    for idx, inst_row in enumerate(installed, 1):
        app_name = inst_row.get("Application", "").strip()
        mfr_installed = inst_row.get("Manufacturer", "").strip()

        if not app_name:
            continue

        best_score = 0
        best_product = ""
        best_mfr = ""

        for lic_row in licenses:
            product_name = lic_row.get("Nombre del producto", "").strip()
            mfr_license = lic_row.get("Fabricante", "").strip()

            score = compute_match_score(
                app_name, product_name, mfr_installed, mfr_license
            )

            if score > best_score:
                best_score = score
                best_product = product_name
                best_mfr = mfr_license

        if best_score > THRESHOLD:
            results.append({
                "Application": app_name,
                "Manufacturer": mfr_installed,
                "Nombre del producto (match)": best_product,
                "Fabricante (match)": best_mfr,
                "% coincidencia": best_score,
            })

        if idx % 100 == 0 or idx == total:
            print(f"  -> Processed {idx}/{total} applications...")

    # Sort by match percentage descending
    results.sort(key=lambda x: x["% coincidencia"], reverse=True)

    # Write output
    print(f"\nWriting results to {OUTPUT_FILE}...")
    fieldnames = [
        "Application",
        "Manufacturer",
        "Nombre del producto (match)",
        "Fabricante (match)",
        "% coincidencia",
    ]
    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=DELIMITER)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nDone! {len(results)} matches found (>{THRESHOLD}% similarity)")
    print(f"Results saved to: {OUTPUT_FILE}")

    # Quick stats
    if results:
        scores = [r["% coincidencia"] for r in results]
        print(f"\nMatch statistics:")
        print(f"  100%  matches: {sum(1 for s in scores if s == 100)}")
        print(f"  90-99% matches: {sum(1 for s in scores if 90 <= s < 100)}")
        print(f"  70-89% matches: {sum(1 for s in scores if 70 <= s < 90)}")
        print(f"  50-69% matches: {sum(1 for s in scores if 50 <= s < 70)}")


if __name__ == "__main__":
    main()
