"""

Usage:
    python generate_candidate_pairs.py train
    python generate_candidate_pairs.py test
"""

import re
import os
import sys
import time
import pickle
import itertools
import pandas as pd

KEY_CAP = 5000     # a single blocking key's bucket is ignored if bigger than this
ROW_CAP = 1000     # hard ceiling on candidates written per Source 1 entity
PROGRESS_EVERY = 200_000  # print a progress line every N rows while streaming

STOPWORDS = {
    "inc", "llc", "ltd", "limited", "corp", "corporation", "co", "company",
    "pvt", "private", "llp", "group", "enterprises", "enterprise", "the",
    "and", "of", "services", "solutions", "international", "holdings",
}


def normalize(text):
    return str(text).strip().lower()


def meaningful_prefixes(name):
    text = normalize(name)
    words = re.findall(r"[a-z]+", text)
    words = [w for w in words if w not in STOPWORDS and len(w) >= 3]
    return sorted(set(w[:4] for w in words))


def name_pair_keys(name, country):
    prefixes = meaningful_prefixes(name)
    if len(prefixes) >= 2:
        return {f"{country}_{a}_{b}" for a, b in itertools.combinations(prefixes, 2)}
    elif len(prefixes) == 1:
        return {f"{country}_{prefixes[0]}"}
    return set()


def address_keys(address, country):
    text = normalize(address)
    if text in ("nan", "none", ""):
        return set()
    digit_groups = re.findall(r"\d+", text)
    if not digit_groups:
        return set()
    longest = max(digit_groups, key=len)
    if len(longest) < 3:
        return set()
    return {f"{country}_{longest}"}


def build_raw_index(rows, key_func, use_address):
    index = {}
    for row in rows:
        country = normalize(row.country)
        text = row.business_address if use_address else row.business_name
        for key in key_func(text, country):
            index.setdefault(key, []).append(row.entity_id)
    return index


def get_or_build_indexes(s23, cache_path):
    if os.path.isfile(cache_path):
        print(f"  Found cached index at {cache_path}, loading instantly...")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("  No cache found, building from scratch (this is the slow step)...")
    s23_rows = list(s23.itertuples())
    by_name_pair = build_raw_index(s23_rows, name_pair_keys, use_address=False)
    by_addr_digit = build_raw_index(s23_rows, address_keys, use_address=True)

    print(f"  Saving index to {cache_path} so future runs skip this step...")
    with open(cache_path, "wb") as f:
        pickle.dump((by_name_pair, by_addr_digit), f, protocol=pickle.HIGHEST_PROTOCOL)

    return by_name_pair, by_addr_digit


def candidates_for_row(row, by_name_pair, by_addr_digit):
    """
    Returns a bounded (<= ROW_CAP) set of candidate ids for one Source 1 row.
    Keys with a bucket bigger than KEY_CAP are skipped entirely (too generic
    to be useful). If the union is still bigger than ROW_CAP, we keep the
    smallest-bucket keys' contributions first (more specific = more likely
    to be a real match) until we hit the cap.
    """
    country = normalize(row.country)
    contributing_buckets = []

    for key in name_pair_keys(row.business_name, country):
        bucket = by_name_pair.get(key)
        if bucket and len(bucket) <= KEY_CAP:
            contributing_buckets.append(bucket)

    for key in address_keys(row.business_address, country):
        bucket = by_addr_digit.get(key)
        if bucket and len(bucket) <= KEY_CAP:
            contributing_buckets.append(bucket)

    # Smallest (most specific) buckets first, so if we have to stop partway
    # through due to ROW_CAP, we've kept the most selective evidence.
    contributing_buckets.sort(key=len)

    found = set()
    for bucket in contributing_buckets:
        found.update(bucket)
        if len(found) >= ROW_CAP:
            break

    if len(found) > ROW_CAP:
        found = set(sorted(found)[:ROW_CAP])  # deterministic trim to the cap

    return found


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    assert split in ("train", "test"), "argument must be 'train' or 'test'"

    data_dir = f"dataset/{split}"
    output_path = "output/candidate_pairs.tsv"
    cache_path = f"block_index_cache_{split}.pkl"

    t0 = time.time()
    print(f"Loading {split} data from {data_dir}/ ...")
    s1 = pd.read_csv(f"{data_dir}/{split}_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{data_dir}/{split}_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{data_dir}/{split}_source3.tsv", sep="\t")
    s23 = pd.concat([s2, s3])
    print(f"  Source 1: {len(s1)} | Source 2: {len(s2)} | Source 3: {len(s3)} "
          f"({time.time() - t0:.1f}s)\n")

    print("Getting the blocking index (cached if available)...")
    t1 = time.time()
    by_name_pair, by_addr_digit = get_or_build_indexes(s23, cache_path)
    print(f"  done ({time.time() - t1:.1f}s)\n")

    print(f"Streaming candidates to {output_path} "
          f"(KEY_CAP={KEY_CAP}, ROW_CAP={ROW_CAP})...")
    t2 = time.time()
    os.makedirs("output", exist_ok=True)

    total_candidates = 0
    zero_candidate_rows = 0
    truncated_rows = 0
    n_rows = len(s1)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for i, row in enumerate(s1.itertuples(), start=1):
            found = candidates_for_row(row, by_name_pair, by_addr_digit)
            f.write(f"{row.entity_id}\t{','.join(sorted(found))}\n")

            total_candidates += len(found)
            if len(found) == 0:
                zero_candidate_rows += 1
            if len(found) >= ROW_CAP:
                truncated_rows += 1

            if i % PROGRESS_EVERY == 0:
                elapsed = time.time() - t2
                rate = i / elapsed
                remaining = (n_rows - i) / rate
                print(f"  {i}/{n_rows} rows written "
                      f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    print(f"\nDone writing {output_path} ({time.time() - t2:.1f}s)\n")
    print(f"Average candidates per entity: {total_candidates / n_rows:.1f}")
    print(f"Entities with zero candidates: {zero_candidate_rows}")
    print(f"Entities hitting the ROW_CAP ({ROW_CAP}): {truncated_rows} "
          f"({truncated_rows / n_rows:.1%})")
    print(f"\nTotal time: {time.time() - t0:.1f}s\n")

    print("Next: validate the file —")
    print(f"  python3 utils/validate_submission.py --candidate {output_path} "
          f"--test-dir {data_dir}")