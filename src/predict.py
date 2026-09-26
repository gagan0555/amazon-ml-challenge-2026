import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
import os
import time
from preprocessing import clean_and_romanize_series # Importing your cleanup logic

MATCH_THRESHOLD = 0.65
CHUNK_SIZE = 50000

CANDIDATES_PATH = "../outputs/candidate_pairs.tsv"
S1_PATH = "../data/test_source1.tsv"
S2_PATH = "../data/test_source2.tsv"
S3_PATH = "../data/test_source3.tsv"
OUTPUT_PATH = "../outputs/matching_results.tsv"

def run_predictions():
    print("Loading and Pre-Cleaning databases S1, S2, and S3...")
    start_global_time = time.time()

    df_s1 = pd.read_csv(S1_PATH, sep='\t').set_index('entity_id')
    df_targets = pd.concat([
        pd.read_csv(S2_PATH, sep='\t').set_index('entity_id'),
        pd.read_csv(S3_PATH, sep='\t').set_index('entity_id')
    ])

    df_s1['clean_name'] = clean_and_romanize_series(df_s1['business_name'])
    df_s1['clean_addr'] = clean_and_romanize_series(df_s1['business_address'])
    df_targets['clean_name'] = clean_and_romanize_series(df_targets['business_name'])
    df_targets['clean_addr'] = clean_and_romanize_series(df_targets['business_address'])
    
    print("Starting RapidFuzz cpdist Engine...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n") 

    chunk_counter = 0

    for chunk in pd.read_csv(CANDIDATES_PATH, sep='\t', chunksize=CHUNK_SIZE):
        chunk = chunk[chunk['candidate_entity_ids'].notna()].copy()
        
        chunk['cand_id_list'] = chunk['candidate_entity_ids'].astype(str).str.split(',')
        exploded = chunk.explode('cand_id_list')
        exploded['cand_id'] = exploded['cand_id_list'].str.strip()
        exploded = exploded[exploded['cand_id'] != ""]
        
        if exploded.empty: continue

        s1_names = exploded['source1_entity_id'].map(df_s1['clean_name']).fillna("").to_numpy(dtype=str)
        s1_addrs = exploded['source1_entity_id'].map(df_s1['clean_addr']).fillna("").to_numpy(dtype=str)
        cand_names = exploded['cand_id'].map(df_targets['clean_name']).fillna("").to_numpy(dtype=str)
        cand_addrs = exploded['cand_id'].map(df_targets['clean_addr']).fillna("").to_numpy(dtype=str)

        name_scores = process.cpdist(s1_names, cand_names, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        addr_scores = process.cpdist(s1_addrs, cand_addrs, scorer=fuzz.token_set_ratio, workers=-1, dtype=np.uint8)
        
        addr_empty_mask = (s1_addrs == "") | (cand_addrs == "")
        final_scores = np.where(
            addr_empty_mask,
            name_scores.astype(np.float32) / 100.0,
            (0.5 * name_scores.astype(np.float32) + 0.5 * addr_scores.astype(np.float32)) / 100.0
        )
        
        exploded['score'] = final_scores
        matches = exploded[exploded['score'] >= MATCH_THRESHOLD]
        matches = matches.sort_values('score', ascending=False).drop_duplicates(subset=['cand_id'], keep='first')
        agg_matches = matches.groupby('source1_entity_id')['cand_id'].apply(lambda x: ','.join(x)).reset_index()
        agg_matches.rename(columns={'cand_id': 'matched_entity_ids'}, inplace=True)
        
        final_chunk_df = pd.merge(chunk[['source1_entity_id']], agg_matches, on='source1_entity_id', how='left')
        final_chunk_df['matched_entity_ids'] = final_chunk_df['matched_entity_ids'].fillna("")
        final_chunk_df.to_csv(OUTPUT_PATH, mode='a', header=False, index=False, sep='\t', quoting=3)
        
        chunk_counter += 1
        print(f"Processed {(chunk_counter * CHUNK_SIZE)} S1 records...")

if __name__ == "__main__":
    run_predictions()
