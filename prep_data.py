import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def prepare_data(subset_learners=None):
    config = load_config()
    raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
    proc_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    logs_dir = config.get("paths", {}).get("logs_dir", "outputs/logs")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 1. Load column mapping
    map_path = os.path.join(logs_dir, "column_map.json")
    if not os.path.exists(map_path):
        from inspect_data import inspect
        mapping = inspect()
    else:
        with open(map_path, "r", encoding="utf-8") as f:
            mapping = json.load(f)
            
    # 2. Find and read raw data
    raw_file = os.path.join(raw_dir, "skill_builder_data.csv")
    if not os.path.exists(raw_file):
        import glob
        files = glob.glob(os.path.join(raw_dir, "*.csv")) + glob.glob(os.path.join(raw_dir, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"No raw dataset found in {raw_dir}")
        raw_file = files[0]
        
    print(f"Loading raw data from: {raw_file}")
    try:
        df = pd.read_csv(raw_file, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(raw_file, encoding="latin-1", low_memory=False)
        
    # Canonical rename
    rename_dict = {
        mapping["learner_id"]: "learner_id",
        mapping["question_id"]: "question_id",
        mapping["concept_id"]: "concept_id",
        mapping["concept_name"]: "concept_name",
        mapping["correct"]: "correct"
    }
    if mapping.get("order"):
        rename_dict[mapping["order"]] = "order"
        
    df = df.rename(columns=rename_dict)
    
    # Ensure order column exists
    if "order" not in df.columns or mapping.get("order") is None:
        df["order"] = np.arange(len(df))
        
    # Keep only necessary canonical columns
    canonical_cols = ["learner_id", "order", "question_id", "concept_id", "concept_name", "correct"]
    df = df[canonical_cols].copy()
    
    # 3. Clean and filter
    initial_rows = len(df)
    df = df.dropna(subset=["learner_id", "concept_id", "correct"]).copy()
    
    # Binarize correct if fractional
    df["correct"] = (df["correct"].astype(float) >= 0.5).astype(int)
    
    # Handle multi-skill duplicates: keep the first interaction per learner per order key
    df = df.drop_duplicates(subset=["learner_id", "order"], keep="first").copy()
    
    # Sort by learner and order key
    df = df.sort_values(by=["learner_id", "order"]).reset_index(drop=True)
    
    # Add step index per learner
    df["step"] = df.groupby("learner_id").cumcount()
    
    # Top N concepts
    top_n = config.get("preprocessing", {}).get("top_n_concepts", 20)
    concept_counts = df["concept_name"].value_counts()
    top_concepts = concept_counts.head(top_n).index.tolist()
    
    print(f"\nFiltering to top {top_n} concepts out of {df['concept_name'].nunique()} unique concepts...")
    df = df[df["concept_name"].isin(top_concepts)].copy()
    
    # Build concepts.json: continuous concept index 0 to top_n-1
    concept_to_idx = {name: idx for idx, name in enumerate(sorted(top_concepts))}
    concepts_meta = {
        str(idx): {
            "concept_index": idx,
            "concept_name": name,
            "interaction_count": int(concept_counts[name])
        }
        for name, idx in concept_to_idx.items()
    }
    concepts_json_path = os.path.join(proc_dir, "concepts.json")
    with open(concepts_json_path, "w", encoding="utf-8") as f:
        json.dump(concepts_meta, f, indent=2)
    print(f"Saved concepts metadata to: {concepts_json_path}")
    
    df["concept_idx"] = df["concept_name"].map(concept_to_idx)
    
    # Filter learners by min_seq_len
    min_seq_len = config.get("preprocessing", {}).get("min_seq_len", 20)
    max_seq_len = config.get("preprocessing", {}).get("max_seq_len", 200)
    
    learner_counts = df["learner_id"].value_counts()
    valid_learners = learner_counts[learner_counts >= min_seq_len].index.tolist()
    
    if subset_learners is not None and subset_learners > 0:
        print(f"Subsetting to first {subset_learners} eligible learners for quick testing...")
        valid_learners = valid_learners[:subset_learners]
        
    df = df[df["learner_id"].isin(valid_learners)].copy()
    
    # Re-sort and re-index step per learner
    df = df.sort_values(by=["learner_id", "order"]).reset_index(drop=True)
    df["step"] = df.groupby("learner_id").cumcount()
    
    # Chunk/truncate sequences to max_seq_len
    # Create sequence chunks (seq_id = learner_id + '_' + str(step // max_seq_len))
    df["seq_chunk"] = df["step"] // max_seq_len
    df["seq_id"] = df["learner_id"].astype(str) + "_c" + df["seq_chunk"].astype(str)
    
    # Filter out residual chunks that are shorter than min_seq_len
    chunk_counts = df["seq_id"].value_counts()
    valid_chunks = chunk_counts[chunk_counts >= min_seq_len].index.tolist()
    df = df[df["seq_id"].isin(valid_chunks)].copy()
    
    # Reset step within each chunk
    df = df.sort_values(by=["seq_id", "order"]).reset_index(drop=True)
    df["step"] = df.groupby("seq_id").cumcount()
    
    # 4. Train/Test Split (75/25 learner-level seeded split)
    seed = config.get("seed", 42)
    unique_learners = sorted(df["learner_id"].unique().tolist())
    train_learners, test_learners = train_test_split(
        unique_learners, test_size=1.0 - config.get("preprocessing", {}).get("train_split_ratio", 0.75),
        random_state=seed, shuffle=True
    )
    
    split_meta = {
        "seed": seed,
        "train_learner_ids": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in train_learners],
        "test_learner_ids": [int(x) if isinstance(x, (np.integer, int)) else str(x) for x in test_learners]
    }
    split_json_path = os.path.join(proc_dir, "split.json")
    with open(split_json_path, "w", encoding="utf-8") as f:
        json.dump(split_meta, f, indent=2)
    print(f"Saved learner split to: {split_json_path} (Train: {len(train_learners)}, Test: {len(test_learners)})")
    
    # 5. Derive question difficulty from TRAIN split only
    train_mask = df["learner_id"].isin(train_learners)
    train_df = df[train_mask]
    
    q_stats = train_df.groupby("question_id")["correct"].agg(["mean", "count"]).reset_index()
    q_stats.columns = ["question_id", "train_correct_rate", "train_attempts"]
    
    # Terciles on question correct rate
    t33 = q_stats["train_correct_rate"].quantile(0.3333)
    t67 = q_stats["train_correct_rate"].quantile(0.6667)
    
    def assign_difficulty(rate):
        if pd.isna(rate):
            return "Medium"
        if rate >= t67:
            return "Easy"
        elif rate <= t33:
            return "Hard"
        else:
            return "Medium"
            
    q_stats["difficulty"] = q_stats["train_correct_rate"].apply(assign_difficulty)
    q_diff_map = dict(zip(q_stats["question_id"], q_stats["difficulty"]))
    
    df["difficulty"] = df["question_id"].map(q_diff_map).fillna("Medium")
    
    # Map questions to continuous integer index for embeddings
    unique_questions = sorted(df["question_id"].unique().tolist())
    q_to_idx = {qid: idx for idx, qid in enumerate(unique_questions)}
    df["question_idx"] = df["question_id"].map(q_to_idx)
    
    # Save processed dataframe
    interactions_path = os.path.join(proc_dir, "interactions.parquet")
    df.to_parquet(interactions_path, index=False)
    print(f"Saved processed interactions to: {interactions_path} ({len(df):,} rows)")
    
    # 6. Print Stats Comparison Table (Beside Paper Table 1)
    n_learners = df["learner_id"].nunique()
    n_concepts = df["concept_name"].nunique()
    n_questions = df["question_id"].nunique()
    total_interactions = len(df)
    avg_per_learner = total_interactions / n_learners if n_learners > 0 else 0
    avg_correct = df["correct"].mean() * 100
    
    stats_table = pd.DataFrame([
        {
            "Metric": "Concepts",
            "Paper (AMPLE LMS)": "382",
            "Reproduced Dataset": f"{n_concepts}"
        },
        {
            "Metric": "Learners",
            "Paper (AMPLE LMS)": "1,276",
            "Reproduced Dataset": f"{n_learners:,}"
        },
        {
            "Metric": "Questions / Problems",
            "Paper (AMPLE LMS)": "N/A",
            "Reproduced Dataset": f"{n_questions:,}"
        },
        {
            "Metric": "Total Interactions",
            "Paper (AMPLE LMS)": "359,681",
            "Reproduced Dataset": f"{total_interactions:,}"
        },
        {
            "Metric": "Avg Interactions / Learner",
            "Paper (AMPLE LMS)": "~281",
            "Reproduced Dataset": f"{avg_per_learner:.1f}"
        },
        {
            "Metric": "Avg Correct Rate",
            "Paper (AMPLE LMS)": "61.67%",
            "Reproduced Dataset": f"{avg_correct:.2f}%"
        }
    ])
    
    print("\n" + "="*80)
    print("DATASET COMPARISON TABLE (Paper Table 1 vs Reproduced Dataset)")
    print("="*80)
    print(stats_table.to_string(index=False))
    print("="*80 + "\n")
    
    # Save stats table
    tables_dir = config.get("paths", {}).get("tables_dir", "outputs/tables")
    os.makedirs(tables_dir, exist_ok=True)
    stats_table.to_csv(os.path.join(tables_dir, "dataset_stats_comparison.csv"), index=False)
    
    return df

if __name__ == "__main__":
    prepare_data()
