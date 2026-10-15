import os
import sys
import glob
import json
import yaml
import pandas as pd
import numpy as np

def load_config(config_path="config.yaml"):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}

def find_data_file(raw_dir="data/raw"):
    extensions = ["*.csv", "*.tsv", "*.parquet", "*.xlsx", "*.xls"]
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(raw_dir, ext)))
    if not files:
        raise FileNotFoundError(f"No data files found in {raw_dir} with extensions {extensions}")
    return files[0]

def read_data(file_path):
    print(f"Reading file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".parquet":
        return pd.read_parquet(file_path), "parquet"
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(file_path), "excel"
    elif ext in [".csv", ".tsv"]:
        sep = "\t" if ext == ".tsv" else ","
        try:
            df = pd.read_csv(file_path, sep=sep, encoding="utf-8", low_memory=False)
            return df, "utf-8"
        except UnicodeDecodeError:
            print("UTF-8 decoding failed. Retrying with latin-1 encoding...")
            df = pd.read_csv(file_path, sep=sep, encoding="latin-1", low_memory=False)
            return df, "latin-1"
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def auto_map_columns(df_columns, config_overrides=None):
    if config_overrides is None:
        config_overrides = {}
        
    canonical_aliases = {
        "learner_id": ["user_id", "student_id", "userid", "studentid", "learner"],
        "order": ["order_id", "timestamp", "start_time", "time", "step", "attempt_id"],
        "question_id": ["problem_id", "question_id", "item_id", "exercise_id", "assignment_id"],
        "concept_id": ["skill_id", "skill_name", "concept", "knowledge_component", "kc", "topic"],
        "concept_name": ["skill_name", "concept_name", "skill_id", "concept", "knowledge_component", "topic"],
        "correct": ["correct", "is_correct", "response", "score"],
        "question_text": ["question_text", "problem_text", "body", "text"]
    }
    
    col_lower_map = {col.lower(): col for col in df_columns}
    final_mapping = {}
    
    # 1. Apply overrides first
    for canonical_name, override_col in config_overrides.items():
        if override_col in df_columns:
            final_mapping[canonical_name] = override_col
        else:
            print(f"Warning: Override column '{override_col}' for '{canonical_name}' not found in dataset columns.")
            
    # 2. Auto-map remaining canonical fields
    for canonical_name, aliases in canonical_aliases.items():
        if canonical_name in final_mapping:
            continue
        matched = None
        for alias in aliases:
            if alias in col_lower_map:
                matched = col_lower_map[alias]
                break
        if matched is not None:
            final_mapping[canonical_name] = matched

    # Specific refinement for skill_id vs skill_name if both exist
    if "skill_id" in col_lower_map and "skill_name" in col_lower_map:
        final_mapping["concept_id"] = col_lower_map["skill_id"]
        final_mapping["concept_name"] = col_lower_map["skill_name"]
    elif "concept_id" in final_mapping and "concept_name" not in final_mapping:
        final_mapping["concept_name"] = final_mapping["concept_id"]

    # Check order/time column
    if "order" not in final_mapping:
        print("Note: No explicit order/timestamp column found. Using file row order as order key.")
        final_mapping["order"] = None
        
    # Check question_text
    if "question_text" not in final_mapping:
        print("Note: No question_text column found. Will use question_id embeddings as exercise representation.")
        final_mapping["question_text"] = None

    # Required fields check: learner_id, question_id, concept_id, correct
    required = ["learner_id", "question_id", "concept_id", "correct"]
    missing = [req for req in required if req not in final_mapping or final_mapping[req] is None]
    
    if missing:
        print("\n" + "="*60)
        print("CRITICAL ERROR: Failed to map required canonical columns!")
        print(f"Missing required fields: {missing}")
        print(f"Dataset columns found: {list(df_columns)}")
        print("Please specify manual mappings in config.yaml under 'column_map'.")
        print("="*60 + "\n")
        sys.exit(1)
        
    return final_mapping

def inspect():
    config = load_config()
    raw_dir = config.get("paths", {}).get("raw_dir", "data/raw")
    logs_dir = config.get("paths", {}).get("logs_dir", "outputs/logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    file_path = find_data_file(raw_dir)
    df, encoding = read_data(file_path)
    
    print("\n" + "="*80)
    print("DATASET INSPECTION REPORT")
    print("="*80)
    print(f"File Name: {os.path.basename(file_path)} (Path: {file_path})")
    print(f"File Size: {os.path.getsize(file_path) / (1024 * 1024):.2f} MB")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Encoding used: {encoding}")
    print("\n--- Columns and Data Types ---")
    dtypes_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(t) for t in df.dtypes],
        "Null Count": [int(df[c].isnull().sum()) for c in df.columns],
        "Null %": [f"{(df[c].isnull().sum() / len(df) * 100):.2f}%" for c in df.columns]
    })
    print(dtypes_df.to_string(index=False))
    
    print("\n--- 5 Sample Rows ---")
    print(df.head(5).to_string())
    
    config_overrides = config.get("column_map", {})
    mapping = auto_map_columns(df.columns, config_overrides)
    
    print("\n--- Canonical Column Mapping ---")
    for k, v in mapping.items():
        print(f"  {k:15s} -> {v}")
        
    # Check correctness values
    correct_col = mapping["correct"]
    unique_correct = df[correct_col].dropna().unique()
    print(f"\nValues in '{correct_col}': {unique_correct}")
    is_binary = set(unique_correct).issubset({0, 1, 0.0, 1.0})
    if is_binary:
        print(f"Correctness column '{correct_col}' is already strictly binary (0/1).")
    else:
        print(f"Notice: Correctness column contains non-binary/fractional values. Values >= 0.5 will be binarized to 1, else 0.")
        
    # Save column_map.json
    out_map_path = os.path.join(logs_dir, "column_map.json")
    with open(out_map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"\nFinal column mapping saved to: {out_map_path}")
    print("="*80 + "\n")
    return mapping

if __name__ == "__main__":
    inspect()
