import os
import json
import yaml
import numpy as np
import pandas as pd
from ekt import get_knowledge_state, load_config

def classify_concept(mastery, threshold=0.50, target=0.70):
    """
    Paper OBE Classification:
    mastery < threshold -> Beginner (Gap)
    threshold <= mastery <= target -> Intermediate (Gap)
    mastery > target -> Expert (Proficient)
    """
    if mastery < threshold:
        level = "Beginner"
        is_gap = True
    elif mastery <= target:
        level = "Intermediate"
        is_gap = True
    else:
        level = "Expert"
        is_gap = False
    return level, is_gap

def run_paper_demo(threshold=0.50, target=0.70):
    demo_vector = [0.82, 0.35, 0.76, 0.60, 0.91]
    demo_concepts = [
        "Concept 1 (Foundations)",
        "Concept 2 (Core Mechanics)",
        "Concept 3 (Problem Solving)",
        "Concept 4 (Advanced Synthesis)",
        "Concept 5 (Real-World Application)"
    ]
    
    print("\n" + "="*80)
    print("OBE KNOWLEDGE GAP ENGINE: PAPER DEMO VECTOR EVALUATION")
    print("="*80)
    print(f"Mastery Threshold: {threshold*100:.1f}% | Target Level: {target*100:.1f}%")
    print(f"{'Concept':<35} | {'Mastery':<9} | {'OBE Level':<14} | {'Gap Flagged?':<12} | {'Action':<15}")
    print("-" * 90)
    
    demo_results = []
    for i, (c_name, score) in enumerate(zip(demo_concepts, demo_vector)):
        level, is_gap = classify_concept(score, threshold, target)
        action = "Send to Remediation" if is_gap else "Mastered (No Action)"
        flag_str = "YES (GAP)" if is_gap else "NO"
        print(f"{c_name:<35} | {score:<9.2f} | {level:<14} | {flag_str:<12} | {action:<15}")
        demo_results.append({
            "concept": c_name,
            "mastery": score,
            "level": level,
            "is_gap": is_gap,
            "action": action
        })
    print("="*80 + "\n")
    return demo_results

def evaluate_held_out_learner(learner_id=None, threshold=0.50, target=0.70):
    config = load_config()
    proc_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    output_dir = config.get("paths", {}).get("output_dir", "outputs")
    
    interactions_path = os.path.join(proc_dir, "interactions.parquet")
    split_path = os.path.join(proc_dir, "split.json")
    concepts_path = os.path.join(proc_dir, "concepts.json")
    
    df = pd.read_parquet(interactions_path)
    with open(split_path, "r", encoding="utf-8") as f:
        split_meta = json.load(f)
    with open(concepts_path, "r", encoding="utf-8") as f:
        concepts_meta = json.load(f)
        
    test_learners = split_meta["test_learner_ids"]
    if learner_id is None or learner_id not in test_learners:
        learner_id = test_learners[0]
        
    learner_df = df[df["learner_id"] == learner_id].sort_values(by="order").reset_index(drop=True)
    history = list(zip(
        learner_df["question_idx"].values,
        learner_df["concept_idx"].values,
        learner_df["correct"].values
    ))
    
    # Compute knowledge state via EKT model
    mastery_vector = get_knowledge_state(history)
    
    # Classify each concept
    results = []
    for idx_str, meta in concepts_meta.items():
        idx = int(idx_str)
        c_name = meta["concept_name"]
        score = float(mastery_vector[idx])
        level, is_gap = classify_concept(score, threshold, target)
        
        # Collect recent questions for this concept
        concept_questions = learner_df[learner_df["concept_idx"] == idx]
        total_attempts = len(concept_questions)
        correct_attempts = int(concept_questions["correct"].sum()) if total_attempts > 0 else 0
        
        results.append({
            "concept_idx": idx,
            "concept_name": c_name,
            "mastery": score,
            "level": level,
            "is_gap": is_gap,
            "total_attempts": total_attempts,
            "correct_attempts": correct_attempts,
            "empirical_accuracy": (correct_attempts / total_attempts) if total_attempts > 0 else None
        })
        
    results_df = pd.DataFrame(results).sort_values(by="mastery").reset_index(drop=True)
    
    print("\n" + "="*80)
    print(f"OBE KNOWLEDGE GAP REPORT: HELD-OUT LEARNER #{learner_id}")
    print("="*80)
    print(f"Total Interactions: {len(learner_df)} | Total Concepts Tracked: {len(results_df)}")
    print(f"Threshold (Beginner/Intermediate): {threshold*100:.0f}% | Target (Intermediate/Expert): {target*100:.0f}%")
    print("\n" + f"{'Index':<6} {'Concept Name':<35} {'Mastery':<9} {'Level':<13} {'Gap Flag':<10} {'Attempts':<10}")
    print("-" * 85)
    for _, row in results_df.iterrows():
        gap_tag = "FLAGGED" if row["is_gap"] else "MASTERED"
        print(f"{row['concept_idx']:<6} {row['concept_name'][:34]:<35} {row['mastery']*100:<8.1f}% {row['level']:<13} {gap_tag:<10} {row['correct_attempts']}/{row['total_attempts']}")
        
    flagged_gaps = results_df[results_df["is_gap"]].copy()
    print("\n" + f"Total Concepts Flagged for Remediation: {len(flagged_gaps)} of {len(results_df)}")
    print("="*80 + "\n")
    
    # Generate Markdown Report
    gap_report_md = f"""# OBE Knowledge Gap Engine Report

## Evaluation Parameters
- **Target Learner ID**: `{learner_id}` (Held-out Test Split)
- **Total Historical Interactions**: {len(learner_df)}
- **Mastery Threshold (Beginner cutoff)**: {threshold*100:.1f}%
- **Mastery Target (Proficiency cutoff)**: {target*100:.1f}%
- **Gap Condition**: Any concept with Mastery < Target ({target*100:.1f}%) is flagged for remedial recommendation.

---

## 1. Paper Demonstration Vector Verification
Vector provided in base paper: `[0.82, 0.35, 0.76, 0.60, 0.91]`

| Concept ID | Synthetic Concept Name | Mastery Score | OBE Level | Gap Status | Action Triggered |
|---|---|---|---|---|---|
| C1 | Concept 1 (Foundations) | 82.0% | **Expert** | Proficient | No Action |
| C2 | Concept 2 (Core Mechanics) | 35.0% | **Beginner** | **FLAGGED GAP** | Send to Remediation |
| C3 | Concept 3 (Problem Solving) | 76.0% | **Expert** | Proficient | No Action |
| C4 | Concept 4 (Advanced Synthesis) | 60.0% | **Intermediate** | **FLAGGED GAP** | Send to Remediation |
| C5 | Concept 5 (Real-World Application) | 91.0% | **Expert** | Proficient | No Action |

---

## 2. Real Held-Out Learner Knowledge State Evaluation
Evaluated using the trained EKT knowledge tracing model on learner `#{learner_id}`.

| Concept Index | Concept Name | EKT Mastery Probability | OBE Classification | Remediation Flag | Learner History (Correct / Total) |
|---|---|---|---|---|---|
"""
    for _, row in results_df.iterrows():
        flag_badge = "**FLAGGED FOR REMEDIATION**" if row["is_gap"] else "Proficient (Expert)"
        gap_report_md += f"| {row['concept_idx']} | {row['concept_name']} | {row['mastery']*100:.2f}% | **{row['level']}** | {flag_badge} | {row['correct_attempts']} / {row['total_attempts']} |\n"

    gap_report_md += f"""
---

## 3. Summary & Remedial Routing
- **Total Tracked Concepts**: {len(results_df)}
- **Expert Concepts (Mastery > {target*100:.0f}%)**: {len(results_df[~results_df['is_gap']])}
- **Concepts Requiring Remediation (Mastery < {target*100:.0f}%)**: {len(flagged_gaps)}
  - **Beginner Concepts (< {threshold*100:.0f}%)**: {len(results_df[results_df['level'] == 'Beginner'])}
  - **Intermediate Concepts ({threshold*100:.0f}% - {target*100:.0f}%)**: {len(results_df[results_df['level'] == 'Intermediate'])}
- **Top Priority Concept for Remediation**: `{flagged_gaps.iloc[0]['concept_name']}` (Mastery: {flagged_gaps.iloc[0]['mastery']*100:.2f}%)
"""
    out_md_path = os.path.join(output_dir, "gap_report.md")
    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write(gap_report_md)
    print(f"Knowledge gap report saved to: {out_md_path}")
    
    return learner_id, learner_df, results_df, flagged_gaps

if __name__ == "__main__":
    config = load_config()
    thresh = config.get("gap_engine", {}).get("threshold", 0.50)
    targ = config.get("gap_engine", {}).get("target", 0.70)
    run_paper_demo(thresh, targ)
    evaluate_held_out_learner(threshold=thresh, target=targ)
