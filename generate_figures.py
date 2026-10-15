import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from ekt import EKTModel, load_config

def plot_metrics_curves(metrics_csv_path="outputs/tables/ekt_metrics.csv", output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(metrics_csv_path):
        print(f"Metrics CSV not found at {metrics_csv_path}")
        return
        
    df = pd.read_csv(metrics_csv_path)
    epochs = df["epoch"].values
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # 1. Loss
    axes[0].plot(epochs, df["train_loss"], linestyle="--", color="#1f77b4", linewidth=2.2, label="Train Loss (BCE)", marker="o", markersize=4)
    axes[0].plot(epochs, df["test_loss"], linestyle="-", color="#d62728", linewidth=2.4, label="Test Loss (BCE)", marker="s", markersize=4)
    axes[0].set_title("Binary Cross-Entropy Loss vs Epoch", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("BCE Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(fontsize=10, loc="upper right")
    
    # 2. AUC
    axes[1].plot(epochs, df["train_auc"], linestyle="--", color="#1f77b4", linewidth=2.2, label="Train AUC", marker="o", markersize=4)
    axes[1].plot(epochs, df["test_auc"], linestyle="-", color="#2ca02c", linewidth=2.4, label="Test AUC", marker="s", markersize=4)
    axes[1].set_title("ROC-AUC vs Epoch", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Area Under ROC Curve", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(fontsize=10, loc="lower right")
    
    # 3. MAE
    axes[2].plot(epochs, df["train_mae"], linestyle="--", color="#1f77b4", linewidth=2.2, label="Train MAE", marker="o", markersize=4)
    axes[2].plot(epochs, df["test_mae"], linestyle="-", color="#ff7f0e", linewidth=2.4, label="Test MAE", marker="s", markersize=4)
    axes[2].set_title("Mean Absolute Error (MAE) vs Epoch", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch", fontsize=11)
    axes[2].set_ylabel("MAE", fontsize=11)
    axes[2].grid(True, linestyle=":", alpha=0.6)
    axes[2].legend(fontsize=10, loc="upper right")
    
    plt.tight_layout()
    out_file = os.path.join(output_dir, "metrics_curves.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved metric curves to: {out_file}")

def plot_knowledge_state_heatmap(learner_id=None, num_concepts=8, max_steps=40, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    config = load_config()
    proc_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    models_dir = os.path.join(config.get("paths", {}).get("output_dir", "outputs"), "models")
    checkpoint_path = os.path.join(models_dir, "ekt_checkpoint.pt")
    
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
    steps_count = min(len(learner_df), max_steps)
    learner_sub = learner_df.iloc[:steps_count]
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
    model = EKTModel(
        num_questions=checkpoint["num_questions"],
        num_concepts=checkpoint["num_concepts"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Select top N most frequent concepts in learner's interactions (or top N concepts overall)
    c_counts = learner_sub["concept_idx"].value_counts()
    selected_c_indices = c_counts.index.tolist()[:num_concepts]
    if len(selected_c_indices) < num_concepts:
        extra = [i for i in range(len(concepts_meta)) if i not in selected_c_indices]
        selected_c_indices.extend(extra[:num_concepts - len(selected_c_indices)])
    selected_c_indices = sorted(selected_c_indices)
    
    concept_labels = [concepts_meta[str(idx)]["concept_name"] for idx in selected_c_indices]
    
    # Compute mastery across all time steps
    q_ids = torch.tensor(np.array([learner_sub["question_idx"].values]), dtype=torch.long).to(device)
    c_ids = torch.tensor(np.array([learner_sub["concept_idx"].values]), dtype=torch.long).to(device)
    r_ids = torch.tensor(np.array([learner_sub["correct"].values]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        mastery_prob, _ = model(q_ids, c_ids, r_ids)
        # mastery_prob: (1, seq_len, total_concepts)
        full_trajectory = mastery_prob[0].cpu().numpy()  # (seq_len, total_concepts)
        
    heatmap_matrix = full_trajectory[:, selected_c_indices].T  # (num_selected_concepts, seq_len)
    
    plt.figure(figsize=(14, 6), dpi=300)
    plt.imshow(heatmap_matrix, aspect="auto", cmap="YlGnBu", vmin=0.3, vmax=0.9, origin="upper")
    cbar = plt.colorbar()
    cbar.set_label("EKT Mastery Probability $y_t$", fontsize=11, fontweight="bold")
    
    plt.yticks(range(len(concept_labels)), concept_labels, fontsize=10)
    step_labels = [f"Step {t+1}" for t in range(steps_count)]
    # Subsample step x-ticks if many steps
    tick_step = max(1, steps_count // 15)
    plt.xticks(range(0, steps_count, tick_step), [f"t={t+1}" for t in range(0, steps_count, tick_step)], fontsize=10)
    
    plt.title(f"Dynamic Knowledge State Heatmap Over Time: Learner #{learner_id}", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Interaction Step (Timeline)", fontsize=11)
    plt.ylabel("Tracked Concept", fontsize=11)
    
    out_file = os.path.join(output_dir, "knowledge_state_heatmap.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved knowledge state heatmap to: {out_file}")

def plot_mastery_radar_chart(learner_id=None, num_concepts=8, output_dir="outputs/figures"):
    os.makedirs(output_dir, exist_ok=True)
    config = load_config()
    proc_dir = config.get("paths", {}).get("processed_dir", "data/processed")
    models_dir = os.path.join(config.get("paths", {}).get("output_dir", "outputs"), "models")
    checkpoint_path = os.path.join(models_dir, "ekt_checkpoint.pt")
    
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
    model = EKTModel(
        num_questions=checkpoint["num_questions"],
        num_concepts=checkpoint["num_concepts"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"]
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    q_ids = torch.tensor(np.array([learner_df["question_idx"].values]), dtype=torch.long).to(device)
    c_ids = torch.tensor(np.array([learner_df["concept_idx"].values]), dtype=torch.long).to(device)
    r_ids = torch.tensor(np.array([learner_df["correct"].values]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        mastery_prob, _ = model(q_ids, c_ids, r_ids)
        initial_mastery = mastery_prob[0, 0].cpu().numpy()
        final_mastery = mastery_prob[0, -1].cpu().numpy()
        
    # Pick concepts with most variance or top concepts
    c_indices = list(range(min(num_concepts, len(concepts_meta))))
    labels = [concepts_meta[str(i)]["concept_name"] for i in c_indices]
    
    init_vals = [float(initial_mastery[i]) for i in c_indices]
    final_vals = [float(final_mastery[i]) for i in c_indices]
    
    # Close the radar polygon
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    init_vals += init_vals[:1]
    final_vals += final_vals[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], labels, fontsize=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.4, 0.6, 0.8, 1.0], ["40%", "60%", "80%", "100%"], color="#666666", size=8)
    plt.ylim(0.2, 1.0)
    
    # Initial state
    ax.plot(angles, init_vals, linewidth=2, linestyle="--", color="#1f77b4", label="Initial Mastery (Step 1)")
    ax.fill(angles, init_vals, color="#1f77b4", alpha=0.15)
    
    # Final state
    ax.plot(angles, final_vals, linewidth=2.5, linestyle="-", color="#2ca02c", label=f"Final Mastery (Step {len(learner_df)})")
    ax.fill(angles, final_vals, color="#2ca02c", alpha=0.25)
    
    # Target Line
    target_line = [0.70] * (num_vars + 1)
    ax.plot(angles, target_line, linewidth=1.5, linestyle=":", color="#d62728", label="OBE Target (70%)")
    
    plt.title(f"Radar Chart: Initial vs Final Concept Mastery (Learner #{learner_id})", fontsize=12, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), fontsize=9)
    
    out_file = os.path.join(output_dir, "mastery_radar_chart.png")
    plt.savefig(out_file, bbox_inches="tight")
    plt.close()
    print(f"Saved mastery radar chart to: {out_file}")

def generate_all_figures():
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-QUALITY FIGURES")
    print("="*80)
    plot_metrics_curves()
    plot_knowledge_state_heatmap()
    plot_mastery_radar_chart()
    print("="*80 + "\n")

if __name__ == "__main__":
    generate_all_figures()
