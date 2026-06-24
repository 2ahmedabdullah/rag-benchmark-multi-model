# evaluate_metrics.py

import pandas as pd
import numpy as np
import os

def calculate_ndcg_at_k(r, k=10):
    r = np.asarray(r, dtype=float)[:k]
    if r.size == 0: return 0.0
    dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
    ideal_r = sorted(r, reverse=True)
    idcg = np.sum(ideal_r / np.log2(np.arange(2, len(ideal_r) + 2)))
    if idcg == 0: return 0.0
    return dcg / idcg

def calculate_rag_metrics(preds_path="./results/raw_search_predictions.xlsx", lat_path="./results/system_latency_scorecard.xlsx"):
    print(f"Reading evaluation predictions from {preds_path}...")
    df = pd.read_excel(preds_path)
    
    metrics_summary = []
    for approach, group in df.groupby("Search Approach"):
        recall_per_query = group.groupby("Query ID")["True Hit?"].any().astype(int)
        precision_per_query = group.groupby("Query ID")["True Hit?"].sum() / 10.0
        
        mrr_list, ndcg_list = [], []
        for q_id, q_group in group.groupby("Query ID"):
            sorted_q_group = q_group.sort_values("Rank")
            binary_hits = sorted_q_group["True Hit?"].astype(int).values
            
            hit_ranks = sorted_q_group[sorted_q_group["True Hit?"] == True]["Rank"].values
            mrr_list.append(1.0 / hit_ranks[0] if len(hit_ranks) > 0 else 0.0)
            ndcg_list.append(calculate_ndcg_at_k(binary_hits, k=10))
        
        metrics_summary.append({
            "Search Approach": approach,
            "Total True Hits": group["True Hit?"].sum(),
            "Precision@10": round(precision_per_query.mean(), 4),
            "Recall@10": round(recall_per_query.mean(), 4),
            "MRR@10": round(np.mean(mrr_list), 4),
            "NDCG@10": round(np.mean(ndcg_list), 4)
        })
        
    metrics_df = pd.DataFrame(metrics_summary)
    
    # --- AUTOMATIC LATENCY MERGE LAYER ---
    if os.path.exists(lat_path):
        print(f"Reading recorded latency files from {lat_path}...")
        lat_df = pd.read_excel(lat_path)
        # Standardize join column names
        lat_df = lat_df.rename(columns={"Strategy": "Search Approach"})
        metrics_df = pd.merge(metrics_df, lat_df, on="Search Approach", how="left")
    
    print("\n======================= FINAL UNIFIED RAG PERFORMANCE SCORECARD =======================")
    print(metrics_df.to_string(index=False))
    print("=======================================================================================")
    
    metrics_df.to_excel("./results/final_metric_scorecard.xlsx", index=False)

if __name__ == "__main__":
    calculate_rag_metrics()