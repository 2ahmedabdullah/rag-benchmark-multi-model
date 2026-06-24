# evaluation/metrics.py


import os
import pandas as pd
from tqdm import tqdm  # Import the progress bar library

def get_predictions_dataframe(qrels, strategies_dict, output_path="./results/raw_search_predictions.xlsx", top_k=10):
    """
    Dynamically compiles any dictionary of search strategies into a unified 
    evaluation DataFrame, safely matching data types against ground truth.
    """
    all_rows = []
    
    # Loop dynamically through whatever strategies are passed in
    for strategy_name, results in strategies_dict.items():
        for q_id, doc_scores in results.items():
            # Sort scores and grab the specified top_k slice
            top_slice = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            for rank, (doc_id, score) in enumerate(top_slice, start=1):
                # String cast protection to prevent FAISS int vs BM25 str mismatches
                q_id_str = str(q_id).strip()
                doc_id_str = str(doc_id).strip()
                
                # Check against ground truth qrels safely
                is_true_hit = qrels.get(q_id_str, {}).get(doc_id_str, 0) > 0
                
                all_rows.append({
                    "Model": "msmarco-distilbert-base-v4",
                    "Search Approach": strategy_name,
                    "Query ID": q_id_str,
                    "Rank": rank,
                    "Document ID": doc_id_str,
                    "Similarity Score": float(score),
                    "True Hit?": is_true_hit
                })
                
    df = pd.DataFrame(all_rows)
    df.to_excel(output_path, index=False)
    return df