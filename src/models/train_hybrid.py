import os

import polars as pl
import torch

from src.models.train_baseline import (
    feature_engineering,
    train_accumulating_window,
)


def load_embeddings(df, model_name="hgt"):
    """
    Loads GNN embeddings and merges them with the DataFrame.
    """
    embedding_path = f"artifacts/embeddings_{model_name}.pt"
    print(f"Loading Embeddings from {embedding_path}...")
    
    if not os.path.exists(embedding_path):
        raise FileNotFoundError(f"Embeddings not found at {embedding_path}. Run train_embeddings.py --model {model_name} first.")
        
    # Load Tensor
    embeddings = torch.load(embedding_path).numpy()
    
    # We need to ensure alignment.
    # train_embeddings.py loaded 'nodes_listing.parquet' and used it in that order.
    # So embeddings[i] corresponds to the i-th row in nodes_listing.parquet.
    # We must load nodes_listing.parquet again and attach embeddings by index.
    
    df_listing = pl.read_parquet("artifacts/nodes_listing.parquet")
    
    # Create a DataFrame of embeddings
    # Column names: embed_0, embed_1, ...
    embed_cols = [f"embed_{i}" for i in range(embeddings.shape[1])]
    df_embed = pl.DataFrame(embeddings, schema=embed_cols)
    
    # Horizontally stack (polars hstack or just with_columns if lengths match)
    if len(df_listing) != len(df_embed):
        raise ValueError(f"Mismatch: Listings {len(df_listing)} vs Embeddings {len(df_embed)}")
        
    df_listing = df_listing.hstack(df_embed)
    
    # Now we join this enriched listing DF with users, just like in baseline
    df_users = pl.read_parquet("artifacts/nodes_user.parquet")
    df = df_listing.join(df_users, on="user_id", how="left")
    
    return df, embed_cols


def main(model_name="hgt"):
    """
    Train hybrid model with accumulating window and MLflow tracking.
    
    Always enabled:
    - MLflow tracking
    - Model registration
    - Accumulating window
    
    Args:
        model_name: GNN model type (hgt, gat, gcn, hgt_rte)
    """
    if not os.path.exists("artifacts/nodes_listing.parquet"):
        print("Artifacts not found. Please run ETL.py first.")
        return
    
    # Load Data & Embeddings
    df, embed_cols = load_embeddings(None, model_name)
    
    # Feature Engineering (Tabular)
    df = feature_engineering(df)
    
    # Train with accumulating window + MLflow
    print(f"Training hybrid model with {len(embed_cols)} embedding features...")
    result = train_accumulating_window(
        df,
        extra_features=embed_cols,
        model_name=f"hybrid_{model_name}"
    )
    
    return result


if __name__ == "__main__":
    import typer
    typer.run(main)
