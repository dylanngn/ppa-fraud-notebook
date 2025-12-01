"""
Data Loading Utilities - Single Source of Truth

This module provides the canonical data loading function used across
all training modules (XGBoost, GNN, hyperopt).
"""
import polars as pl

from src.utils.hydra_utils import resolve_path


def load_data() -> pl.DataFrame:
    """
    Load and join listing and user data for training.
    
    Returns:
        DataFrame with listings joined to users on user_id
        
    Raises:
        FileNotFoundError: If required parquet files don't exist
    """
    listing_path = resolve_path("artifacts/nodes_listing.parquet")
    user_path = resolve_path("artifacts/nodes_user.parquet")
    
    if not listing_path.exists():
        raise FileNotFoundError(
            f"Listing nodes not found: {listing_path}. Run 'make etl' first."
        )
    if not user_path.exists():
        raise FileNotFoundError(
            f"User nodes not found: {user_path}. Run 'make etl' first."
        )
    
    df_listings = pl.read_parquet(listing_path)
    df_users = pl.read_parquet(user_path)
    
    return df_listings.join(df_users, on="user_id", how="left")

