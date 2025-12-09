"""
Experiment 10C: Association Rule Mining

Research Gap: Supervisor feedback (Section 2.3) requires interpretable fraud patterns.

Methods:
    - Apriori algorithm
    - FP-Growth (faster)

Run with:
    python -m src.experiments.exp10_association_rules

Output:
    - artifacts/unsupervised/association_rules.csv
    - artifacts/unsupervised/top_fraud_rules.csv
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig

from src.models.utils.common import setup_mlflow
from src.utils.hydra_utils import resolve_path

# Try to import mlxtend for association rules
try:
    from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False
    logging.warning("mlxtend not installed. pip install mlxtend for association rules.")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
ARTIFACTS_DIR = resolve_path("artifacts")
OUTPUT_DIR = ARTIFACTS_DIR / "unsupervised"
RAW_INSERTIONS = ARTIFACTS_DIR / "raw_insertions.parquet"


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare data for association rule mining."""
    logger.info("Loading data...")
    
    df = pl.read_parquet(RAW_INSERTIONS)
    
    # Add fraud label
    df = df.with_columns(
        pl.col("fraud_flag").is_not_null().cast(pl.Int8).alias("is_fraud")
    )
    
    # Select key categorical/boolean features for rule mining
    # Focus on features that can be discretized meaningfully
    feature_mapping = {
        # Payment & Bundle
        'listing.lister.billing.payment.paymentType': 'payment_type',
        'bundle.tier': 'bundle_tier',
        # Platform
        'listing_platform': 'platform',
        'listing.offerType': 'offer_type',
        # Boolean features (already binary)
        'listing.characteristics.hasBalcony': 'has_balcony',
        'listing.characteristics.hasParking': 'has_parking',
        'listing.characteristics.hasElevator': 'has_elevator',
        'listing.characteristics.isNewBuilding': 'is_new_building',
        'listing.characteristics.hasGarage': 'has_garage',
        # Location
        'listing.address.region': 'region',
    }
    
    # Select available columns
    available = [col for col in feature_mapping.keys() if col in df.columns]
    
    # Also add is_fraud
    pdf = df.select(available + ["is_fraud"]).to_pandas()
    
    # Rename columns
    rename_map = {k: v for k, v in feature_mapping.items() if k in available}
    pdf = pdf.rename(columns=rename_map)
    
    return pdf, pdf["is_fraud"]


def discretize_features(df: pd.DataFrame) -> pd.DataFrame:
    """Convert features to binary for association rule mining."""
    logger.info("Discretizing features...")
    
    binary_df = pd.DataFrame()
    
    for col in df.columns:
        if col == 'is_fraud':
            binary_df['FRAUD'] = df[col].astype(bool)
            continue
        
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            # One-hot encode categorical
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
            # Keep only top categories (limit columns)
            top_cats = df[col].value_counts().head(5).index
            for cat in top_cats:
                col_name = f"{col}_{cat}"
                if col_name in dummies.columns:
                    binary_df[col_name] = dummies[col_name]
        elif df[col].dtype == 'bool' or set(df[col].dropna().unique()).issubset({0, 1, True, False}):
            # Already binary
            binary_df[f"{col}_TRUE"] = df[col].fillna(False).astype(bool)
        else:
            # Numeric - discretize into bins
            try:
                binary_df[f"{col}_HIGH"] = df[col] > df[col].median()
            except:
                pass
    
    logger.info(f"Created {len(binary_df.columns)} binary features")
    return binary_df


def run_fpgrowth(binary_df: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    """Run FP-Growth for frequent itemset mining."""
    logger.info(f"Running FP-Growth (min_support={min_support})...")
    
    # Run FP-Growth
    frequent_itemsets = fpgrowth(binary_df, min_support=min_support, use_colnames=True)
    logger.info(f"Found {len(frequent_itemsets)} frequent itemsets")
    
    return frequent_itemsets


def generate_rules(frequent_itemsets: pd.DataFrame, min_confidence: float = 0.5) -> pd.DataFrame:
    """Generate association rules from frequent itemsets."""
    logger.info(f"Generating rules (min_confidence={min_confidence})...")
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    logger.info(f"Generated {len(rules)} rules")
    
    return rules


def filter_fraud_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Filter rules that predict fraud."""
    # Rules where consequent contains FRAUD
    fraud_rules = rules[rules['consequents'].apply(lambda x: 'FRAUD' in x)]
    
    # Sort by lift (how much more likely fraud is given the antecedent)
    fraud_rules = fraud_rules.sort_values('lift', ascending=False)
    
    # Format for readability
    fraud_rules['antecedent_str'] = fraud_rules['antecedents'].apply(lambda x: ' AND '.join(sorted(x)))
    fraud_rules['consequent_str'] = fraud_rules['consequents'].apply(lambda x: ' AND '.join(sorted(x)))
    
    return fraud_rules


def run_association_rules(config: DictConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full association rule mining."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 10C: ASSOCIATION RULE MINING")
    logger.info("=" * 70)
    
    if not HAS_MLXTEND:
        logger.error("mlxtend not installed. Run: pip install mlxtend")
        return pd.DataFrame(), pd.DataFrame()
    
    # Load data
    df, y = load_data()
    logger.info(f"Loaded {len(df):,} samples")
    logger.info(f"Fraud rate: {y.mean()*100:.2f}%")
    
    # Discretize features
    binary_df = discretize_features(df)
    
    # Sample for faster processing (association rules can be slow)
    if len(binary_df) > 100000:
        logger.info(f"Sampling 100,000 records for speed...")
        binary_df = binary_df.sample(n=100000, random_state=42)
    
    # Run FP-Growth
    frequent_itemsets = run_fpgrowth(binary_df, min_support=0.01)
    
    if len(frequent_itemsets) == 0:
        logger.warning("No frequent itemsets found. Try lowering min_support.")
        return pd.DataFrame(), pd.DataFrame()
    
    # Generate rules
    all_rules = generate_rules(frequent_itemsets, min_confidence=0.1)
    
    if len(all_rules) == 0:
        logger.warning("No rules generated. Try lowering min_confidence.")
        return pd.DataFrame(), frequent_itemsets
    
    # Filter fraud-related rules
    fraud_rules = filter_fraud_rules(all_rules)
    logger.info(f"Found {len(fraud_rules)} fraud-predicting rules")
    
    return fraud_rules, all_rules


def create_summary(fraud_rules: pd.DataFrame, overall_fraud_rate: float) -> pd.DataFrame:
    """Create human-readable summary of top rules."""
    if len(fraud_rules) == 0:
        return pd.DataFrame()
    
    top_rules = fraud_rules.head(20)[['antecedent_str', 'support', 'confidence', 'lift']].copy()
    top_rules.columns = ['Rule (IF)', 'Support', 'Confidence', 'Lift']
    top_rules['Support'] = top_rules['Support'].apply(lambda x: f"{x*100:.2f}%")
    top_rules['Confidence'] = top_rules['Confidence'].apply(lambda x: f"{x*100:.1f}%")
    top_rules['Lift'] = top_rules['Lift'].apply(lambda x: f"{x:.2f}x")
    
    return top_rules


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Main entry point."""
    logger.info("=" * 70)
    logger.info("EXPERIMENT 10C: ASSOCIATION RULE MINING")
    logger.info("=" * 70)
    
    if not HAS_MLXTEND:
        logger.error("mlxtend not installed. Run: pip install mlxtend")
        return None
    
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_name = cfg.get('experiment_name', 'unsupervised-pattern-exp10')
    setup_mlflow(experiment_name)
    
    start_time = time.time()
    
    with mlflow.start_run(
        run_name=f"exp10c_association_{datetime.now().strftime('%Y%m%d_%H%M')}",
        tags={"experiment_type": "unsupervised", "sub_experiment": "10C"}
    ):
        # Run association rule mining
        fraud_rules, all_rules = run_association_rules(cfg)
        
        if len(fraud_rules) == 0:
            logger.warning("No fraud rules found")
            return None
        
        # Log metrics
        mlflow.log_metrics({
            'n_total_rules': len(all_rules),
            'n_fraud_rules': len(fraud_rules),
            'max_lift': fraud_rules['lift'].max(),
            'avg_confidence': fraud_rules['confidence'].mean()
        })
        
        # Save all rules
        rules_path = OUTPUT_DIR / "association_rules.csv"
        all_rules.to_csv(rules_path, index=False)
        logger.info(f"Saved all rules: {rules_path}")
        
        # Save fraud rules
        fraud_rules_path = OUTPUT_DIR / "fraud_association_rules.csv"
        fraud_rules[['antecedent_str', 'support', 'confidence', 'lift']].to_csv(fraud_rules_path, index=False)
        logger.info(f"Saved fraud rules: {fraud_rules_path}")
        
        # Create summary
        summary = create_summary(fraud_rules, 0.082)
        summary_path = OUTPUT_DIR / "top_fraud_rules.csv"
        summary.to_csv(summary_path, index=False)
        logger.info(f"Saved summary: {summary_path}")
        
        # Log artifacts
        mlflow.log_artifacts(str(OUTPUT_DIR))
        
        elapsed = time.time() - start_time
        mlflow.log_metric('total_time_seconds', elapsed)
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("TOP FRAUD-PREDICTING RULES")
        logger.info("=" * 70)
        logger.info("(Read as: IF <antecedent> THEN FRAUD with <confidence> confidence)")
        print(summary.to_string(index=False))
        
        logger.info(f"\nTotal time: {elapsed:.1f}s")
        
        # Key finding
        if len(fraud_rules) > 0:
            best_rule = fraud_rules.iloc[0]
            logger.info(f"\n🔑 Best Rule: IF {best_rule['antecedent_str']}")
            logger.info(f"   THEN FRAUD with {best_rule['confidence']*100:.1f}% confidence ({best_rule['lift']:.1f}x lift)")
    
    return fraud_rules


if __name__ == "__main__":
    main()
