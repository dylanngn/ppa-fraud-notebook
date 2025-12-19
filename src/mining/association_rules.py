"""
Association Rule Mining for Fraud Pattern Discovery.

Uses the Apriori algorithm to find frequent itemsets and association rules
that indicate fraud patterns. This is a classic Data Mining technique.

Example patterns discovered:
- {disposable_email, vpn} -> fraud (high confidence)
- {small_cluster, public_proxy} -> fraud
"""

import polars as pl
import numpy as np
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from itertools import combinations
import logging

logger = logging.getLogger(__name__)


@dataclass
class AssociationRuleConfig:
    """Configuration for association rule mining."""
    
    min_support: float = 0.001  # Minimum support threshold
    min_confidence: float = 0.5  # Minimum confidence threshold
    min_lift: float = 2.0  # Minimum lift threshold
    max_itemset_size: int = 4  # Maximum itemset size


@dataclass
class AssociationRule:
    """Represents a discovered association rule."""
    antecedent: Tuple[str, ...]
    consequent: str  # Always "is_fraud" for our use case
    support: float
    confidence: float
    lift: float
    count: int


class FraudPatternMiner:
    """
    Mines association rules from fraud data to discover patterns.
    
    Uses a simplified Apriori-like approach focused on fraud detection.
    """
    
    def __init__(self, config: AssociationRuleConfig = None):
        self.config = config or AssociationRuleConfig()
        self.rules: List[AssociationRule] = []
        self.frequent_itemsets: Dict[Tuple[str, ...], float] = {}
    
    def fit(self, df: pl.DataFrame, binary_columns: List[str]) -> "FraudPatternMiner":
        """
        Mine association rules from the data.
        
        Args:
            df: DataFrame with binary features and is_fraud label
            binary_columns: List of binary feature column names
        """
        logger.info(f"Mining association rules from {len(df)} records")
        
        n_records = len(df)
        fraud_rate = df["is_fraud"].mean()
        
        # Convert to transactions (list of active items per record)
        logger.info("Converting to transactions...")
        
        # Filter to available columns
        available_cols = [c for c in binary_columns if c in df.columns]
        logger.info(f"Using {len(available_cols)} binary features")
        
        # Step 1: Find frequent 1-itemsets
        item_support = {}
        for col in available_cols:
            support = df[col].sum() / n_records
            if support >= self.config.min_support:
                item_support[(col,)] = support
        
        logger.info(f"Found {len(item_support)} frequent 1-itemsets")
        
        # Step 2: Find frequent k-itemsets (up to max_itemset_size)
        for k in range(2, self.config.max_itemset_size + 1):
            candidates = self._generate_candidates(list(item_support.keys()), k)
            
            new_itemsets = {}
            for candidate in candidates:
                # Calculate support
                mask = pl.lit(True)
                for item in candidate:
                    mask = mask & pl.col(item)
                
                support = df.filter(mask).height / n_records
                
                if support >= self.config.min_support:
                    new_itemsets[candidate] = support
            
            if not new_itemsets:
                break
                
            item_support.update(new_itemsets)
            logger.info(f"Found {len(new_itemsets)} frequent {k}-itemsets")
        
        self.frequent_itemsets = item_support
        
        # Step 3: Generate association rules for fraud
        logger.info("Generating association rules for fraud...")
        
        for itemset, support in item_support.items():
            # Calculate confidence: P(fraud | itemset)
            mask = pl.lit(True)
            for item in itemset:
                mask = mask & pl.col(item)
            
            itemset_df = df.filter(mask)
            if itemset_df.height < 10:  # Minimum support count
                continue
            
            fraud_in_itemset = itemset_df["is_fraud"].sum()
            confidence = fraud_in_itemset / itemset_df.height
            
            # Calculate lift
            lift = confidence / fraud_rate
            
            if confidence >= self.config.min_confidence and lift >= self.config.min_lift:
                rule = AssociationRule(
                    antecedent=itemset,
                    consequent="is_fraud",
                    support=support,
                    confidence=confidence,
                    lift=lift,
                    count=itemset_df.height,
                )
                self.rules.append(rule)
        
        # Sort by lift
        self.rules.sort(key=lambda r: r.lift, reverse=True)
        logger.info(f"Discovered {len(self.rules)} association rules")
        
        return self
    
    def _generate_candidates(
        self, 
        itemsets: List[Tuple[str, ...]], 
        k: int
    ) -> List[Tuple[str, ...]]:
        """Generate candidate k-itemsets from (k-1)-itemsets."""
        items = set()
        for itemset in itemsets:
            items.update(itemset)
        
        candidates = []
        for combo in combinations(sorted(items), k):
            candidates.append(combo)
        
        return candidates
    
    def get_top_rules(self, n: int = 20) -> List[AssociationRule]:
        """Get top N rules by lift."""
        return self.rules[:n]
    
    def print_rules(self, n: int = 20):
        """Print top rules in readable format."""
        print(f"\nTop {n} Association Rules for Fraud:")
        print("-" * 80)
        print(f"{'Antecedent':<45} {'Conf':>8} {'Lift':>8} {'Count':>8}")
        print("-" * 80)
        
        for rule in self.get_top_rules(n):
            ant_str = " & ".join(rule.antecedent)
            if len(ant_str) > 43:
                ant_str = ant_str[:40] + "..."
            print(f"{ant_str:<45} {rule.confidence:>7.1%} {rule.lift:>7.1f}x {rule.count:>8}")
    
    def create_rule_features(
        self, 
        df: pl.DataFrame, 
        top_n: int = 10
    ) -> pl.DataFrame:
        """
        Create binary features from top association rules.
        
        Each rule becomes a feature indicating if the record matches the pattern.
        """
        result = df
        
        for i, rule in enumerate(self.get_top_rules(top_n)):
            # Create binary feature for this rule
            mask = pl.lit(True)
            for item in rule.antecedent:
                if item in df.columns:
                    mask = mask & pl.col(item)
            
            feature_name = f"rule_{i+1}_match"
            result = result.with_columns(mask.alias(feature_name))
        
        return result
