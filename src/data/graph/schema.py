"""
Graph Schema Definition - Single Source of Truth

This module defines the graph structure used throughout the system.
Any changes to node types or edge types should be made HERE ONLY.

Used by:
- graph_builder.py (building the graph)
- sage.py, hgt.py (GNN model initialization)
- create_artifacts.py (creating node/edge parquet files)
"""
from typing import Tuple, List

# =============================================================================
# NODE TYPES
# =============================================================================
# Each node type represents an entity in our fraud detection graph

NODE_TYPES: List[str] = [
    "user",      # Platform users who post listings
    "listing",   # Real estate listings (target node for fraud prediction)
    "email",     # Email addresses (contact, billing)
    "phone",     # Phone numbers (contact, billing)
    "address",   # Physical addresses (listing location, billing)
    "ip",        # IP addresses used during submission
]

# =============================================================================
# EDGE TYPES
# =============================================================================
# Each edge type represents a relationship between entities
# Format: (source_type, relation_name, target_type)

EDGE_TYPES: List[Tuple[str, str, str]] = [
    # User relationships
    ("user", "posts", "listing"),           # User posts a listing
    ("user", "has_email", "email"),         # User's account email
    ("user", "uses", "ip"),                 # User's IP addresses
    
    # Listing → Email relationships
    ("listing", "has_contact_email", "email"),   # Contact email on listing
    ("listing", "has_billing_email", "email"),   # Billing email for payment
    
    # Listing → Phone relationships (unified: billing + lister phone)
    ("listing", "has_phone", "phone"),      # Phone on listing (coalesced)
    
    # Listing → Address relationships
    ("listing", "located_at", "address"),       # Listing location
    ("listing", "has_billing_addr", "address"), # Billing address
]

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

# Number of node and edge types
NUM_NODE_TYPES: int = len(NODE_TYPES)
NUM_EDGE_TYPES: int = len(EDGE_TYPES)

# PyTorch Geometric metadata format: (node_types, edge_types)
GRAPH_METADATA: Tuple[List[str], List[Tuple[str, str, str]]] = (
    NODE_TYPES,
    EDGE_TYPES,
)


def get_metadata() -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Get graph metadata in PyTorch Geometric format.
    
    Returns:
        Tuple of (node_types, edge_types)
    
    Example:
        >>> metadata = get_metadata()
        >>> model = SAGEWrapper(metadata=metadata, ...)
    """
    return GRAPH_METADATA


def validate_graph(data) -> bool:
    """
    Validate that a HeteroData graph matches the expected schema.
    
    Args:
        data: PyTorch Geometric HeteroData object
        
    Returns:
        True if valid, raises ValueError if not
    """
    # Check node types
    actual_node_types = set(data.node_types)
    expected_node_types = set(NODE_TYPES)
    
    if actual_node_types != expected_node_types:
        missing = expected_node_types - actual_node_types
        extra = actual_node_types - expected_node_types
        raise ValueError(
            f"Graph node types mismatch.\n"
            f"  Missing: {missing}\n"
            f"  Extra: {extra}"
        )
    
    # Check edge types
    actual_edge_types = set(data.edge_types)
    expected_edge_types = set(EDGE_TYPES)
    
    if actual_edge_types != expected_edge_types:
        missing = expected_edge_types - actual_edge_types
        extra = actual_edge_types - expected_edge_types
        raise ValueError(
            f"Graph edge types mismatch.\n"
            f"  Missing: {missing}\n"
            f"  Extra: {extra}"
        )
    
    return True

