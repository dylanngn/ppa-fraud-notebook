"""
Data extraction module.
Handles fetching data from the database in chunks.
"""
import polars as pl
from typing import Optional
import logging

logger = logging.getLogger(__name__)

def build_chunk_query(schema: str, start_date: str, end_date: str) -> str:
    """
    Build SQL query for a date range chunk.
    
    Args:
        schema: Database schema name
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        SQL query string
    """
    return f"""
    SELECT 
        i.object_reference,
        i.user_ip_address,
        i.listing::text as listing_json,
        i.fraud_flag,
        i.auto_approval_criteria::text as auto_approval_criteria_json,
        i.first_published_date,
        i.customer_segment,
        i.selected_bundle::text as selected_bundle_json,
        i.created_at as listing_created_at,
        i.platform as listing_platform,
        sh.transition_timestamp as submission_at,
        u.owner_id,
        u.platform as user_platform,
        u.created_at as account_created_at,
        u.contact_emails
    FROM {schema}.insertions i
    JOIN (
        SELECT insertion_id, min(transition_timestamp) as transition_timestamp
        FROM {schema}.status_history
        WHERE status_from = 'DRAFT' AND status_to = 'PENDING_APPROVAL'
        GROUP BY insertion_id
    ) sh ON i.id = sh.insertion_id
    LEFT JOIN {schema}.users u ON i.listing->'legacy'->>'personId' = u.owner_id
    WHERE sh.transition_timestamp >= '{start_date}' 
    AND sh.transition_timestamp < '{end_date}'
    AND i.platform <> 're.smg'
    AND meta -> 'migratedFromPersonId' is null
    ORDER BY submission_at
    """

def fetch_chunk(
    db_uri: str,
    query: str
) -> Optional[pl.DataFrame]:
    """
    Fetch a single chunk of data from the database.
    
    Args:
        db_uri: Database URI
        query: SQL query
        
    Returns:
        DataFrame with fetched data, or None if empty
    """
    try:
        df = pl.read_database_uri(query, db_uri, engine="connectorx")
        if len(df) == 0:
            return None
        return df
    except Exception as e:
        logger.error(f"Error fetching chunk: {e}")
        raise
