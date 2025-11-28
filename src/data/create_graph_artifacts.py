"""
Creates graph artifacts (nodes and edges) from flattened and anonymized data.

This module creates the parquet files used by both:
- GNN models (via graph_builder.py)
- XGBoost models (via graph_features.py)

The input data should already be flattened and anonymized (from fetch_raw_insertions).
"""

import polars as pl
import os


def create_nodes_and_edges(df_listings: pl.DataFrame):
    """
    Creates the nodes and edges for the Heterogeneous Graph.
    
    Works directly with flattened data that uses dot-notation field names.
    The data should already be flattened and anonymized from fetch_raw_insertions().
    
    Args:
        df_listings: DataFrame with flattened and anonymized data (dot-notation columns)
    """
    print("Creating Nodes and Edges from flattened data...")
    
    # Helper to safely get column with fallback
    def get_col(col_name: str, fallback_col: str = None):
        """Get column expression, with optional fallback if primary doesn't exist."""
        if col_name in df_listings.columns:
            return pl.col(col_name)
        elif fallback_col and fallback_col in df_listings.columns:
            return pl.col(fallback_col)
        else:
            # Return None literal if column doesn't exist
            return pl.lit(None).cast(pl.Utf8)
    
    # Helper to get platform (coalesce listing_platform and user_platform)
    platform_col = pl.coalesce([
        get_col("listing_platform"),
        get_col("user_platform")
    ]).alias("platform")
    
    # Helper to get fraud flags
    fraud_flag_col = get_col("fraud_flag")
    is_fraud_col = fraud_flag_col.is_not_null().alias("is_fraud")
    first_published_col = get_col("first_published_date")
    
    # Create user nodes
    nodes_user = df_listings.select([
        pl.col("owner_id").alias("user_id"),
        pl.col("account_created_at"),
        pl.lit(None).cast(pl.Utf8).alias("email_domain")
    ]).drop_nulls(subset=["user_id"]).unique(subset=["user_id"])
    
    # Create listing nodes (using dot-notation fields)
    nodes_listing = df_listings.select([
        pl.col("object_reference").alias("insertion_id"),
        pl.col("owner_id").alias("user_id"),
        pl.col("account_created_at"),
        platform_col,
        get_col("listing.offerType").alias("offer_type"),
        get_col("listing.prices.buy.price").alias("price_buy"),
        get_col("listing.prices.rent.gross").alias("price_rent_gross"),
        get_col("listing.prices.rent.net").alias("price_rent_net"),
        get_col("listing.characteristics.livingSpace").alias("living_space"),
        get_col("listing.characteristics.numberOfRooms").alias("rooms"),
        get_col("listing.characteristics.yearBuilt").alias("year_built"),
        get_col("listing.characteristics.floor").alias("floor"),
        get_col("listing.characteristics.numberOfFloors").alias("num_floors"),
        get_col("listing.characteristics.isNewBuilding").alias("is_new"),
        get_col("listing.characteristics.hasBalcony").alias("has_balcony"),
        get_col("listing.characteristics.hasElevator").alias("has_elevator"),
        get_col("listing.characteristics.hasParking").alias("has_parking"),
        get_col("listing.characteristics.isOldBuilding").alias("is_old"),
        # Address fields (use hash columns from anonymization)
        get_col("listing.address.postalCode", "listing.address.postalCode.hash").alias("zip_code"),
        get_col("listing.address.locality", "listing.address.city_hash").alias("city"),
        get_col("listing.address.street", "listing.address.street_hash").alias("street"),
        get_col("listing.address.country", "listing.address.country_hash").alias("country"),
        get_col("listing.address.region").alias("region"),
        get_col("listing.address.geoCoordinates.latitude").alias("latitude"),
        get_col("listing.address.geoCoordinates.longitude").alias("longitude"),
        get_col("bundle.period").alias("bundle_period"),
        get_col("bundle.tier").alias("bundle_tier"),
        get_col("listing.lister.billing.payment.paymentType").alias("payment_type"),
        pl.col("customer_segment"),
        get_col("listing.localization.primary").alias("language"),
        # Description - coalesce multiple language options
        pl.coalesce([
            get_col("listing.localization.de.text.description"),
            get_col("listing.localization.en.text.description"),
            get_col("listing.localization.fr.text.description"),
            get_col("listing.localization.it.text.description"),
            get_col("listing.descriptions.description")
        ]).alias("description_text"),
        is_fraud_col,
        pl.col("submission_at"),
        # Extract seon_approved from flattened auto_approval_criteria (may be mostly null)
        # This is needed for Seon baseline evaluation even if coverage is low
        get_col("auto_approval_criteria.criteria.seonApproved").alias("seon_approved"),
        fraud_flag_col,
        first_published_col
    ]).unique(subset=["insertion_id"])
    
    # IP nodes: use hashed IP address
    nodes_ip = df_listings.select(
        get_col("user_ip_address_hash", "user_ip_address").alias("user_ip_address")
    ).unique().drop_nulls()
    
    # Email nodes: collect from all email hash columns
    # NOTE: Removed inquiry/viewing email fields (99.89%+ null coverage per data quality report)
    email_cols = [
        "listing.lister.email.hash",  # 99.976% coverage
        "listing.lister.billing.email.hash",  # 98.002% coverage
    ]
    
    emails_from_users = df_listings.select(
        get_col("contact_emails_hash", "contact_emails")
        .str.split(",")
        .explode()
        .str.strip_chars()
        .alias("email")
    ).drop_nulls()
    
    emails_from_listings = []
    for col in email_cols:
        if col in df_listings.columns:
            emails_from_listings.append(
                df_listings.select(pl.col(col).alias("email"))
            )
    
    nodes_email = pl.concat([emails_from_users] + emails_from_listings).unique().drop_nulls()
    
    # Phone nodes: collect from all phone hash columns
    # NOTE: Removed low-coverage fields per data quality report:
    # - phoneMobile.hash: 99.78% null
    # - contacts.inquiry/viewing phone fields: 99.89%+ null
    phone_cols = [
        "listing.lister.phone.hash",  # 69.978% coverage
        "listing.lister.billing.phoneDay.hash",  # 97.995% coverage (best)
    ]
    
    phones_from_listings = []
    for col in phone_cols:
        if col in df_listings.columns:
            phones_from_listings.append(
                df_listings.select(pl.col(col).alias("phone"))
            )
    
    nodes_phone = pl.concat(phones_from_listings).unique().drop_nulls() if phones_from_listings else pl.DataFrame({"phone": []})
    
    # Address nodes
    def create_address_df(df, street_col, zip_col, city_col, country_col, lat_col=None, lon_col=None):
        cols = [
            get_col(street_col).fill_null("").alias("street"),
            get_col(zip_col).fill_null("").alias("zip"),
            get_col(city_col).fill_null("").alias("city"),
            get_col(country_col).fill_null("").alias("country")
        ]
        if lat_col and lon_col:
            cols.append(get_col(lat_col).alias("latitude"))
            cols.append(get_col(lon_col).alias("longitude"))
        else:
            cols.append(pl.lit(None).cast(pl.Float64).alias("latitude"))
            cols.append(pl.lit(None).cast(pl.Float64).alias("longitude"))
            
        return df.select(cols).with_columns(
            (pl.col("country") + "_" + pl.col("zip") + "_" + pl.col("city") + "_" + pl.col("street")).alias("address_id")
        ).drop_nulls(subset=["address_id"]).unique(subset=["address_id"])
    
    # Property address (use hash columns if available)
    addr_property = create_address_df(
        df_listings,
        "listing.address.street.hash", "listing.address.postalCode.hash",
        "listing.address.city_hash", "listing.address.country_hash",
        "listing.address.geoCoordinates.latitude", "listing.address.geoCoordinates.longitude"
    )
    
    # Billing address (use hash columns if available)
    addr_billing = create_address_df(
        df_listings,
        "listing.lister.billing.address.street_hash",
        "listing.lister.billing.address.zip_hash",
        "listing.lister.billing.address.city_hash",
        "listing.lister.billing.address.country_hash"
    )
    
    nodes_address = pl.concat([addr_property, addr_billing]).unique(subset=["address_id"])
    
    # Person nodes (from name hash if available)
    # NOTE: inquiry.name_hash has 99.98% null coverage, so person nodes will be sparse
    # Keeping for now but may want to remove if not useful
    name_hash_col = "listing.lister.contacts.inquiry.name_hash"
    if name_hash_col in df_listings.columns:
        nodes_person = df_listings.select([
            pl.col(name_hash_col).alias("person_name")
        ]).filter(pl.col("person_name").is_not_null()).unique()
    else:
        # Fallback: try to construct from given/family names (though they should be anonymized)
        nodes_person = df_listings.select([
            (
                get_col("listing.lister.contacts.inquiry.givenName").fill_null("") + " " +
                get_col("listing.lister.contacts.inquiry.familyName").fill_null("")
            ).str.strip_chars().alias("person_name")
        ]).filter(pl.col("person_name") != "").unique()
    
    # Create edges
    def create_edge_df(src_col, dst_col, src_name="source", dst_name="target"):
        return df_listings.select([
            pl.col(src_col).alias(src_name),
            get_col(dst_col).alias(dst_name)
        ]).drop_nulls().unique()
    
    edges_user_posts = df_listings.select([
        pl.col("owner_id").alias("source"),
        pl.col("object_reference").alias("target")
    ]).drop_nulls().unique()
    
    edges_user_ip = df_listings.select([
        pl.col("owner_id").alias("source"),
        get_col("user_ip_address_hash", "user_ip_address").alias("target")
    ]).drop_nulls().unique()
    
    edges_user_email = df_listings.select([
        pl.col("owner_id").alias("source"),
        get_col("contact_emails_hash", "contact_emails")
        .str.split(",")
        .explode()
        .str.strip_chars()
        .alias("target")
    ]).drop_nulls().unique()
    
    # Listing-email edges
    # NOTE: Removed inquiry_email edge (99.89% null coverage per data quality report)
    edges_listing_contact_email = create_edge_df(
        "object_reference",
        "listing.lister.email.hash"  # 99.976% coverage
    )
    edges_listing_billing_email = create_edge_df(
        "object_reference",
        "listing.lister.billing.email.hash"  # 98.002% coverage
    )
    
    # Listing-phone edges
    edges_listing_contact_phone = create_edge_df(
        "object_reference",
        "listing.lister.phone.hash"
    )
    edges_listing_billing_phone = create_edge_df(
        "object_reference",
        "listing.lister.billing.phoneDay.hash"
    )
    
    # Listing-address edges
    edges_listing_located_at = df_listings.select([
        pl.col("object_reference").alias("source"),
        (
            get_col("listing.address.country_hash").fill_null("") + "_" +
            get_col("listing.address.postalCode.hash").fill_null("") + "_" +
            get_col("listing.address.city_hash").fill_null("") + "_" +
            get_col("listing.address.street_hash").fill_null("")
        ).alias("target")
    ]).drop_nulls().unique()
    
    edges_listing_billing_addr = df_listings.select([
        pl.col("object_reference").alias("source"),
        (
            get_col("listing.lister.billing.address.country_hash").fill_null("") + "_" +
            get_col("listing.lister.billing.address.zip_hash").fill_null("") + "_" +
            get_col("listing.lister.billing.address.city_hash").fill_null("") + "_" +
            get_col("listing.lister.billing.address.street_hash").fill_null("")
        ).alias("target")
    ]).drop_nulls().unique()
    
    # Listing-person edges
    edges_listing_person = df_listings.select([
        pl.col("object_reference").alias("source"),
        get_col("listing.lister.contacts.inquiry.name_hash").alias("target")
    ]).filter(pl.col("target").is_not_null()).unique()
    
    # Save all artifacts
    print("Saving Parquet Artifacts...")
    os.makedirs("artifacts", exist_ok=True)
    
    nodes_user.write_parquet("artifacts/nodes_user.parquet")
    nodes_listing.write_parquet("artifacts/nodes_listing.parquet")
    nodes_ip.write_parquet("artifacts/nodes_ip.parquet")
    nodes_email.write_parquet("artifacts/nodes_email.parquet")
    nodes_phone.write_parquet("artifacts/nodes_phone.parquet")
    nodes_address.write_parquet("artifacts/nodes_address.parquet")
    nodes_person.write_parquet("artifacts/nodes_person.parquet")
    
    edges_user_posts.write_parquet("artifacts/edges_user_posts_listing.parquet")
    edges_user_ip.write_parquet("artifacts/edges_user_uses_ip.parquet")
    edges_user_email.write_parquet("artifacts/edges_user_has_email.parquet")
    
    edges_listing_contact_email.write_parquet("artifacts/edges_listing_contact_email.parquet")
    edges_listing_billing_email.write_parquet("artifacts/edges_listing_billing_email.parquet")
    # NOTE: Removed inquiry_email edge (low coverage)
    
    edges_listing_contact_phone.write_parquet("artifacts/edges_listing_contact_phone.parquet")
    edges_listing_billing_phone.write_parquet("artifacts/edges_listing_billing_phone.parquet")
    
    edges_listing_located_at.write_parquet("artifacts/edges_listing_located_at.parquet")
    edges_listing_billing_addr.write_parquet("artifacts/edges_listing_billing_addr.parquet")
    
    edges_listing_person.write_parquet("artifacts/edges_listing_has_person.parquet")
    
    print("Graph artifacts created successfully!")
    
    return (
        nodes_user, nodes_listing, nodes_ip, nodes_email, nodes_phone, nodes_address, nodes_person,
        edges_user_posts, edges_user_ip, edges_user_email,
        edges_listing_contact_email, edges_listing_billing_email,
        edges_listing_contact_phone, edges_listing_billing_phone,
        edges_listing_located_at, edges_listing_billing_addr,
        edges_listing_person
    )

