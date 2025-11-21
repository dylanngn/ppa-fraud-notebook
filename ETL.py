import polars as pl
import os
from sentence_transformers import SentenceTransformer
import torch
from dotenv import load_dotenv

# Database Connection URI
load_dotenv()

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("DB_URI environment variable not set")

def extract_data():
    """
    Extracts data from the Aurora Postgres database using ConnectorX for speed.
    Saves raw data to artifacts/raw_*.parquet for inspection and checkpointing.
    """
    
    # Check if raw data already exists to avoid re-running heavy SQL
    if os.path.exists("artifacts/raw_users.parquet") and os.path.exists("artifacts/raw_insertions.parquet"):
        print("Loading raw data from Parquet artifacts...")
        df_users = pl.read_parquet("artifacts/raw_users.parquet")
        df_insertions = pl.read_parquet("artifacts/raw_insertions.parquet")
        return df_users, df_insertions

    print("Extracting data from Database...")
    
    # 1. Users
    query_users = "SELECT id as user_id, created_at, contact_emails, user_type, platform FROM hginsertionapiprod.users"
    df_users = pl.read_database_uri(query_users, DB_URI, engine="connectorx")
    
    # 2. Insertions (Listings) - Chunked Extraction
    print("Extracting Insertions (Chunked)...")
    
    chunk_size = 10000
    temp_dir = "artifacts/temp_raw_insertions"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Check existing chunks to resume
    existing_chunks = [f for f in os.listdir(temp_dir) if f.endswith(".parquet")]
    next_chunk_idx = len(existing_chunks)
    offset = next_chunk_idx * chunk_size
    
    if next_chunk_idx > 0:
        print(f"Resuming from chunk {next_chunk_idx} (Offset {offset})...")
    
    while True:
        print(f"Fetching chunk {next_chunk_idx} (Offset {offset})...")
        
        # We add ORDER BY i.id to ensure deterministic pagination
        query_chunk = f"""
        SELECT 
            i.id as insertion_id,
            i.object_reference,
            i.platform,
            i.user_id,
            i.user_ip_address,
            i.listing::text as listing_json,
            i.customer_segment,
            i.fraud_flag,
            i.auto_approval_criteria,
            i.first_published_date,
            i.created_at as listing_created_at,
            sh.transition_timestamp as submission_at
        FROM hginsertionapiprod.insertions i
        JOIN (
            SELECT insertion_id, min(transition_timestamp) as transition_timestamp
            FROM hginsertionapiprod.status_history
            WHERE status_from = 'DRAFT' AND status_to = 'PENDING_APPROVAL'
            GROUP BY insertion_id
        ) sh ON i.id = sh.insertion_id
        WHERE sh.transition_timestamp BETWEEN '2023-11-01' AND '2025-11-01'
        AND platform <> 're.smg'
        AND meta -> 'migratedFromPersonId' is null
        ORDER BY i.id
        LIMIT {chunk_size} OFFSET {offset}
        """
        
        try:
            df_chunk = pl.read_database_uri(query_chunk, DB_URI, engine="connectorx")
            
            if len(df_chunk) == 0:
                print("No more data to fetch.")
                break
                
            chunk_path = os.path.join(temp_dir, f"chunk_{next_chunk_idx}.parquet")
            df_chunk.write_parquet(chunk_path)
            print(f"Saved {chunk_path} ({len(df_chunk)} rows)")
            
            if len(df_chunk) < chunk_size:
                print("Last chunk fetched.")
                break
                
            offset += chunk_size
            next_chunk_idx += 1
            
        except Exception as e:
            print(f"Error fetching chunk {next_chunk_idx}: {e}")
            print("Stopping. You can resume later.")
            raise e

    # Assemble all chunks
    print("Assembling chunks...")
    df_insertions = pl.read_parquet(f"{temp_dir}/*.parquet")
    
    # Save raw artifacts
    os.makedirs("artifacts", exist_ok=True)
    print(f"Saving {len(df_users)} users to artifacts/raw_users.parquet...")
    df_users.write_parquet("artifacts/raw_users.parquet")
    print(f"Saving {len(df_insertions)} insertions to artifacts/raw_insertions.parquet...")
    df_insertions.write_parquet("artifacts/raw_insertions.parquet")
    
    return df_users, df_insertions

def process_listings(df_insertions):
    """
    Parses the JSON listing column and creates the Listing Node DataFrame.
    """
    print("Processing Listings...")
    
    # Define the schema for the JSON structure we want to extract
    listing_dtype = pl.Struct({
        "lister": pl.Struct({
            "username": pl.Utf8,
            "email": pl.Utf8,
            "phone": pl.Utf8,
            "mobile": pl.Utf8,
            "address": pl.Struct({
                "postalCode": pl.Utf8,
                "locality": pl.Utf8
            }),
            "billing": pl.Struct({
                "email": pl.Utf8,
                "phoneDay": pl.Utf8,
                "phoneEvening": pl.Utf8,
                "phoneMobile": pl.Utf8,
                "address": pl.Struct({
                    "postalCode": pl.Utf8,
                    "locality": pl.Utf8
                })
            }),
            "contacts": pl.Struct({
                "inquiry": pl.Struct({
                    "email": pl.Utf8,
                    "phone": pl.Utf8,
                    "mobile": pl.Utf8
                }),
                "viewing": pl.Struct({
                    "email": pl.Utf8,
                    "phone": pl.Utf8,
                    "mobile": pl.Utf8
                })
            })
        }),
        "prices": pl.Struct({
            "buy": pl.Struct({"price": pl.Float64}),
            "rent": pl.Struct({
                "gross": pl.Float64,
                "net": pl.Float64
            })
        }),
        "characteristics": pl.Struct({
            "livingSpace": pl.Float64,
            "numberOfRooms": pl.Float64
        }),
        "address": pl.Struct({
            "postalCode": pl.Utf8,
            "locality": pl.Utf8
        }),
        "descriptions": pl.Struct({
            "description": pl.Utf8
        })
    })

    # Parse JSON
    df = df_insertions.lazy().with_columns(
        pl.col("listing_json").str.json_decode(listing_dtype).alias("listing_struct")
    )
    
    # Extract relevant fields
    df_processed = df.select([
        pl.col("insertion_id"),
        pl.col("object_reference"),
        pl.col("platform"),
        pl.col("user_id"),
        pl.col("user_ip_address"),
        pl.col("auto_approval_criteria"),
        pl.col("first_published_date"),
        pl.col("listing_created_at"),
        pl.col("submission_at"),
        
        # Refined Fraud Logic:
        # 1. Must have fraud_flag
        # 2. If fraud_flag > first_published_date, it slipped through Seon (Critical Target)
        pl.col("fraud_flag"),
        (pl.col("fraud_flag").is_not_null()).alias("is_fraud"),
        (
            pl.col("fraud_flag").is_not_null() & 
            (pl.col("fraud_flag") > pl.col("first_published_date"))
        ).alias("is_slip_through_fraud"),
        
        # Lister Info
        pl.col("listing_struct").struct.field("lister").struct.field("username").alias("lister_username"),
        pl.col("listing_struct").struct.field("lister").struct.field("email").alias("lister_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("phone").alias("lister_phone"),
        pl.col("listing_struct").struct.field("lister").struct.field("mobile").alias("lister_mobile"),
        pl.col("listing_struct").struct.field("lister").struct.field("address").struct.field("postalCode").alias("lister_zip"),
        pl.col("listing_struct").struct.field("lister").struct.field("address").struct.field("locality").alias("lister_city"),

        # Billing Info
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("email").alias("billing_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("phoneDay").alias("billing_phone_day"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("phoneEvening").alias("billing_phone_evening"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("phoneMobile").alias("billing_phone_mobile"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("address").struct.field("postalCode").alias("billing_zip"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("address").struct.field("locality").alias("billing_city"),

        # Contact Info
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("email").alias("contact_inquiry_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("phone").alias("contact_inquiry_phone"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("mobile").alias("contact_inquiry_mobile"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("viewing").struct.field("email").alias("contact_viewing_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("viewing").struct.field("phone").alias("contact_viewing_phone"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("viewing").struct.field("mobile").alias("contact_viewing_mobile"),
        
        # Listing Details
        pl.col("listing_struct").struct.field("prices").struct.field("buy").struct.field("price").alias("price_buy"),
        pl.col("listing_struct").struct.field("prices").struct.field("rent").struct.field("gross").alias("price_rent_gross"),
        pl.col("listing_struct").struct.field("prices").struct.field("rent").struct.field("net").alias("price_rent_net"),
        
        pl.col("listing_struct").struct.field("characteristics").struct.field("livingSpace").alias("living_space"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("numberOfRooms").alias("rooms"),
        pl.col("listing_struct").struct.field("address").struct.field("postalCode").alias("zip_code"),
        pl.col("listing_struct").struct.field("address").struct.field("locality").alias("city"),
        
        # Text for Embedding
        pl.col("listing_struct").struct.field("descriptions").struct.field("description").alias("description_text")
    ]).collect()
    
    # Coalesce Price (Buy vs Rent) for a single 'price' feature if needed, or keep separate
    # For now, let's keep them but maybe fill nulls
    
    return df_processed

def generate_embeddings(df_listings):
    """
    Generates text embeddings for listing descriptions.
    """
    print("Generating Text Embeddings (this may take a while)...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    # Handle null descriptions
    texts = df_listings["description_text"].fill_null("").to_list()
    
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    # Convert embeddings to a list of lists (or keep as numpy/tensor for saving)
    # For Parquet, list of lists is okay, or we can save as a separate numpy file.
    # Let's save as a column of lists for simplicity in Parquet
    
    df_with_embeddings = df_listings.with_columns(
        pl.Series(name="description_embedding", values=embeddings)
    )
    
    return df_with_embeddings

def create_nodes_and_edges(df_users, df_listings):
    """
    Creates the final Node and Edge DataFrames.
    """
    print("Creating Nodes and Edges...")
    
    # --- NODES ---
    
    # 1. User Nodes
    # Feature Engineering for Users could happen here (e.g. email domain extraction)
    nodes_user = df_users.select([
        pl.col("user_id"),
        pl.col("account_created_at"),
        pl.col("user_type"),
        pl.col("contact_emails").str.split("@").list.get(1).alias("email_domain") # Simple domain extraction
    ]).unique(subset=["user_id"])
    
    # 2. Listing Nodes
    nodes_listing = df_listings.select([
        "insertion_id", 
        "object_reference", 
        "platform",
        "is_fraud", 
        "is_slip_through_fraud",
        "listing_created_at", 
        "submission_at",
        "first_published_date",
        "auto_approval_criteria",
        "lister_username",
        "price_buy", "price_rent_gross", "living_space", "rooms", 
        "zip_code", "city", "description_embedding"
    ])
    
    # 3. IP Address Nodes
    nodes_ip = df_listings.select("user_ip_address").unique().drop_nulls()

    # 4. Email Nodes
    # Collect emails from Users and all Listing sources
    email_cols = [
        "lister_email", "billing_email", 
        "contact_inquiry_email", "contact_viewing_email"
    ]
    
    emails_from_users = df_users.select(pl.col("contact_emails").str.split(",").explode().str.strip_chars().alias("email"))
    emails_from_listings = [df_listings.select(pl.col(c).alias("email")) for c in email_cols]
    
    nodes_email = pl.concat([emails_from_users] + emails_from_listings).unique().drop_nulls()

    # 5. Phone Nodes
    # Collect phones from all Listing sources
    phone_cols = [
        "lister_phone", "lister_mobile",
        "billing_phone_day", "billing_phone_evening", "billing_phone_mobile",
        "contact_inquiry_phone", "contact_inquiry_mobile",
        "contact_viewing_phone", "contact_viewing_mobile"
    ]
    phones_from_listings = [df_listings.select(pl.col(c).alias("phone")) for c in phone_cols]
    nodes_phone = pl.concat(phones_from_listings).unique().drop_nulls()

    # 6. Location Nodes (Zip + City)
    # Create a composite key "Zip_City"
    # Sources: Property Address, Lister Address, Billing Address
    def create_loc_df(df, zip_col, city_col):
        return df.select([
            pl.col(zip_col).alias("zip"),
            pl.col(city_col).alias("city")
        ]).with_columns(
            (pl.col("zip") + "_" + pl.col("city")).alias("location_id")
        ).drop_nulls()

    loc_property = create_loc_df(df_listings, "zip_code", "city")
    loc_lister = create_loc_df(df_listings, "lister_zip", "lister_city")
    loc_billing = create_loc_df(df_listings, "billing_zip", "billing_city")
    
    nodes_location = pl.concat([loc_property, loc_lister, loc_billing]).unique(subset=["location_id"])

    # --- EDGES ---
    
    # 1. User -> Posts -> Listing
    edges_user_posts_listing = df_listings.select([
        pl.col("user_id").alias("source"),
        pl.col("insertion_id").alias("target")
    ])
    
    # 2. User -> Uses -> IP
    edges_user_uses_ip = df_listings.select([
        pl.col("user_id").alias("source"),
        pl.col("user_ip_address").alias("target")
    ]).drop_nulls().unique()

    # 3. User -> Has -> Email
    edges_user_has_email = df_users.select([
        pl.col("user_id").alias("source"),
        pl.col("contact_emails").str.split(",").explode().str.strip_chars().alias("target")
    ]).drop_nulls().unique()

    # 4. Listing -> Has -> Email (from any source)
    edge_dfs = []
    for col in email_cols:
        edge_dfs.append(df_listings.select([
            pl.col("insertion_id").alias("source"),
            pl.col(col).alias("target")
        ]))
    edges_listing_has_email = pl.concat(edge_dfs).drop_nulls().unique()

    # 5. Listing -> Has -> Phone (from any source)
    edge_dfs = []
    for col in phone_cols:
        edge_dfs.append(df_listings.select([
            pl.col("insertion_id").alias("source"),
            pl.col(col).alias("target")
        ]))
    edges_listing_has_phone = pl.concat(edge_dfs).drop_nulls().unique()

    # 6. Listing -> Located_At -> Location (Property)
    edges_listing_located_at = df_listings.select([
        pl.col("insertion_id").alias("source"),
        (pl.col("zip_code") + "_" + pl.col("city")).alias("target")
    ]).drop_nulls().unique()

    # 7. User -> Located_At -> Location (via Lister/Billing)
    # This is a bit tricky as we don't have direct User address, but we can infer from their listings
    # Let's link User to the locations found in their listings (Lister/Billing)
    user_loc_dfs = []
    user_loc_dfs.append(df_listings.select([
        pl.col("user_id").alias("source"),
        (pl.col("lister_zip") + "_" + pl.col("lister_city")).alias("target")
    ]))
    user_loc_dfs.append(df_listings.select([
        pl.col("user_id").alias("source"),
        (pl.col("billing_zip") + "_" + pl.col("billing_city")).alias("target")
    ]))
    edges_user_located_at = pl.concat(user_loc_dfs).drop_nulls().unique()
    
    return (
        nodes_user, nodes_listing, nodes_ip, nodes_email, nodes_phone, nodes_location,
        edges_user_posts_listing, edges_user_uses_ip, 
        edges_user_has_email, edges_listing_has_email, edges_listing_has_phone,
        edges_listing_located_at, edges_user_located_at
    )

def main():
    # Create artifacts directory if not exists
    os.makedirs("artifacts", exist_ok=True)
    
    # 1. Extract
    df_users, df_insertions = extract_data()
    print(f"Extracted {len(df_users)} users and {len(df_insertions)} insertions.")
    
    # 2. Process Listings
    df_listings_processed = process_listings(df_insertions)
    
    # 3. Generate Embeddings
    # Note: This can be slow. For testing, you might want to sample.
    # df_listings_processed = df_listings_processed.head(1000) 
    df_listings_with_embeddings = generate_embeddings(df_listings_processed)
    
    # 4. Create Graph Elements
    (
        nodes_user, nodes_listing, nodes_ip, nodes_email, nodes_phone, nodes_location,
        edges_user_posts, edges_user_uses_ip,
        edges_user_has_email, edges_listing_has_email, edges_listing_has_phone,
        edges_listing_located_at, edges_user_located_at
    ) = create_nodes_and_edges(df_users, df_listings_with_embeddings)
    
    # 5. Save to Parquet
    print("Saving to Parquet...")
    nodes_user.write_parquet("artifacts/nodes_user.parquet")
    nodes_listing.write_parquet("artifacts/nodes_listing.parquet")
    nodes_ip.write_parquet("artifacts/nodes_ip.parquet")
    nodes_email.write_parquet("artifacts/nodes_email.parquet")
    nodes_phone.write_parquet("artifacts/nodes_phone.parquet")
    nodes_location.write_parquet("artifacts/nodes_location.parquet")
    
    edges_user_posts.write_parquet("artifacts/edges_user_posts_listing.parquet")
    edges_user_uses_ip.write_parquet("artifacts/edges_user_uses_ip.parquet")
    edges_user_has_email.write_parquet("artifacts/edges_user_has_email.parquet")
    edges_listing_has_email.write_parquet("artifacts/edges_listing_has_email.parquet")
    edges_listing_has_phone.write_parquet("artifacts/edges_listing_has_phone.parquet")
    edges_listing_located_at.write_parquet("artifacts/edges_listing_located_at.parquet")
    edges_user_located_at.write_parquet("artifacts/edges_user_located_at.parquet")
    
    print("ETL Complete. Data saved to 'artifacts/' directory.")

if __name__ == "__main__":
    main()
