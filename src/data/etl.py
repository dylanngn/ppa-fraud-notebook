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

DB_SCHEMA = os.getenv("DB_SCHEMA")
if not DB_SCHEMA:
    raise ValueError("DB_SCHEMA environment variable not set")

def fetch_raw_users():
    """
    Fetches users from DB and saves to artifacts/raw_users.parquet.
    Skips if file exists.
    """
    if os.path.exists("artifacts/raw_users.parquet"):
        print("raw_users.parquet exists. Skipping fetch.")
        return

    print("Extracting Users from Database...")
    query_users = f"""
    SELECT 
        id as user_id, 
        owner_id, 
        created_at, 
        contact_emails
    FROM {DB_SCHEMA}.users
    WHERE
        created_at BETWEEN '2020-12-17' AND '2025-11-01'
    ORDER BY created_at
    """
    df_users = pl.read_database_uri(query_users, DB_URI, engine="connectorx")
    
    os.makedirs("artifacts", exist_ok=True)
    print(f"Saving {len(df_users)} users to artifacts/raw_users.parquet...")
    df_users.write_parquet("artifacts/raw_users.parquet")
    
    # Free memory
    del df_users

def fetch_raw_insertions():
    """
    Fetches insertions from DB in chunks and saves to artifacts/raw_insertions.parquet.
    Skips if file exists. Resumes from last chunk if interrupted.
    """
    if os.path.exists("artifacts/raw_insertions.parquet"):
        print("raw_insertions.parquet exists. Skipping fetch.")
        return

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
        i.object_reference,
        i.platform,
        i.user_id,
        i.user_ip_address,
        i.listing::text as listing_json,
        i.fraud_flag,
        i.auto_approval_criteria::text as auto_approval_criteria_json,
        i.first_published_date,
        i.customer_segment,
        i.selected_bundle::text as selected_bundle_json,
        i.created_at as listing_created_at,
        sh.transition_timestamp as submission_at
    FROM {DB_SCHEMA}.insertions i
    JOIN (
        SELECT insertion_id, min(transition_timestamp) as transition_timestamp
        FROM {DB_SCHEMA}.status_history
        WHERE status_from = 'DRAFT' AND status_to = 'PENDING_APPROVAL'
        GROUP BY insertion_id
    ) sh ON i.id = sh.insertion_id
    WHERE sh.transition_timestamp BETWEEN '2023-01-01' AND '2025-11-02'
    AND platform <> 're.smg'
    AND meta -> 'migratedFromPersonId' is null
    ORDER BY submission_at
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
    print(f"Saving {len(df_insertions)} insertions to artifacts/raw_insertions.parquet...")
    df_insertions.write_parquet("artifacts/raw_insertions.parquet") 

def extract_data():
    """
    Orchestrates the ETL process.
    """
    # 1. Fetch Raw Data (Checkpointing)
    fetch_raw_users()
    fetch_raw_insertions()
    
    # 2. Load Raw Data
    print("Loading raw data from Parquet artifacts...")
    df_users = pl.read_parquet("artifacts/raw_users.parquet")
    df_insertions = pl.read_parquet("artifacts/raw_insertions.parquet")
    
    return df_users, df_insertions

def process_listings(df_insertions):
    """
    Parses the listing JSON and extracts features.
    """
    # Define Schema for JSON parsing (Enhanced)
    listing_dtype = pl.Struct({
        "offerType": pl.Utf8,
        "lister": pl.Struct({
            "username": pl.Utf8,
            "email": pl.Utf8,
            "phone": pl.Utf8,
            "mobile": pl.Utf8,
            "address": pl.Struct({
                "street": pl.Utf8,
                "postalCode": pl.Utf8,
                "locality": pl.Utf8,
                "country": pl.Utf8,
                "region": pl.Utf8,
                "geoCoordinates": pl.Struct({
                    "latitude": pl.Float64,
                    "longitude": pl.Float64
                })
            }),
            "billing": pl.Struct({
                "email": pl.Utf8,
                "phoneDay": pl.Utf8,
                "phoneMobile": pl.Utf8,
                "address": pl.Struct({
                    "street": pl.Utf8,
                    "postalCode": pl.Utf8,
                    "locality": pl.Utf8,
                    "country": pl.Utf8
                }),
                "payment": pl.Struct({
                    "paymentType": pl.Utf8
                })
            }),
            "contacts": pl.Struct({
                "inquiry": pl.Struct({
                    "givenName": pl.Utf8,
                    "familyName": pl.Utf8,
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
            "numberOfRooms": pl.Float64,
            "yearBuilt": pl.Float64,
            "floor": pl.Float64,
            "numberOfFloors": pl.Float64,
            "isNewBuilding": pl.Boolean,
            "hasBalcony": pl.Boolean,
            "hasElevator": pl.Boolean,
            "hasParking": pl.Boolean,
            "isOldBuilding": pl.Boolean
        }),
        "address": pl.Struct({
            "street": pl.Utf8,
            "postalCode": pl.Utf8,
            "locality": pl.Utf8,
            "country": pl.Utf8,
            "region": pl.Utf8,
            "geoCoordinates": pl.Struct({
                "latitude": pl.Float64,
                "longitude": pl.Float64
            })
        }),
        "descriptions": pl.Struct({
            "description": pl.Utf8
        }),
        "localization": pl.Struct({
            "primary": pl.Utf8,
            "de": pl.Struct({"text": pl.Struct({"title": pl.Utf8, "description": pl.Utf8})}),
            "en": pl.Struct({"text": pl.Struct({"title": pl.Utf8, "description": pl.Utf8})}),
            "fr": pl.Struct({"text": pl.Struct({"title": pl.Utf8, "description": pl.Utf8})}),
            "it": pl.Struct({"text": pl.Struct({"title": pl.Utf8, "description": pl.Utf8})})
        })
    })

    # Schema for Bundle JSON
    bundle_dtype = pl.Struct({
        "period": pl.Int64,
        "tier": pl.Utf8
    })

    auto_approval_criteria_dtype = pl.Struct({
        "criteria": pl.Struct({
            "criteria": pl.Struct({
                "seonApproved": pl.Boolean
            })
        })
    })

    # Parse JSON
    df = df_insertions.lazy().with_columns([
        pl.col("listing_json").str.json_decode(listing_dtype).alias("listing_struct"),
        pl.col("selected_bundle_json").str.json_decode(bundle_dtype).alias("bundle_struct"),
        pl.col("auto_approval_criteria_json").str.json_decode(auto_approval_criteria_dtype).alias("auto_approval_criteria_struct")
    ])
    
    # Extract relevant fields
    df_processed = df.select([
        pl.col("object_reference"),
        pl.col("platform"),
        pl.col("user_id"),
        pl.col("user_ip_address"),
        pl.col("auto_approval_criteria"),
        pl.col("first_published_date"),
        pl.col("listing_created_at"),
        pl.col("submission_at"),
        
        # Fraud Logic
        pl.col("fraud_flag"),
        (pl.col("fraud_flag").is_not_null()).alias("is_fraud"),
        (
            pl.col("fraud_flag").is_not_null() & 
            (pl.col("fraud_flag") > pl.col("first_published_date"))
        ).alias("is_slip_through_fraud"),
        
        # Customer Segment
        pl.col("customer_segment"),

        # Bundle Info
        pl.col("bundle_struct").struct.field("period").alias("bundle_period"),
        pl.col("bundle_struct").struct.field("tier").alias("bundle_tier"),

        # Offer Type
        pl.col("listing_struct").struct.field("offerType").alias("offer_type"),

        # Lister Info
        pl.col("listing_struct").struct.field("lister").struct.field("username").alias("lister_username"),
        pl.col("listing_struct").struct.field("lister").struct.field("email").alias("lister_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("phone").alias("lister_phone"),
        pl.col("listing_struct").struct.field("lister").struct.field("mobile").alias("lister_mobile"),
        pl.col("listing_struct").struct.field("lister").struct.field("address").struct.field("street").alias("lister_street"),
        pl.col("listing_struct").struct.field("lister").struct.field("address").struct.field("postalCode").alias("lister_zip"),
        pl.col("listing_struct").struct.field("lister").struct.field("address").struct.field("locality").alias("lister_city"),
        pl.col("listing_struct").struct.field("lister").struct.field("address").struct.field("country").alias("lister_country"),

        # Billing Info
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("email").alias("billing_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("phoneDay").alias("billing_phone_day"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("phoneMobile").alias("billing_phone_mobile"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("address").struct.field("street").alias("billing_street"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("address").struct.field("postalCode").alias("billing_zip"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("address").struct.field("locality").alias("billing_city"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("address").struct.field("country").alias("billing_country"),
        pl.col("listing_struct").struct.field("lister").struct.field("billing").struct.field("payment").struct.field("paymentType").alias("payment_type"),

        # Contact Info
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("givenName").alias("inquiry_given_name"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("familyName").alias("inquiry_family_name"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("email").alias("contact_inquiry_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("phone").alias("contact_inquiry_phone"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("inquiry").struct.field("mobile").alias("contact_inquiry_mobile"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("viewing").struct.field("email").alias("contact_viewing_email"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("viewing").struct.field("phone").alias("contact_viewing_phone"),
        pl.col("listing_struct").struct.field("lister").struct.field("contacts").struct.field("viewing").struct.field("mobile").alias("contact_viewing_mobile"),
        
        # Listing Details (Prices)
        pl.col("listing_struct").struct.field("prices").struct.field("buy").struct.field("price").alias("price_buy"),
        pl.col("listing_struct").struct.field("prices").struct.field("rent").struct.field("gross").alias("price_rent_gross"),
        pl.col("listing_struct").struct.field("prices").struct.field("rent").struct.field("net").alias("price_rent_net"),
        
        # Characteristics
        pl.col("listing_struct").struct.field("characteristics").struct.field("livingSpace").alias("living_space"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("numberOfRooms").alias("rooms"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("yearBuilt").alias("year_built"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("floor").alias("floor"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("numberOfFloors").alias("num_floors"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("isNewBuilding").alias("is_new"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("hasBalcony").alias("has_balcony"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("hasElevator").alias("has_elevator"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("hasParking").alias("has_parking"),
        pl.col("listing_struct").struct.field("characteristics").struct.field("isOldBuilding").alias("is_old"),

        # Location
        pl.col("listing_struct").struct.field("address").struct.field("street").alias("street"),
        pl.col("listing_struct").struct.field("address").struct.field("postalCode").alias("zip_code"),
        pl.col("listing_struct").struct.field("address").struct.field("locality").alias("city"),
        pl.col("listing_struct").struct.field("address").struct.field("country").alias("country"),
        pl.col("listing_struct").struct.field("address").struct.field("region").alias("region"),
        pl.col("listing_struct").struct.field("address").struct.field("geoCoordinates").struct.field("latitude").alias("latitude"),
        pl.col("listing_struct").struct.field("address").struct.field("geoCoordinates").struct.field("longitude").alias("longitude"),
        
        # Localization & Text
        pl.col("listing_struct").struct.field("localization").struct.field("primary").alias("language"),
        pl.coalesce([
            pl.col("listing_struct").struct.field("localization").struct.field("de").struct.field("text").struct.field("description"),
            pl.col("listing_struct").struct.field("localization").struct.field("en").struct.field("text").struct.field("description"),
            pl.col("listing_struct").struct.field("localization").struct.field("fr").struct.field("text").struct.field("description"),
            pl.col("listing_struct").struct.field("localization").struct.field("it").struct.field("text").struct.field("description"),
            pl.col("listing_struct").struct.field("descriptions").struct.field("description") # Fallback to legacy
        ]).alias("description_text")
    ]).collect()
    
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
    Creates the nodes and edges for the Heterogeneous Graph.
    """
    print("Creating Nodes and Edges...")
    
    # --- NODES ---
    
    # Prepare mapping DataFrames (include account_created_at for listings)
    user_id_map = df_users.select([
        "user_id", 
        "owner_id",
        pl.col("created_at").alias("account_created_at")
    ])
    
    # 1. User Nodes
    nodes_user = df_users.select([
        pl.col("owner_id").alias("user_id"),
        pl.col("created_at").alias("account_created_at"),
        pl.col("contact_emails").str.extract(r"@([^@,]+)", 1).alias("email_domain")
    ]).unique(subset=["user_id"])
    
    # 2. Listing Nodes (Enhanced)
    nodes_listing = df_listings.join(user_id_map, on="user_id", how="left").select([
        pl.col("object_reference").alias("insertion_id"),
        pl.col("owner_id").alias("user_id"),
        pl.col("account_created_at"),  # Join from user_id_map
        pl.col("platform"),  # For analysis
        pl.col("offer_type"),
        pl.col("price_buy"),
        pl.col("price_rent_gross"),
        pl.col("price_rent_net"),
        pl.col("living_space"),
        pl.col("rooms"),
        pl.col("year_built"),
        pl.col("floor"),
        pl.col("num_floors"),
        pl.col("is_new"),
        pl.col("has_balcony"),
        pl.col("has_elevator"),
        pl.col("has_parking"),
        pl.col("is_old"),
        pl.col("zip_code"),
        pl.col("city"),
        pl.col("street"),
        pl.col("country"),
        pl.col("region"),
        pl.col("latitude"),
        pl.col("longitude"),
        pl.col("bundle_period"),
        pl.col("bundle_tier"),
        pl.col("payment_type"),
        pl.col("customer_segment"),
        pl.col("language"),
        pl.col("description_embedding"),
        pl.col("fraud_flag").is_not_null().alias("is_fraud"),
        pl.col("submission_at"),
        # Add Seon evaluation columns
        pl.col("auto_approval_criteria"),
        pl.col("fraud_flag"),
        pl.col("first_published_date")
    ]).unique(subset=["insertion_id"])
    
    # 3. IP Address Nodes
    nodes_ip = df_listings.select("user_ip_address").unique().drop_nulls()

    # 4. Email Nodes
    email_cols = [
        "lister_email", "billing_email", 
        "contact_inquiry_email", "contact_viewing_email"
    ]
    emails_from_users = df_users.select(pl.col("contact_emails").str.split(",").explode().str.strip_chars().alias("email"))
    emails_from_listings = [df_listings.select(pl.col(c).alias("email")) for c in email_cols]
    nodes_email = pl.concat([emails_from_users] + emails_from_listings).unique().drop_nulls()

    # 5. Phone Nodes
    phone_cols = [
        "lister_phone", "lister_mobile",
        "billing_phone_day", "billing_phone_mobile",
        "contact_inquiry_phone", "contact_inquiry_mobile",
        "contact_viewing_phone", "contact_viewing_mobile"
    ]
    phones_from_listings = [df_listings.select(pl.col(c).alias("phone")) for c in phone_cols]
    nodes_phone = pl.concat(phones_from_listings).unique().drop_nulls()

    # 6. Address Nodes (Granular: Country + Zip + City + Street)
    # ID: Country_Zip_City_Street
    def create_address_df(df, street_col, zip_col, city_col, country_col, lat_col=None, lon_col=None):
        cols = [
            pl.col(street_col).fill_null("").alias("street"),
            pl.col(zip_col).fill_null("").alias("zip"),
            pl.col(city_col).fill_null("").alias("city"),
            pl.col(country_col).fill_null("").alias("country")
        ]
        if lat_col and lon_col:
            cols.append(pl.col(lat_col).alias("latitude"))
            cols.append(pl.col(lon_col).alias("longitude"))
        else:
            cols.append(pl.lit(None).cast(pl.Float64).alias("latitude"))
            cols.append(pl.lit(None).cast(pl.Float64).alias("longitude"))
            
        return df.select(cols).with_columns(
            (pl.col("country") + "_" + pl.col("zip") + "_" + pl.col("city") + "_" + pl.col("street")).alias("address_id")
        ).drop_nulls(subset=["address_id"]).unique(subset=["address_id"])

    addr_property = create_address_df(df_listings, "street", "zip_code", "city", "country", "latitude", "longitude")
    addr_lister = create_address_df(df_listings, "lister_street", "lister_zip", "lister_city", "lister_country")
    addr_billing = create_address_df(df_listings, "billing_street", "billing_zip", "billing_city", "billing_country")
    
    nodes_address = pl.concat([addr_property, addr_lister, addr_billing]).unique(subset=["address_id"])

    # 7. Person Nodes (Name)
    nodes_person = df_listings.select([
        (pl.col("inquiry_given_name").fill_null("") + " " + pl.col("inquiry_family_name").fill_null("")).str.strip_chars().alias("person_name")
    ]).filter(pl.col("person_name") != "").unique()

    # --- EDGES ---
    
    # Helper to create edge DF
    def create_edge_df(src_col, dst_col, src_name="source", dst_name="target"):
        return df_listings.select([
            pl.col(src_col).alias(src_name),
            pl.col(dst_col).alias(dst_name)
        ]).drop_nulls().unique()

    # 1. User -> Posts -> Listing
    edges_user_posts = df_listings.join(user_id_map, on="user_id", how="left").select([
        pl.col("owner_id").alias("source"),
        pl.col("object_reference").alias("target")
    ]).drop_nulls().unique()
    
    # 2. User -> Uses -> IP
    edges_user_ip = df_listings.join(user_id_map, on="user_id", how="left").select([
        pl.col("owner_id").alias("source"),
        pl.col("user_ip_address").alias("target")
    ]).drop_nulls().unique()
    
    # 3. User -> Has -> Email
    edges_user_email = df_users.select([
        pl.col("owner_id").alias("source"),
        pl.col("contact_emails").str.split(",").explode().str.strip_chars().alias("target")
    ]).drop_nulls().unique()

    # 4. Listing -> Has -> Email (Typed)
    edges_listing_contact_email = create_edge_df("object_reference", "lister_email")
    edges_listing_billing_email = create_edge_df("object_reference", "billing_email")
    edges_listing_inquiry_email = create_edge_df("object_reference", "contact_inquiry_email")
    
    # 5. Listing -> Has -> Phone (Typed)
    edges_listing_contact_phone = create_edge_df("object_reference", "lister_phone")
    edges_listing_billing_phone = create_edge_df("object_reference", "billing_phone_day") # Using day phone as primary billing
    
    # 6. Listing -> Located_At -> Address
    edges_listing_located_at = df_listings.select([
        pl.col("object_reference").alias("source"),
        (pl.col("country").fill_null("") + "_" + pl.col("zip_code").fill_null("") + "_" + pl.col("city").fill_null("") + "_" + pl.col("street").fill_null("")).alias("target")
    ]).drop_nulls().unique()
    
    # 7. Listing -> Lister_Address -> Address
    edges_listing_lister_addr = df_listings.select([
        pl.col("object_reference").alias("source"),
        (pl.col("lister_country").fill_null("") + "_" + pl.col("lister_zip").fill_null("") + "_" + pl.col("lister_city").fill_null("") + "_" + pl.col("lister_street").fill_null("")).alias("target")
    ]).drop_nulls().unique()
    
    # 8. Listing -> Billing_Address -> Address
    edges_listing_billing_addr = df_listings.select([
        pl.col("object_reference").alias("source"),
        (pl.col("billing_country").fill_null("") + "_" + pl.col("billing_zip").fill_null("") + "_" + pl.col("billing_city").fill_null("") + "_" + pl.col("billing_street").fill_null("")).alias("target")
    ]).drop_nulls().unique()
    
    # 9. Listing -> Has_Contact_Person -> Person
    edges_listing_person = df_listings.select([
        pl.col("object_reference").alias("source"),
        (pl.col("inquiry_given_name").fill_null("") + " " + pl.col("inquiry_family_name").fill_null("")).str.strip_chars().alias("target")
    ]).filter(pl.col("target") != "").unique()

    # Save Artifacts
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
    edges_listing_inquiry_email.write_parquet("artifacts/edges_listing_inquiry_email.parquet")
    
    edges_listing_contact_phone.write_parquet("artifacts/edges_listing_contact_phone.parquet")
    edges_listing_billing_phone.write_parquet("artifacts/edges_listing_billing_phone.parquet")
    
    edges_listing_located_at.write_parquet("artifacts/edges_listing_located_at.parquet")
    edges_listing_lister_addr.write_parquet("artifacts/edges_listing_lister_addr.parquet")
    edges_listing_billing_addr.write_parquet("artifacts/edges_listing_billing_addr.parquet")
    
    edges_listing_person.write_parquet("artifacts/edges_listing_has_person.parquet")
    
    print("ETL Complete. Artifacts saved.")
    
    return (
        nodes_user, nodes_listing, nodes_ip, nodes_email, nodes_phone, nodes_address, nodes_person,
        edges_user_posts, edges_user_ip, edges_user_email,
        edges_listing_contact_email, edges_listing_billing_email, edges_listing_inquiry_email,
        edges_listing_contact_phone, edges_listing_billing_phone,
        edges_listing_located_at, edges_listing_lister_addr, edges_listing_billing_addr,
        edges_listing_person
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
    
    # 4. Create Graph Elements and Save
    create_nodes_and_edges(df_users, df_listings_with_embeddings)
    
    print("ETL Complete. Data saved to 'artifacts/' directory.")

if __name__ == "__main__":
    main()
