import polars as pl
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import os
import numpy as np
import pickle

def load_node_mapping(df, id_col, node_type):
    """
    Creates a mapping from Raw ID -> PyG Index (0..N-1).
    Returns:
        mapping (dict): Raw ID -> Index
        x (torch.Tensor): Node features (if any)
    """
    print(f"Processing {node_type} nodes...")
    
    # Ensure unique
    df = df.unique(subset=[id_col])
    
    # Create mapping
    ids = df[id_col].to_list()
    mapping = {raw_id: i for i, raw_id in enumerate(ids)}
    
    return mapping, df

def build_graph():
    print("Loading Parquet artifacts...")
    
    # --- Load Nodes ---
    df_user = pl.read_parquet("artifacts/nodes_user.parquet")
    df_listing = pl.read_parquet("artifacts/nodes_listing.parquet")
    df_ip = pl.read_parquet("artifacts/nodes_ip.parquet")
    df_email = pl.read_parquet("artifacts/nodes_email.parquet")
    df_phone = pl.read_parquet("artifacts/nodes_phone.parquet")
    df_address = pl.read_parquet("artifacts/nodes_address.parquet")
    df_person = pl.read_parquet("artifacts/nodes_person.parquet")
    
    data = HeteroData()
    
    # --- Process Mappings & Features ---
    
    # 1. User
    user_map, df_user = load_node_mapping(df_user, "user_id", "user")
    # Features: Account Age (days)
    # We can compute account age relative to a fixed date or just use 1s for now.
    # Let's stick to dummy features for MVP to avoid complex date parsing here (handled in ETL/Features)
    data['user'].x = torch.ones(len(df_user), 1)
    data['user'].num_nodes = len(df_user)
    
    # 2. Listing
    listing_map, df_listing = load_node_mapping(df_listing, "insertion_id", "listing")
    
    # Features: Price, Size, Rooms, OfferType, Characteristics
    # Handle Nulls
    df_listing = df_listing.with_columns([
        pl.col("price_rent_gross").fill_null(0),
        pl.col("price_buy").fill_null(0),
        pl.col("living_space").fill_null(0),
        pl.col("rooms").fill_null(0),
        pl.col("offer_type").fill_null("RENT"), # Default
        pl.col("is_new").fill_null(False).cast(pl.Int8),
        pl.col("has_balcony").fill_null(False).cast(pl.Int8),
        pl.col("has_elevator").fill_null(False).cast(pl.Int8),
        pl.col("has_parking").fill_null(False).cast(pl.Int8)
    ])
    
    # One-hot encode offer_type (RENT=0, BUY=1)
    offer_type_feat = (df_listing["offer_type"] == "BUY").cast(pl.Int8).to_numpy().reshape(-1, 1)
    
    # Numerical Features
    # Combine rent and buy price into one 'price' column? Or keep separate?
    # Let's keep separate to allow model to learn distinction.
    num_feats = df_listing.select([
        "price_rent_gross", "price_buy", "living_space", "rooms",
        "is_new", "has_balcony", "has_elevator", "has_parking"
    ]).to_numpy()
    
    # Embeddings
    embeddings = np.stack(df_listing["description_embedding"].to_numpy())
    
    # Concatenate
    x_listing = np.concatenate([num_feats, offer_type_feat, embeddings], axis=1)
    data['listing'].x = torch.from_numpy(x_listing).float()
    
    # Labels (Target)
    y = df_listing["is_fraud"].cast(pl.Int64).fill_null(0).to_numpy()
    data['listing'].y = torch.from_numpy(y).long()
    
    # Timestamps
    timestamps = df_listing["submission_at"].cast(pl.Int64).to_numpy() # ns
    data['listing'].timestamp = torch.from_numpy(timestamps)
    data['listing'].num_nodes = len(df_listing)

    # 3. IP
    ip_map, _ = load_node_mapping(df_ip, "user_ip_address", "ip")
    data['ip'].num_nodes = len(ip_map)
    data['ip'].x = torch.ones(len(ip_map), 1)

    # 4. Email
    email_map, _ = load_node_mapping(df_email, "email", "email")
    data['email'].num_nodes = len(email_map)
    data['email'].x = torch.ones(len(email_map), 1)

    # 5. Phone
    phone_map, _ = load_node_mapping(df_phone, "phone", "phone")
    data['phone'].num_nodes = len(phone_map)
    data['phone'].x = torch.ones(len(phone_map), 1)

    # 6. Address (Granular)
    addr_map, _ = load_node_mapping(df_address, "address_id", "address")
    data['address'].num_nodes = len(addr_map)
    data['address'].x = torch.ones(len(addr_map), 1)

    # 7. Person
    person_map, _ = load_node_mapping(df_person, "person_name", "person")
    data['person'].num_nodes = len(person_map)
    data['person'].x = torch.ones(len(person_map), 1)

    # --- Process Edges ---
    
    # Mapping for edge timestamps (Listing Time)
    listing_time_map = dict(zip(df_listing["insertion_id"], df_listing["submission_at"].cast(pl.Int64)))
    
    def add_edge(filename, src_col, dst_col, src_type, dst_type, rel_name, time_source_col=None):
        print(f"Processing edge: {src_type} - {rel_name} - {dst_type}")
        if not os.path.exists(f"artifacts/{filename}"):
            print(f"Warning: {filename} not found. Skipping.")
            return

        df_edge = pl.read_parquet(f"artifacts/{filename}")
        
        # Map IDs to Indices
        src_indices = [src_map.get(i) for i in df_edge[src_col].to_list()]
        dst_indices = [dst_map.get(i) for i in df_edge[dst_col].to_list()]
        
        # Filter Nones
        valid_mask = [(s is not None and d is not None) for s, d in zip(src_indices, dst_indices)]
        src_indices = [s for s, v in zip(src_indices, valid_mask) if v]
        dst_indices = [d for d, v in zip(dst_indices, valid_mask) if v]
        
        if not src_indices:
            return

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data[src_type, rel_name, dst_type].edge_index = edge_index
        
        # Assign Timestamps
        edge_times = []
        if time_source_col:
            if dst_type == 'listing':
                valid_dst_ids = [i for i, v in zip(df_edge[dst_col].to_list(), valid_mask) if v]
                edge_times = [listing_time_map.get(i, 0) for i in valid_dst_ids]
            elif src_type == 'listing':
                valid_src_ids = [i for i, v in zip(df_edge[src_col].to_list(), valid_mask) if v]
                edge_times = [listing_time_map.get(i, 0) for i in valid_src_ids]
                
        if edge_times:
            data[src_type, rel_name, dst_type].timestamp = torch.tensor(edge_times, dtype=torch.long)

    # Define mappings
    maps = {
        "user": user_map,
        "listing": listing_map,
        "ip": ip_map,
        "email": email_map,
        "phone": phone_map,
        "address": addr_map,
        "person": person_map
    }

    # 1. User -> Posts -> Listing
    src_map, dst_map = maps["user"], maps["listing"]
    add_edge("edges_user_posts_listing.parquet", "source", "target", "user", "listing", "posts", time_source_col="target")
    
    # 2. User -> Uses -> IP
    src_map, dst_map = maps["user"], maps["ip"]
    add_edge("edges_user_uses_ip.parquet", "source", "target", "user", "ip", "uses")
    
    # 3. User -> Has -> Email
    src_map, dst_map = maps["user"], maps["email"]
    add_edge("edges_user_has_email.parquet", "source", "target", "user", "email", "has_email")
    
    # 4. Listing -> Has -> Email (Typed)
    src_map, dst_map = maps["listing"], maps["email"]
    add_edge("edges_listing_contact_email.parquet", "source", "target", "listing", "email", "has_contact_email", time_source_col="source")
    add_edge("edges_listing_billing_email.parquet", "source", "target", "listing", "email", "has_billing_email", time_source_col="source")
    add_edge("edges_listing_inquiry_email.parquet", "source", "target", "listing", "email", "has_inquiry_email", time_source_col="source")
    
    # 5. Listing -> Has -> Phone (Typed)
    src_map, dst_map = maps["listing"], maps["phone"]
    add_edge("edges_listing_contact_phone.parquet", "source", "target", "listing", "phone", "has_contact_phone", time_source_col="source")
    add_edge("edges_listing_billing_phone.parquet", "source", "target", "listing", "phone", "has_billing_phone", time_source_col="source")
    
    # 6. Listing -> Located_At -> Address
    src_map, dst_map = maps["listing"], maps["address"]
    add_edge("edges_listing_located_at.parquet", "source", "target", "listing", "address", "located_at", time_source_col="source")
    add_edge("edges_listing_lister_addr.parquet", "source", "target", "listing", "address", "lister_address", time_source_col="source")
    add_edge("edges_listing_billing_addr.parquet", "source", "target", "listing", "address", "billing_address", time_source_col="source")
    
    # 7. Listing -> Has -> Person
    src_map, dst_map = maps["listing"], maps["person"]
    add_edge("edges_listing_has_person.parquet", "source", "target", "listing", "person", "has_contact_person", time_source_col="source")

    # --- Reverse Edges ---
    transform = T.ToUndirected()
    data = transform(data)
    
    print("Graph construction complete!")
    print(data)
    
    # Save
    torch.save(data, "artifacts/graph.pt")
    
    # Save Mappings
    with open("artifacts/mappings.pkl", "wb") as f:
        pickle.dump(maps, f)

if __name__ == "__main__":
    build_graph()
