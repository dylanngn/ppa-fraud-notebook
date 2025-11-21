import polars as pl
import torch
from torch_geometric.data import HeteroData
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
    df_location = pl.read_parquet("artifacts/nodes_location.parquet")
    
    data = HeteroData()
    
    # --- Process Mappings & Features ---
    
    # 1. User
    user_map, df_user = load_node_mapping(df_user, "user_id", "user")
    # Features: Account Age (days), User Type (encoded)
    # We need to handle dates and categoricals
    # For simplicity in this MVP, let's just use a dummy feature or simple ones
    # Let's compute account_age_days relative to a fixed date or just raw timestamp normalized?
    # Better: Account Age at time of snapshot? 
    # Let's just use 1s for now if features are complex, OR implement the same logic as baseline.
    # Let's use a simple feature: user_type encoded
    user_types = df_user["user_type"].cast(pl.Categorical).to_physical().to_numpy()
    data['user'].x = torch.from_numpy(user_types).view(-1, 1).float()
    data['user'].num_nodes = len(df_user)
    
    # 2. Listing
    listing_map, df_listing = load_node_mapping(df_listing, "insertion_id", "listing")
    
    # Features: Price, Size, Rooms, Description Embedding
    # Handle Nulls
    df_listing = df_listing.with_columns([
        pl.col("price_rent_gross").fill_null(0),
        pl.col("living_space").fill_null(0),
        pl.col("rooms").fill_null(0)
    ])
    
    # Numerical Features
    num_feats = df_listing.select(["price_rent_gross", "living_space", "rooms"]).to_numpy()
    
    # Embeddings (List of floats -> Tensor)
    # Polars stores list as Series of lists. Need to convert to numpy 2D array.
    # This can be slow.
    embeddings = np.stack(df_listing["description_embedding"].to_numpy())
    
    # Concatenate
    x_listing = np.concatenate([num_feats, embeddings], axis=1)
    data['listing'].x = torch.from_numpy(x_listing).float()
    
    # Labels (Target)
    # is_fraud might be boolean or int. Handle nulls (assume 0).
    y = df_listing["is_fraud"].fill_null(0).cast(pl.Int64).to_numpy()
    data['listing'].y = torch.from_numpy(y).long()
    
    # Train/Test Split Mask (Time-based)
    # We can add masks here or do it in training loop.
    # Let's add 'submission_at' as a timestamp attribute for splitting later
    # PyG doesn't standardly store timestamps, but we can store it as an attribute
    timestamps = df_listing["submission_at"].cast(pl.Int64).to_numpy() # ns
    data['listing'].timestamp = torch.from_numpy(timestamps)
    
    data['listing'].num_nodes = len(df_listing)

    # 3. IP
    ip_map, _ = load_node_mapping(df_ip, "user_ip_address", "ip")
    data['ip'].num_nodes = len(ip_map)
    data['ip'].x = torch.ones(len(ip_map), 1) # Dummy feature

    # 4. Email
    email_map, _ = load_node_mapping(df_email, "email", "email")
    data['email'].num_nodes = len(email_map)
    data['email'].x = torch.ones(len(email_map), 1)

    # 5. Phone
    phone_map, _ = load_node_mapping(df_phone, "phone", "phone")
    data['phone'].num_nodes = len(phone_map)
    data['phone'].x = torch.ones(len(phone_map), 1)

    # 6. Location
    loc_map, _ = load_node_mapping(df_location, "location_id", "location")
    data['location'].num_nodes = len(loc_map)
    data['location'].x = torch.ones(len(loc_map), 1)

    # --- Process Edges ---
    
    def add_edge(filename, src_col, dst_col, src_type, dst_type, rel_name):
        print(f"Processing edge: {src_type} - {rel_name} - {dst_type}")
        df_edge = pl.read_parquet(f"artifacts/{filename}")
        
        # Map IDs to Indices
        # We need to filter edges where nodes might be missing (if any)
        # But our ETL should be consistent.
        
        # Use the maps
        src_indices = [src_map.get(i) for i in df_edge[src_col].to_list()]
        dst_indices = [dst_map.get(i) for i in df_edge[dst_col].to_list()]
        
        # Filter Nones (if any missing nodes)
        valid_mask = [(s is not None and d is not None) for s, d in zip(src_indices, dst_indices)]
        src_indices = [s for s, v in zip(src_indices, valid_mask) if v]
        dst_indices = [d for d, v in zip(dst_indices, valid_mask) if v]
        
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        data[src_type, rel_name, dst_type].edge_index = edge_index

    # Define mappings for easy access
    maps = {
        "user": user_map,
        "listing": listing_map,
        "ip": ip_map,
        "email": email_map,
        "phone": phone_map,
        "location": loc_map
    }

    # 1. User -> Posts -> Listing
    src_map, dst_map = maps["user"], maps["listing"]
    add_edge("edges_user_posts_listing.parquet", "source", "target", "user", "listing", "posts")
    
    # 2. User -> Uses -> IP
    src_map, dst_map = maps["user"], maps["ip"]
    add_edge("edges_user_uses_ip.parquet", "source", "target", "user", "ip", "uses")
    
    # 3. User -> Has -> Email
    src_map, dst_map = maps["user"], maps["email"]
    add_edge("edges_user_has_email.parquet", "source", "target", "user", "email", "has_email")
    
    # 4. Listing -> Has -> Email
    src_map, dst_map = maps["listing"], maps["email"]
    add_edge("edges_listing_has_email.parquet", "source", "target", "listing", "email", "has_email")
    
    # 5. Listing -> Has -> Phone
    src_map, dst_map = maps["listing"], maps["phone"]
    add_edge("edges_listing_has_phone.parquet", "source", "target", "listing", "phone", "has_phone")
    
    # 6. Listing -> Located_At -> Location
    src_map, dst_map = maps["listing"], maps["location"]
    add_edge("edges_listing_located_at.parquet", "source", "target", "listing", "location", "located_at")
    
    # 7. User -> Located_At -> Location
    src_map, dst_map = maps["user"], maps["location"]
    add_edge("edges_user_located_at.parquet", "source", "target", "user", "location", "located_at")

    # --- Reverse Edges ---
    # PyG GNNs usually need undirected or reverse edges for message passing in both directions
    transform = torch_geometric.transforms.ToUndirected()
    data = transform(data)
    
    print("Graph construction complete!")
    print(data)
    
    # Save
    torch.save(data, "artifacts/graph.pt")
    
    # Save Mappings (Optional, for inference later)
    with open("artifacts/mappings.pkl", "wb") as f:
        pickle.dump(maps, f)

if __name__ == "__main__":
    import torch_geometric.transforms
    build_graph()
