
import logging
import sys
import os
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.append(".")

from src.data.schema import FEATURE_SCHEMA

def create_full_dummy_data(path: str):
    print(f"Creating dummy data at {path}")
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i*6) for i in range(2000)] # ~500 days
    
    # Create required columns matching Schema
    data = {
        "listing_id": [f"L{i}" for i in range(len(dates))],
        "user_id": [f"U{i%100}" for i in range(len(dates))],
        "submission_at": dates, # New timestamp
        # Derived: fraud_flag (presence implies fraud)
        "fraud_flag": [dates[i] if i % 50 == 0 else None for i in range(len(dates))],
        
        # ID Columns
        "user_ip_address_hash": [f"192.168.1.{i%20}" for i in range(len(dates))],
        "listing.lister.phone.hash": [f"ph_{i%50}" for i in range(len(dates))],
        "listing.lister.email.hash": [f"domain_{i%10}.com" for i in range(len(dates))],
        "listing.lister.billing.phoneDay.hash": [f"bill_ph_{i%50}" for i in range(len(dates))], # NEW Billing Phone
        "listing.lister.billing.email.hash": [f"bill_email_{i%10}.com" for i in range(len(dates))], # NEW Billing Email
        "listing.lister.billing.address.city_hash": [f"city_{i%5}" for i in range(len(dates))],
        "listing.lister.billing.phoneDay.area_hash": [f"area_{i%5}" for i in range(len(dates))],
        "listing.lister.email.domain_hash": [f"dom_{i%5}" for i in range(len(dates))],
        "listing.type": ["rent"] * len(dates),
        "bundle.tier": ["basic"] * len(dates),
        "listing.platforms": ["['is24']"] * len(dates),
        
        # Benchmarks
        "auto_approval_criteria.criteria.seonApproved": [True if i % 10 != 0 else False for i in range(len(dates))],
        
        # Features
        "listing.prices.rent.gross": [100.0] * len(dates),
        "listing.prices.rent.area": [50.0] * len(dates),
        "listing.characteristics.numberOfRooms": [3.5] * len(dates),
        "listing.characteristics.numberOfBathrooms": [1.0] * len(dates),
        "listing.characteristics.yearBuilt": [2000.0] * len(dates),
        "listing.characteristics.numberOfFloors": [2.0] * len(dates),
    }
    
    df = pl.DataFrame(data)
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print("Dummy data created.")

def run_dry_run():
    print(">>> Running Dry Run of Training Pipeline")
    
    dummy_path = "tmp/full_dummy.parquet"
    create_full_dummy_data(dummy_path)
    
    # Run Trainer via simplified CLI call or direct instantiation
    # Calling script via subprocess to test Hydra integration
    import subprocess
    
    print("Vanilla Done. Testing GNN Variant...")
    cmd_gnn = [
        "python", "-m", "src.training.trainer",
        f"data.path={dummy_path}",
        "data.end_date=2023-02-01",
        "training.initial_train_months=1",
        "model.variant=graphsage_xgboost",
        "model.gnn.epochs=1",
        "hydra.run.dir=tmp/hydra_out_gnn"
    ]
    print(f"Executing: {' '.join(cmd_gnn)}")
    subprocess.check_call(cmd_gnn)
    
    print(">>> All Dry Runs Passed!")

if __name__ == "__main__":
    try:
        run_dry_run()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
