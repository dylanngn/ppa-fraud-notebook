import torch
from datetime import datetime

def check_graph_timestamps():
    try:
        data = torch.load("artifacts/graph.pt", weights_only=False)
        ts = data['listing'].timestamp
        
        print(f"Total timestamps: {len(ts)}")
        print(f"Min value: {ts.min().item()}")
        print(f"Max value: {ts.max().item()}")
        
        # Check if ns or us
        # 2023 in ns is approx 1.67e18
        # 2023 in us is approx 1.67e15
        
        val = ts.max().item()
        print(f"Max value (scientific): {val:.2e}")
        
        if val > 1e17:
            print("Likely Nanoseconds (Correct)")
            print(f"Max Date: {datetime.fromtimestamp(val / 1e9)}")
        elif val > 1e14:
            print("Likely Microseconds (Incorrect for GNN)")
            print(f"Max Date (if treated as ns): {datetime.fromtimestamp(val / 1e9)}")
            print(f"Max Date (if treated as us): {datetime.fromtimestamp(val / 1e6)}")
        else:
            print("Unknown scale")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_graph_timestamps()
