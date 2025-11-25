"""
Demo Script - Test Fraud Detection API

Shows how to use the API endpoints locally.
"""
import requests
import json
from pathlib import Path


BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("\n=== Testing /health endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.json()


def test_predict(listing_id: int):
    """Test prediction endpoint."""
    print(f"\n=== Testing /predict endpoint (listing_id={listing_id}) ===")
    
    payload = {
        "listing_id": listing_id
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        print(f"\n📊 Fraud Score: {result['fraud_score']:.4f}")
        print(f"🎯 Confidence: {result['confidence']:.2f}")
        print(f"🆕 Cold Start: {result['is_cold_start']}")
        return result
    else:
        print(f"Error: {response.text}")
        return None


def test_explain(listing_id: int, generate_plot: bool = False):
    """Test explanation endpoint."""
    print(f"\n=== Testing /explain endpoint (listing_id={listing_id}) ===")
    
    payload = {
        "listing_id": listing_id,
        "generate_plot": generate_plot
    }
    
    response = requests.post(f"{BASE_URL}/explain", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        
        # Show top features (without plot data)
        result_clean = {k: v for k, v in result.items() if k != "waterfall_plot_base64"}
        print(f"Response: {json.dumps(result_clean, indent=2)}")
        
        print(f"\n📈 Top Contributing Features:")
        for i, feat_dict in enumerate(result['top_features'][:5], 1):
            for feat_name, shap_value in feat_dict.items():
                direction = "📈" if shap_value > 0 else "📉"
                print(f"  {i}. {feat_name}: {shap_value:+.4f} {direction}")
                
        if result.get('waterfall_plot_base64'):
            print(f"\n🎨 Waterfall plot generated ({len(result['waterfall_plot_base64'])} bytes)")
            
        return result
    else:
        print(f"Error: {response.text}")
        return None


def main():
    """Run demo tests."""
    print("=" * 60)
    print("Fraud Detection API - Demo")
    print("=" * 60)
    
    # Test health
    health = test_health()
    
    if health.get('status') != 'healthy':
        print("\n❌ Service is not healthy. Exiting.")
        return
        
    # Get a sample listing ID from data
    print("\n📋 Loading sample listing ID...")
    import polars as pl
    listings = pl.read_parquet("artifacts/nodes_listing.parquet")
    sample_id = int(listings["insertion_id"][0])
    print(f"Using listing_id: {sample_id}")
    
    # Test prediction
    pred_result = test_predict(sample_id)
    
    # Test explanation (without plot for speed)
    if pred_result:
        explain_result = test_explain(sample_id, generate_plot=False)
        
    # Test explanation WITH plot
    if input("\nGenerate waterfall plot? (y/n): ").lower() == 'y':
        explain_with_plot = test_explain(sample_id, generate_plot=True)
        
        # Save plot to file
        if explain_with_plot and explain_with_plot.get('waterfall_plot_base64'):
            import base64
            plot_data = base64.b64decode(explain_with_plot['waterfall_plot_base64'])
            with open('waterfall_plot.png', 'wb') as f:
                f.write(plot_data)
            print("\n💾 Waterfall plot saved to waterfall_plot.png")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
