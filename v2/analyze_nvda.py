
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import DataLoader
from v2.regime_detector import RegimeDetector
import pandas as pd
import numpy as np

def analyze_nvda_regime():
    print("Fetching NVDA data...")
    loader = DataLoader()
    df = loader.fetch_ticker('NVDA', period='2y')
    
    detector = RegimeDetector()
    print("extracting features...")
    features = detector.extract_features(df)
    
    print("\nFeature Statistics:")
    print(features[['rolling_return', 'rolling_vol', 'efficiency']].describe())
    
    print("\nFitting model...")
    detector.fit(df)
    
    print("\nCluster Centers (Scaled):")
    print(detector.model.means_)
    
    print("\nCluster Centers (Unscaled):")
    centers = pd.DataFrame(
        detector.scaler.inverse_transform(detector.model.means_),
        columns=features.columns
    )
    print(centers)
    
    print("\nRegime Mapping:")
    print(detector.regime_mapping)
    for cluster, regime_id in detector.regime_mapping.items():
        regime_name = detector.REGIME_NAMES[regime_id]
        print(f"  Cluster {cluster} -> {regime_name}")
        
    # Analyze why a specific high-growth period was misclassified
    # Let's look at late 2023/early 2024 for NVDA
    print("\nAnalyzing specific period (Jan 2024):")
    subset = df['2024-01-01':'2024-02-01']
    if not subset.empty:
        probs = detector.predict_proba(subset)
        print(probs.head())
        
        subset_features = detector.extract_features(subset)
        print("\nFeatures during this period:")
        print(subset_features.head())

if __name__ == "__main__":
    analyze_nvda_regime()
