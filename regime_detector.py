"""
Market Regime Detection Module for MoA Trading Framework
Uses Gaussian Mixture Models to classify market into 4 regimes:
- Growth (Bull market with positive drift)
- Stagnation (Sideways, mean-reverting)
- Transition (High volatility, uncertain direction)
- Crisis (Crash, extreme negative returns)
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class RegimeDetector:
    """
    GMM-based market regime detector.
    Provides soft probability assignments across 4 market states.
    """
    
    # Regime labels
    GROWTH = 0
    STAGNATION = 1
    TRANSITION = 2
    CRISIS = 3
    
    REGIME_NAMES = {
        GROWTH: 'Growth',
        STAGNATION: 'Stagnation', 
        TRANSITION: 'Transition',
        CRISIS: 'Crisis'
    }
    
    def __init__(
        self, 
        n_regimes: int = 4,
        lookback: int = 20,
        retrain_frequency: int = 60
    ):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            lookback: Rolling window for feature calculation
            retrain_frequency: Days between model retraining
        """
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.retrain_frequency = retrain_frequency
        self.model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_mapping: Dict[int, int] = {}  # Map GMM labels to semantic regimes
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime classification.
        
        Features:
        - Rolling return (mean momentum)
        - Rolling volatility  
        - Rolling skewness
        - Rolling kurtosis
        - Volatility trend (vol change)
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Rolling features
        features['rolling_return'] = df['returns'].rolling(self.lookback).mean()
        features['rolling_vol'] = df['returns'].rolling(self.lookback).std() * np.sqrt(252)
        features['rolling_skew'] = df['returns'].rolling(self.lookback).skew()
        features['rolling_kurt'] = df['returns'].rolling(self.lookback).kurt()
        
        # Volatility trend
        features['vol_change'] = features['rolling_vol'].pct_change(5)
        
        # Price trend (normalized)
        sma_50 = df['close'].rolling(50).mean()
        features['price_vs_sma'] = (df['close'] - sma_50) / sma_50
        
        # Drop NaN rows
        features = features.dropna()
        
        return features
    
    def _assign_regime_labels(self, features: pd.DataFrame) -> None:
        """
        Assign semantic meaning to GMM cluster labels.
        Based on cluster characteristics:
        - Growth: High return, low-medium vol
        - Crisis: Negative return, high vol, negative skew
        - Transition: High vol change, high kurtosis
        - Stagnation: Low return, low vol
        """
        # Get cluster centers
        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.model.means_),
            columns=features.columns
        )
        
        # Score each cluster for each regime type
        regime_scores = np.zeros((self.n_regimes, 4))  # clusters x regime_types
        
        for i in range(self.n_regimes):
            ret = centers.iloc[i]['rolling_return']
            vol = centers.iloc[i]['rolling_vol']
            skew = centers.iloc[i]['rolling_skew']
            vol_change = centers.iloc[i]['vol_change']
            
            # Growth score: high return, moderate vol
            regime_scores[i, self.GROWTH] = ret * 3 - abs(vol - 0.15)
            
            # Stagnation score: low abs return, low vol
            regime_scores[i, self.STAGNATION] = -abs(ret) * 2 - vol
            
            # Transition score: high vol change, high vol
            regime_scores[i, self.TRANSITION] = abs(vol_change) * 2 + vol
            
            # Crisis score: negative return, high vol, negative skew
            regime_scores[i, self.CRISIS] = -ret * 3 + vol - skew
        
        # Assign each cluster to highest-scoring regime
        # First pass: assign greedily
        used_regimes = set()
        self.regime_mapping = {}
        
        for _ in range(self.n_regimes):
            # Find best unassigned cluster-regime pair
            best_score = -np.inf
            best_cluster = -1
            best_regime = -1
            
            for cluster in range(self.n_regimes):
                if cluster in self.regime_mapping:
                    continue
                for regime in range(4):
                    if regime in used_regimes:
                        continue
                    if regime_scores[cluster, regime] > best_score:
                        best_score = regime_scores[cluster, regime]
                        best_cluster = cluster
                        best_regime = regime
            
            if best_cluster >= 0:
                self.regime_mapping[best_cluster] = best_regime
                used_regimes.add(best_regime)
    
    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        """
        Fit the GMM model on historical data.
        """
        features = self.extract_features(df)
        
        if len(features) < 50:
            raise ValueError("Not enough data to fit regime detector")
        
        # Scale features
        X = self.scaler.fit_transform(features)
        
        # Fit GMM
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        self.model.fit(X)
        
        # Assign semantic labels to clusters
        self._assign_regime_labels(features)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get probability distribution over regimes.
        
        Returns:
            DataFrame with columns: P(Growth), P(Stagnation), P(Transition), P(Crisis)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = self.extract_features(df)
        X = self.scaler.transform(features)
        
        # Get GMM probabilities
        gmm_probs = self.model.predict_proba(X)
        
        # Map to semantic regimes
        regime_probs = np.zeros((len(features), 4))
        for gmm_label, regime_label in self.regime_mapping.items():
            regime_probs[:, regime_label] = gmm_probs[:, gmm_label]
        
        # Create DataFrame
        prob_df = pd.DataFrame(
            regime_probs,
            index=features.index,
            columns=['P(Growth)', 'P(Stagnation)', 'P(Transition)', 'P(Crisis)']
        )
        
        return prob_df
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Get most likely regime for each time step.
        """
        probs = self.predict_proba(df)
        regime_ids = probs.values.argmax(axis=1)
        regime_names = [self.REGIME_NAMES[r] for r in regime_ids]
        return pd.Series(regime_names, index=probs.index, name='regime')
    
    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """
        Get the current regime and probabilities.
        
        Returns:
            (regime_name, {regime: probability})
        """
        probs = self.predict_proba(df)
        if len(probs) == 0:
            return 'Unknown', {}
        
        last_probs = probs.iloc[-1]
        regime_id = last_probs.values.argmax()
        regime_name = self.REGIME_NAMES[regime_id]
        
        prob_dict = {
            'Growth': last_probs['P(Growth)'],
            'Stagnation': last_probs['P(Stagnation)'],
            'Transition': last_probs['P(Transition)'],
            'Crisis': last_probs['P(Crisis)']
        }
        
        return regime_name, prob_dict


def add_regime_features(df: pd.DataFrame, detector: RegimeDetector) -> pd.DataFrame:
    """
    Add regime detection results to DataFrame.
    """
    probs = detector.predict_proba(df)
    regimes = detector.predict(df)
    
    # Merge back
    df = df.copy()
    for col in probs.columns:
        df[col] = probs[col]
    df['regime'] = regimes
    
    return df


if __name__ == "__main__":
    # Test the regime detector
    from data_loader import DataLoader
    
    loader = DataLoader()
    df = loader.fetch_ticker('SPY', period='2y')
    
    print("Training Regime Detector...")
    detector = RegimeDetector()
    detector.fit(df)
    
    regime, probs = detector.get_current_regime(df)
    print(f"\nCurrent Regime: {regime}")
    print("Probabilities:")
    for r, p in probs.items():
        print(f"  {r}: {p:.2%}")
    
    # Show regime distribution
    regimes = detector.predict(df)
    print(f"\nRegime Distribution:")
    print(regimes.value_counts())
