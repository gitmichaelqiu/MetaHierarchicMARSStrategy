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
        - Efficiency (Return / Vol)
        """
        features = pd.DataFrame(index=df.index)
        
        # Returns
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Rolling features
        rolling_return = df['returns'].rolling(self.lookback).mean()
        rolling_vol = df['returns'].rolling(self.lookback).std() * np.sqrt(252)
        
        features['rolling_return'] = rolling_return
        features['rolling_vol'] = rolling_vol
        features['rolling_skew'] = df['returns'].rolling(self.lookback).skew()
        features['rolling_kurt'] = df['returns'].rolling(self.lookback).kurt()
        
        # Volatility trend
        features['vol_change'] = features['rolling_vol'].pct_change(5)
        
        # Price trend (normalized)
        sma_50 = df['close'].rolling(50).mean()
        features['price_vs_sma'] = (df['close'] - sma_50) / sma_50
        
        # Efficiency (Sortino-like proxy: Return / Vol)
        # Add epsilon to avoid div by zero
        features['efficiency'] = rolling_return / (rolling_vol + 1e-6)
        
        # Drop NaN rows
        features = features.dropna()
        
        return features
    
    def _assign_regime_labels(self, features: pd.DataFrame) -> None:
        """
        Assign semantic meaning to GMM cluster labels using Relative Ranking.
        
        Robust Logic:
        1. Cluster with Highest Return -> Growth
        2. Cluster with Lowest Return -> Crisis
        3. Of remaining, Cluster with Lowest Volatility -> Stagnation
        4. Last remaining -> Transition
        """
        # Get cluster centers - unscaled for easier interpretation
        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.model.means_),
            columns=features.columns
        )
        
        self.regime_mapping = {}
        available_clusters = set(range(self.n_regimes))
        
        # 1. Identify Growth (Highest Return)
        # Check: Growth must have positive return
        growth_candidates = centers[centers['rolling_return'] > 0.0001]
        if not growth_candidates.empty:
            growth_cluster = growth_candidates['rolling_return'].idxmax()
        else:
            # No real growth, take best available but treat as Stagnation or Transition if weak
            growth_cluster = centers['rolling_return'].idxmax()
            
        self.regime_mapping[growth_cluster] = self.GROWTH
        available_clusters.remove(growth_cluster)
        
        # 2. Identify Crisis (Lowest Return)
        # Check: Crisis must have negative return
        if available_clusters:
            remaining_centers = centers.loc[list(available_clusters)]
            min_ret_cluster = remaining_centers['rolling_return'].idxmin()
            
            if centers.loc[min_ret_cluster, 'rolling_return'] < -0.0002: # -0.02% daily ~ -5% annualized
                self.regime_mapping[min_ret_cluster] = self.CRISIS
                available_clusters.remove(min_ret_cluster)
            else:
                # Lowest return is not negative enough for Crisis
                # Map to Stagnation or Transition later
                pass
        
        # 3. Identify Stagnation (Lowest Volatility) from remaining
        if available_clusters:
            vol_score = centers['rolling_vol'].loc[list(available_clusters)]
            stagnation_cluster = vol_score.idxmin()
            self.regime_mapping[stagnation_cluster] = self.STAGNATION
            available_clusters.remove(stagnation_cluster)
        
        # 4. Remaining is Transition
        if available_clusters:
            for c in available_clusters:
                self.regime_mapping[c] = self.TRANSITION
            
        # Fallback/Safety: If n_regimes > 4, map others to Transition
        for i in range(self.n_regimes):
            if i not in self.regime_mapping:
                self.regime_mapping[i] = self.TRANSITION
    
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
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
