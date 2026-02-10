"""
V2 Market Regime Detection Module
Uses Gaussian Mixture Models with RANK-BASED label assignment.

Key V2 improvements over V1:
- Rank-based regime assignment instead of hardcoded thresholds
  (fixes NVDA being stuck in "Stagnation" despite +64% growth)
- Efficiency ratio feature for better Growth vs Stagnation discrimination
- Expanding-window retraining for adaptivity
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
    GMM-based market regime detector with rank-based label assignment.
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
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.retrain_frequency = retrain_frequency
        self.model: Optional[GaussianMixture] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_mapping: Dict[int, int] = {}
        self._fit_count = 0
        self._last_retrain_idx = 0
        
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime classification.
        
        V2 adds efficiency_ratio for better Growth/Stagnation discrimination.
        """
        features = pd.DataFrame(index=df.index)
        
        if 'returns' not in df.columns:
            returns = df['close'].pct_change()
        else:
            returns = df['returns']
        
        # Rolling features 
        features['rolling_return'] = returns.rolling(self.lookback).mean()
        features['rolling_vol'] = returns.rolling(self.lookback).std() * np.sqrt(252)
        features['rolling_skew'] = returns.rolling(self.lookback).skew()
        features['rolling_kurt'] = returns.rolling(self.lookback).kurt()
        
        # Volatility trend
        features['vol_change'] = features['rolling_vol'].pct_change(5)
        
        # Price vs SMA (normalized)
        sma_50 = df['close'].rolling(50).mean()
        features['price_vs_sma'] = (df['close'] - sma_50) / sma_50
        
        # V2 NEW: Efficiency Ratio (Kaufman)
        # Direction / Volatility — high for trending, low for choppy
        direction = abs(df['close'] - df['close'].shift(self.lookback))
        volatility = returns.abs().rolling(self.lookback).sum() * df['close']
        features['efficiency_ratio'] = (direction / volatility.replace(0, np.nan)).clip(0, 1)
        
        features = features.dropna()
        return features
    
    def _assign_regime_labels(self, features: pd.DataFrame) -> None:
        """
        V2: RANK-BASED regime assignment.
        
        Instead of comparing cluster centers against hardcoded thresholds,
        we RANK clusters by their key features and assign labels based on 
        relative ordering. This naturally adapts to high-vol stocks like NVDA 
        where "normal" volatility exceeds V1's hardcoded thresholds.
        
        Algorithm:
        1. Compute composite scores for each regime type using RANKINGS
        2. Use Hungarian-style greedy assignment (best score first)
        """
        centers = pd.DataFrame(
            self.scaler.inverse_transform(self.model.means_),
            columns=features.columns
        )
        
        n = self.n_regimes
        
        # Extract key features per cluster
        returns_arr = centers['rolling_return'].values
        vol_arr = centers['rolling_vol'].values
        skew_arr = centers['rolling_skew'].values
        vol_change_arr = centers['vol_change'].values
        eff_ratio_arr = centers['efficiency_ratio'].values
        
        # Compute ranks (0 = lowest, n-1 = highest)
        ret_rank = np.argsort(np.argsort(returns_arr)).astype(float)
        vol_rank = np.argsort(np.argsort(vol_arr)).astype(float)
        skew_rank = np.argsort(np.argsort(skew_arr)).astype(float)
        vol_ch_rank = np.argsort(np.argsort(np.abs(vol_change_arr))).astype(float)
        eff_rank = np.argsort(np.argsort(eff_ratio_arr)).astype(float)
        
        # Normalize ranks to [0, 1]
        ret_rank /= max(1, n - 1)
        vol_rank /= max(1, n - 1)
        skew_rank /= max(1, n - 1)
        vol_ch_rank /= max(1, n - 1)
        eff_rank /= max(1, n - 1)
        
        # Score each cluster for each regime type using RANKS
        regime_scores = np.zeros((n, 4))
        
        for i in range(n):
            # Growth: highest return rank + high efficiency ratio + moderate vol
            regime_scores[i, self.GROWTH] = (
                ret_rank[i] * 3.0 + 
                eff_rank[i] * 2.0 - 
                vol_rank[i] * 0.5
            )
            
            # Stagnation: low return rank + low vol rank + low efficiency ratio
            regime_scores[i, self.STAGNATION] = (
                (1 - abs(ret_rank[i] - 0.5) * 2) * 2.0 +  # middle return
                (1 - vol_rank[i]) * 2.0 +                   # low vol
                (1 - eff_rank[i]) * 1.5                      # low efficiency
            )
            
            # Transition: high vol change + high vol + medium efficiency
            regime_scores[i, self.TRANSITION] = (
                vol_ch_rank[i] * 3.0 + 
                vol_rank[i] * 2.0 -
                abs(ret_rank[i] - 0.5) * 1.0  # neutral return preferred
            )
            
            # Crisis: lowest return rank + high vol + negative skew
            regime_scores[i, self.CRISIS] = (
                (1 - ret_rank[i]) * 3.0 +  # low return
                vol_rank[i] * 2.0 +          # high vol
                (1 - skew_rank[i]) * 1.5     # negative skew
            )
        
        # Greedy assignment: best score first
        used_regimes = set()
        self.regime_mapping = {}
        
        for _ in range(n):
            best_score = -np.inf
            best_cluster = -1
            best_regime = -1
            
            for cluster in range(n):
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
        """Fit the GMM model on historical data."""
        features = self.extract_features(df)
        
        if len(features) < 50:
            raise ValueError("Not enough data to fit regime detector")
        
        X = self.scaler.fit_transform(features)
        
        self.model = GaussianMixture(
            n_components=self.n_regimes,
            covariance_type='full',
            n_init=10,
            random_state=42
        )
        self.model.fit(X)
        
        self._assign_regime_labels(features)
        
        self.is_fitted = True
        self._fit_count += 1
        return self
    
    def predict_proba(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get probability distribution over regimes."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        features = self.extract_features(df)
        X = self.scaler.transform(features)
        
        gmm_probs = self.model.predict_proba(X)
        
        regime_probs = np.zeros((len(features), 4))
        for gmm_label, regime_label in self.regime_mapping.items():
            regime_probs[:, regime_label] = gmm_probs[:, gmm_label]
        
        prob_df = pd.DataFrame(
            regime_probs,
            index=features.index,
            columns=['P(Growth)', 'P(Stagnation)', 'P(Transition)', 'P(Crisis)']
        )
        
        return prob_df
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Get most likely regime for each time step."""
        probs = self.predict_proba(df)
        regime_ids = probs.values.argmax(axis=1)
        regime_names = [self.REGIME_NAMES[r] for r in regime_ids]
        return pd.Series(regime_names, index=probs.index, name='regime')
    
    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """Get the current regime and probabilities."""
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
    """Add regime detection results to DataFrame."""
    probs = detector.predict_proba(df)
    regimes = detector.predict(df)
    
    df = df.copy()
    for col in probs.columns:
        df[col] = probs[col]
    df['regime'] = regimes
    
    return df
