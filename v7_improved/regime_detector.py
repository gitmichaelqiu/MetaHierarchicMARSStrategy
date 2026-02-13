"""
V5 Regime Detector — Volume-confirmed transitions.

Key V5 changes over V4:
- Added volume_ratio and volume_trend features
- Directional Crisis filter (from V4)
- Volume helps distinguish real breakouts from fakeouts
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """V5 Regime Detector with volume confirmation."""
    
    REGIME_NAMES = ['Growth', 'Stagnation', 'Transition', 'Crisis']
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 20,
        retrain_frequency: int = 20,
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.retrain_frequency = retrain_frequency
        
        self.model = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            n_init=5,
            max_iter=200,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.label_map: Dict[int, str] = {}
        self.is_fitted: bool = False
        self._bars_since_retrain: int = 0
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features including volume metrics."""
        close = df['close']
        features = pd.DataFrame(index=df.index)
        
        features['return_20d'] = close.pct_change(self.lookback)
        features['volatility_20d'] = close.pct_change().rolling(self.lookback).std() * np.sqrt(252)
        features['skewness_20d'] = close.pct_change().rolling(self.lookback).skew()
        features['kurtosis_20d'] = close.pct_change().rolling(self.lookback).apply(
            lambda x: stats.kurtosis(x, fisher=True), raw=True
        )
        
        # Efficiency ratio
        direction = (close - close.shift(self.lookback)).abs()
        volatility_sum = close.diff().abs().rolling(self.lookback).sum()
        features['efficiency_ratio'] = direction / volatility_sum.replace(0, np.nan)
        
        # V5 NEW: Volume features
        if 'volume' in df.columns:
            vol = df['volume'].replace(0, np.nan)
            vol_20d_avg = vol.rolling(self.lookback).mean()
            features['volume_ratio'] = vol / vol_20d_avg  # Current vs avg
            features['volume_trend'] = vol.rolling(5).mean() / vol_20d_avg  # Short vs long
        else:
            features['volume_ratio'] = 1.0
            features['volume_trend'] = 1.0
        
        features = features.dropna()
        return features
    
    def _assign_regime_labels(self, features: pd.DataFrame):
        """Rank-based regime assignment with directional + volume filters."""
        X = self.scaler.transform(features.values)
        labels = self.model.predict(X)
        
        cluster_stats = {}
        for i in range(self.n_regimes):
            mask = labels == i
            if mask.sum() == 0:
                cluster_stats[i] = {
                    'mean_return': 0, 'mean_volatility': 0.5,
                    'mean_skewness': 0, 'mean_efficiency': 0.5,
                    'mean_volume_ratio': 1.0, 'count': 0
                }
                continue
            
            cluster_data = features.iloc[mask.nonzero()[0]]
            cluster_stats[i] = {
                'mean_return': cluster_data['return_20d'].mean(),
                'mean_volatility': cluster_data['volatility_20d'].mean(),
                'mean_skewness': cluster_data['skewness_20d'].mean(),
                'mean_efficiency': cluster_data['efficiency_ratio'].mean(),
                'mean_volume_ratio': cluster_data['volume_ratio'].mean(),
                'count': mask.sum(),
            }
        
        clusters = list(range(self.n_regimes))
        scores = {regime: {} for regime in self.REGIME_NAMES}
        
        ret_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_return'])
        vol_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_volatility'])
        eff_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_efficiency'])
        skew_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_skewness'])
        
        for c in clusters:
            ret_rank = ret_ranked.index(c)
            vol_rank = vol_ranked.index(c)
            eff_rank = eff_ranked.index(c)
            skew_rank = skew_ranked.index(c)
            
            # Growth: high return, high efficiency
            scores['Growth'][c] = ret_rank * 3 + eff_rank * 2 + (3 - vol_rank) * 0.5
            
            # Stagnation: low return, low volatility, low efficiency
            scores['Stagnation'][c] = (3 - ret_rank) * 1.5 + (3 - vol_rank) * 2 + (3 - eff_rank) * 2
            
            # Crisis: negative return, high vol, negative skew
            crisis_base = (3 - ret_rank) * 2 + vol_rank * 3 + (3 - skew_rank) * 1.5
            # V4/V5: Directional penalty
            if cluster_stats[c]['mean_return'] > 0.02:
                crisis_base *= 0.3
            scores['Crisis'][c] = crisis_base
            
            # Transition: moderate everything
            center_ret = abs(ret_rank - 1.5)
            center_vol = abs(vol_rank - 1.5)
            scores['Transition'][c] = (3 - center_ret) * 2 + (3 - center_vol) * 1.5 + vol_rank * 0.5
        
        # Greedy assignment
        assigned = {}
        used_clusters = set()
        used_regimes = set()
        
        all_assignments = []
        for regime in self.REGIME_NAMES:
            for c in clusters:
                all_assignments.append((scores[regime][c], regime, c))
        
        all_assignments.sort(key=lambda x: -x[0])
        
        for score, regime, cluster in all_assignments:
            if regime not in used_regimes and cluster not in used_clusters:
                assigned[cluster] = regime
                used_clusters.add(cluster)
                used_regimes.add(regime)
        
        for c in clusters:
            if c not in assigned:
                for regime in self.REGIME_NAMES:
                    if regime not in used_regimes:
                        assigned[c] = regime
                        used_regimes.add(regime)
                        break
                else:
                    assigned[c] = 'Transition'
        
        self.label_map = assigned
    
    def fit(self, df: pd.DataFrame) -> 'RegimeDetector':
        features = self.extract_features(df)
        if len(features) < self.n_regimes * 10:
            self.is_fitted = False
            return self
        
        X = features.values
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model.fit(X_scaled)
        self._assign_regime_labels(features)
        self.is_fitted = True
        self._bars_since_retrain = 0
        return self
    
    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        if not self.is_fitted:
            return 'Unknown', {}
        
        self._bars_since_retrain += 1
        if self._bars_since_retrain >= self.retrain_frequency:
            try:
                self.fit(df)
            except Exception:
                pass
        
        features = self.extract_features(df)
        if len(features) == 0:
            return 'Unknown', {}
        
        last_features = features.iloc[[-1]]
        X = self.scaler.transform(last_features.values)
        probs = self.model.predict_proba(X)[0]
        
        regime_probs = {}
        for cluster_idx, prob in enumerate(probs):
            regime_name = self.label_map.get(cluster_idx, 'Unknown')
            regime_probs[regime_name] = regime_probs.get(regime_name, 0) + prob
        
        current_regime = max(regime_probs, key=regime_probs.get)
        return current_regime, regime_probs
