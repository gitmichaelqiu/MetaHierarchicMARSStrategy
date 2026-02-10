"""
V4 Regime Detector — Improved directional filter.

Key V4 change: adds directional bias to prevent misclassifying
volatile uptrends as Crisis. If recent returns are positive AND
volatility is high, score favors Growth/Transition over Crisis.

Based on V2/V3 rank-based regime detector.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RegimeDetector:
    """V4 Regime Detector with directional Crisis filter."""
    
    REGIME_NAMES = ['Growth', 'Stagnation', 'Transition', 'Crisis']
    
    def __init__(
        self,
        n_regimes: int = 4,
        lookback: int = 20,
        features_lookback: int = 60,
        retrain_frequency: int = 20,
    ):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.features_lookback = features_lookback
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
        """Extract features including efficiency ratio."""
        close = df['close']
        
        features = pd.DataFrame(index=df.index)
        
        # Rolling returns
        features['return_20d'] = close.pct_change(self.lookback)
        
        # Rolling volatility
        features['volatility_20d'] = close.pct_change().rolling(self.lookback).std() * np.sqrt(252)
        
        # Rolling skewness
        features['skewness_20d'] = close.pct_change().rolling(self.lookback).skew()
        
        # Rolling kurtosis
        features['kurtosis_20d'] = close.pct_change().rolling(self.lookback).apply(
            lambda x: stats.kurtosis(x, fisher=True), raw=True
        )
        
        # Efficiency ratio (Kaufman)
        direction = (close - close.shift(self.lookback)).abs()
        volatility_sum = close.diff().abs().rolling(self.lookback).sum()
        features['efficiency_ratio'] = direction / volatility_sum.replace(0, np.nan)
        
        features = features.dropna()
        return features
    
    def _assign_regime_labels(self, features: pd.DataFrame):
        """Rank-based regime assignment with V4 directional filter."""
        X = self.scaler.transform(features.values)
        labels = self.model.predict(X)
        
        cluster_stats = {}
        for i in range(self.n_regimes):
            mask = labels == i
            if mask.sum() == 0:
                cluster_stats[i] = {
                    'mean_return': 0, 'mean_volatility': 0.5,
                    'mean_skewness': 0, 'mean_efficiency': 0.5, 'count': 0
                }
                continue
            
            cluster_data = features.iloc[mask.nonzero()[0]]
            cluster_stats[i] = {
                'mean_return': cluster_data['return_20d'].mean(),
                'mean_volatility': cluster_data['volatility_20d'].mean(),
                'mean_skewness': cluster_data['skewness_20d'].mean(),
                'mean_efficiency': cluster_data['efficiency_ratio'].mean(),
                'count': mask.sum(),
            }
        
        # Rank-based scoring
        clusters = list(range(self.n_regimes))
        
        # Score each cluster for each regime using ranks
        scores = {regime: {} for regime in self.REGIME_NAMES}
        
        ret_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_return'])
        vol_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_volatility'])
        eff_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_efficiency'])
        skew_ranked = sorted(clusters, key=lambda c: cluster_stats[c]['mean_skewness'])
        
        for c in clusters:
            ret_rank = ret_ranked.index(c)  # 0=lowest return, 3=highest
            vol_rank = vol_ranked.index(c)
            eff_rank = eff_ranked.index(c)
            skew_rank = skew_ranked.index(c)
            
            # Growth: high return, high efficiency, moderate-high vol OK
            scores['Growth'][c] = ret_rank * 3 + eff_rank * 2 + (3 - vol_rank) * 0.5
            
            # Stagnation: low return, low volatility, low efficiency
            scores['Stagnation'][c] = (3 - ret_rank) * 1.5 + (3 - vol_rank) * 2 + (3 - eff_rank) * 2
            
            # Crisis: negative return, high volatility, negative skew
            # V4 CHANGE: also penalize if returns are actually positive (directional filter)
            crisis_base = (3 - ret_rank) * 2 + vol_rank * 3 + (3 - skew_rank) * 1.5
            # Directional penalty: if mean return is positive, this isn't a Crisis
            if cluster_stats[c]['mean_return'] > 0.02:
                crisis_base *= 0.3  # Heavy penalty — positive returns ≠ Crisis
            scores['Crisis'][c] = crisis_base
            
            # Transition: moderate everything
            center_ret = abs(ret_rank - 1.5)
            center_vol = abs(vol_rank - 1.5)
            scores['Transition'][c] = (3 - center_ret) * 2 + (3 - center_vol) * 1.5 + vol_rank * 0.5
        
        # Assign regimes greedily (highest score wins, no duplicates)
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
        
        # Fill any unassigned
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
        """Train the regime detector."""
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
    
    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Predict regimes for all rows."""
        if not self.is_fitted:
            return pd.Series('Unknown', index=df.index)
        
        features = self.extract_features(df)
        X = self.scaler.transform(features.values)
        raw_labels = self.model.predict(X)
        
        regime_names = pd.Series(
            [self.label_map.get(l, 'Unknown') for l in raw_labels],
            index=features.index
        )
        
        return regime_names.reindex(df.index, fill_value='Unknown')
    
    def get_current_regime(self, df: pd.DataFrame) -> Tuple[str, Dict[str, float]]:
        """Get current regime with probabilities."""
        if not self.is_fitted:
            return 'Unknown', {}
        
        # Expanding-window retraining
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
            if regime_name in regime_probs:
                regime_probs[regime_name] += prob
            else:
                regime_probs[regime_name] = prob
        
        current_regime = max(regime_probs, key=regime_probs.get)
        return current_regime, regime_probs
