## Générer
import numpy as np
import pandas as pd

class FinancialDataDiscretizer:
    """Convert continuous financial data to discrete symbols"""
    
    def __init__(self):
        self.volatility_thresholds = None
        self.volume_thresholds = None
        
    def fit_thresholds(self, volatility_data, volume_data):
        """Learn quantile-based thresholds from training data"""
        # Volatility: Low, Medium, High (tertiles)
        self.volatility_thresholds = np.percentile(volatility_data, [33, 67])
        
        # Volume: Low, Normal, High (tertiles)
        self.volume_thresholds = np.percentile(volume_data, [33, 67])
        
        return self
    
    def discretize_returns(self, returns):
        """Convert returns to binary: 0=down, 1=up"""
        return (returns > 0).astype(int)
    
    def discretize_volatility(self, volatility):
        """Convert volatility to ternary: 0=Low, 1=Medium, 2=High"""
        vol_discrete = np.zeros_like(volatility, dtype=int)
        vol_discrete[volatility >= self.volatility_thresholds[0]] = 1
        vol_discrete[volatility >= self.volatility_thresholds[1]] = 2
        return vol_discrete
    
    def discretize_volume(self, volume):
        """Convert volume to ternary: 0=Low, 1=Normal, 2=High"""
        vol_discrete = np.zeros_like(volume, dtype=int)
        vol_discrete[volume >= self.volume_thresholds[0]] = 1
        vol_discrete[volume >= self.volume_thresholds[1]] = 2
        return vol_discrete
    
    def prepare_data(self, price_data, volatility_data, volume_data):
        """
        Prepare all features for CTW
        
        Returns:
            dict with discretized sequences
        """
        # Calculate returns
        returns = np.diff(np.log(price_data))
        
        # Align all sequences (returns is 1 shorter)
        volatility_aligned = volatility_data[1:]
        volume_aligned = volume_data[1:]
        
        # Discretize
        return {
            'returns': self.discretize_returns(returns),
            'volatility': self.discretize_volatility(volatility_aligned),
            'volume': self.discretize_volume(volume_aligned)
        }
