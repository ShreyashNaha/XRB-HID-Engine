"""
preprocessor.py
Astrophysical Signal Filtering and Alignment.
Filters data based on Signal-to-Noise (SNR) thresholds to remove 
instrumental artifacts and "clipping" lines from diagrams.
"""

import pandas as pd
import numpy as np

class DataPreprocessor:
    """Handles data cleaning, SNR filtering, and feature engineering."""
    
    def __init__(self, maxi_df: pd.DataFrame, bat_df: pd.DataFrame, target_name: str):
        self.maxi_df = maxi_df
        self.bat_df = bat_df
        self.target_name = target_name

    def _apply_snr_filter(self, df: pd.DataFrame, flux_col: str, err_col: str, sigma: float = 3.0) -> pd.DataFrame:
        """
        Removes 'noise' days. Only keeps data where the flux is 
        significantly higher (default 3x) than the instrument error.
        """
        # Astronomers call this the 3-sigma detection limit.
        return df[df[flux_col] > (sigma * df[err_col])]

    def _align_timeseries(self) -> pd.DataFrame:
        """
        Filters noise from both instruments and aligns them onto a 1-day grid.
        """
        # 1. Filter out days where the star was too dim to be 'real' data
        clean_maxi = self._apply_snr_filter(self.maxi_df, 'Soft_Flux', 'Soft_Err')
        clean_bat = self._apply_snr_filter(self.bat_df, 'RATE', 'ERROR')

        if clean_maxi.empty or clean_bat.empty:
            return pd.DataFrame()

        # 2. Floor the MJD to integers to create 1-day universal bins
        clean_maxi['MJD_grid'] = np.floor(clean_maxi['MJD']).astype(int)
        clean_bat['MJD_grid'] = np.floor(clean_bat['MJD']).astype(int)

        # 3. Aggregate multiple passes within the same day
        maxi_daily = clean_maxi.groupby('MJD_grid')[['Soft_Flux', 'Hard_Flux']].mean().reset_index()
        bat_daily = clean_bat.groupby('MJD_grid')[['RATE']].mean().reset_index()

        # 4. Join the observatories (Inner join ensures we only keep 'Active' days)
        merged = pd.merge(maxi_daily, bat_daily, on='MJD_grid', how='inner')
        return merged

    def process(self) -> pd.DataFrame:
        """
        Main execution pipeline for feature engineering.
        Calculates physical properties only on high-confidence detection days.
        """
        df = self._align_timeseries()
        
        # Minimum points needed for a meaningful plot and AI clustering
        if df.empty or len(df) < 5:
            raise ValueError(f"No high-confidence overlapping days found for {self.target_name}.")

        # Physics Calculations
        # Hardness Ratio: High Energy (BAT) / Low Energy (MAXI Soft)
        df['Hardness_Ratio'] = df['RATE'] / df['Soft_Flux']
        
        # Total Intensity: Sum of Soft and Hard X-rays
        df['Total_Intensity'] = df['RATE'] + df['Soft_Flux']
        
        # Final scrub for any mathematical anomalies
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Hardness_Ratio', 'Total_Intensity'])
        
        # Sort by time for the plot lines
        return df.sort_values('MJD_grid')