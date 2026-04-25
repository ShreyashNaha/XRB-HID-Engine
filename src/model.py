"""
model.py
Unsupervised Machine Learning module.
Applies Gaussian Mixture Models (GMM) to classify X-ray states dynamically.
Translates mathematical clusters into astrophysical terminology based on centroid analysis.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from src.config import PipelineConfig

class StateClusterer:
    """Handles unsupervised GMM clustering of Hardness-Intensity data."""
    
    def __init__(self):
        self.max_components = PipelineConfig.GMM_MAX_COMPONENTS
        self.random_state = PipelineConfig.RANDOM_STATE
        self.covariance_type = PipelineConfig.GMM_COVARIANCE_TYPE

    def _determine_optimal_clusters(self, X: np.ndarray) -> int:
        """
        Calculates the Bayesian Information Criterion (BIC) to automatically 
        find the true number of physical states without overfitting.
        """
        lowest_bic = np.infty
        best_n = 2  # An X-ray binary will have at least a Hard and a Soft state

        # Cap the search to prevent the model from splitting noise into fake states
        max_search = min(self.max_components, len(X) // 10) 
        max_search = max(2, max_search)

        for n in range(2, max_search + 1):
            gmm = GaussianMixture(
                n_components=n, 
                covariance_type=self.covariance_type, 
                random_state=self.random_state
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            
            if bic < lowest_bic:
                lowest_bic = bic
                best_n = n

        return best_n

    def _physically_name_states(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Maps generic cluster IDs to actual astrophysics states by analyzing 
        where their mathematical centroids sit on the Hardness Ratio axis.
        """
        centroids = df.groupby('Cluster_ID')[['Hardness_Ratio', 'Total_Intensity']].mean()
        
        # Sort clusters by their X-axis (Hardness Ratio)
        sorted_by_hardness = centroids.sort_values(by='Hardness_Ratio')
        
        state_mapping = {}
        n_clusters = len(centroids)
        
        for rank, (cluster_id, row) in enumerate(sorted_by_hardness.iterrows()):
            if rank == 0:
                name = "Soft State (Disk)"
            elif rank == n_clusters - 1:
                name = "Hard State (Corona)"
            else:
                name = f"Intermediate State {rank}"
                
            state_mapping[cluster_id] = name
            
        df['Physical_State'] = df['Cluster_ID'].map(state_mapping)
        return df

    def fit_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main execution method. Transforms data, optimizes GMM, and labels rows.
        """
        features = ['Hardness_Ratio', 'Total_Intensity']
        
        # Log-transform is mandatory because X-ray intensities span 
        # multiple orders of magnitude (from dim quiscence to violent outbursts)
        X = np.log10(df[features].values)

        best_n = self._determine_optimal_clusters(X)
        
        gmm = GaussianMixture(
            n_components=best_n, 
            covariance_type=self.covariance_type, 
            random_state=self.random_state
        )
        
        # Predict generates the raw integer labels (0, 1, 2...)
        df['Cluster_ID'] = gmm.fit_predict(X)
        
        # Translate the math into physics
        final_df = self._physically_name_states(df)
        
        return final_df