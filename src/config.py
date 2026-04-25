"""
config.py
Central configuration file for the BH-NS HID Pipeline.
Defines directory structures, observatory URL templates, and Machine Learning hyperparameters.
"""

from pathlib import Path

# Base directory paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CATALOG_DIR = DATA_DIR / "catalogs"

# UI/Dashboard directories
DASHBOARD_DIR = ROOT_DIR / "dashboard"
ASSETS_DIR = DASHBOARD_DIR / "assets"

# Automatically ensure all required directories exist on startup
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CATALOG_DIR, ASSETS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class PipelineConfig:
    """Settings for data ingestion, caching, and ML processing."""
    
    # Data Fetching Settings
    REQUEST_TIMEOUT_SECONDS = 30
    # 24 hours expiry ensures the GitHub Action downloads fresh daily data 
    # but local testing won't spam the servers if run multiple times a day.
    CACHE_EXPIRY_HOURS = 24.0  
    
    # Observatory Master URLs
    MAXI_BASE_URL = "http://maxi.riken.jp/star_data"
    BAT_BASE_URL = "https://swift.gsfc.nasa.gov/results/transients"
    BAT_WEAK_URL = "https://swift.gsfc.nasa.gov/results/transients/weak"
    
    # Gaussian Mixture Model (GMM) Hyperparameters
    # We cap components at 5 to prevent overfitting; physical states rarely exceed 4 or 5
    GMM_MAX_COMPONENTS = 5 
    GMM_COVARIANCE_TYPE = 'full'
    RANDOM_STATE = 42 # Ensures mathematical reproducibility across GitHub and Local