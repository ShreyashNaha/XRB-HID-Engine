"""
data_loader.py
Object-Oriented data ingestion module.
Downloads daily raw telemetry from JAXA and NASA, using a safe caching mechanism 
to prevent rate-limiting or server abuse.
"""

import os
import time
import requests
import pandas as pd
from src.config import RAW_DATA_DIR, PipelineConfig

class BaseDataLoader:
    """Base class handling robust file downloading and cache validation."""
    
    def __init__(self, target_dict: dict, observatory_name: str, filename_key: str):
        self.target = target_dict
        self.name = f"{observatory_name} ({self.target['Target_Name']})"
        self.local_path = RAW_DATA_DIR / self.target[filename_key]

    def _is_cache_valid(self) -> bool:
        """Checks if the local data is fresh enough to avoid re-downloading."""
        if not self.local_path.exists():
            return False
        file_mod_time = os.path.getmtime(self.local_path)
        hours_since_update = (time.time() - file_mod_time) / 3600.0
        return hours_since_update < PipelineConfig.CACHE_EXPIRY_HOURS

    def _download_data(self, url: str) -> bool:
        """Safely pulls data from the target URL."""
        try:
            response = requests.get(url, timeout=PipelineConfig.REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            with open(self.local_path, 'wb') as f:
                f.write(response.content)
            return True
        except Exception as e:
            # We silently catch connection errors so the pipeline can skip the target 
            # rather than crashing the entire daily run.
            return False

    def get_data(self) -> pd.DataFrame:
        """Main entry point: fetches data if needed, then parses it."""
        if not self._is_cache_valid():
            if not self._fetch_logic():
                if not self.local_path.exists():
                    raise RuntimeError(f"Data unavailable for {self.name}")
        return self._parse_data()

    def _fetch_logic(self) -> bool:
        raise NotImplementedError

    def _parse_data(self) -> pd.DataFrame:
        raise NotImplementedError

class MaxiLoader(BaseDataLoader):
    """Handles JAXA MAXI soft X-ray telemetry (2-20 keV)."""
    
    def __init__(self, target_dict: dict):
        super().__init__(target_dict, "MAXI", "MAXI_Filename")
        
    def _fetch_logic(self) -> bool:
        print(f"   [MAXI] Fetching daily telemetry for {self.target['Target_Name']}...")
        return self._download_data(self.target['MAXI_URL'])
    
    def _parse_data(self) -> pd.DataFrame:
        # MAXI provides multiple energy bands. We extract MJD (Time), 
        # and standard soft/hard bands to allow accurate physical intensity scaling.
        # Column 0: MJD, Column 1: 2-4 keV, Column 3: 4-10 keV
        cols = ["MJD", "Soft_Flux", "Soft_Err", "Hard_Flux", "Hard_Err"]
        df = pd.read_csv(
            self.local_path, 
            sep=r'\s+', 
            comment='#', 
            names=cols, 
            usecols=[0, 1, 2, 3, 4], 
            engine='c'
        )
        return df.dropna().astype('float32')

class BatLoader(BaseDataLoader):
    """Handles NASA Swift-BAT hard X-ray telemetry (15-50 keV)."""
    
    def __init__(self, target_dict: dict):
        super().__init__(target_dict, "Swift-BAT", "BAT_Filename")
        
    def _fetch_logic(self) -> bool:
        print(f"   [BAT] Fetching daily telemetry for {self.target['Target_Name']}...")
        base = PipelineConfig.BAT_BASE_URL
        primary_url = f"{base}/{self.target['BAT_ID']}.lc.txt"
        
        # If the target is dim, NASA puts it in the 'weak' directory. We check both.
        if self._download_data(primary_url): 
            return True
        weak_url = f"{PipelineConfig.BAT_WEAK_URL}/{self.target['BAT_ID']}.lc.txt"
        return self._download_data(weak_url)
    
    def _parse_data(self) -> pd.DataFrame:
        # BAT provides MJD, Count Rate, and Error.
        df = pd.read_csv(
            self.local_path, 
            sep=r'\s+', 
            comment='#', 
            names=["MJD", "RATE", "ERROR"], 
            usecols=[0, 1, 2], 
            engine='c'
        )
        return df.dropna().astype('float32')