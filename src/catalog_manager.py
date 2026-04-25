"""
catalog_manager.py
The dynamic X-CORE Cross-Matching Engine.
Scrapes NASA and JAXA for new X-ray transients, resolves their celestial 
coordinates, and dynamically appends new discoveries to the master registry.
"""

import pandas as pd
import requests
import io
import re
from astropy.coordinates import SkyCoord
import astropy.units as u
from src.config import CATALOG_DIR, PipelineConfig

class CatalogManager:
    """Manages the dynamic discovery and logging of celestial targets."""
    
    def __init__(self):
        self.catalog_file = CATALOG_DIR / "master_catalog.csv"
        self.nasa_url = "https://swift.gsfc.nasa.gov/results/transients/index.html"
        self.jaxa_url = "http://maxi.riken.jp/top/slist.html"
        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self._ensure_catalog_exists()

    def _scrape_nasa_dynamic(self) -> pd.DataFrame:
        """Safely extracts NASA targets, immune to HTML table column shifting."""
        print("[*] Scanning NASA Swift-BAT transient monitor...")
        try:
            response = requests.get(self.nasa_url, headers=self.headers, timeout=PipelineConfig.REQUEST_TIMEOUT_SECONDS)
            dfs = pd.read_html(io.StringIO(response.text))
            
            # Find the largest table (which will be the massive transient list)
            df = max(dfs, key=len)
            
            valid_targets = []
            # We iterate blindly through rows, mathematically validating the data types 
            # instead of trusting NASA's column headers.
            for _, row in df.iterrows():
                row_vals = [str(x).strip() for x in row.values]
                
                # A valid row must have a Name (String) and RA/Dec (Floats)
                name, ra, dec = None, None, None
                
                for val in row_vals:
                    if not val or val.lower() in ['nan', 'none', '---']:
                        continue
                    
                    # Try to parse as coordinates
                    try:
                        num = float(val)
                        if 0 <= num <= 360 and ra is None:
                            ra = num
                        elif -90 <= num <= 90 and dec is None:
                            dec = num
                    except ValueError:
                        # If it's not a number, and it's not a generic category, it's the Name
                        if len(val) > 2 and "LMXB" not in val and "HMXB" not in val and name is None:
                            name = val
                
                if name and ra is not None and dec is not None:
                    valid_targets.append({'name': name, 'ra': ra, 'dec': dec})
                    
            return pd.DataFrame(valid_targets)
        except Exception as e:
            print(f"[!] NASA Scrape Failed (Website might be down): {e}")
            return pd.DataFrame()

    def _scrape_jaxa_dynamic(self) -> list:
        """Uses Regex to hunt for J-coordinate identifiers in JAXA's raw HTML."""
        print("[*] Scanning JAXA MAXI archives...")
        try:
            response = requests.get(self.jaxa_url, headers=self.headers, timeout=PipelineConfig.REQUEST_TIMEOUT_SECONDS)
            # Find all instances of J[HHMM][+/-][DDD]
            sources = re.findall(r"J\d{4}[+-]\d{3}", response.text)
            return list(set(sources))
        except Exception as e:
            print(f"[!] JAXA Scrape Failed: {e}")
            return []

    def _parse_jaxa_coords(self, j_id: str):
        """Converts J-names into Astropy SkyCoord objects."""
        try:
            match = re.match(r"J(\d{2})(\d{2})([+-])(\d{3})", j_id)
            if not match: return None
            hh, mm, sign, dd_dec = match.groups()
            
            ra_deg = (float(hh) + float(mm)/60.0) * 15.0
            dec_deg = float(dd_dec) / 10.0
            if sign == '-': dec_deg = -dec_deg
            
            return SkyCoord(ra=ra_deg*u.degree, dec=dec_deg*u.degree, frame='icrs')
        except:
            return None

    def _cross_match_and_update(self) -> None:
        """Finds overlaps and appends them to the master CSV safely."""
        nasa_df = self._scrape_nasa_dynamic()
        jaxa_ids = self._scrape_jaxa_dynamic()
        
        if nasa_df.empty or not jaxa_ids:
            print("[!] Could not complete dynamic scan today. Using existing database.")
            return

        print(f"[*] Cross-matching {len(nasa_df)} NASA sources against {len(jaxa_ids)} JAXA targets...")
        
        # Load existing targets so we don't duplicate
        existing_targets = []
        if self.catalog_file.exists() and self.catalog_file.stat().st_size > 10:
            existing_df = pd.read_csv(self.catalog_file)
            existing_targets = existing_df['Target_Name'].tolist()

        nasa_coords = SkyCoord(ra=nasa_df['ra'].values*u.degree, dec=nasa_df['dec'].values*u.degree)
        new_discoveries = []

        for j_id in jaxa_ids:
            jaxa_coord = self._parse_jaxa_coords(j_id)
            if jaxa_coord is None: continue

            d2d = jaxa_coord.separation(nasa_coords)
            
            # 0.5 degrees tolerance for matching
            if len(d2d) > 0 and d2d.min() < 0.5 * u.degree:
                idx = d2d.argmin()
                match = nasa_df.iloc[idx]
                target_name = str(match['name'])
                
                # Only add if we haven't seen this system before
                if target_name not in existing_targets:
                    clean_bat = target_name.replace(" ", "")
                    new_discoveries.append({
                        "Target_Name": target_name,
                        "Type": "X-Ray Binary", 
                        "BAT_ID": clean_bat,
                        "MAXI_ID": j_id,
                        "BAT_Filename": f"{clean_bat}.lc.txt",
                        "MAXI_Filename": f"{j_id}_g_lc_1day_all.dat",
                        "MAXI_URL": f"http://maxi.riken.jp/star_data/{j_id}/{j_id}_g_lc_1day_all.dat"
                    })

        if new_discoveries:
            print(f"[*] Discovery! Added {len(new_discoveries)} new targets to the registry.")
            new_df = pd.DataFrame(new_discoveries)
            
            if self.catalog_file.exists() and self.catalog_file.stat().st_size > 10:
                new_df.to_csv(self.catalog_file, mode='a', header=False, index=False)
            else:
                new_df.to_csv(self.catalog_file, index=False)
        else:
            print("[*] No new targets discovered today. Registry is up to date.")

    def _ensure_catalog_exists(self) -> None:
        """Triggers the cross-match update automatically."""
        self._cross_match_and_update()

    def get_targets(self) -> list[dict]:
        """Returns the full master catalog for the pipeline to process."""
        if self.catalog_file.exists() and self.catalog_file.stat().st_size > 10:
            df = pd.read_csv(self.catalog_file)
            return df.to_dict('records')
        return []