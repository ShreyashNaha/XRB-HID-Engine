"""
run_pipeline.py
The master orchestrator script for the X-CORE Pipeline.
Coordinates catalog generation, data downloading, preprocessing, 
machine learning clustering, and dashboard generation.
Includes a CI/CD-safe colored progress tracker.
"""

import sys
import traceback
from src.catalog_manager import CatalogManager
from src.data_loader import MaxiLoader, BatLoader
from src.preprocessor import DataPreprocessor
from src.model import StateClusterer
from src.visualization import DashboardBuilder

# --- ANSI Color Codes for Terminal UI ---
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

def main():
    print(f"\n{BOLD}{CYAN}=== STARTING: Unsupervised Black Hole & Neutron Star XRB HID Clustering Engine==={RESET}\n")
    
    # Step 1: Initialize the dynamic catalog and get targets
    catalog_manager = CatalogManager()
    targets = catalog_manager.get_targets()
    
    total_targets = len(targets)
    if total_targets == 0:
        print(f"{RED}[!] Critical Error: Master catalog is empty. Exiting.{RESET}")
        sys.exit(1)
        
    print(f"{GREEN}[*] Loaded {total_targets} targets from master registry.{RESET}\n")
    
    # Step 2: Initialize Machine Learning and UI Builders
    clusterer = StateClusterer()
    dashboard_builder = DashboardBuilder()
    
    success_count = 0
    skip_count = 0
    
    # Step 3: Process each target with a colored progress tracker
    for i, target in enumerate(targets, 1):
        target_name = target['Target_Name']
        target_type = target.get('Type', 'Unknown')
        progress_pct = (i / total_targets) * 100
        
        # The Progress Header
        print(f"{CYAN}[{i}/{total_targets} | {progress_pct:.1f}%] {BOLD}Analyzing: {target_name} {RESET}({target_type})")
        
        try:
            # 3a. Download/Load Data
            maxi_loader = MaxiLoader(target)
            bat_loader = BatLoader(target)
            
            maxi_df = maxi_loader.get_data()
            bat_df = bat_loader.get_data()
            
            # 3b. Preprocess and Align Timeseries
            preprocessor = DataPreprocessor(maxi_df, bat_df, target_name)
            aligned_df = preprocessor.process()
            
            # 3c. Unsupervised ML Clustering
            clustered_df = clusterer.fit_predict(aligned_df)
            
            # 3d. Build Interactive Plot
            dashboard_builder.build_target_plot(clustered_df, target_name)
            success_count += 1
            print(f"   {GREEN}↳ SUCCESS: Interactive plot generated.{RESET}\n")
            
        except ValueError as ve:
            # Handles "Insufficient overlapping daily telemetry" gracefully
            skip_count += 1
            print(f"   {YELLOW}↳ SKIPPED: {str(ve)}{RESET}\n")
            continue
        except Exception as e:
            # Handles unexpected crashes
            skip_count += 1
            print(f"   {RED}↳ FAILED: {str(e)}{RESET}\n")
            # traceback.print_exc() # Uncomment for deep debugging
            continue

    # Step 4: Finalize Dashboard
    print(f"{BOLD}{CYAN}=== PIPELINE COMPLETE ==={RESET}")
    print(f"[*] Processed: {GREEN}{success_count} Successful{RESET} | {YELLOW}{skip_count} Skipped/Dim{RESET}")
    
    print("\n[*] Generating main dashboard UI...")
    dashboard_builder.build_index_html()
    print(f"{GREEN}[*] All systems nominal. You can now open dashboard/index.html{RESET}\n")

if __name__ == "__main__":
    main()