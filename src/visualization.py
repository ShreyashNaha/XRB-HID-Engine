"""
visualization.py
UI and Dashboard generation module.
Creates interactive Plotly Hardness-Intensity Diagrams and populates a 
static HTML template to generate the final dashboard.
"""

import plotly.express as px
import pandas as pd
from datetime import datetime, timezone
import shutil
from src.config import DASHBOARD_DIR, ASSETS_DIR

class DashboardBuilder:
    """Builds interactive visualizations and injects data into the HTML template."""
    
    def __init__(self):
        self.processed_targets = []
        self.css_dir = DASHBOARD_DIR / "css"
        self.css_dir.mkdir(parents=True, exist_ok=True)
        self.template_path = DASHBOARD_DIR / "template.html"

    def build_target_plot(self, df: pd.DataFrame, target_name: str) -> None:
        """
        Generates the Hardness-Intensity Diagram (HID) for a single target.
        Uses logarithmic axes and colors points by their AI-determined physical state.
        """
        clean_name = str(target_name).replace(" ", "_").replace("+", "p").replace("-", "m")
        file_path = ASSETS_DIR / f"{clean_name}.html"
        
        fig = px.scatter(
            df, 
            x='Hardness_Ratio', 
            y='Total_Intensity', 
            color='Physical_State',
            hover_data=['MJD_grid'],
            title=f"Hardness-Intensity Diagram: {target_name}",
            labels={
                'Hardness_Ratio': 'Hardness Ratio (NASA BAT / JAXA MAXI)',
                'Total_Intensity': 'Total Intensity (Proxy for Mass Accretion)',
                'Physical_State': 'Accretion State'
            },
            log_x=True, 
            log_y=True,
            template='plotly_dark'
        )
        
        fig.update_traces(mode='lines+markers', line=dict(width=1, color='rgba(255,255,255,0.2)'))
        fig.write_html(file_path, include_plotlyjs='cdn')
        
        self.processed_targets.append({
            'name': target_name,
            'file': f"assets/{clean_name}.html"
        })
        print(f"   [UI] Generated interactive plot for {target_name}")

    def _write_css(self) -> None:
        """Generates the external style.css file for clean separation of concerns."""
        css_content = """
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #111; color: #fff; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
.header { text-align: center; margin-bottom: 20px; }
h1 { color: #00d2ff; margin-bottom: 5px; }
.timestamp { color: #888; font-size: 0.9em; }
.controls { margin-bottom: 20px; padding: 15px; background-color: #222; border-radius: 8px; border: 1px solid #333; }
select { padding: 10px; font-size: 16px; background-color: #333; color: #fff; border: 1px solid #555; border-radius: 5px; cursor: pointer; outline: none; }
iframe { width: 100%; max-width: 1200px; height: 750px; border: none; border-radius: 8px; background-color: #000; }
"""
        css_path = self.css_dir / "style.css"
        with open(css_path, "w", encoding="utf-8") as f:
            f.write(css_content.strip())

    def build_index_html(self) -> None:
        """
        Reads the template HTML, injects the dynamic dropdown options and timestamp, 
        and saves the final dashboard as index.html.
        """
        if not self.processed_targets:
            print("[!] No targets processed. Skipping dashboard generation.")
            return
            
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found at {self.template_path}. Please create template.html.")

        self.processed_targets.sort(key=lambda x: x['name'])
        default_file = self.processed_targets[0]['file']
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S GMT")

        options_html = ""
        for target in self.processed_targets:
            options_html += f'            <option value="{target["file"]}">{target["name"]}</option>\n'

        self._write_css()

        # Read template, replace placeholders, and save as index.html
        with open(self.template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        final_html = template_content.replace("{{ TIMESTAMP }}", current_time)
        final_html = final_html.replace("{{ DROPDOWN_OPTIONS }}", options_html.strip())
        final_html = final_html.replace("{{ DEFAULT_PLOT }}", default_file)
        
        index_path = DASHBOARD_DIR / "index.html"
        with open(index_path, "w", encoding="utf-8") as f:
            f.write(final_html)
            
        print(f"[*] Dashboard successfully generated at {index_path}")