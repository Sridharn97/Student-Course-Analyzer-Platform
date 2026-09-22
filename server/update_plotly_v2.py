import os
import re

views_dir = r"d:\Projects\Student Course Analyzer Platform\views"

for root, _, files in os.walk(views_dir):
    for file in files:
        if file.endswith(".py") and file not in ("ui_utils.py", "__init__.py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            
            if "st.plotly_chart" in content:
                orig_content = content
                if "apply_premium_plotly_layout" not in content:
                    content = content.replace("import streamlit as st", "import streamlit as st\nfrom .ui_utils import apply_premium_plotly_layout")
                
                # Replace
                content = re.sub(r'st\.plotly_chart\s*\(\s*([a-zA-Z0-9_]+)\s*,', r'st.plotly_chart(apply_premium_plotly_layout(\1),', content)
                
                if content != orig_content:
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(content)
                    print(f"Updated {filepath}")
                else:
                    print(f"No changes made to {filepath}")
