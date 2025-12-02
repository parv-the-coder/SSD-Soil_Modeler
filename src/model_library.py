"""
Model Library Page - View and download all trained models
"""

import streamlit as st
import os
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime
import base64
import re
import zipfile
import tempfile
import matplotlib.pyplot as plt

def get_file_creation_time(filepath):
    """Get file creation/modification time"""
    try:
        stat = os.stat(filepath)
        return datetime.fromtimestamp(stat.st_mtime)
    except:
        return None

def get_model_info(model_path):
    """Extract basic information from a pickled model"""
    info = {
        'size_kb': round(os.path.getsize(model_path) / 1024, 2),
        'created': get_file_creation_time(model_path),
        'features_count': None,
        'performance': None
    }
    
    # Try to extract model name and additional info
    model_name = Path(model_path).stem
    
    # Check for associated files
    base_path = Path(model_path).parent
    model_id = model_name.replace('best_model_', '')
    
    # Check for feature file
    feature_file = base_path / f"best_model_{model_id}_features.txt"
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            features = f.readlines()
            info['features_count'] = len(features)
    
    # Check for log file
    log_file = base_path / f"best_model_{model_id}_log.txt"
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                log_content = f.read()
                # Extract R² and RMSE from log
                r2_match = re.search(r'R2=([0-9.-]+)', log_content)
                rmse_match = re.search(r'RMSE=([0-9.-]+)', log_content)
                if r2_match and rmse_match:
                    info['performance'] = f"R²: {r2_match.group(1)}, RMSE: {rmse_match.group(1)}"
        except:
            pass
    
    return info

def show_model_library():
    """Main function for the Model Library page - modified to work within existing app"""
    
    # Custom CSS for model library
    st.markdown("""
    <style>
        .model-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 5px solid #2a7143;
        }
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .model-name {
            font-size: 1.2rem;
            font-weight: bold;
            color: #2a7143;
        }
        .model-size {
            color: #666;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📚 Model Library")
    st.markdown("""
    Browse and download all trained machine learning models. Each model includes:
    - **Model file (.pkl)** - The trained model for predictions
    - **Features file (.txt)** - List of wavelengths/features used
    - **Log file (.txt)** - Training performance metrics and details
    """)
    
    # Check if models directory exists
    models_dir = Path("models")
    if not models_dir.exists():
        models_dir.mkdir(exist_ok=True)
    
    # Find all model files
    model_files = list(models_dir.glob("best_model_*.pkl"))
    
    if not model_files:
        st.warning("📭 No trained models found in the models directory.")
        st.info("Go to the **Train Models** tab to train your first model.")
        return
    
    st.success(f"✅ Found {len(model_files)} trained model(s)")
    
    # Create tabs for different views
    tab1, tab2 = st.tabs(["📊 Model Dashboard", "📥 Bulk Download"])
    
    with tab1:
        # Display all models in a dashboard
        for model_file in sorted(model_files):
            model_path = str(model_file)
            model_name = model_file.stem.replace('best_model_', '').upper()
            model_info = get_model_info(model_path)
            
            # Get associated files
            model_id = model_file.stem.replace('best_model_', '')
            feature_file = models_dir / f"best_model_{model_id}_features.txt"
            log_file = models_dir / f"best_model_{model_id}_log.txt"
            
            # Create model card using columns instead of HTML
            with st.container():
                header_col1, header_col2 = st.columns([0.8, 0.2])
                
                with header_col1:
                    st.markdown(f"**Model {model_name}**")
                with header_col2:
                    st.markdown(f"📦 {model_info['size_kb']} KB")
                
                # Combined metrics and download in aligned columns
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if model_info['created']:
                        st.metric("Created", model_info['created'].strftime("%Y-%m-%d"))
                    st.markdown("##### ")
                    if os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            st.download_button(
                                label="📥 Download Model (.pkl)",
                                data=f,
                                file_name=f"best_model_{model_id}.pkl",
                                mime="application/octet-stream",
                                use_container_width=True,
                                key=f"dl_model_{model_id}"
                            )
                
                with col2:
                    if model_info['features_count']:
                        st.metric("Features", model_info['features_count'])
                    st.markdown("##### ")
                    if feature_file.exists():
                        with open(feature_file, "rb") as f:
                            st.download_button(
                                label="📥 Download Features (.txt)",
                                data=f,
                                file_name=f"best_model_{model_id}_features.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key=f"dl_features_{model_id}"
                            )
                    else:
                        st.button(
                            "❌ Features missing",
                            disabled=True,
                            use_container_width=True,
                            key=f"no_features_{model_id}"
                        )
                
                with col3:
                    st.metric("File", model_file.suffix)
                    st.markdown("##### ")
                    if log_file.exists():
                        with open(log_file, "rb") as f:
                            st.download_button(
                                label="📥 Download Log (.txt)",
                                data=f,
                                file_name=f"best_model_{model_id}_log.txt",
                                mime="text/plain",
                                use_container_width=True,
                                key=f"dl_log_{model_id}"
                            )
                    else:
                        st.button(
                            "❌ Log missing",
                            disabled=True,
                            use_container_width=True,
                            key=f"no_log_{model_id}"
                        )
                
                st.markdown("---")
    
    with tab2:
        st.subheader("📦 Bulk Download All Models")
        
        # Create zip file of all models
        if st.button("📦 Create Zip Archive of All Models", use_container_width=True):
            with st.spinner("Creating archive..."):
                # Create temporary zip file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
                    with zipfile.ZipFile(tmp_zip.name, 'w') as zipf:
                        # Add all model files
                        for model_file in model_files:
                            model_id = model_file.stem.replace('best_model_', '')
                            zipf.write(model_file, f"models/{model_file.name}")
                            
                            # Add feature file if exists
                            feature_file = models_dir / f"best_model_{model_id}_features.txt"
                            if feature_file.exists():
                                zipf.write(feature_file, f"models/{feature_file.name}")
                            
                            # Add log file if exists
                            log_file = models_dir / f"best_model_{model_id}_log.txt"
                            if log_file.exists():
                                zipf.write(log_file, f"models/{log_file.name}")
                    
                    # Create download button for zip
                    with open(tmp_zip.name, "rb") as f:
                        st.download_button(
                            label="📥 Download All Models as ZIP",
                            data=f,
                            file_name=f"spectral_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
        
        st.markdown("---")
        
        # Display all files in a table
        st.subheader("📋 File Inventory")
        
        files_data = []
        for model_file in sorted(model_files):
            model_id = model_file.stem.replace('best_model_', '')
            model_info = get_model_info(str(model_file))
            
            files_data.append({
                'Model': model_id.upper(),
                'Type': 'Model (.pkl)',
                'File': model_file.name,
                'Size (KB)': f"{model_info['size_kb']:.2f}" if model_info['size_kb'] is not None else 'Unknown',
                'Features': str(model_info['features_count']) if model_info['features_count'] else 'N/A',
                'Created': model_info['created'].strftime("%Y-%m-%d %H:%M") if model_info['created'] else 'Unknown'
            })
            
            # Add feature file info
            feature_file = models_dir / f"best_model_{model_id}_features.txt"
            if feature_file.exists():
                size_kb = round(os.path.getsize(feature_file) / 1024, 2)
                created = get_file_creation_time(feature_file)
                files_data.append({
                    'Model': model_id.upper(),
                    'Type': 'Features (.txt)',
                    'File': feature_file.name,
                    'Size (KB)': f"{size_kb:.2f}",
                    'Features': 'N/A',
                    'Created': created.strftime("%Y-%m-%d %H:%M") if created else 'Unknown'
                })
            
            # Add log file info
            log_file = models_dir / f"best_model_{model_id}_log.txt"
            if log_file.exists():
                size_kb = round(os.path.getsize(log_file) / 1024, 2)
                created = get_file_creation_time(log_file)
                files_data.append({
                    'Model': model_id.upper(),
                    'Type': 'Log (.txt)',
                    'File': log_file.name,
                    'Size (KB)': f"{size_kb:.2f}",
                    'Features': 'N/A',
                    'Created': created.strftime("%Y-%m-%d %H:%M") if created else 'Unknown'
                })
        
        if files_data:
            df = pd.DataFrame(files_data)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Model": st.column_config.TextColumn("Model"),
                    "Type": st.column_config.TextColumn("File Type"),
                    "File": st.column_config.TextColumn("File Name"),
                    "Size (KB)": st.column_config.TextColumn("Size (KB)"),
                    "Features": st.column_config.TextColumn("Features"),
                    "Created": st.column_config.TextColumn("Created"),
                }
            )
            
            # Export table as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download File Inventory (CSV)",
                data=csv,
                file_name="model_inventory.csv",
                mime="text/csv",
                use_container_width=True
            )