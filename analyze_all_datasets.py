#!/usr/bin/env python3
"""
🔬 COMPREHENSIVE DATASET ANALYSIS
Analyze all spectral datasets to understand characteristics and optimize preprocessing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

def analyze_dataset(filepath):
    """Analyze a single dataset and return characteristics"""
    print(f"\n🔬 ANALYZING: {os.path.basename(filepath)}")
    print("="*60)
    
    # Load dataset
    try:
        data = pd.read_csv(filepath)
        print(f"✅ Dataset loaded successfully!")
    except:
        try:
            data = pd.read_excel(filepath)
            print(f"✅ Dataset loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    # Basic structure
    print(f"📊 DATASET STRUCTURE:")
    print(f"   Samples: {data.shape[0]}")
    print(f"   Features: {data.shape[1] - 1} (spectral bands)")
    print(f"   Total columns: {data.shape[1]}")
    
    # Extract wavelengths and target
    wavelengths = data.columns[:-1]
    target_values = data.iloc[:, -1]
    spectral_data = data.iloc[:, :-1]
    
    try:
        wavelength_nums = wavelengths.astype(float)
        print(f"   Wavelength range: {wavelength_nums.min():.0f} - {wavelength_nums.max():.0f} nm")
    except:
        print(f"   Wavelength columns: {len(wavelengths)}")
    
    # Target analysis
    print(f"\n🎯 TARGET VARIABLE ANALYSIS:")
    print(f"   Column name: {data.columns[-1]}")
    print(f"   Mean: {target_values.mean():.3f}")
    print(f"   Std Dev: {target_values.std():.3f}")
    print(f"   Min: {target_values.min():.3f}")
    print(f"   Max: {target_values.max():.3f}")
    print(f"   Range: {target_values.max() - target_values.min():.3f}")
    print(f"   CV (Coeff. Variation): {(target_values.std()/target_values.mean()*100):.1f}%")
    
    # Data distribution analysis
    skewness = stats.skew(target_values)
    kurtosis = stats.kurtosis(target_values)
    print(f"   Skewness: {skewness:.3f}")
    print(f"   Kurtosis: {kurtosis:.3f}")
    
    # Spectral data analysis
    print(f"\n🌈 SPECTRAL DATA ANALYSIS:")
    print(f"   Reflectance range: {spectral_data.min().min():.4f} - {spectral_data.max().max():.4f}")
    print(f"   Mean reflectance: {spectral_data.mean().mean():.4f}")
    print(f"   Reflectance variability (CV): {(spectral_data.std().mean()/spectral_data.mean().mean()*100):.1f}%")
    
    # Data quality assessment
    print(f"\n🔍 DATA QUALITY:")
    missing_values = data.isnull().sum().sum()
    duplicate_rows = data.duplicated().sum()
    print(f"   Missing values: {missing_values}")
    print(f"   Duplicate rows: {duplicate_rows}")
    
    # Outlier detection
    z_scores = np.abs(stats.zscore(target_values))
    outliers_3std = np.sum(z_scores > 3)
    outliers_2std = np.sum(z_scores > 2)
    print(f"   Outliers (Z > 3): {outliers_3std} ({outliers_3std/len(target_values)*100:.1f}%)")
    print(f"   Outliers (Z > 2): {outliers_2std} ({outliers_2std/len(target_values)*100:.1f}%)")
    
    # Spectral characteristics analysis
    print(f"\n📈 SPECTRAL CHARACTERISTICS:")
    
    # Calculate spectral slopes (general trend)
    spectral_slopes = []
    for i in range(len(spectral_data)):
        spectrum = spectral_data.iloc[i].values
        if len(wavelength_nums) == len(spectrum):
            slope = np.polyfit(wavelength_nums, spectrum, 1)[0]
            spectral_slopes.append(slope)
    
    if spectral_slopes:
        print(f"   Mean spectral slope: {np.mean(spectral_slopes):.6f}")
        print(f"   Spectral slope variability: {np.std(spectral_slopes):.6f}")
    
    # Find absorption bands (local minima regions)
    mean_spectrum = spectral_data.mean(axis=0).values
    if len(wavelength_nums) == len(mean_spectrum):
        # Common soil absorption regions
        absorption_regions = {
            "Iron_oxides_400-700": (400, 700),
            "Water_1400-1500": (1400, 1500), 
            "Water_1900-2000": (1900, 2000),
            "Clay_minerals_2200-2250": (2200, 2250)
        }
        
        print(f"   Key absorption regions:")
        for region_name, (start_wl, end_wl) in absorption_regions.items():
            try:
                mask = (wavelength_nums >= start_wl) & (wavelength_nums <= end_wl)
                if mask.any():
                    region_values = mean_spectrum[mask]
                    min_refl = np.min(region_values)
                    print(f"     {region_name}: Min reflectance = {min_refl:.4f}")
            except:
                pass
    
    # Correlation with target
    correlations = []
    for col in spectral_data.columns:
        corr = spectral_data[col].corr(target_values)
        if not np.isnan(corr):
            correlations.append(abs(corr))
    
    if correlations:
        max_corr = max(correlations)
        mean_corr = np.mean(correlations)
        print(f"\n🔗 SPECTRAL-TARGET CORRELATION:")
        print(f"   Max correlation: {max_corr:.3f}")
        print(f"   Mean correlation: {mean_corr:.3f}")
        print(f"   Strong correlations (>0.5): {sum(c > 0.5 for c in correlations)}")
        print(f"   Moderate correlations (0.3-0.5): {sum(0.3 <= c <= 0.5 for c in correlations)}")
    
    # Determine optimal preprocessing recommendations
    print(f"\n💡 PREPROCESSING RECOMMENDATIONS:")
    
    # Based on data characteristics, recommend preprocessing
    recommendations = []
    
    if target_values.max() / target_values.min() > 10:
        recommendations.append("Log transformation for target (high dynamic range)")
    
    if spectral_data.max().max() > 1.0:
        recommendations.append("Normalization to [0,1] range")
    
    if outliers_2std > len(target_values) * 0.05:  # >5% outliers
        recommendations.append("Robust outlier removal (Z-score < 2)")
    
    if mean_corr < 0.3:
        recommendations.append("Advanced feature engineering (derivatives, ratios)")
    
    if spectral_data.mean().mean() < 0.1:
        recommendations.append("Absorbance transformation (log(1/R))")
    else:
        recommendations.append("Continuum removal for mineral identification")
    
    recommendations.append("Savitzky-Golay smoothing for noise reduction")
    recommendations.append("First/Second derivatives for band enhancement")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Performance prediction
    print(f"\n🎯 EXPECTED PERFORMANCE:")
    
    if max_corr > 0.7:
        performance_level = "EXCELLENT (R² > 0.8 expected)"
    elif max_corr > 0.5:
        performance_level = "VERY GOOD (R² 0.6-0.8 expected)"
    elif max_corr > 0.3:
        performance_level = "MODERATE (R² 0.4-0.6 expected)"
    else:
        performance_level = "CHALLENGING (R² < 0.4, needs advanced techniques)"
    
    print(f"   Performance outlook: {performance_level}")
    
    # Return analysis results
    return {
        'filepath': filepath,
        'dataset_name': os.path.basename(filepath),
        'n_samples': data.shape[0],
        'n_features': data.shape[1] - 1,
        'target_mean': target_values.mean(),
        'target_std': target_values.std(),
        'target_range': target_values.max() - target_values.min(),
        'target_cv': target_values.std()/target_values.mean()*100,
        'target_skewness': skewness,
        'outliers_2std': outliers_2std,
        'outliers_3std': outliers_3std,
        'max_correlation': max_corr if correlations else 0,
        'mean_correlation': mean_corr if correlations else 0,
        'reflectance_range': spectral_data.max().max() - spectral_data.min().min(),
        'recommendations': recommendations,
        'performance_outlook': performance_level,
        'data': data
    }

def analyze_all_datasets():
    """Analyze all datasets in the data directory"""
    
    print("🔬 COMPREHENSIVE SPECTRAL DATASET ANALYSIS")
    print("="*80)
    print("📊 Analyzing all datasets for optimal preprocessing and modeling")
    print()
    
    data_dir = '/home/gaurav-patel/SSD/End_Project/SSD-Soil_Modeler/data'
    
    # Find all data files
    data_files = []
    for file in os.listdir(data_dir):
        if file.endswith(('.xls', '.xlsx', '.csv')):
            data_files.append(os.path.join(data_dir, file))
    
    print(f"📁 Found {len(data_files)} datasets to analyze:")
    for file in data_files:
        print(f"   - {os.path.basename(file)}")
    
    # Analyze each dataset
    analysis_results = []
    for filepath in data_files:
        result = analyze_dataset(filepath)
        if result:
            analysis_results.append(result)
    
    # Summary comparison
    print(f"\n📊 DATASET COMPARISON SUMMARY")
    print("="*80)
    
    if analysis_results:
        print(f"{'Dataset':<25} {'Samples':<8} {'Features':<9} {'Target_CV':<10} {'Max_Corr':<10} {'Outlook':<15}")
        print("-" * 80)
        
        for result in analysis_results:
            dataset_name = result['dataset_name'][:23]
            samples = result['n_samples']
            features = result['n_features']
            target_cv = result['target_cv']
            max_corr = result['max_correlation']
            outlook = result['performance_outlook'].split()[0]
            
            print(f"{dataset_name:<25} {samples:<8} {features:<9} {target_cv:<10.1f} {max_corr:<10.3f} {outlook:<15}")
    
    print(f"\n💡 OVERALL RECOMMENDATIONS:")
    print("1. 🔧 Implement adaptive preprocessing based on dataset characteristics")
    print("2. 📈 Use ensemble methods for robust performance across datasets")
    print("3. 🎯 Apply dataset-specific optimization strategies")
    print("4. 🧹 Implement smart outlier detection and removal")
    print("5. ⚡ Use cross-validation for reliable performance estimates")
    
    return analysis_results

if __name__ == "__main__":
    results = analyze_all_datasets()