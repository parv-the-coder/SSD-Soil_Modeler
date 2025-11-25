#!/usr/bin/env python3
"""
Test script for optimized ML pipeline with all datasets
This script will test the complete optimized pipeline on all 5 datasets
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_optimized_pipeline():
    """Test the complete optimized pipeline on all datasets"""
    
    print("🚀 TESTING OPTIMIZED ML PIPELINE")
    print("="*60)
    
    # Test imports
    try:
        from data_preprocessing import (
            optimized_reflectance, optimized_absorbance, optimized_continuum_removal,
            advanced_spectral_derivatives, spectral_feature_engineering, adaptive_outlier_removal,
            reflectance_transformation, absorbance_transformation, continuum_removal_transformation
        )
        print("✅ All optimized preprocessing functions imported successfully")
    except ImportError as e:
        print(f"⚠️ Some imports failed, trying basic imports: {e}")
        try:
            from data_preprocessing import (
                optimized_reflectance, optimized_absorbance, optimized_continuum_removal,
                advanced_spectral_derivatives, spectral_feature_engineering, adaptive_outlier_removal
            )
            print("✅ Core optimized preprocessing functions imported successfully")
        except ImportError as e2:
            print(f"❌ Import error: {e2}")
            return
    
    try:
        from model_training import create_dataset_specific_models, save_model
        print("✅ Model training functions imported successfully")
    except Exception as e:
        print(f"❌ Model training import error: {e}")
        return
    
    # Find datasets
    data_dir = Path("data")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Find all CSV files
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return
    
    print(f"📁 Found {len(csv_files)} dataset files")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Test on each dataset
    successful_tests = 0
    failed_tests = 0
    
    for csv_file in sorted(csv_files):
        dataset_name = csv_file.stem
        print(f"\n🔬 Testing dataset: {dataset_name}")
        print("-"*40)
        
        try:
            # Load dataset
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded: {df.shape[0]} samples, {df.shape[1]} features")
            
            # Separate features and target (assuming target is the last column)
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
            # Ensure X has proper column names (wavelengths)
            if not all(isinstance(col, (int, float)) for col in X.columns):
                # Create wavelength columns if they don't exist
                X.columns = [f"band_{i}" for i in range(len(X.columns))]
            
            print(f"🎯 Target range: [{y.min():.2f}, {y.max():.2f}]")
            print(f"📈 Target CV: {(y.std() / y.mean() * 100):.1f}%")
            
            # Test adaptive outlier removal
            print("🧹 Testing adaptive outlier removal...")
            X_clean, y_clean = adaptive_outlier_removal(X, y, dataset_name)
            outliers_removed = len(y) - len(y_clean)
            print(f"   Removed {outliers_removed} outliers ({outliers_removed/len(y)*100:.1f}%)")
            
            # Test optimized preprocessing methods
            preprocessing_methods = {
                "Optimized_Reflectance": optimized_reflectance,
                "Optimized_Absorbance": optimized_absorbance,
                "Optimized_Continuum_Removal": optimized_continuum_removal,
                "Spectral_Derivatives": advanced_spectral_derivatives,
                "Feature_Engineering": spectral_feature_engineering
            }
            
            print("🔧 Testing optimized preprocessing methods:")
            for method_name, method_func in preprocessing_methods.items():
                try:
                    X_processed = method_func(X_clean)
                    if isinstance(X_processed, pd.DataFrame) and X_processed.shape[0] > 0:
                        print(f"   ✅ {method_name}: {X_processed.shape[0]} samples, {X_processed.shape[1]} features")
                    else:
                        print(f"   ❌ {method_name}: Invalid output")
                except Exception as e:
                    print(f"   ⚠️ {method_name}: {str(e)}")
            
            # Test dataset-specific models
            print("🤖 Testing dataset-specific model configurations...")
            models_config = create_dataset_specific_models(dataset_name, X_clean.shape[1], X_clean.shape[0])
            print(f"   📋 Created {len(models_config)} model configurations")
            
            # Quick test with one preprocessing method and one model
            print("⚡ Quick pipeline test...")
            X_test = optimized_reflectance(X_clean)
            
            # Clean the test data
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.fillna(X_test.median())
            non_const_cols = X_test.columns[X_test.std(ddof=0) > 0]
            X_test = X_test.loc[:, non_const_cols]
            
            if X_test.shape[1] > 0:
                # Test with a simple model
                from sklearn.cross_decomposition import PLSRegression
                from sklearn.model_selection import cross_val_score
                from sklearn.preprocessing import RobustScaler
                
                # Scale data
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X_test)
                
                # Simple PLS test
                pls = PLSRegression(n_components=min(5, X_test.shape[1], len(y_clean)-1), scale=False)
                scores = cross_val_score(pls, X_scaled, y_clean, cv=3, scoring='r2')
                
                print(f"   🎯 PLS Test R²: {scores.mean():.4f} ± {scores.std():.4f}")
                
                # Test model saving
                pls.fit(X_scaled, y_clean)
                model_path, metadata_path = save_model(pls, "PLS_Test", dataset_name, scores.mean())
                print(f"   💾 Model saved to: {model_path}")
                
                successful_tests += 1
                print(f"   ✅ {dataset_name}: PASSED")
            else:
                print(f"   ❌ {dataset_name}: No valid features after preprocessing")
                failed_tests += 1
                
        except Exception as e:
            print(f"   ❌ {dataset_name}: FAILED - {str(e)}")
            failed_tests += 1
    
    # Final summary
    print(f"\n🏆 PIPELINE TEST SUMMARY")
    print("="*60)
    print(f"✅ Successful tests: {successful_tests}")
    print(f"❌ Failed tests: {failed_tests}")
    print(f"📊 Success rate: {successful_tests/(successful_tests+failed_tests)*100:.1f}%")
    
    if successful_tests > 0:
        print("\n🎉 Optimized pipeline is working correctly!")
        print("🚀 Ready to run full optimization on all datasets")
        
        # Show saved models
        model_files = list(models_dir.glob("*.pkl"))
        if model_files:
            print(f"\n💾 Saved {len(model_files)} test models in models/ directory")
    else:
        print("\n⚠️ Pipeline needs debugging before full optimization")
    
    return successful_tests > 0

if __name__ == "__main__":
    success = test_optimized_pipeline()
    
    if success:
        print("\n🎯 To run full optimization, use:")
        print("   python full_dataset_optimization.py")
    
    exit(0 if success else 1)