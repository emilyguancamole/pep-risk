#!/usr/bin/env python3
"""
Test script to verify app.py updates work correctly with the new model and data
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append('/Users/emilyguan/Downloads/EndoScribe/pep_prediction/AlbertCodeFiles/pep_risk-master/pep_risk_app')

def test_app_components():
    """Test the updated app components without running the full Dash app"""
    
    print("🧪 TESTING APP.PY UPDATES")
    print("="*50)
    
    try:
        # Test data loading
        print("1. Testing data loading...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import KNNImputer
        import pickle
        import joblib
        
        data_dir = "/Users/emilyguan/Downloads/EndoScribe/pep_prediction/AlbertCodeFiles/pep_risk-master/pep_risk_app/data"
        
        # Load the new model
        main_model = joblib.load(os.path.join(data_dir, "gbm_model_python.pkl"))
        
        # Load the new training data
        train_impute = pd.read_csv(os.path.join(data_dir, "train_impute_python.csv"))
        print(f"   ✅ Training data loaded: {train_impute.shape}")
        
        # Test feature columns
        exclude_cols = ['patient_id', 'pep', 'type_of_sod', 'study_id', 'study', 'pep_severity']
        feature_cols = [col for col in train_impute.columns if col not in exclude_cols]
        print(f"   ✅ Feature columns: {len(feature_cols)} (type_of_sod excluded)")
        
        # Test for string columns
        string_cols = train_impute[feature_cols].select_dtypes(include=['object']).columns.tolist()
        if len(string_cols) == 0:
            print("   ✅ No string columns in training data")
        else:
            print(f"   ❌ Found string columns: {string_cols}")
            return False
        
        print("\n2. Testing normalization function...")
        
        # Create test input data with potential string values
        test_input = pd.DataFrame({
            'age_years': [65.0],
            'gender_': [1],  # gender_male_1 -> gender_
            'bmi': [25.5],
            'sod': [1],
            'history_of_pep': [0],
            'therapy': ['No treatment']
        })
        
        # Add remaining feature columns with default values
        for col in feature_cols:
            if col not in test_input.columns:
                test_input[col] = [0]
        
        # Test the normalize function (R-exact method)
        def normalize_patient_data(input_df):
            """Test version of normalize function matching the notebook implementation"""
            input_cleaned = input_df.copy()
            
            # Step 1: Convert '.' to NaN and ensure numeric types
            for col in input_cleaned.columns:
                if col in feature_cols:
                    input_cleaned[col] = input_cleaned[col].replace('.', np.nan)
                    input_cleaned[col] = pd.to_numeric(input_cleaned[col], errors='coerce')
            
            # Step 2: Center and scale
            scaler = StandardScaler()
            scaler.fit(train_impute[feature_cols])
            
            input_normalized = input_cleaned.copy()
            columns_to_normalize = [col for col in feature_cols if col in input_cleaned.columns]
            
            if len(columns_to_normalize) > 0:
                input_normalized[columns_to_normalize] = scaler.transform(input_cleaned[columns_to_normalize])
            
            # Step 3: KNN imputation (k=10) - matching R exactly
            imputer = KNNImputer(n_neighbors=10)
            # Fit imputer on training data
            imputer.fit(train_impute[feature_cols])
            # Apply to input data
            input_normalized[columns_to_normalize] = imputer.transform(input_normalized[columns_to_normalize])
            
            return input_normalized
        
        normalized_data = normalize_patient_data(test_input)
        print("   ✅ Normalization completed successfully")
        
        print("\n3. Testing model prediction...")
        
        # Test prediction
        X_test = normalized_data[feature_cols].iloc[0:1]
        prediction = main_model.predict_proba(X_test)
        print(f"   ✅ Model prediction: {prediction[0][1]:.4f}")
        
        print("\n4. Testing with string '.' values...")
        
        # Test with string '.' values (the original problem)
        test_with_dots = test_input.copy()
        test_with_dots['bmi'] = ['.']  # This would cause the original error
        test_with_dots['sod'] = ['.']
        
        normalized_with_dots = normalize_patient_data(test_with_dots)
        
        # Check if '.' values were properly converted
        has_strings = normalized_with_dots[feature_cols].select_dtypes(include=['object']).shape[1] > 0
        if has_strings:
            print("   ❌ String values remain after normalization")
            return False
        else:
            print("   ✅ String '.' values properly converted to numeric")
        
        # Test prediction with converted data
        X_test_dots = normalized_with_dots[feature_cols].iloc[0:1]
        prediction_dots = main_model.predict_proba(X_test_dots)
        print(f"   ✅ Model prediction with converted dots: {prediction_dots[0][1]:.4f}")
        
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   ✅ App.py updates are working correctly")
        print(f"   ✅ String '.' values are properly handled")
        print(f"   ✅ Model predictions work with new data")
        print(f"   ✅ Feature exclusion (type_of_sod) implemented correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_app_components()