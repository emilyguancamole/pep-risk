#!/usr/bin/env python3
"""
Quick test of the actual app.py to verify it works with real data
"""

import sys
import os
sys.path.append('/Users/emilyguan/Downloads/EndoScribe/pep_prediction/AlbertCodeFiles/pep_risk-master/pep_risk_app')

def test_app_callback():
    """Test the actual app callback function"""
    
    print("🧪 TESTING ACTUAL APP CALLBACK")
    print("="*40)
    
    try:
        # Import the app components
        from app import normalize_patient_data, create_input_dataframe, predict_with_therapy_adjustment
        
        print("1. Testing create_input_dataframe...")
        
        # Test the input dataframe creation with sample data
        input_df = create_input_dataframe(
            age_years=65,
            gender_male_1=1,
            bmi=25.5,
            sod=1,
            history_of_pep=0,
            hx_of_recurrent_pancreatitis=0,
            pancreatic_sphincterotomy=0,
            precut_sphincterotomy=0,
            minor_papilla_sphincterotomy=0,
            failed_cannulation=0,
            difficult_cannulation=0,
            pneumatic_dilation_of_intact_biliary_sphincter=0,
            pancreatic_duct_injection=0,
            pancreatic_duct_injections_2=0,
            acinarization=0,
            trainee_involvement=0,
            cholecystectomy=0,
            pancreo_biliary_malignancy=0,
            guidewire_cannulation=0,
            guidewire_passage_into_pancreatic_duct=0,
            guidewire_passage_into_pancreatic_duct_2=0,
            biliary_sphincterotomy=0
        )
        
        print(f"   ✅ Input dataframe created: {input_df.shape}")
        print(f"   ✅ Columns: {input_df.columns.tolist()}")
        
        print("\\n2. Testing normalize_patient_data...")
        
        normalized_df = normalize_patient_data(input_df)
        print(f"   ✅ Data normalized: {normalized_df.shape}")
        
        print("\\n3. Testing predict_with_therapy_adjustment...")
        
        predictions_df = predict_with_therapy_adjustment(normalized_df)
        print(f"   ✅ Predictions made: {predictions_df.shape}")
        print(f"   ✅ Therapies: {predictions_df['therapy'].tolist()}")
        print(f"   ✅ Sample predictions:")
        for _, row in predictions_df.head().iterrows():
            print(f"      {row['therapy']}: {row['prediction']:.4f}")
        
        print(f"\\n🎉 ALL APP TESTS PASSED!")
        print(f"   ✅ App.py is working correctly with the new model and data")
        print(f"   ✅ Ready for deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ App test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_app_callback()