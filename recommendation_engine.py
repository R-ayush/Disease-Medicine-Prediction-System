"""
Recommendation Engine
Provides personalized medicine, diet, workout, and precaution recommendations
"""

import pandas as pd
import numpy as np
import joblib
import ast
from pathlib import Path

class HealthRecommendationEngine:
    """Engine for generating health recommendations based on predicted disease"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = None
        self.feature_names = None
        self.recommendations_data = {}
        self.symptom_severity = {}
        
    def load_model(self):
        """Load trained model and encoders"""
        try:
            self.model = joblib.load('models/disease_predictor.pkl')
            self.label_encoder = joblib.load('models/label_encoder.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            print("✓ Model loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def load_recommendation_data(self):
        """Load all recommendation datasets"""
        try:
            data_path = Path('datasets')
            
            # Load recommendation data
            self.recommendations_data['description'] = pd.read_csv(
                data_path / 'description.csv'
            )
            self.recommendations_data['medications'] = pd.read_csv(
                data_path / 'medications.csv'
            )
            self.recommendations_data['diets'] = pd.read_csv(
                data_path / 'diets.csv'
            )
            self.recommendations_data['precautions'] = pd.read_csv(
                data_path / 'precautions_df.csv'
            )
            self.recommendations_data['workouts'] = pd.read_csv(
                data_path / 'workout_df.csv'
            )
            
            # Load symptom severity
            severity_df = pd.read_csv(data_path / 'Symptom-severity.csv')
            self.symptom_severity = dict(zip(
                severity_df['Symptom'], 
                severity_df['weight']
            ))
            
            print("✓ Recommendation data loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading recommendation data: {e}")
            return False
    
    def predict_disease(self, symptoms_dict):
        """
        Predict disease based on symptoms
        
        Args:
            symptoms_dict: Dictionary with symptom names as keys and 1/0 as values
        
        Returns:
            Tuple of (predicted_disease, probability, top_3_predictions)
        """
        # Create feature vector
        feature_vector = np.zeros(len(self.feature_names))
        
        for symptom, value in symptoms_dict.items():
            if symptom in self.feature_names:
                idx = self.feature_names.index(symptom)
                feature_vector[idx] = value
        
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Predict
        prediction = self.model.predict(feature_vector)[0]
        probabilities = self.model.predict_proba(feature_vector)[0]
        
        # Get disease name
        disease_name = self.label_encoder.inverse_transform([prediction])[0]
        disease_probability = probabilities[prediction]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(probabilities)[-3:][::-1]
        top_3_diseases = self.label_encoder.inverse_transform(top_3_indices)
        top_3_probs = probabilities[top_3_indices]
        
        top_3_predictions = [
            {'disease': disease, 'probability': float(prob)}
            for disease, prob in zip(top_3_diseases, top_3_probs)
        ]
        
        return disease_name, float(disease_probability), top_3_predictions
    
    def calculate_severity_score(self, symptoms_dict):
        """Calculate overall severity score based on symptoms"""
        total_severity = 0
        symptom_count = 0
        
        for symptom, value in symptoms_dict.items():
            if value == 1 and symptom in self.symptom_severity:
                total_severity += self.symptom_severity[symptom]
                symptom_count += 1
        
        if symptom_count == 0:
            return 0, "Low"
        
        avg_severity = total_severity / symptom_count
        
        # Categorize severity
        if avg_severity < 3:
            severity_level = "Low"
        elif avg_severity < 5:
            severity_level = "Moderate"
        else:
            severity_level = "High"
        
        return round(avg_severity, 2), severity_level
    
    def get_disease_description(self, disease):
        """Get disease description"""
        desc_df = self.recommendations_data['description']
        result = desc_df[desc_df['Disease'] == disease]
        
        if not result.empty:
            return result.iloc[0]['Description']
        return "Description not available."
    
    def get_medications(self, disease):
        """Get medication recommendations"""
        med_df = self.recommendations_data['medications']
        result = med_df[med_df['Disease'] == disease]
        
        if not result.empty:
            medications_str = result.iloc[0]['Medication']
            try:
                # Parse the string representation of list
                medications = ast.literal_eval(medications_str)
                return medications
            except:
                return [medications_str]
        return []
    
    def get_diet_recommendations(self, disease):
        """Get diet recommendations"""
        diet_df = self.recommendations_data['diets']
        result = diet_df[diet_df['Disease'] == disease]
        
        if not result.empty:
            diet_str = result.iloc[0]['Diet']
            try:
                diets = ast.literal_eval(diet_str)
                return diets
            except:
                return [diet_str]
        return []
    
    def get_workout_recommendations(self, disease):
        """Get workout/lifestyle recommendations"""
        workout_df = self.recommendations_data['workouts']
        result = workout_df[workout_df['disease'] == disease]
        
        if not result.empty:
            workouts = result['workout'].tolist()
            return workouts
        return []
    
    def get_precautions(self, disease):
        """Get precaution recommendations"""
        precaution_df = self.recommendations_data['precautions']
        result = precaution_df[precaution_df['Disease'] == disease]
        
        if not result.empty:
            precautions = []
            for i in range(1, 5):
                col_name = f'Precaution_{i}'
                if col_name in result.columns:
                    precaution = result.iloc[0][col_name]
                    if pd.notna(precaution) and precaution.strip():
                        precautions.append(precaution)
            return precautions
        return []
    
    def generate_comprehensive_report(self, symptoms_dict):
        """
        Generate comprehensive health report with all recommendations
        
        Args:
            symptoms_dict: Dictionary with symptom names as keys and 1/0 as values
        
        Returns:
            Dictionary containing all recommendations
        """
        # Predict disease
        disease, probability, top_3 = self.predict_disease(symptoms_dict)
        
        # Calculate severity
        severity_score, severity_level = self.calculate_severity_score(symptoms_dict)
        
        # Get active symptoms
        active_symptoms = [s for s, v in symptoms_dict.items() if v == 1]
        
        # Generate report
        report = {
            'prediction': {
                'disease': disease,
                'confidence': probability,
                'top_3_predictions': top_3
            },
            'severity': {
                'score': severity_score,
                'level': severity_level
            },
            'symptoms': {
                'active_symptoms': active_symptoms,
                'count': len(active_symptoms)
            },
            'description': self.get_disease_description(disease),
            'medications': self.get_medications(disease),
            'diet': self.get_diet_recommendations(disease),
            'workouts': self.get_workout_recommendations(disease),
            'precautions': self.get_precautions(disease)
        }
        
        return report
    
    def get_symptom_suggestions(self, partial_symptom):
        """Get symptom suggestions based on partial input"""
        suggestions = [
            symptom for symptom in self.feature_names 
            if partial_symptom.lower() in symptom.lower()
        ]
        return suggestions[:10]  # Return top 10 matches
    
    def get_all_symptoms(self):
        """Get list of all available symptoms"""
        return self.feature_names

def test_recommendation_engine():
    """Test the recommendation engine"""
    print("\n" + "=" * 80)
    print("TESTING RECOMMENDATION ENGINE")
    print("=" * 80)
    
    # Initialize engine
    engine = HealthRecommendationEngine()
    
    # Load model and data
    if not engine.load_model():
        print("Failed to load model. Please train the model first.")
        return
    
    if not engine.load_recommendation_data():
        print("Failed to load recommendation data.")
        return
    
    # Test with sample symptoms
    print("\nTest Case: Patient with itching, skin_rash, and nodal_skin_eruptions")
    
    test_symptoms = {
        'itching': 1,
        'skin_rash': 1,
        'nodal_skin_eruptions': 1
    }
    
    # Generate report
    report = engine.generate_comprehensive_report(test_symptoms)
    
    # Display report
    print("\n" + "-" * 80)
    print("HEALTH RECOMMENDATION REPORT")
    print("-" * 80)
    
    print(f"\n🔍 PREDICTION:")
    print(f"   Disease: {report['prediction']['disease']}")
    print(f"   Confidence: {report['prediction']['confidence']:.2%}")
    
    print(f"\n📊 TOP 3 PREDICTIONS:")
    for i, pred in enumerate(report['prediction']['top_3_predictions'], 1):
        print(f"   {i}. {pred['disease']}: {pred['probability']:.2%}")
    
    print(f"\n⚠️  SEVERITY:")
    print(f"   Score: {report['severity']['score']}")
    print(f"   Level: {report['severity']['level']}")
    
    print(f"\n📝 DESCRIPTION:")
    print(f"   {report['description']}")
    
    print(f"\n💊 MEDICATIONS:")
    for med in report['medications']:
        print(f"   • {med}")
    
    print(f"\n🥗 DIET RECOMMENDATIONS:")
    for diet in report['diet']:
        print(f"   • {diet}")
    
    print(f"\n🏃 WORKOUT/LIFESTYLE:")
    for workout in report['workouts'][:5]:  # Show first 5
        print(f"   • {workout}")
    
    print(f"\n⚕️  PRECAUTIONS:")
    for precaution in report['precautions']:
        print(f"   • {precaution}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    test_recommendation_engine()
