import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


class DiseasePredictor:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        # Create necessary directories
        Path('models').mkdir(exist_ok=True)
        Path('visualizations').mkdir(exist_ok=True)
        
    def load_data(self):
        """Load and prepare training data with validation and cleaning"""
        print("Loading training data...")
        df = pd.read_csv('datasets/Training.csv')
 
        initial_count = len(df)
        df = df.drop_duplicates()
        duplicates_removed = initial_count - len(df)
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate rows")

        constant_columns = [col for col in df.columns if df[col].nunique() == 1]
        if constant_columns:
            print(f"Removing constant features: {constant_columns}")
            df = df.drop(columns=constant_columns)
        
        # 3. Check for features with too many unique values (potential IDs)
        high_cardinality = [col for col in df.columns 
                        if df[col].nunique() > len(df) * 0.9 and 
                        col != df.columns[-1]]  # Exclude target column
        if high_cardinality:
            print(f"Warning: High cardinality features (potential IDs): {high_cardinality}")
        
        # 4. Separate features and target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 5. Check for missing values
        missing = X.isnull().sum()
        if missing.any():
            print("Warning: Missing values found in features:")
            print(missing[missing > 0])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\nFinal dataset size: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X, y_encoded, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print(f"Data split: {len(X_train)} train, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self):
        """Initialize ML models"""
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=0 
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=42,
                probability=True
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(
                n_neighbors=5
            ),
            'Naive Bayes': GaussianNB()
        }
        print(f"Initialized {len(self.models)} models")
    
    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        results = {}
        
        print("\nTraining models...")
        print("-" * 50)
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Calculate accuracies
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc
            }
            
            print(f"  Training Accuracy: {train_acc:.2%}")
            print(f"  Testing Accuracy:  {test_acc:.2%}")
        
        return results
    
    def select_best_model(self, results):
        """Select the best performing model"""
        best_name = max(results, key=lambda x: results[x]['test_accuracy'])
        self.best_model = results[best_name]['model']
        
        print("\n" + "=" * 50)
        print(f"BEST MODEL: {best_name}")
        print("=" * 50)
        print(f"\nTraining Accuracy: {results[best_name]['train_accuracy']:.2%}")
        print(f"Testing Accuracy:  {results[best_name]['test_accuracy']:.2%}")
        
        return best_name, results[best_name]
    
    
    def visualize_results(self, results, X_test, y_test):
        """Create visualizations of model performance"""
        # Model comparison bar chart
        plt.figure(figsize=(10, 6))
        model_names = list(results.keys())
        test_accs = [results[m]['test_accuracy'] for m in model_names]
        
        plt.bar(model_names, test_accs, color='steelblue')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.title('Model Performance Comparison', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0.7, 1.0)
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=150, bbox_inches='tight')
        print("\nSaved model comparison chart")
        plt.close()
        
        # Feature importance for the best model
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], color='mediumseagreen')
            plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
            plt.xlabel('Importance', fontsize=12)
            plt.title('Top 10 Important Features', fontsize=16)
            plt.tight_layout()
            plt.savefig('visualizations/feature_importance.png', dpi=150, bbox_inches='tight')
            print("Saved feature importance chart")
    
    def generate_classification_report(self, X_test, y_test):
        """Generate classification report"""
        from sklearn.metrics import classification_report
        
        y_pred = self.best_model.predict(X_test)
        
        print("\n" + "=" * 50)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 50)
        
        # Simple accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nOverall Accuracy: {accuracy:.2%}")
        
        # Save report to file
        with open('model_performance_report.txt', 'w') as f:
            f.write(f"Disease Prediction Model - Test Accuracy: {accuracy:.2%}\n")
        
        print("\nDetailed report saved to: model_performance_report.txt")
    
    def save_model(self):
        """Save the trained model and label encoder"""
        Path('models').mkdir(exist_ok=True)
        
        joblib.dump(self.best_model, 'models/disease_predictor.pkl')
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        joblib.dump(self.feature_names, 'models/feature_names.pkl')
        
        print("\nModel saved: models/disease_predictor.pkl")
        print("Label encoder saved: models/label_encoder.pkl")
        print("Feature names saved: models/feature_names.pkl")

def main():
    print("\n" + "=" * 50)
    print("DISEASE PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = DiseasePredictor()
        
        # Load data
        print("\nLoading data...")
        X, y_encoded, _ = predictor.load_data()
        
        # Split data
        X_train, X_test, y_train, y_test = predictor.split_data(X, y_encoded)
        
        # Initialize and train models
        print("\nInitializing models...")
        predictor.initialize_models()
        
        # Train and evaluate models
        print("\nTraining models...")
        results = predictor.train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Select best model
        best_name, _ = predictor.select_best_model(results)
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        predictor.visualize_results(results, X_test, y_test)
        
        # Generate and save report
        predictor.generate_classification_report(X_test, y_test)
        
        # Save the best model
        predictor.save_model()
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nNext Steps:")
        print("1. Review visualizations in 'visualizations/' directory")
        print("2. Check model performance in 'model_performance_report.txt'")
        print("3. Run the Streamlit app: streamlit run app.py")
        
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        return

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
