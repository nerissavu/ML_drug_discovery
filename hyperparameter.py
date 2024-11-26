import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           f1_score, precision_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

class ResultStorage:
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'final_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def save_results(self, property_name, model, train_metrics, test_metrics, best_params):
        results = {
            'model': model,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params
        }
        
        # Save model and metrics separately
        model_path = os.path.join(self.results_dir, f'{property_name}_model.pkl')
        metrics_path = os.path.join(self.results_dir, f'{property_name}_metrics.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        metrics = {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'best_params': best_params
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
            
        return results
        
    def load_results(self, property_name):
        model_path = os.path.join(self.results_dir, f'{property_name}_model.pkl')
        metrics_path = os.path.join(self.results_dir, f'{property_name}_metrics.json')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        return {
            'model': model,
            'train_metrics': metrics['train_metrics'],
            'test_metrics': metrics['test_metrics'],
            'best_params': metrics['best_params']
        }

class DrugRFPipeline:
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.storage = ResultStorage(base_dir)
        os.makedirs(base_dir, exist_ok=True)
        
        self.rf_params = {
            'regressor__n_estimators': [100, 200, 300],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        
        self.rf_class_params = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }

    def setup_model(self, is_classification=False):
        if is_classification:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ]), self.rf_class_params
        else:
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(random_state=42))
            ]), self.rf_params

    def plot_feature_importance(self, model, feature_names, property_name, is_classification):
        plt.figure(figsize=(12, 6))
        estimator = model.named_steps['regressor' if not is_classification else 'classifier']
        importance = estimator.feature_importances_
        sorted_idx = importance.argsort()
        
        plt.barh(range(len(sorted_idx)), importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.title(f'Random Forest Feature Importance for {property_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        plot_path = os.path.join(self.storage.results_dir, f'{property_name}_feature_importance.png')
        plt.savefig(plot_path)
        plt.close()

    def train_evaluate_model(self, X, y, property_name, is_classification=False):
        # Remove Morgan fingerprints if present
        feature_cols = [col for col in X.columns if not col.startswith('Morgan_')]
        X = X[feature_cols]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Setup model and parameters
        model, params = self.setup_model(is_classification)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(
            model,
            params,
            cv=5,
            scoring='f1_weighted' if is_classification else 'r2',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\nTraining Random Forest for {property_name}...")
        grid_search.fit(X_train, y_train)
        
        # Get best model results
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
        # Calculate metrics
        if is_classification:
            train_metrics = {
                'f1': f1_score(y_train, y_train_pred, average='weighted'),
                'precision': precision_score(y_train, y_train_pred, average='weighted')
            }
            test_metrics = {
                'f1': f1_score(y_test, y_test_pred, average='weighted'),
                'precision': precision_score(y_test, y_test_pred, average='weighted')
            }
        else:
            train_metrics = {
                'r2': r2_score(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'mae': mean_absolute_error(y_train, y_train_pred)
            }
            test_metrics = {
                'r2': r2_score(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'mae': mean_absolute_error(y_test, y_test_pred)
            }
        
        # Plot and save feature importance
        self.plot_feature_importance(best_model, feature_cols, property_name, is_classification)
        
        # Save results
        results = self.storage.save_results(
            property_name,
            best_model,
            train_metrics,
            test_metrics,
            grid_search.best_params_
        )
        
        return results

def main():
    print("Loading cleaned data...")
    with open('cleaned_drug_data.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)
    
    pipeline = DrugRFPipeline(base_dir='rf_drug_analysis_results')
    
    all_results = {}
    for property_name, df in cleaned_data.items():
        print(f"\nProcessing {property_name}")
        X = df.drop(['SMILES', 'Property_Value'], axis=1)
        print("Feature order:", X.columns.tolist())

        y = df['Property_Value']
        is_classification = len(y.unique()) <= 2
        
        results = pipeline.train_evaluate_model(X, y, property_name, is_classification)
        all_results[property_name] = results
        
        print(f"\nResults for {property_name}:")
        print(f"Train metrics: {results['train_metrics']}")
        print(f"Test metrics: {results['test_metrics']}")
        print(f"Best parameters: {results['best_params']}")
    
    return all_results

if __name__ == "__main__":
    results = main()