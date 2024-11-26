import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                           f1_score, precision_score, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

class DrugMLPipeline:
    def __init__(self):
        self.regression_models = {
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', RandomForestRegressor(random_state=42))
            ]),
            'Decision Tree': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', DecisionTreeRegressor(random_state=42))
            ]),
            'XGBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', XGBRegressor(random_state=42))
            ])
        }
        
        self.classification_models = {
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(random_state=42))
            ]),
            'Decision Tree': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', DecisionTreeClassifier(random_state=42))
            ]),
            'XGBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', XGBClassifier(random_state=42))
            ])
        }
    
    def setup_directories(self, property_name):
        """Create directories for storing results"""
        plot_dir = os.path.join('model_plots', property_name)
        metrics_dir = 'model_metrics_nov_15'
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        return plot_dir, metrics_dir
    
    def store_metrics(self, metrics, property_name):
        """Store metrics in CSV format"""
        _, metrics_dir = self.setup_directories(property_name)
        metrics_data = []
        for model_name, model_metrics in metrics.items():
            row = {'Model': model_name}
            row.update(model_metrics)
            metrics_data.append(row)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(metrics_dir, f'{property_name}_metrics.csv'), index=False)
    
    def plot_feature_importance(self, model, X, y, feature_names, model_name, target, 
                              is_classifier=False, save_dir=None):
        plt.figure(figsize=(12, 6))
        estimator = model.named_steps['regressor' if not is_classifier else 'classifier']
        if hasattr(estimator, 'feature_importances_'):
            importance = estimator.feature_importances_
            sorted_idx = importance.argsort()
            plt.barh(range(len(sorted_idx)), importance[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
            plt.title(f'{model_name} Feature Importance for {target}')
            plt.xlabel('Importance')
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'{model_name}_feature_importance.png'))
            plt.close()
    
    def evaluate_model_performance(self, model, X_train, X_test, y_train, y_test, 
                                 property_name, model_name, save_dir=None, is_classification=False):
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if is_classification:
            metrics = {
                'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
                'Test F1': f1_score(y_test, y_test_pred, average='weighted'),
                'Train Precision': precision_score(y_train, y_train_pred, average='weighted'),
                'Test Precision': precision_score(y_test, y_test_pred, average='weighted')
            }
            
            # Store detailed classification report
            train_report = classification_report(y_train, y_train_pred)
            test_report = classification_report(y_test, y_test_pred)
            
            print(f"\nDetailed Classification Report for {model_name} on {property_name}")
            print("\nTraining Set:")
            print(train_report)
            print("\nTest Set:")
            print(test_report)
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
        else:
            metrics = {
                'Train R2': r2_score(y_train, y_train_pred),
                'Test R2': r2_score(y_test, y_test_pred),
                'Train RMSE': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                'Test RMSE': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                'Train MAE': mean_absolute_error(y_train, y_train_pred),
                'Test MAE': mean_absolute_error(y_test, y_test_pred)
            }
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        metrics['CV Mean'] = cv_scores.mean()
        metrics['CV Std'] = cv_scores.std()
        
        plt.figure(figsize=(10, 6))
        performance_data = [
            [metrics[f'Train {"F1" if is_classification else "R2"}']] * 5,
            [metrics[f'Test {"F1" if is_classification else "R2"}']] * 5,
            cv_scores
        ]
        plt.boxplot(performance_data, labels=['Train', 'Test', 'CV'])
        plt.title(f'{model_name} Performance on {property_name}')
        plt.ylabel('F1 Score' if is_classification else 'R² Score')
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{model_name}_performance.png'))
        plt.close()
        
        print(f"\nPerformance Metrics for {model_name} on {property_name}:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics
    
    def train_evaluate_models(self, X, y, property_name, is_classification=False):
        plot_dir, _ = self.setup_directories(property_name)
        feature_cols = [col for col in X.columns if not col.startswith('Morgan_')]
        X = X[feature_cols]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        models = self.classification_models if is_classification else self.regression_models
        all_metrics = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model_performance(
                model, X_train, X_test, y_train, y_test, 
                property_name, name, plot_dir, is_classification
            )
            all_metrics[name] = metrics
            
            self.plot_feature_importance(model, X, y, feature_cols, name, property_name, 
                                      is_classifier=is_classification, save_dir=plot_dir)
        
        self.store_metrics(all_metrics, property_name)
        return all_metrics

def main():
    print("Loading cleaned data...")
    with open('cleaned_drug_data.pkl', 'rb') as f:
        cleaned_data = pickle.load(f)
    
    pipeline = DrugMLPipeline()
    
    for property_name, df in cleaned_data.items():
        print(f"\nProcessing {property_name}")
        X = df.drop(['SMILES', 'Property_Value'], axis=1)
        y = df['Property_Value']
        is_classification = len(y.unique()) <= 2
        
        metrics = pipeline.train_evaluate_models(X, y, property_name, is_classification)

if __name__ == "__main__":
    main()