import os
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
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

class MLVisualization:
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'stored_results')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def store_results(self, results_dict, property_name):
        """Store cross-validation results for later use"""
        filename = os.path.join(self.results_dir, f'{property_name}_results.json')
        with open(filename, 'w') as f:
            json.dump(results_dict, f)
            
    def load_results(self, property_name):
        """Load previously stored results"""
        filename = os.path.join(self.results_dir, f'{property_name}_results.json')
        with open(filename, 'r') as f:
            return json.load(f)

    def plot_model_comparison(self, results_dict=None, property_name=None, load_stored=False, is_classification=False):
        """
        Create an enhanced boxplot comparing train and test scores
        """
        if load_stored and property_name:
            results_dict = self.load_results(property_name)
        
        # Set style
        plt.style.use('seaborn')
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Color palette
        colors = {'Random Forest': '#3498db', 
                 'Decision Tree': '#f39c12', 
                 'XGBoost': '#2ecc71'}
        
        # Prepare data for plotting
        model_names = list(results_dict.keys())
        positions = []
        labels = []
        data = []
        colors_list = []
        
        for i, name in enumerate(model_names):
            base_pos = i * 3
            positions.extend([base_pos, base_pos + 1])
            
            # Extract train and test scores
            data.extend([
                results_dict[name]['train_scores'],
                results_dict[name]['test_scores']
            ])
            
            # Add labels
            labels.extend(['Train', 'Test'])
            
            # Add colors
            colors_list.extend([colors[name], colors[name]])
        
        # Create boxplot
        bp = plt.boxplot(data, positions=positions,
                        patch_artist=True,  # Fill boxes with color
                        medianprops=dict(color='black', linewidth=1.5),
                        flierprops=dict(marker='o', markerfacecolor='black', markersize=4),
                        whiskerprops=dict(linewidth=1),
                        boxprops=dict(linewidth=1),
                        capprops=dict(linewidth=1))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize plot
        plt.grid(True, axis='y', linestyle='--', alpha=0.3)
        
        # Add model labels
        model_positions = [i * 3 + 0.5 for i in range(len(model_names))]
        plt.xticks(model_positions, model_names, fontsize=10)
        
        # Add train/test labels
        for i, pos in enumerate(positions):
            plt.text(pos, plt.ylim()[0] - 0.05, labels[i],
                    horizontalalignment='center', fontsize=9)
        
        metric_name = 'F1 Score' if is_classification else 'R² Score'
        plt.title(f'Model Performance Comparison - {property_name}' if property_name else 'Model Performance Comparison',
                 pad=20, fontsize=12, fontweight='bold')
        plt.ylabel(metric_name, fontsize=10)
        
        # Remove top and right spines
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Add some padding at the bottom for labels
        plt.subplots_adjust(bottom=0.15)
        
        return plt.gcf()

class DrugMLPipeline:
    def __init__(self, base_dir='results'):
        self.base_dir = base_dir
        self.visualization = MLVisualization(base_dir)
        
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
        
        os.makedirs(self.base_dir, exist_ok=True)
    
    def setup_directories(self, property_name):
        """Create directories for storing results"""
        plot_dir = os.path.join(self.base_dir, 'model_plots_4', property_name)
        metrics_dir = os.path.join(self.base_dir, 'model_metrics_4')
        
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        return plot_dir, metrics_dir

    def create_model_comparison(self, X, y, models_dict, is_classification=False, n_splits=10):
        """Create and store cross-validation results for each model"""
        results = {}
        
        # First split: Create held-out test set
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Setup cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for name, model in models_dict.items():
            print(f"Evaluating {name}...")
            train_scores = []
            test_scores = []
            
            for train_idx, val_idx in kf.split(X_train_full):
                X_train = X_train_full.iloc[train_idx]
                y_train = y_train_full.iloc[train_idx]
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Calculate scores based on task type
                if is_classification:
                    train_score = f1_score(y_train, model.predict(X_train), average='weighted')
                    test_score = f1_score(y_test, model.predict(X_test), average='weighted')
                else:
                    train_score = r2_score(y_train, model.predict(X_train))
                    test_score = r2_score(y_test, model.predict(X_test))
                
                train_scores.append(train_score)
                test_scores.append(test_score)
            
            results[name] = {
                'train_scores': train_scores,
                'test_scores': test_scores
            }
        
        return results

    def evaluate_model_performance(self, model, X_train, X_test, y_train, y_test, 
                                 property_name, model_name, save_dir=None, is_classification=False):
        """Evaluate model performance with various metrics"""
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        if is_classification:
            metrics = {
                'Train F1': f1_score(y_train, y_train_pred, average='weighted'),
                'Test F1': f1_score(y_test, y_test_pred, average='weighted'),
                'Train Precision': precision_score(y_train, y_train_pred, average='weighted'),
                'Test Precision': precision_score(y_test, y_test_pred, average='weighted')
            }
            
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
        
        return metrics
    
    def plot_feature_importance(self, model, X, y, feature_names, model_name, target, 
                              is_classifier=False, save_dir=None):
        """Plot feature importance for the model"""
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
                plt.savefig(os.path.join(save_dir, f'{model_name.replace(" ", "_")}_feature_importance.png'))
            plt.close()
    
    def store_metrics(self, metrics, property_name):
        """Store metrics in CSV format"""
        try:
            _, metrics_dir = self.setup_directories(property_name)
            metrics_data = []
            
            for model_name, model_metrics in metrics.items():
                row = {'Model': model_name}
                row.update(model_metrics)
                metrics_data.append(row)
            
            metrics_df = pd.DataFrame(metrics_data)
            output_path = os.path.join(metrics_dir, f'{property_name}_metrics.csv')
            
            metrics_df.to_csv(output_path, index=False)
            print(f"Successfully saved metrics to: {output_path}")
            
            return output_path, metrics_df
        except Exception as e:
            print(f"Error storing metrics: {str(e)}")
            raise

    def train_evaluate_models(self, X, y, property_name, is_classification=False):
        """Main method to train and evaluate models"""
        plot_dir, _ = self.setup_directories(property_name)
        
        # Remove Morgan fingerprints if present
        feature_cols = [col for col in X.columns if not col.startswith('Morgan_')]
        X = X[feature_cols]
        
        # Select appropriate models
        models = self.classification_models if is_classification else self.regression_models
        
        # Get cross-validation results and create plot
        cv_results = self.create_model_comparison(X, y, models, is_classification=is_classification)
        self.visualization.store_results(cv_results, property_name)
        
        # Create and save comparison plot
        comparison_fig = self.visualization.plot_model_comparison(
            cv_results, 
            property_name,
            is_classification=is_classification
        )
        comparison_plot_path = os.path.join(plot_dir, 'model_comparison.png')
        comparison_fig.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        all_metrics = {}
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model_performance(
                model, X_train, X_test, y_train, y_test,
                property_name, name, plot_dir, is_classification
            )
            all_metrics[name] = metrics
            
            self.plot_feature_importance(
                model, X, y, feature_cols, name, property_name, 
                is_classifier=is_classification, save_dir=plot_dir
            )
        
        # Store metrics and get DataFrame
        metrics_path, metrics_df = self.store_metrics(all_metrics, property_name)
        print(f"Complete analysis results saved in:\n{plot_dir}\n{metrics_path}")
        
        return metrics_df

def main():
    print("Loading cleaned data...")
    try:
        with open('cleaned_drug_data.pkl', 'rb') as f:
            cleaned_data = pickle.load(f)
        
        # Initialize pipeline
        pipeline = DrugMLPipeline(base_dir='drug_analysis_results')
        
        all_results = {}
        for property_name, df in cleaned_data.items():
            print(f"\nProcessing {property_name}")
            X = df.drop(['SMILES', 'Property_Value'], axis=1)
            y = df['Property_Value']
            is_classification = len(y.unique()) <= 2
            
            metrics_df = pipeline.train_evaluate_models(X, y, property_name, is_classification)
            all_results[property_name] = metrics_df
            print(f"\nResults for {property_name}:")
            print(metrics_df)
            
        return all_results
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    results = main()