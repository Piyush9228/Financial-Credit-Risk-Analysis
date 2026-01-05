#!/usr/bin/env python3
"""
MAIN SCRIPT: Financial Credit Risk Prediction with Preprocessing Analysis
Complete analysis comparing Normalization vs Standardization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)

def load_german_credit_dataset():
    """
    Load German Credit Dataset from UCI
    """
    print("=" * 70)
    print("FINANCIAL CREDIT RISK PREDICTION")
    print("Normalization vs Standardization Analysis")
    print("=" * 70)
    
    print("\n📥 Loading German Credit Dataset...")
    
    # Load dataset directly from UCI
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # Column names as per UCI documentation
    columns = [
        'checking_account', 'duration', 'credit_history', 'purpose', 
        'credit_amount', 'savings_account', 'employment', 
        'installment_rate', 'personal_status', 'debtors', 
        'residence_since', 'property', 'age', 'other_installment_plans',
        'housing', 'existing_credits', 'job', 'liable_people', 
        'telephone', 'foreign_worker', 'credit_risk'
    ]
    
    # Load the data
    df = pd.read_csv(url, sep=' ', names=columns, header=None)
    
    # Convert target: 1=Good → 0, 2=Bad → 1
    df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
    df = df.rename(columns={'credit_risk': 'default_risk'})
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   Samples: {df.shape[0]}")
    print(f"   Features: {df.shape[1] - 1}")
    print(f"   Good credit (0): {(df['default_risk'] == 0).sum()} samples")
    print(f"   Bad credit (1): {(df['default_risk'] == 1).sum()} samples")
    
    return df

def explore_dataset(df):
    """
    Perform exploratory data analysis
    """
    print("\n📊 EXPLORATORY DATA ANALYSIS")
    print("-" * 50)
    
    # Display basic info
    print("\n1. Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    # Target distribution
    print("\n2. Target Distribution:")
    target_counts = df['default_risk'].value_counts()
    for value, count in target_counts.items():
        percentage = (count / len(df)) * 100
        label = "Good Credit" if value == 0 else "Bad Credit"
        print(f"   {label} ({value}): {count} ({percentage:.1f}%)")
    
    # Numerical features summary
    numerical_cols = ['duration', 'credit_amount', 'age', 'installment_rate']
    print("\n3. Key Numerical Features:")
    for col in numerical_cols:
        if col in df.columns:
            stats = df[col].describe()
            print(f"   {col}:")
            print(f"     Min: {stats['min']:.1f}, Max: {stats['max']:.1f}")
            print(f"     Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
    
    return df

def prepare_features(df):
    """
    Prepare features for machine learning
    """
    print("\n🔧 PREPARING FEATURES")
    print("-" * 50)
    
    # Select numerical features that need scaling
    numerical_features = [
        'duration', 'credit_amount', 'installment_rate',
        'residence_since', 'age', 'existing_credits', 'liable_people'
    ]
    
    # Select only features that exist in dataset
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    # Separate features and target
    X_numerical = df[numerical_features]
    X_categorical = df.drop(numerical_features + ['default_risk'], axis=1)
    
    # One-hot encode categorical features
    X_categorical_encoded = pd.get_dummies(X_categorical, drop_first=True)
    
    # Combine all features
    X = pd.concat([X_numerical, X_categorical_encoded], axis=1)
    y = df['default_risk']
    
    print(f"✅ Features prepared:")
    print(f"   Total features: {X.shape[1]}")
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Categorical features (encoded): {X_categorical_encoded.shape[1]}")
    
    return X, y, numerical_features

def compare_scaling_methods(X, y, numerical_features):
    """
    Compare different scaling methods
    """
    print("\n⚖️ COMPARING SCALING METHODS")
    print("-" * 50)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Define scaling methods to compare
    scalers = {
        'No Scaling': None,
        'Standardization (Z-score)': StandardScaler(),
        'Normalization (Min-Max)': MinMaxScaler(),
        'Robust Scaling': RobustScaler()
    }
    
    # Define models to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Store all results
    all_results = []
    
    for scaler_name, scaler in scalers.items():
        print(f"\n📊 Scaling Method: {scaler_name}")
        print("-" * 30)
        
        # Apply scaling if applicable
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Show scaling statistics
        if scaler:
            print(f"   After {scaler_name}:")
            print(f"     Mean: {X_train_scaled.mean():.4f}")
            print(f"     Std: {X_train_scaled.std():.4f}")
            print(f"     Min: {X_train_scaled.min():.4f}")
            print(f"     Max: {X_train_scaled.max():.4f}")
        
        # Train and evaluate each model
        for model_name, model in models.items():
            print(f"   Training {model_name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Store results
            result = {
                'Scaling Method': scaler_name,
                'Model': model_name,
                'Accuracy': accuracy,
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall
            }
            all_results.append(result)
            
            print(f"     ✓ Accuracy: {accuracy:.4f}")
            print(f"     ✓ F1-Score: {f1:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    return results_df

def visualize_results(df, X, y, numerical_features, results_df):
    """
    Create visualizations for the analysis
    """
    print("\n📈 CREATING VISUALIZATIONS")
    print("-" * 50)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Target Distribution
    ax1 = plt.subplot(3, 3, 1)
    target_counts = df['default_risk'].value_counts()
    colors = ['lightgreen', 'lightcoral']
    bars = ax1.bar(['Good Credit (0)', 'Bad Credit (1)'], target_counts.values, 
                   color=colors, edgecolor='black', linewidth=2)
    ax1.set_title('Target Variable Distribution', fontweight='bold')
    ax1.set_ylabel('Count')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for bar, count in zip(bars, target_counts.values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}\n({count/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature Distributions (first 4 numerical features)
    for idx, feature in enumerate(numerical_features[:4]):
        ax = plt.subplot(3, 3, idx + 2)
        ax.hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.axvline(df[feature].mean(), color='red', linestyle='--', 
                  label=f'Mean: {df[feature].mean():.1f}')
        ax.set_title(f'{feature} Distribution', fontweight='bold')
        ax.set_xlabel(feature)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Scaling Effects on Credit Amount
    ax6 = plt.subplot(3, 3, 6)
    if 'credit_amount' in numerical_features:
        credit_data = X['credit_amount'].values.reshape(-1, 1)
        
        # Apply different scalings
        scalers = {
            'Original': None,
            'Min-Max': MinMaxScaler(),
            'Z-score': StandardScaler()
        }
        
        for i, (name, scaler) in enumerate(scalers.items()):
            if scaler:
                scaled_data = scaler.fit_transform(credit_data)
            else:
                scaled_data = credit_data
            
            # Plot distribution
            ax6.hist(scaled_data.flatten(), bins=30, alpha=0.6, 
                    label=name, density=True)
        
        ax6.set_title('Scaling Effects on Credit Amount', fontweight='bold')
        ax6.set_xlabel('Scaled Value')
        ax6.set_ylabel('Density')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    # 4. Accuracy Comparison
    ax7 = plt.subplot(3, 3, 7)
    pivot_acc = results_df.pivot_table(index='Scaling Method', 
                                       columns='Model', 
                                       values='Accuracy')
    
    # Create bar chart
    x = np.arange(len(pivot_acc))
    width = 0.2
    
    for i, model in enumerate(pivot_acc.columns):
        ax7.bar(x + i*width, pivot_acc[model], width, label=model)
    
    ax7.set_xlabel('Scaling Method')
    ax7.set_ylabel('Accuracy')
    ax7.set_title('Accuracy by Scaling Method and Model', fontweight='bold')
    ax7.set_xticks(x + width)
    ax7.set_xticklabels(pivot_acc.index, rotation=45, ha='right')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 5. F1-Score Comparison Heatmap
    ax8 = plt.subplot(3, 3, 8)
    pivot_f1 = results_df.pivot_table(index='Scaling Method', 
                                      columns='Model', 
                                      values='F1-Score')
    
    im = ax8.imshow(pivot_f1.values, cmap='RdYlGn', aspect='auto')
    ax8.set_title('F1-Score Heatmap', fontweight='bold')
    ax8.set_xlabel('Model')
    ax8.set_ylabel('Scaling Method')
    ax8.set_xticks(np.arange(len(pivot_f1.columns)))
    ax8.set_yticks(np.arange(len(pivot_f1.index)))
    ax8.set_xticklabels(pivot_f1.columns, rotation=45, ha='right')
    ax8.set_yticklabels(pivot_f1.index)
    
    # Add text annotations
    for i in range(len(pivot_f1.index)):
        for j in range(len(pivot_f1.columns)):
            text = ax8.text(j, i, f'{pivot_f1.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax8)
    
    # 6. Best Model for Each Scaling Method
    ax9 = plt.subplot(3, 3, 9)
    best_results = results_df.loc[results_df.groupby('Scaling Method')['Accuracy'].idxmax()]
    
    colors = plt.cm.Set3(np.arange(len(best_results)))
    bars = ax9.bar(range(len(best_results)), best_results['Accuracy'], 
                   color=colors, edgecolor='black')
    
    ax9.set_title('Best Model for Each Scaling Method', fontweight='bold')
    ax9.set_ylabel('Accuracy')
    ax9.set_xlabel('Scaling Method')
    ax9.set_xticks(range(len(best_results)))
    ax9.set_xticklabels(best_results['Scaling Method'], rotation=45, ha='right')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add model names on bars
    for bar, model in zip(bars, best_results['Model']):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height - 0.02,
                model, ha='center', va='top', rotation=90, fontsize=8,
                color='white', fontweight='bold')
    
    plt.suptitle('Financial Credit Risk: Normalization vs Standardization Analysis', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved as 'results/comprehensive_analysis.png'")
    
    plt.show()
    
    return fig

def generate_report(results_df):
    """
    Generate comprehensive report of results
    """
    print("\n📋 GENERATING FINAL REPORT")
    print("=" * 70)
    
    # Find best overall result
    best_overall = results_df.loc[results_df['Accuracy'].idxmax()]
    
    print("\n🏆 BEST OVERALL PERFORMANCE")
    print("-" * 40)
    print(f"Model: {best_overall['Model']}")
    print(f"Scaling Method: {best_overall['Scaling Method']}")
    print(f"Accuracy: {best_overall['Accuracy']:.4f}")
    print(f"F1-Score: {best_overall['F1-Score']:.4f}")
    print(f"Precision: {best_overall['Precision']:.4f}")
    print(f"Recall: {best_overall['Recall']:.4f}")
    
    # Compare normalization vs standardization
    print("\n📊 NORMALIZATION VS STANDARDIZATION COMPARISON")
    print("-" * 40)
    
    norm_results = results_df[results_df['Scaling Method'] == 'Normalization (Min-Max)']
    std_results = results_df[results_df['Scaling Method'] == 'Standardization (Z-score)']
    
    if len(norm_results) > 0 and len(std_results) > 0:
        norm_avg_acc = norm_results['Accuracy'].mean()
        std_avg_acc = std_results['Accuracy'].mean()
        
        improvement = ((std_avg_acc - norm_avg_acc) / norm_avg_acc) * 100
        
        print(f"Average Accuracy with Normalization: {norm_avg_acc:.4f}")
        print(f"Average Accuracy with Standardization: {std_avg_acc:.4f}")
        print(f"Improvement with Standardization: {improvement:+.2f}%")
    
    # Performance by model
    print("\n🤖 PERFORMANCE BY MODEL (Best Scaling)")
    print("-" * 40)
    
    for model in results_df['Model'].unique():
        model_results = results_df[results_df['Model'] == model]
        best_scaling = model_results.loc[model_results['Accuracy'].idxmax(), 'Scaling Method']
        best_acc = model_results['Accuracy'].max()
        print(f"{model}:")
        print(f"  Best Scaling: {best_scaling}")
        print(f"  Best Accuracy: {best_acc:.4f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 40)
    print("1. Use Standardization for financial datasets with outliers")
    print("2. Tree-based models (Random Forest) work well without scaling")
    print("3. Distance-based algorithms benefit most from standardization")
    print("4. Always test multiple preprocessing techniques")
    
    # Save results to CSV
    results_df.to_csv('results/model_performance.csv', index=False)
    print(f"\n💾 Results saved to 'results/model_performance.csv'")
    
    return best_overall

def main():
    """
    Main execution function
    """
    try:
        # Step 1: Load dataset
        df = load_german_credit_dataset()
        
        # Step 2: Explore data
        explore_dataset(df)
        
        # Step 3: Prepare features
        X, y, numerical_features = prepare_features(df)
        
        # Step 4: Compare scaling methods
        results_df = compare_scaling_methods(X, y, numerical_features)
        
        # Step 5: Visualize results
        visualize_results(df, X, y, numerical_features, results_df)
        
        # Step 6: Generate report
        best_result = generate_report(results_df)
        
        print("\n" + "=" * 70)
        print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
        print(f"\n🎯 Key Finding:")
        print(f"   {best_result['Model']} with {best_result['Scaling Method']}")
        print(f"   achieved the highest accuracy: {best_result['Accuracy']:.4f}")
        
        print("\n📁 Output files created:")
        print("   1. results/comprehensive_analysis.png - All visualizations")
        print("   2. results/model_performance.csv - Performance metrics")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()