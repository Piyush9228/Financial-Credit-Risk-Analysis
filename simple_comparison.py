"""
SIMPLE VISUAL COMPARISON: Normalization vs Standardization
Shows only the most important differences
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (15, 5)

def create_simple_comparison():
    """Create simple but clear comparison"""
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Load data
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    columns = ['checking_account', 'duration', 'credit_history', 'purpose', 
               'credit_amount', 'savings_account', 'employment', 
               'installment_rate', 'personal_status', 'debtors', 
               'residence_since', 'property', 'age', 'other_installment_plans',
               'housing', 'existing_credits', 'job', 'liable_people', 
               'telephone', 'foreign_worker', 'credit_risk']
    
    df = pd.read_csv(url, sep=' ', names=columns, header=None)
    
    # Use credit_amount for demonstration
    data = df[['credit_amount']].values
    
    # Plot 1: Original Data
    axes[0].hist(data, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():,.0f}')
    axes[0].axvline(data.mean() + data.std(), color='green', linestyle=':', label='±1 Std')
    axes[0].axvline(data.mean() - data.std(), color='green', linestyle=':')
    axes[0].set_title('Original Credit Amount', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Amount (DM)')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Min: {data.min():,.0f}\nMax: {data.max():,.0f}\nRange: {data.max()-data.min():,.0f}'
    axes[0].text(0.05, 0.95, stats_text, transform=axes[0].transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: After Normalization
    scaler_minmax = MinMaxScaler()
    normalized = scaler_minmax.fit_transform(data)
    
    axes[1].hist(normalized, bins=30, color='green', alpha=0.7, edgecolor='black')
    axes[1].axvline(normalized.mean(), color='red', linestyle='--', label=f'Mean: {normalized.mean():.3f}')
    axes[1].set_title('After Normalization (Min-Max)', fontweight='bold', fontsize=12)
    axes[1].set_xlabel('Scaled Value [0, 1]')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Min: {normalized.min():.3f}\nMax: {normalized.max():.3f}\nRange: {normalized.max()-normalized.min():.3f}'
    axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: After Standardization
    scaler_standard = StandardScaler()
    standardized = scaler_standard.fit_transform(data)
    
    axes[2].hist(standardized, bins=30, color='red', alpha=0.7, edgecolor='black')
    axes[2].axvline(standardized.mean(), color='red', linestyle='--', label=f'Mean: {standardized.mean():.3f}')
    axes[2].axvline(1, color='blue', linestyle=':', label='±1 Std Dev')
    axes[2].axvline(-1, color='blue', linestyle=':')
    axes[2].set_title('After Standardization (Z-score)', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Z-score')
    axes[2].set_ylabel('Frequency')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f'Min: {standardized.min():.2f}\nMax: {standardized.max():.2f}\nStd: {standardized.std():.2f}'
    axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add overall title
    plt.suptitle('Normalization vs Standardization: Visual Comparison', 
                fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('results/simple_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Simple comparison saved as 'results/simple_comparison.png'")
    
    plt.show()
    
    # Print comparison table
    print_comparison_table()

def print_comparison_table():
    """Print a clear comparison table"""
    
    print("\n" + "="*70)
    print("📊 NORMALIZATION vs STANDARDIZATION: KEY DIFFERENCES")
    print("="*70)
    
    print("\nMATHEMATICAL FORMULAS:")
    print("-" * 40)
    print("Normalization (Min-Max):  (x - min) / (max - min)")
    print("Standardization (Z-score): (x - μ) / σ")
    print("Robust Scaling:           (x - median) / IQR")
    
    print("\nOUTPUT CHARACTERISTICS:")
    print("-" * 40)
    print("| Method         | Range        | Mean | Std | Best For                |")
    print("|----------------|--------------|------|-----|-------------------------|")
    print("| Normalization  | [0, 1]       | varies | varies | Neural Networks, KNN  |")
    print("| Standardization| (-∞, +∞)     | 0    | 1   | SVM, Linear Models      |")
    print("| No Scaling     | original     | original | original | Tree-based models |")
    print("| Robust Scaling | (-∞, +∞)     | 0*   | 1*  | Data with outliers      |")
    print("  * median=0, IQR-based scale")
    
    print("\nYOUR RESULTS (Average Accuracy):")
    print("-" * 40)
    print("No Scaling:      76.42% 🏆 (Best overall)")
    print("Robust Scaling:  76.33% (Best for KNN)")
    print("Standardization: 75.25% (Best for SVM)")
    print("Normalization:   75.25%")
    
    print("\n" + "="*70)
    print("🎯 RECOMMENDATIONS:")
    print("="*70)
    print("1. Always test NO SCALING first (worked best in your case)")
    print("2. Use STANDARDIZATION for SVM and linear models")
    print("3. Use ROBUST SCALING for data with outliers (financial data)")
    print("4. Use NORMALIZATION for neural networks and bounded data")
    print("5. Tree-based algorithms (Random Forest) rarely need scaling")

# Run the simple comparison
if __name__ == "__main__":
    import os
    os.makedirs('results', exist_ok=True)
    
    create_simple_comparison()