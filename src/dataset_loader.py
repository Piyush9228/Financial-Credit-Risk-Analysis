"""
Dataset Loader Module for Financial Credit Risk Project
Handles loading of German Credit and other financial datasets
"""

import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import warnings
warnings.filterwarnings('ignore')

class DatasetLoader:
    """
    Loads various financial datasets for normalization vs standardization analysis
    """
    
    @staticmethod
    def load_german_credit_uci():
        """
        Load German Credit dataset from UCI using ucimlrepo package
        Returns: pandas DataFrame with features and target
        """
        print("📥 Loading German Credit Dataset from UCI...")
        
        try:
            # Fetch dataset from UCI repository
            german_credit = fetch_ucirepo(id=144)
            
            # Get features and target
            X = german_credit.data.features
            y = german_credit.data.targets
            
            # Combine into single DataFrame
            df = pd.concat([X, y], axis=1)
            
            # Print dataset information
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {df.shape}")
            print(f"   Features: {len(X.columns)}")
            print(f"   Target variable: {y.columns[0]}")
            print(f"   Target distribution:")
            print(f"     Good credit (1): {(y.iloc[:, 0] == 1).sum()} samples")
            print(f"     Bad credit (2): {(y.iloc[:, 0] == 2).sum()} samples")
            
            # Convert target to binary (1=good, 2=bad -> 0=good, 1=bad)
            df[y.columns[0]] = df[y.columns[0]].map({1: 0, 2: 1})
            
            # Rename target column for consistency
            df = df.rename(columns={y.columns[0]: 'default_risk'})
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print("Trying alternative method...")
            return DatasetLoader.load_german_credit_direct()
    
    @staticmethod
    def load_german_credit_direct():
        """
        Load German Credit dataset directly from UCI URL
        Alternative method if ucimlrepo fails
        """
        print("📥 Loading German Credit Dataset directly from UCI...")
        
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
            
            # Column names as per dataset documentation
            columns = [
                'checking_account', 'duration', 'credit_history', 'purpose', 
                'credit_amount', 'savings_account', 'employment', 
                'installment_rate', 'personal_status', 'debtors', 
                'residence_since', 'property', 'age', 'other_installment_plans',
                'housing', 'existing_credits', 'job', 'liable_people', 
                'telephone', 'foreign_worker', 'credit_risk'
            ]
            
            # Load data
            df = pd.read_csv(url, sep=' ', names=columns, header=None)
            
            # Convert target to binary (1=good, 2=bad -> 0=good, 1=bad)
            df['credit_risk'] = df['credit_risk'].map({1: 0, 2: 1})
            
            # Rename for consistency
            df = df.rename(columns={'credit_risk': 'default_risk'})
            
            print(f"✅ Dataset loaded successfully!")
            print(f"   Shape: {df.shape}")
            print(f"   Target distribution:")
            print(f"     Good credit (0): {(df['default_risk'] == 0).sum()} samples")
            print(f"     Bad credit (1): {(df['default_risk'] == 1).sum()} samples")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading direct dataset: {e}")
            return None
    
    @staticmethod
    def load_credit_card_default():
        """
        Load Credit Card Default dataset from UCI
        Larger dataset for comparison
        """
        print("📥 Loading Credit Card Default Dataset...")
        
        try:
            # Direct download from UCI
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
            
            # Load Excel file (skip first row which has extra header)
            df = pd.read_excel(url, header=1)
            
            # Clean column names
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Rename target column
            df = df.rename(columns={'default_payment_next_month': 'default_risk'})
            
            print(f"✅ Credit Card Default Dataset loaded successfully!")
            print(f"   Shape: {df.shape}")
            print(f"   Default rate: {df['default_risk'].mean():.2%}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading credit card dataset: {e}")
            return None
    
    @staticmethod
    def load_dataset(dataset_name='german'):
        """
        Main method to load dataset by name
        Args:
            dataset_name: 'german' or 'credit_card'
        Returns:
            pandas DataFrame
        """
        datasets = {
            'german': DatasetLoader.load_german_credit_uci,
            'german_direct': DatasetLoader.load_german_credit_direct,
            'credit_card': DatasetLoader.load_credit_card_default
        }
        
        if dataset_name in datasets:
            print(f"\n{'='*60}")
            print(f"LOADING DATASET: {dataset_name.upper()}")
            print('='*60)
            return datasets[dataset_name]()
        else:
            print(f"❌ Dataset '{dataset_name}' not found.")
            print(f"Available datasets: {list(datasets.keys())}")
            return None
    
    @staticmethod
    def analyze_dataset(df):
        """
        Perform basic analysis on dataset
        Args:
            df: pandas DataFrame
        """
        if df is None:
            print("❌ No dataset to analyze")
            return
        
        print("\n📊 DATASET ANALYSIS")
        print("-" * 40)
        
        # Basic info
        print(f"Dataset Shape: {df.shape}")
        print(f"Number of Features: {df.shape[1] - 1}")
        print(f"Number of Samples: {df.shape[0]}")
        
        # Check for target column
        target_col = 'default_risk' if 'default_risk' in df.columns else None
        if not target_col:
            # Try to find target column
            possible_targets = ['class', 'target', 'y', 'default']
            for col in possible_targets:
                if col in df.columns:
                    target_col = col
                    break
        
        if target_col:
            # Target distribution
            target_counts = df[target_col].value_counts()
            print(f"\n🎯 TARGET DISTRIBUTION ({target_col}):")
            for value, count in target_counts.items():
                percentage = (count / len(df)) * 100
                label = "Good Credit" if value == 0 else "Bad Credit"
                print(f"  {label} ({value}): {count} samples ({percentage:.1f}%)")
        
        # Data types
        print(f"\n📝 DATA TYPES:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # Missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            print(f"\n⚠️  MISSING VALUES: {missing_values} total")
            missing_cols = df.isnull().sum()
            missing_cols = missing_cols[missing_cols > 0]
            for col, count in missing_cols.items():
                percentage = (count / len(df)) * 100
                print(f"  {col}: {count} ({percentage:.1f}%)")
        else:
            print(f"\n✅ NO MISSING VALUES")
        
        # Numerical features summary
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            print(f"\n🔢 NUMERICAL FEATURES ({len(numerical_cols)}):")
            print("  " + ", ".join(numerical_cols))
            
            # Show statistics for first few numerical columns
            print(f"\n📈 STATISTICS FOR KEY NUMERICAL FEATURES:")
            key_cols = [col for col in ['credit_amount', 'duration', 'age', 'installment_rate'] 
                       if col in df.columns][:3]
            for col in key_cols:
                stats = df[col].describe()
                print(f"  {col}:")
                print(f"    Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
                print(f"    Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}")
                print(f"    25%: {stats['25%']:.2f}, 50%: {stats['50%']:.2f}, 75%: {stats['75%']:.2f}")
        
        return df

# Example usage and testing
if __name__ == "__main__":
    # Test the dataset loader
    print("🧪 TESTING DATASET LOADER")
    print("=" * 60)
    
    # Load German Credit dataset
    df = DatasetLoader.load_dataset('german')
    
    if df is not None:
        # Analyze the dataset
        DatasetLoader.analyze_dataset(df)
        
        # Save to CSV for future use
        df.to_csv('../data/processed/german_credit.csv', index=False)
        print(f"\n💾 Dataset saved to: ../data/processed/german_credit.csv")
        
        # Show first few rows
        print(f"\n👀 FIRST 5 ROWS:")
        print(df.head())