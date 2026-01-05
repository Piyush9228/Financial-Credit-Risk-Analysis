#!/usr/bin/env python3
"""
MAIN SCRIPT using modular approach
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from model_trainer import ModelTrainer
from evaluator import Evaluator
from visualizer import Visualizer

def main():
    """Main execution function using modular approach"""
    print("=" * 70)
    print("FINANCIAL CREDIT RISK - SCALING COMPARISON ANALYSIS")
    print("=" * 70)
    
    # Step 1: Initialize components
    print("\n1️⃣ INITIALIZING COMPONENTS")
    processor = DataProcessor(random_state=42)
    trainer = ModelTrainer(random_state=42)
    evaluator = Evaluator()
    visualizer = Visualizer()
    
    # Step 2: Load and prepare data
    print("\n2️⃣ LOADING AND PREPARING DATA")
    df = processor.load_german_credit_data()
    X, y, numerical_features = processor.prepare_features(df)
    
    # Step 3: Split data
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    # Step 4: Compare scaling methods
    print("\n3️⃣ COMPARING SCALING METHODS")
    all_results = []
    
    for method in processor.get_scaling_methods():
        print(f"\n📊 Method: {method}")
        
        # Apply scaling
        X_train_scaled, X_test_scaled, scaler = processor.apply_scaling(
            X_train, X_test, method
        )
        
        # Train and evaluate models
        results = trainer.train_and_evaluate_all_models(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Add scaling method to results
        for result in results:
            result['Scaling Method'] = method
            all_results.append(result)
    
    # Step 5: Analyze results
    print("\n4️⃣ ANALYZING RESULTS")
    results_df = pd.DataFrame(all_results)
    
    # Generate comprehensive analysis
    summary = evaluator.generate_summary(results_df)
    
    # Step 6: Create visualizations
    print("\n5️⃣ CREATING VISUALIZATIONS")
    visualizer.create_comprehensive_analysis(
        df, results_df, numerical_features, save=True
    )
    
    # Step 7: Generate final report
    print("\n6️⃣ GENERATING FINAL REPORT")
    report = evaluator.generate_report(results_df)
    print(report)
    
    # Save results
    results_df.to_csv('results/final_results.csv', index=False)
    print("\n✅ Analysis completed! Results saved to 'results/final_results.csv'")

if __name__ == "__main__":
    main()