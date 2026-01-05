import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Suppress warnings
warnings.filterwarnings('ignore')

# Import custom modules
from src.data_processor import DataProcessor
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer

def setup_directories():
    """Create necessary directories for the project."""
    directories = ['data', 'results', 'logs', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    print("✅ Directories created successfully")

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("FINANCIAL CREDIT RISK PREDICTION WITH PREPROCESSING ANALYSIS")
    print("="*80)
    
    # Setup
    setup_directories()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize components
    print("\n📊 Initializing components...")
    data_processor = DataProcessor(random_state=42)
    model_trainer = ModelTrainer(random_state=42)
    evaluator = Evaluator()
    visualizer = Visualizer(output_dir='results')
    
    # Step 1: Generate or load data
    print("\n1️⃣  DATA PREPARATION")
    print("-"*40)
    data_path = 'data/financial_data.csv'
    
    if not os.path.exists(data_path):
        print("Generating synthetic financial dataset...")
        df = data_processor.generate_financial_data(n_samples=2000)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset saved to {data_path}")
    else:
        print(f"Loading existing dataset from {data_path}")
        df = data_processor.load_data(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {', '.join(df.columns[:-1])}")
    print(f"Target variable: {df.columns[-1]}")
    
    # Step 2: Exploratory Data Analysis
    print("\n2️⃣  EXPLORATORY DATA ANALYSIS")
    print("-"*40)
    visualizer.plot_feature_distributions(df, save=True)
    visualizer.plot_correlation_matrix(df, save=True)
    visualizer.plot_target_distribution(df, save=True)
    
    # Step 3: Preprocess data
    print("\n3️⃣  DATA PREPROCESSING")
    print("-"*40)
    X = df.drop('default_risk', axis=1)
    y = df['default_risk']
    
    # Split data
    X_train, X_test, y_train, y_test = data_processor.split_data(X, y, test_size=0.3)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Step 4: Define scaling methods
    scaling_methods = data_processor.get_scaling_methods()
    print(f"\nScaling methods to compare: {len(scaling_methods)}")
    for name in scaling_methods.keys():
        print(f"  - {name}")
    
    # Step 5: Model training and evaluation
    print("\n4️⃣  MODEL TRAINING AND EVALUATION")
    print("-"*40)
    
    all_results = []
    for scaling_name, scaler in scaling_methods.items():
        print(f"\n📈 Processing: {scaling_name}")
        
        # Apply scaling
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        # Train models
        models = model_trainer.get_models()
        results = model_trainer.train_and_evaluate_models(
            models, X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # Add scaling information
        for result in results:
            result['Scaling Method'] = scaling_name
            all_results.append(result)
    
    # Step 6: Results analysis
    print("\n5️⃣  RESULTS ANALYSIS")
    print("-"*40)
    
    # Convert results to DataFrame
    results_df = evaluator.results_to_dataframe(all_results)
    
    # Save results
    results_file = f'results/performance_results_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✅ Results saved to {results_file}")
    
    # Generate summary statistics
    summary = evaluator.generate_summary(results_df)
    summary_file = f'results/summary_{timestamp}.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Summary saved to {summary_file}")
    
    # Step 7: Visualizations
    print("\n6️⃣  GENERATING VISUALIZATIONS")
    print("-"*40)
    
    # Performance comparison heatmaps
    visualizer.plot_performance_heatmap(
        results_df, metric='Accuracy', 
        title='Accuracy by Scaling Method and Model',
        save=True
    )
    
    visualizer.plot_performance_heatmap(
        results_df, metric='F1-Score', 
        title='F1-Score by Scaling Method and Model',
        save=True
    )
    
    # Comparison charts
    visualizer.plot_scaling_comparison(results_df, save=True)
    visualizer.plot_model_comparison(results_df, save=True)
    
    # Detailed analysis
    visualizer.plot_detailed_analysis(results_df, save=True)
    
    # Step 8: Generate report
    print("\n7️⃣  GENERATING FINAL REPORT")
    print("-"*40)
    
    report = evaluator.generate_comprehensive_report(results_df)
    report_file = f'results/final_report_{timestamp}.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Report saved to {report_file}")
    
    # Print key findings
    print("\n" + "="*80)
    print("🎯 KEY FINDINGS")
    print("="*80)
    
    best_result = results_df.loc[results_df['Accuracy'].idxmax()]
    print(f"\n🏆 BEST PERFORMING COMBINATION:")
    print(f"  Model: {best_result['Model']}")
    print(f"  Scaling: {best_result['Scaling Method']}")
    print(f"  Accuracy: {best_result['Accuracy']:.4f}")
    print(f"  F1-Score: {best_result['F1-Score']:.4f}")
    
    avg_standardization = results_df[
        results_df['Scaling Method'] == 'Standardization (Z-score)'
    ]['Accuracy'].mean()
    
    avg_normalization = results_df[
        results_df['Scaling Method'] == 'Normalization (Min-Max)'
    ]['Accuracy'].mean()
    
    improvement = ((avg_standardization - avg_normalization) / avg_normalization) * 100
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"  Average Accuracy with Standardization: {avg_standardization:.4f}")
    print(f"  Average Accuracy with Normalization: {avg_normalization:.4f}")
    print(f"  Improvement with Standardization: {improvement:+.2f}%")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    # Save best configuration
    best_config = {
        'timestamp': timestamp,
        'best_model': best_result['Model'],
        'best_scaling': best_result['Scaling Method'],
        'accuracy': float(best_result['Accuracy']),
        'f1_score': float(best_result['F1-Score']),
        'recommendation': 'Use Standardization for financial datasets'
    }
    
    config_file = 'results/best_model_config.json'
    with open(config_file, 'w') as f:
        json.dump(best_config, f, indent=4)
    
    print(f"\n📁 Output files saved in 'results/' directory")
    print(f"📄 Final report: {report_file}")
    print(f"⚙️  Best configuration: {config_file}")

if __name__ == "__main__":
    main()