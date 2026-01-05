#!/usr/bin/env python3
"""
Setup script for the project
Creates all necessary files and folders
"""

import os
import sys

def create_structure():
    """Create project structure"""
    structure = {
        'Financial_Scaling_Analysis': [
            'main.py',
            'run_analysis.py',
            'requirements.txt',
            'README.md',
            ['data', []],
            ['results', ['figures', 'tables']],
            ['src', [
                '__init__.py',
                'dataset_loader.py',
                'data_processor.py',
                'model_trainer.py',
                'evaluator.py',
                'visualizer.py'
            ]]
        ]
    }
    
    print("📁 Creating project structure...")
    
    for root, contents in structure.items():
        os.makedirs(root, exist_ok=True)
        print(f"✅ Created: {root}/")
        
        for item in contents:
            if isinstance(item, list):
                # It's a directory
                dir_path = os.path.join(root, item[0])
                os.makedirs(dir_path, exist_ok=True)
                print(f"✅ Created: {dir_path}/")
                
                # Create subdirectories
                for subdir in item[1]:
                    subdir_path = os.path.join(dir_path, subdir)
                    os.makedirs(subdir_path, exist_ok=True)
                    print(f"✅ Created: {subdir_path}/")
            else:
                # It's a file
                file_path = os.path.join(root, item)
                with open(file_path, 'w') as f:
                    f.write(f"# {item}\n")
                print(f"✅ Created: {file_path}")
    
    print("\n✅ Project structure created successfully!")

def create_requirements():
    """Create requirements.txt file"""
    requirements = """numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
"""
    
    with open('Financial_Scaling_Analysis/requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ Created requirements.txt")

def create_readme():
    """Create README.md file"""
#     readme = """# Financial Credit Risk Prediction with Preprocessing Analysis

# ## Project Overview
# This project compares Normalization vs Standardization techniques for financial credit risk prediction using the German Credit dataset.

# ## Quick Start

# 1. Install dependencies:
# ```bash
# #pip install -r requirements.txt'''