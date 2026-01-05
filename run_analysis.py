#!/usr/bin/env python3
"""
Simple script to run the complete analysis
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['pandas', 'numpy', 'sklearn', 'matplotlib']
    
    print("🔍 Checking dependencies...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is NOT installed")
            print(f"   Run: pip install {package}")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    directories = ['data', 'results']
    
    print("\n📁 Creating directories...")
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created: {directory}/")
        else:
            print(f"📁 Already exists: {directory}/")
    
    return True

def main():
    """Main function to run analysis"""
    print("\n" + "="*60)
    print("FINANCIAL SCALING ANALYSIS RUNNER")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first.")
        sys.exit(1)
    
    # Setup directories
    setup_directories()
    
    # Import and run main analysis
    print("\n🚀 Starting analysis...")
    
    try:
        # Import the main script
        from main import main as run_main_analysis
        
        # Run the analysis
        run_main_analysis()
        
    except Exception as e:
        print(f"\n❌ Error running analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()