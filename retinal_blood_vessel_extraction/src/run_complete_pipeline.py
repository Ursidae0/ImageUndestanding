import cv2
import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train_model
from reproduce_paper_figures import run_reproduction

def main():
    parser = argparse.ArgumentParser(description="Run Complete Retinal Vessel Extraction Pipeline")
    parser.add_argument('--train_dir', default="dataset/DRIVE/training", help="Path to training data")
    parser.add_argument('--test_dir', default="dataset/DRIVE/test", help="Path to test data")
    parser.add_argument('--output_model', default="retina_model.pkl", help="Path to save trained model")
    parser.add_argument('--output_figures', default="paper_figures_reproduction", help="Directory to save figures")
    parser.add_argument('--skip_train', action='store_true', help="Skip training and use existing model")
    
    args = parser.parse_args()
    
    print("=== Step 1: Training Model ===")
    if args.skip_train:
        print("Skipping training (--skip_train set). checking for model...")
        if not os.path.exists(args.output_model):
            print(f"Error: Model {args.output_model} not found. Cannot skip training.")
            sys.exit(1)
        print(f"Using existing model: {args.output_model}")
    else:
        if not os.path.exists(args.train_dir):
            print(f"Error: Training directory {args.train_dir} not found.")
            sys.exit(1)
            
        train_model(args.train_dir, args.output_model)

    
    print("\n=== Step 2: Generating Paper Figures ===")
    if not os.path.exists(args.test_dir):
        print(f"Error: Test directory {args.test_dir} not found.")
        sys.exit(1)
        
    run_reproduction(data_root=args.test_dir, 
                     output_root=args.output_figures, 
                     model_path=args.output_model)
                     
    print("\n=== Pipeline Complete ===")
    print(f"Model saved to: {args.output_model}")
    print(f"Figures saved to: {args.output_figures}")

if __name__ == "__main__":
    main()
