import argparse
import torch
import sys
import os

# Get the parent directory of the current script (i.e., the root directory of your project)
project_root = os.path.dirname(os.path.abspath(__file__))
# Append the project root to the Python path
sys.path.append(project_root)

from flava.trainer import ModelRunner

def main(args):
    current_seed = torch.initial_seed()
    print("Current PyTorch seed:", current_seed)
    # Create a ModelRunner instance with specified arguments
    runner = ModelRunner(dataset_path=args.dataset_path,
                         lr=args.learning_rate, 
                         batch_size=args.batch_size,
                         dropout=args.dropout, 
                         mask_text=args.mask_text, 
                         mask_image=args.mask_image)

    # Run training and validation for the specified number of epochs
    for epoch in range(args.epochs):
        print(f"Starting Epoch {epoch + 1}")
        runner.train(epoch + 1)
        runner.validate()
        if args.predict:
            output_file = f"predict-epoch{epoch + 1}.txt"
            runner.predict(args.dataset_path, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLAVA model training and testing")

    # Argument for setting the number of epochs
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train")

    # Argument for setting the batch size
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and validation")

    # Arguments for unimodal testing
    parser.add_argument("--mask_text", action="store_true", help="Mask text input for unimodal testing")
    parser.add_argument("--mask_image", action="store_true", help="Mask image input for unimodal testing")

    # Argument for setting the learning rate
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate for training")

    parser.add_argument("learning_rate", type=float, default=0.3, help="Dropout rate for training")

    # Argument for setting the path for prediction data
    parser.add_argument("--predict", action="store_true", help="Run prediction on the specified dataset")

    # Argument for specifying the dataset path
    parser.add_argument("--dataset_path", type=str, default="dataset", help="Path to the dataset directory")


    args = parser.parse_args()
    main(args)
