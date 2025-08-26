#!/usr/bin/env python3
"""
MNIST Dataset Image Extractor

This script reads MNIST .ubyte files and extracts individual images.
It can visualize images, save them as PNG files, or convert the dataset to other formats.

MNIST File Format:
- Images: .idx3-ubyte files contain 28x28 grayscale images (0-255 pixel values)
- Labels: .idx1-ubyte files contain corresponding digit labels (0-9)

File Structure:
- Image files: [magic][num_images][num_rows][num_cols][pixel_data...]
- Label files: [magic][num_labels][label_data...]

Usage:
    python extract_mnist_images.py --help
    python extract_mnist_images.py --visualize data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte
    python extract_mnist_images.py --extract data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --output extracted_images/ --count 100
"""

import argparse
import struct
import numpy as np
import os
import sys
from pathlib import Path

def read_mnist_images(file_path):
    """
    Read MNIST image file (.idx3-ubyte format)
    
    Returns:
        numpy.ndarray: Array of shape (num_images, height, width) containing image data
    """
    print(f"Reading image file: {file_path}")
    
    with open(file_path, 'rb') as f:
        # Read file header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            raise ValueError(f'Invalid magic number {magic_number}. Expected 2051 for image files.')
        
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]
        
        print(f"  Images: {num_images}")
        print(f"  Dimensions: {num_rows}x{num_cols}")
        
        # Read image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, num_rows, num_cols)
        
    return images

def read_mnist_labels(file_path):
    """
    Read MNIST label file (.idx1-ubyte format)
    
    Returns:
        numpy.ndarray: Array of shape (num_labels,) containing label data
    """
    print(f"Reading label file: {file_path}")
    
    with open(file_path, 'rb') as f:
        # Read file header
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            raise ValueError(f'Invalid magic number {magic_number}. Expected 2049 for label files.')
        
        num_labels = struct.unpack('>I', f.read(4))[0]
        print(f"  Labels: {num_labels}")
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        
    return labels

def visualize_samples(images, labels, num_samples=10):
    """
    Display sample images using matplotlib
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for visualization.")
        print("Install it with: pip install matplotlib")
        return
    
    # Create subplot grid
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        row = i // cols
        col = i % cols
        
        if rows > 1:
            ax = axes[row, col]
        else:
            ax = axes[col]
        
        ax.imshow(images[i], cmap='gray')
        ax.set_title(f'Label: {labels[i]}')
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(num_samples, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1:
            axes[row, col].axis('off')
        else:
            axes[col].axis('off')
    
    plt.tight_layout()
    plt.show()

def extract_images(images, labels, output_dir, count=None):
    """
    Extract and save individual images as PNG files
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow (PIL) is required for saving images.")
        print("Install it with: pip install Pillow")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    num_to_extract = min(count or len(images), len(images))
    
    print(f"Extracting {num_to_extract} images to {output_dir}/")
    
    for i in range(num_to_extract):
        # Create subdirectory for each digit
        digit_dir = output_path / str(labels[i])
        digit_dir.mkdir(exist_ok=True)
        
        # Save image
        img = Image.fromarray(images[i], mode='L')  # 'L' for grayscale
        filename = digit_dir / f"image_{i:05d}.png"
        img.save(filename)
        
        if (i + 1) % 1000 == 0:
            print(f"  Extracted {i + 1}/{num_to_extract} images...")
    
    print(f"Extraction complete! Images saved to {output_dir}/")
    print("Directory structure:")
    for digit in range(10):
        digit_path = output_path / str(digit)
        if digit_path.exists():
            count = len(list(digit_path.glob("*.png")))
            print(f"  {digit}/: {count} images")

def get_dataset_info(images, labels):
    """
    Display information about the dataset
    """
    print("\n=== Dataset Information ===")
    print(f"Number of images: {len(images)}")
    print(f"Image dimensions: {images[0].shape}")
    print(f"Pixel value range: {images.min()} - {images.max()}")
    print(f"Number of labels: {len(labels)}")
    print(f"Label range: {labels.min()} - {labels.max()}")
    
    # Count occurrences of each digit
    print("\nLabel distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for digit, count in zip(unique, counts):
        print(f"  Digit {digit}: {count} images")

def main():
    parser = argparse.ArgumentParser(
        description="Extract and visualize MNIST dataset images from .ubyte files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show dataset information
  python extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --info

  # Visualize first 10 images
  python extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --visualize

  # Extract first 100 images as PNG files
  python extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --extract --output extracted_images/ --count 100

  # Extract all images (warning: creates 60,000 files for training set!)
  python extract_mnist_images.py data/input/train-images-idx3-ubyte data/input/train-labels-idx1-ubyte --extract --output all_images/
        """
    )
    
    parser.add_argument('image_file', help='Path to MNIST image file (.idx3-ubyte)')
    parser.add_argument('label_file', help='Path to MNIST label file (.idx1-ubyte)')
    
    parser.add_argument('--info', action='store_true', 
                       help='Display dataset information')
    parser.add_argument('--visualize', action='store_true', 
                       help='Display sample images using matplotlib')
    parser.add_argument('--extract', action='store_true', 
                       help='Extract images as PNG files')
    parser.add_argument('--output', default='extracted_images/', 
                       help='Output directory for extracted images (default: extracted_images/)')
    parser.add_argument('--count', type=int, 
                       help='Number of images to extract (default: all)')
    parser.add_argument('--samples', type=int, default=10, 
                       help='Number of sample images to visualize (default: 10)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.image_file):
        print(f"Error: Image file not found: {args.image_file}")
        sys.exit(1)
    
    if not os.path.exists(args.label_file):
        print(f"Error: Label file not found: {args.label_file}")
        sys.exit(1)
    
    # Read MNIST data
    try:
        images = read_mnist_images(args.image_file)
        labels = read_mnist_labels(args.label_file)
    except Exception as e:
        print(f"Error reading MNIST files: {e}")
        sys.exit(1)
    
    # Validate data consistency
    if len(images) != len(labels):
        print(f"Error: Mismatch between number of images ({len(images)}) and labels ({len(labels)})")
        sys.exit(1)
    
    print("MNIST data loaded successfully!")
    
    # Default action: show info
    if not any([args.visualize, args.extract]):
        args.info = True
    
    # Execute requested actions
    if args.info:
        get_dataset_info(images, labels)
    
    if args.visualize:
        print(f"\nVisualizing {args.samples} sample images...")
        visualize_samples(images, labels, args.samples)
    
    if args.extract:
        print(f"\nExtracting images...")
        extract_images(images, labels, args.output, args.count)

if __name__ == '__main__':
    main()
