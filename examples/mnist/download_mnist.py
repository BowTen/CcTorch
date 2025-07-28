#!/usr/bin/env python3
"""
Download and process MNIST dataset
Converts MNIST data to binary format that can be easily read by C++
"""

import os
import urllib.request
import gzip
import struct
import numpy as np

def download_file(url, filename):
    """Download file from URL if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

def read_mnist_images(filename):
    """Read MNIST image file"""
    with gzip.open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic}")
        
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows, cols)
        return images

def read_mnist_labels(filename):
    """Read MNIST label file"""
    with gzip.open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic}")
        
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def save_to_binary(images, labels, prefix):
    """Save images and labels to binary format for C++"""
    # Normalize images to [0, 1] range
    images_normalized = images.astype(np.float32) / 255.0
    
    # Save images
    with open(f"{prefix}_images.bin", "wb") as f:
        # Write header: number of images, height, width
        f.write(struct.pack('III', images.shape[0], images.shape[1], images.shape[2]))
        # Write image data
        images_normalized.tobytes('C')
        f.write(images_normalized.tobytes())
    
    # Save labels
    with open(f"{prefix}_labels.bin", "wb") as f:
        # Write header: number of labels
        f.write(struct.pack('I', len(labels)))
        # Write label data
        f.write(labels.tobytes())
    
    print(f"Saved {len(images)} images and labels to {prefix}_images.bin and {prefix}_labels.bin")

def main():
    # MNIST download URLs (using a mirror that works)
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz", 
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    # Create data directory
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created directory: {data_dir}")
    
    # Download files
    for filename in files:
        url = base_url + filename
        filepath = os.path.join(data_dir, filename)
        download_file(url, filepath)
    
    print("\nMNIST dataset download complete!")
    print("Original files are ready for C++ to read directly:")
    print(f"- Training images: {os.path.join(data_dir, 'train-images-idx3-ubyte.gz')}")
    print(f"- Training labels: {os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')}")
    print(f"- Test images: {os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')}")
    print(f"- Test labels: {os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')}")

if __name__ == "__main__":
    main()
