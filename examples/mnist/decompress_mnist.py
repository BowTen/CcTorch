#!/usr/bin/env python3
"""
Decompress MNIST .gz files to raw binary files for C++ to read
"""

import os
import gzip
import shutil

def decompress_file(gz_file, output_file):
    """Decompress a .gz file"""
    if not os.path.exists(gz_file):
        print(f"File not found: {gz_file}")
        return False
        
    print(f"Decompressing {gz_file} -> {output_file}")
    
    with gzip.open(gz_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"Successfully decompressed {output_file}")
    return True

def main():
    data_dir = "data"
    
    # Files to decompress
    files_to_decompress = [
        ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte"),
        ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte"),
        ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte"),
        ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte")
    ]
    
    for gz_name, raw_name in files_to_decompress:
        gz_path = os.path.join(data_dir, gz_name)
        raw_path = os.path.join(data_dir, raw_name)
        
        if os.path.exists(raw_path):
            print(f"{raw_path} already exists, skipping...")
            continue
            
        if not decompress_file(gz_path, raw_path):
            print(f"Failed to decompress {gz_path}")
            return 1
    
    print("\nAll files decompressed successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
