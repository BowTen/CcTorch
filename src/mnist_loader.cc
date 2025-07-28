#include "../include/mnist_loader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <algorithm>

// Simple function to read gzipped files - we'll implement a basic version
namespace
{
    // Function to reverse bytes for big-endian format
    uint32_t reverse_int(uint32_t i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((uint32_t)c1 << 24) + ((uint32_t)c2 << 16) + ((uint32_t)c3 << 8) + c4;
    }
}

namespace cctorch
{

    MNISTData MNISTLoader::load_dataset(const std::string &images_file, const std::string &labels_file)
    {
        MNISTData data;

        // Load images
        data.images = load_images(images_file);

        // Load labels
        data.labels = load_labels(labels_file);

        // Verify data consistency
        if (data.images.size() != data.labels.size())
        {
            throw std::runtime_error("Number of images and labels don't match");
        }

        data.num_images = data.images.size();

        return data;
    }

    MNISTData MNISTLoader::load_train_data(const std::string &data_dir)
    {
        std::string images_file = data_dir + "/train-images-idx3-ubyte";
        std::string labels_file = data_dir + "/train-labels-idx1-ubyte";
        return load_dataset(images_file, labels_file);
    }

    MNISTData MNISTLoader::load_test_data(const std::string &data_dir)
    {
        std::string images_file = data_dir + "/t10k-images-idx3-ubyte";
        std::string labels_file = data_dir + "/t10k-labels-idx1-ubyte";
        return load_dataset(images_file, labels_file);
    }

    MNISTData MNISTLoader::get_batch(const MNISTData &data, int batch_start, int batch_size)
    {
        MNISTData batch;
        batch.image_height = data.image_height;
        batch.image_width = data.image_width;

        int end_idx = std::min(batch_start + batch_size, (int)data.images.size());

        for (int i = batch_start; i < end_idx; i++)
        {
            batch.images.push_back(data.images[i]);
            batch.labels.push_back(data.labels[i]);
        }

        batch.num_images = batch.images.size();
        return batch;
    }

    std::vector<float> MNISTLoader::label_to_onehot(uint8_t label)
    {
        std::vector<float> onehot(10, 0.0f);
        if (label <= 9)
        {
            onehot[label] = 1.0f;
        }
        return onehot;
    }

    void MNISTLoader::print_dataset_info(const MNISTData &data, const std::string &name)
    {
        std::cout << "=== " << name << " Dataset Info ===" << std::endl;
        std::cout << "Number of images: " << data.num_images << std::endl;
        std::cout << "Image dimensions: " << data.image_width << "x" << data.image_height << std::endl;
        std::cout << "Image vector size: " << (data.image_width * data.image_height) << std::endl;

        if (!data.images.empty())
        {
            std::cout << "First image vector size: " << data.images[0].size() << std::endl;
            std::cout << "Sample pixel values (first 10): ";
            for (int i = 0; i < 10 && i < (int)data.images[0].size(); i++)
            {
                std::cout << (int)data.images[0][i] << " ";
            }
            std::cout << std::endl;
        }

        if (!data.labels.empty())
        {
            std::cout << "Sample labels (first 10): ";
            for (int i = 0; i < 10 && i < (int)data.labels.size(); i++)
            {
                std::cout << (int)data.labels[i] << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::vector<std::vector<uint8_t>> MNISTLoader::load_images(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open images file: " + filename);
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        magic = reverse_int(magic);

        if (magic != 2051)
        {
            throw std::runtime_error("Invalid magic number in images file: " + std::to_string(magic));
        }

        // Read dimensions
        uint32_t num_images, rows, cols;
        file.read(reinterpret_cast<char *>(&num_images), sizeof(num_images));
        file.read(reinterpret_cast<char *>(&rows), sizeof(rows));
        file.read(reinterpret_cast<char *>(&cols), sizeof(cols));

        num_images = reverse_int(num_images);
        rows = reverse_int(rows);
        cols = reverse_int(cols);

        std::cout << "Loading " << num_images << " images of size " << rows << "x" << cols << std::endl;

        std::vector<std::vector<uint8_t>> images;
        images.reserve(num_images);

        size_t image_size = rows * cols;
        std::vector<uint8_t> image_buffer(image_size);

        for (uint32_t i = 0; i < num_images; i++)
        {
            file.read(reinterpret_cast<char *>(image_buffer.data()), image_size);

            if (file.gcount() != image_size)
            {
                throw std::runtime_error("Failed to read complete image " + std::to_string(i));
            }

            images.push_back(image_buffer);
        }

        file.close();
        std::cout << "Successfully loaded " << images.size() << " images" << std::endl;
        return images;
    }

    std::vector<uint8_t> MNISTLoader::load_labels(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Cannot open labels file: " + filename);
        }

        // Read magic number
        uint32_t magic;
        file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        magic = reverse_int(magic);

        if (magic != 2049)
        {
            throw std::runtime_error("Invalid magic number in labels file: " + std::to_string(magic));
        }

        // Read number of labels
        uint32_t num_labels;
        file.read(reinterpret_cast<char *>(&num_labels), sizeof(num_labels));
        num_labels = reverse_int(num_labels);

        std::cout << "Loading " << num_labels << " labels" << std::endl;

        std::vector<uint8_t> labels(num_labels);
        file.read(reinterpret_cast<char *>(labels.data()), num_labels);

        if (file.gcount() != num_labels)
        {
            throw std::runtime_error("Failed to read complete labels");
        }

        file.close();
        std::cout << "Successfully loaded " << labels.size() << " labels" << std::endl;
        return labels;
    }

    std::vector<std::vector<float>> MNISTLoader::normalize_image(const std::vector<std::vector<uint8_t>> &images)
    {
        std::vector<std::vector<float>> res;
        for (auto &row : images)
        {
            std::vector<float> float_row;
            for (auto x : row)
            {
                float_row.push_back(x / 255.0); // 归一化到 [0, 1] 而不是 [-1, 1]
            }
            res.push_back(float_row);
        }
        return res;
    }

} // namespace cctorch
