#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <random>
#include "tensor.h"

namespace cctorch
{

    struct MNISTData
    {
        std::vector<std::vector<uint8_t>> images; // Each image is a flattened vector of 784 uint8 values (0-255)
        std::vector<uint8_t> labels;              // Labels (0-9)
        int image_height;
        int image_width;
        int num_images;

        MNISTData() : image_height(28), image_width(28), num_images(0) {}

        void shuffle()
        {
            std::mt19937_64 rng(std::random_device{}());
            for (int i = images.size() - 1; i > 0; --i)
            {
                int j = rng() % (i + 1);
                std::swap(images[i], images[j]);
                std::swap(labels[i], labels[j]);
            }
        }
    };

    class MNISTLoader
    {
    public:
        /**
         * Load MNIST dataset from original MNIST format files
         * @param images_file Path to the MNIST images file (e.g., "data/train-images-idx3-ubyte.gz")
         * @param labels_file Path to the MNIST labels file (e.g., "data/train-labels-idx1-ubyte.gz")
         * @return MNISTData structure containing images and labels
         */
        static MNISTData load_dataset(const std::string &images_file, const std::string &labels_file);

        /**
         * Load training dataset
         * @param data_dir Directory containing the MNIST data files
         * @return Training data
         */
        static MNISTData load_train_data(const std::string &data_dir = "data");

        /**
         * Load test dataset
         * @param data_dir Directory containing the MNIST data files
         * @return Test data
         */
        static MNISTData load_test_data(const std::string &data_dir = "data");

        /**
         * Get a batch of data
         * @param data The full dataset
         * @param batch_start Starting index for the batch
         * @param batch_size Size of the batch
         * @return Subset of the data
         */
        static MNISTData get_batch(const MNISTData &data, int batch_start, int batch_size);

        /**
         * Convert label to one-hot vector
         * @param label Label value (0-9)
         * @return One-hot encoded vector of size 10
         */
        static std::vector<float> label_to_onehot(uint8_t label);

        /**
         * Print dataset statistics
         * @param data The dataset to analyze
         * @param name Name of the dataset (for printing)
         */
        static void print_dataset_info(const MNISTData &data, const std::string &name);

        /**
         * Normalize image data to [0, 1] range
         * @param images Vector of images, each image is a vector of uint8_t values
         * @return Normalized images as a vector of float vectors
         */
        static std::vector<std::vector<float>> normalize_image(const std::vector<std::vector<uint8_t>> &images);

    private:
        static std::vector<std::vector<uint8_t>> load_images(const std::string &filename);
        static std::vector<uint8_t> load_labels(const std::string &filename);
    };

} // namespace cctorch

#endif // MNIST_LOADER_H
