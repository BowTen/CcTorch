#include "../include/layer.h"
#include <iostream>
#include <cmath>
#include <random>
#include <fstream>
#include <stdexcept>

namespace cctorch
{

    // Linear class implementation
    Linear::Linear(int in_features, int out_features)
        : in_features(in_features), out_features(out_features), biases(out_features, Tensor(0.0f))
    {
        float std = std::sqrt(2.0f / in_features);
        // 创建随机数生成器
        std::default_random_engine generator(std::random_device{}());
        // 设置均值和标准差
        std::normal_distribution<float> distribution(0.0f, std);

        weights.resize(in_features, vector<Tensor>());
        for (int i = 0; i < in_features; ++i)
        {
            for (int j = 0; j < out_features; ++j)
            {
                weights[i].push_back(Tensor(distribution(generator)));
            }
        }
    }

    std::vector<Tensor> Linear::forward(const std::vector<Tensor> &input)
    {
        vector<Tensor> outputs;
        outputs.reserve(out_features);
        for (int i = 0; i < out_features; i++)
        {
            outputs.push_back(biases[i]); // Initialize output with bias
            for (int j = 0; j < in_features; j++)
            {
                outputs[i] = outputs[i] + (input[j] * weights[j][i]);
            }
        }
        return outputs;
    }

    std::vector<Tensor> Linear::parameters()
    {
        std::vector<Tensor> params;
        for (const auto &weight_vec : weights)
        {
            params.insert(params.end(), weight_vec.begin(), weight_vec.end());
        }
        params.insert(params.end(), biases.begin(), biases.end());
        return params;
    }

    void Linear::save(const std::string &filename) const
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        save_to_stream(file);
        file.close();
        std::cout << "Linear layer saved to " << filename << std::endl;
    }

    void Linear::load(const std::string &filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        load_from_stream(file);
        file.close();
        std::cout << "Linear layer loaded from " << filename << std::endl;
    }

    void Linear::save_to_stream(std::ofstream &file) const
    {
        // 写入层类型标识符 (1 for Linear)
        int layer_type = 1;
        file.write(reinterpret_cast<const char *>(&layer_type), sizeof(int));

        // 写入输入和输出特征数
        file.write(reinterpret_cast<const char *>(&in_features), sizeof(int));
        file.write(reinterpret_cast<const char *>(&out_features), sizeof(int));

        // 写入权重 (按 weights[i][j] 的顺序)
        for (int i = 0; i < in_features; ++i)
        {
            for (int j = 0; j < out_features; ++j)
            {
                float weight_value = weights[i][j].value();
                file.write(reinterpret_cast<const char *>(&weight_value), sizeof(float));
            }
        }

        // 写入偏置
        for (int i = 0; i < out_features; ++i)
        {
            float bias_value = biases[i].value();
            file.write(reinterpret_cast<const char *>(&bias_value), sizeof(float));
        }
    }

    void Linear::load_from_stream(std::ifstream &file)
    {
        // 读取层类型标识符
        int layer_type;
        file.read(reinterpret_cast<char *>(&layer_type), sizeof(int));
        if (layer_type != 1)
        {
            throw std::runtime_error("Invalid layer type in file. Expected Linear layer (type 1).");
        }

        // 读取输入和输出特征数
        int file_in_features, file_out_features;
        file.read(reinterpret_cast<char *>(&file_in_features), sizeof(int));
        file.read(reinterpret_cast<char *>(&file_out_features), sizeof(int));

        // 验证尺寸匹配
        if (file_in_features != in_features || file_out_features != out_features)
        {
            throw std::runtime_error("Model dimensions mismatch. File: " +
                                     std::to_string(file_in_features) + "x" + std::to_string(file_out_features) +
                                     ", Current: " + std::to_string(in_features) + "x" + std::to_string(out_features));
        }

        // 读取权重
        for (int i = 0; i < in_features; ++i)
        {
            for (int j = 0; j < out_features; ++j)
            {
                float weight_value;
                file.read(reinterpret_cast<char *>(&weight_value), sizeof(float));
                weights[i][j] = Tensor(weight_value);
            }
        }

        // 读取偏置
        for (int i = 0; i < out_features; ++i)
        {
            float bias_value;
            file.read(reinterpret_cast<char *>(&bias_value), sizeof(float));
            biases[i] = Tensor(bias_value);
        }
    }

    // ReLU class implementation
    vector<Tensor> ReLU::operator()(const vector<Tensor> &inputs)
    {
        vector<Tensor> outputs;
        outputs.reserve(inputs.size());

        for (auto &tensor : inputs)
        {
            outputs.push_back(tensor.relu());
        }
        return outputs;
    }

    vector<vector<Tensor>> ReLU::operator()(const vector<vector<Tensor>> &inputs)
    {
        vector<vector<Tensor>> outputs;
        outputs.reserve(inputs.size());
        for (auto &input : inputs)
        {
            outputs.push_back(operator()(input)); // Call the single input operator
        }
        return outputs;
    }

} // namespace cctorch