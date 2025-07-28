#include "../../include/tensor.h"
#include "../../include/layer.h"
#include "../../include/model.h"
#include <fstream>
#include <stdexcept>
#include <filesystem>

// 定义自己的模型
class MLP : public cctorch::Model
{
public:
    // 定义模型的层
    cctorch::Linear linear1;
    cctorch::ReLU relu;
    cctorch::Linear linear2;

    MLP() : linear1(784, 128),
            relu(),
            linear2(128, 10) {}

    // 必须实现，单个样本前向传播
    std::vector<cctorch::Tensor> forward(const std::vector<cctorch::Tensor> &input) override
    {
        auto x = linear1(input);
        x = relu(x);
        x = linear2(x);
        return x;
    }

    // 必须实现，返回所有参数
    std::vector<cctorch::Tensor> parameters() override
    {
        std::vector<cctorch::Tensor> params;
        auto p1 = linear1.parameters();
        auto p2 = linear2.parameters();

        params.insert(params.end(), p1.begin(), p1.end());
        params.insert(params.end(), p2.begin(), p2.end());

        return params;
    }

    // 可选实现，保存模型到文件
    void save(const std::string &filename) const override
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for writing: " + filename);
        }

        // 依次保存每个层
        linear1.save_to_stream(file);
        linear2.save_to_stream(file);

        file.close();
        std::cout << "MLP model saved to " << filename << std::endl;
    }

    // 可选实现，从文件加载模型参数
    void load(const std::string &filename) override
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
        {
            throw std::runtime_error("Failed to open file for reading: " + filename);
        }

        // 依次加载每个层
        linear1.load_from_stream(file);
        linear2.load_from_stream(file);

        file.close();
        std::cout << "MLP model loaded from " << filename << std::endl;
    }
};