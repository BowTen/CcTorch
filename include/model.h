#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include <vector>
#include <string>
#include <iostream>
#include <cstdlib>

namespace cctorch
{
    class Model
    {
    public:
        virtual std::vector<Tensor> forward(const std::vector<Tensor> &input) = 0;
        virtual std::vector<Tensor> parameters() = 0;

        // 保存模型到文件的虚函数，默认实现终止程序
        virtual void save(const std::string &filename) const
        {
            std::cerr << "Error: save() method not implemented for this model type!" << std::endl;
            std::abort();
        }

        // 从文件加载模型参数的虚函数，默认实现终止程序
        virtual void load(const std::string &filename)
        {
            std::cerr << "Error: load() method not implemented for this model type!" << std::endl;
            std::abort();
        }

        std::vector<Tensor> operator()(const std::vector<Tensor> &input)
        {
            return forward(input);
        }
        std::vector<std::vector<Tensor>> operator()(const std::vector<std::vector<Tensor>> &input)
        {
            std::vector<std::vector<Tensor>> output;
            for (const auto &batch : input)
            {
                auto batch_output = forward(batch);
                output.push_back(batch_output);
            }
            return output;
        }
    };

} // namespace cctorch

#endif // MODEL_H