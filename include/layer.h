#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include "tensor.h"
#include "model.h"

using std::vector;

namespace cctorch
{

    // 模型格式
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  1(Linear)        layer类型
    // 0004     32 bit integer  in_features      in_features
    // 0008     32 bit integer  out_features     out_features
    // 0012     float           w[0][0]          模型权重
    // 0016     float           w[0][1]          模型权重
    // ...
    class Linear : public Model
    {
    private:
        int in_features;
        int out_features;

    public:
        vector<vector<Tensor>> weights;
        vector<Tensor> biases;
        Linear(int in_features, int out_features);
        vector<Tensor> forward(const vector<Tensor> &input);
        std::vector<Tensor> parameters();

        // 重写保存和加载方法
        void save(const std::string &filename) const override;
        void load(const std::string &filename) override;

        // 支持追加模式的保存和加载方法（用于多层模型）
        void save_to_stream(std::ofstream &file) const;
        void load_from_stream(std::ifstream &file);

        // 获取输入和输出特征数的公开方法（用于加载时验证）
        int get_in_features() const { return in_features; }
        int get_out_features() const { return out_features; }
    };

    class ReLU
    {
    public:
        vector<Tensor> operator()(const vector<Tensor> &inputs);
        vector<vector<Tensor>> operator()(const vector<vector<Tensor>> &inputs);
    };

} // namespace cctorch

#endif // LAYER_H