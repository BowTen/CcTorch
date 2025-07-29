#include "../../include/tensor.h"
#include "../../include/layer.h"
#include "../../include/mnist_loader.h"
#include "../../include/loss.h"
#include "../../include/optimizer.h"
#include "../../include/model.h"
#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <filesystem>

using cctorch::MNISTLoader;
using std::vector;

class MLP : public cctorch::Model
{
public:
    cctorch::Linear linear1;
    cctorch::ReLU relu;
    cctorch::Linear linear2;

    // 模型格式
    // [offset] [type]          [value]          [description]
    // 0000     32 bit integer  1(Linear)        layer类型
    // 0004     32 bit integer  784              in_features
    // 0008     32 bit integer  128              out_features
    // 0012     float           w[0][0]          模型权重
    // 0016     float           w[0][1]          模型权重
    // ...
    // xxxx     float           b[x]             模型偏置
    // ...
    // xxxx     32 bit integer  1(Linear)        layer类型
    // xxxx+4   32 bit integer  128              in_features
    // xxxx+8   32 bit integer  10               out_features
    // xxxx+12  float           w[0][0]          模型权重
    // xxxx+16  float           w[0][1]          模型权重
    // ...
    // xxxx+... float           b[x]             模型偏置
    // ...
    MLP() : linear1(784, 128),
            relu(),
            linear2(128, 10) {}

    std::vector<cctorch::Tensor> forward(const std::vector<cctorch::Tensor> &input) override
    {
        auto x = linear1(input);
        x = relu(x);
        x = linear2(x);
        return x;
    }

    std::vector<cctorch::Tensor> parameters() override
    {
        // return linear1.parameters();

        std::vector<cctorch::Tensor> params;
        auto p1 = linear1.parameters();
        auto p2 = linear2.parameters();

        params.insert(params.end(), p1.begin(), p1.end());
        params.insert(params.end(), p2.begin(), p2.end());

        return params;
    }

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

int main()
{
    // 使用相对路径指向数据目录
    std::string data_path = "../data";
    std::string models_path = "../models";

    // 创建models目录（如果不存在）
    std::filesystem::create_directories(models_path);

    auto train_data = cctorch::MNISTLoader::load_train_data(data_path);
    auto test_data = cctorch::MNISTLoader::load_test_data(data_path);
    int epochs = 2;
    float learning_rate = 0.001f;
    MLP mlp;

    // mlp.load(models_path + "/mlp_epoch_1_250.bin");

    cctorch::Adam optimizer(mlp.parameters(), learning_rate);
    std::cout << "MLP initialized with " << mlp.parameters().size() << " parameters." << std::endl;
    cctorch::CrossEntropyLoss criterion;
    int batch_size = 64;

    std::cout << "Training MLP on MNIST dataset..." << std::endl;
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        train_data.shuffle();
        int num_batches = 0;
        for (int i = 0; i < train_data.num_images; i += batch_size)
        {
            auto batch = cctorch::MNISTLoader::get_batch(train_data, i, batch_size);
            auto outputs = mlp(cctorch::to_tensor(MNISTLoader::normalize_image(batch.images)));
            auto loss = criterion(outputs, batch.labels);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            num_batches++;
            if (num_batches % 1 == 0)
            {
                int correct = 0;
                for (int j = 0; j < outputs.size(); j++)
                {
                    int mx = 0;
                    for (int i = 0; i < outputs[j].size(); i++)
                    {
                        if (outputs[j][i].value() > outputs[j][mx].value())
                        {
                            mx = i;
                        }
                    }
                    if (mx == batch.labels[j])
                    {
                        correct++;
                    }
                }

                std::cout << "Epoch [" << epoch << "/" << epochs << "], Batch [" << num_batches << "], Loss: " << loss.value() << ", Accuracy: " << (static_cast<float>(correct) / batch.labels.size()) * 100 << "%";
                auto l1p = mlp.linear1.parameters();
                auto l2p = mlp.linear2.parameters();
                auto max_l1 = std::max_element(l1p.begin(), l1p.end(), [&](const cctorch::Tensor &a, const cctorch::Tensor &b)
                                               { return a.grad() < b.grad(); });
                auto min_l1 = std::min_element(l1p.begin(), l1p.end(), [&](const cctorch::Tensor &a, const cctorch::Tensor &b)
                                               { return a.grad() < b.grad(); });
                auto max_l2 = std::max_element(l2p.begin(), l2p.end(), [&](const cctorch::Tensor &a, const cctorch::Tensor &b)
                                               { return a.grad() < b.grad(); });
                auto min_l2 = std::min_element(l2p.begin(), l2p.end(), [&](const cctorch::Tensor &a, const cctorch::Tensor &b)
                                               { return a.grad() < b.grad(); });
                std::cout << ", maxl1: " << max_l1->grad() << ", minl1: " << min_l1->grad() << ", maxl2: " << max_l2->grad() << ", minl2: " << min_l2->grad() << std::endl;
            }
            if (num_batches % 10 == 0)
            {
                std::string model_filename = "mlp_epoch_" + std::to_string(epoch) + "_" + std::to_string(num_batches) + ".bin";
                std::string full_path = models_path + "/" + model_filename;
                mlp.save(full_path);
                std::cout << "Model saved at epoch " << epoch << ", batch " << num_batches << std::endl;
            }
        }
    }

    // 训练结束后保存最终模型
    mlp.save("mlp_final.bin");
    std::cout << "Final model saved as mlp_final.bin" << std::endl;

    return 0;
}