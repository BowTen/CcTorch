# CcTorch ✨

CcTorch 是一个用C++实现的简单深度学习框架。出于学习目的，所以开发了本项目。

## 主要特性

- **自动微分**: 支持反向传播的梯度计算
- **动态计算图**: 运行时构建计算图
- **神经网络层**: 线性层、ReLU激活函数
- **损失函数**: 均方误差、交叉熵损失
- **优化器**: 随机梯度下降(SGD)
- **数据集加载器**: MNIST数据集支持
- **模型序列化**: 保存模型为自定义二进制文件并读取

## 项目结构

```
CcTorch/
├── include/              # 头文件
│   ├── tensor.h          # 带自动微分的张量类
│   ├── layer.h           # 神经网络层 (Linear, ReLU)
│   ├── loss.h            # 损失函数 (MSE, CrossEntropy)
│   ├── optimizer.h       # 优化器 (SGD)
│   ├── model.h           # 基础模型类
│   └── mnist_loader.h    # MNIST数据集加载器
├── src/                  # 实现源文件
├── examples/             # 示例程序
│   ├── linear/           # 线性回归示例
│   └── mnist/            # MNIST分类示例
└── CMakeLists.txt        # 构建配置
```

## 模型文件格式

CcTorch 使用自定义的二进制格式来保存模型，格式如下：

```
[offset] [type]          [value]          [description]
0000     32 bit integer  1(Linear)        层类型
0004     32 bit integer  784              输入特征数
0008     32 bit integer  128              输出特征数
0012     float           w[0][0]          权重矩阵
0016     float           w[0][1]          权重矩阵
...
xxxx     float           b[x]             偏置向量
...
xxxx     32 bit integer  1(Linear)        下一层类型
xxxx+4   32 bit integer  128              输入特征数
xxxx+8   32 bit integer  10               输出特征数
xxxx+12  float           w[0][0]          权重矩阵
xxxx+16  float           w[0][1]          权重矩阵
...
xxxx+... float           b[x]             偏置向量
```
## 构建


```bash
mkdir build
cd build
cmake -G "MinGW Makefiles" ..
make
```

这将创建 `libcctorch.a` - CcTorch主库文件。

## 示例

### 反向传播自动求导
```cpp
#include "include/tensor.h"
#include <iostream>

int main()
{
    // 测试 backward()
    cctorch::Tensor a = cctorch::Tensor(2);
    cctorch::Tensor b = cctorch::Tensor(3);
    cctorch::Tensor x = a * b; // 自动动态构建计算图
    cctorch::Tensor z = x + x;
    z.backward();                                       // 自动找拓扑关系，反向传播自动求导
    std::cout << "a.grad(): " << a.grad() << std::endl; // 6
    std::cout << "b.grad(): " << b.grad() << std::endl; // 4
    std::cout << "x.grad(): " << x.grad() << std::endl; // 2
    std::cout << "z.grad(): " << z.grad() << std::endl; // 1
}
```

### 设计模型
```cpp
#include "include/tensor.h"
#include "include/layer.h"
#include "include/model.h"
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
```

### 线性回归
位于 `examples/linear/` 目录。演示使用梯度下降拟合简单线性函数 y = 2.5x + 1.0。

### MNIST分类
位于 `examples/mnist/` 目录。在MNIST手写数字数据集上训练2层MLP。
