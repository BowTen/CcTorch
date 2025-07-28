#include "../../include/tensor.h"
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