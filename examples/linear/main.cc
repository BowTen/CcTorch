#include "../../include/tensor.h"
#include "../../include/layer.h"
#include "../../include/loss.h"
#include "../../include/optimizer.h"
#include <iostream>

int main()
{
    // y = 2.5x + 1.0
    // X = [1.0, 2.0, 3.0, 4.0]
    // Y = [3.5, 6.0, 8.5, 11.0]
    vector<cctorch::Tensor> X;
    X.push_back(cctorch::Tensor(1.0f));
    X.push_back(cctorch::Tensor(2.0f));
    X.push_back(cctorch::Tensor(3.0f));
    X.push_back(cctorch::Tensor(4.0f));

    vector<cctorch::Tensor> Y;
    Y.push_back(cctorch::Tensor(3.5f));
    Y.push_back(cctorch::Tensor(6.0f));
    Y.push_back(cctorch::Tensor(8.5f));
    Y.push_back(cctorch::Tensor(11.0f));

    int epochs = 40;
    float learning_rate = 0.01f;
    cctorch::Linear linear(1, 1);
    cctorch::SGD optimizer(linear.parameters(), learning_rate);
    cctorch::MSELoss criterion;

    std::cout << "parameters size: " << optimizer.parameters.size() << std::endl;

    for (int epoch = 1; epoch <= epochs; ++epoch)
    {
        float tot_loss = 0;
        // batch gradient descent
        vector<vector<cctorch::Tensor>> batch_inputs;
        for (const auto &x : X)
        {
            batch_inputs.push_back({x});
        }
        vector<vector<cctorch::Tensor>> outputs = linear(batch_inputs);
        auto flat_outputs = flatten(outputs);
        cctorch::Tensor loss = criterion(flat_outputs, Y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        tot_loss += loss.value();

        if (epoch % 1 == 0)
        {
            std::cout << "Epoch [" << epoch << "/" << epochs << "], Loss: " << tot_loss / X.size();
            for (const auto &param : optimizer.parameters)
            {
                std::cout << ", Parameter: " << param.value() << " (grad: " << param.grad() << ")";
            }
            std::cout << std::endl;
        }
    }

    return 0;
}
