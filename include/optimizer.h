#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>

namespace cctorch
{
    class SGD
    {
    public:
        std::vector<Tensor> parameters;
        float learning_rate;

        SGD(std::vector<Tensor> parameters, float learning_rate) : parameters(parameters), learning_rate(learning_rate) {}

        void zero_grad()
        {
            for (int i = 0; i < parameters.size(); i++)
                parameters[i].zero_grad();
        }

        void step()
        {
            for (int i = 0; i < parameters.size(); i++)
                parameters[i].data->value -= parameters[i].data->grad * learning_rate;
        }
    };
}

#endif // OPTIMIZER_H