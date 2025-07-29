#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"
#include <vector>
#include <cmath>

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

    class Adam
    {
    public:
        std::vector<Tensor> parameters;
        float learning_rate;
        float beta1;
        float beta2;
        float epsilon;
        std::vector<float> m; // First moment
        std::vector<float> v; // Second moment
        int t;                // Time step

        Adam(std::vector<Tensor> parameters, float learning_rate = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
            : parameters(parameters), learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon), t(0), m(parameters.size(), 0.0f), v(parameters.size(), 0.0f)
        {
        }

        void zero_grad()
        {
            for (int i = 0; i < parameters.size(); i++)
                parameters[i].zero_grad();
        }

        void step()
        {
            ++t;
            for (int i = 0; i < parameters.size(); ++i)
            {
                m[i] = beta1 * m[i] + (1 - beta1) * parameters[i].grad();
                v[i] = beta2 * v[i] + (1 - beta2) * parameters[i].grad() * parameters[i].grad();

                float m_hat = m[i] / (1 - std::pow(beta1, t));
                float v_hat = v[i] / (1 - std::pow(beta2, t));

                parameters[i].data->value -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
            }
        }
    };
}

#endif // OPTIMIZER_H