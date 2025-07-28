#ifndef LOSS_H
#define LOSS_H

#include "tensor.h"
#include <iostream>

namespace cctorch
{
    class MSELoss
    {
    public:
        Tensor operator()(const std::vector<Tensor> &predictions, const std::vector<Tensor> &targets)
        {
            if (predictions.size() != targets.size())
            {
                throw std::invalid_argument("Predictions and targets must have the same size.");
            }

            Tensor loss = Tensor(0.0f);
            for (size_t i = 0; i < predictions.size(); ++i)
            {
                auto diff = predictions[i] - targets[i];
                loss = loss + (diff * diff);
            }
            return loss / Tensor(static_cast<float>(predictions.size()));
        }
    };

    class CrossEntropyLoss
    {
    public:
        Tensor operator()(const std::vector<std::vector<Tensor>> &predictions, const std::vector<unsigned char> &targets)
        {
            if (predictions.size() != targets.size())
            {
                throw std::invalid_argument("Predictions and targets must have the same size.");
            }

            Tensor total_loss = Tensor(0.0f);
            for (size_t i = 0; i < predictions.size(); ++i)
            {
                Tensor sum_exp = Tensor(0.0f);
                for (const auto &pred : predictions[i])
                {
                    sum_exp = sum_exp + pred.exp();
                }
                auto loss = sum_exp.log() - predictions[i][targets[i]];
                total_loss = total_loss + loss;
            }
            return total_loss / Tensor(static_cast<float>(predictions.size()));
        }
    };
}

#endif // LOSS_H