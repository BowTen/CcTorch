#ifndef TENSOR_H
#define TENSOR_H

#include <utility>
#include <memory>
#include <functional>
#include <vector>

namespace cctorch
{

    struct tensor_data;

    class Tensor
    {
    public:
        enum class back_type
        {
            NONE,
            ADD,
            SUB,
            MUL,
            DIV,
            RELU,
            EXP,
            LOG
        };

        std::shared_ptr<tensor_data> data;

        float value() const;
        float grad() const;

        Tensor() = default;
        Tensor(float value);
        Tensor(float value, Tensor par1, Tensor par2, back_type back);

        bool topo_decent() const;

        void backward();

        Tensor operator+(const Tensor &other) const;
        Tensor operator-(const Tensor &other) const;
        Tensor operator*(const Tensor &other) const;
        Tensor operator/(const Tensor &other) const;
        Tensor relu() const;
        Tensor exp() const;
        Tensor log() const;

        void zero_grad();
        void drop_par(int i);

    private:
        void _backward() const;
        void add_backward() const;
        void sub_backward() const;
        void mul_backward() const;
        void div_backward() const;
        void relu_backward() const;
        void exp_backward() const;
        void log_backward() const;
    };

    struct tensor_data
    {
        float value;
        float grad;
        Tensor par1;
        Tensor par2;
        unsigned int sons;
        Tensor::back_type back;

        tensor_data(float value);
        tensor_data(float value, Tensor par1, Tensor par2, Tensor::back_type back);
    };

    std::vector<Tensor> to_tensor(const std::vector<float> &vec);

    std::vector<std::vector<Tensor>> to_tensor(const std::vector<std::vector<float>> &vec);

    std::vector<cctorch::Tensor> flatten(const std::vector<std::vector<cctorch::Tensor>> &inputs);

    Tensor random_tensor(float min, float max);
} // namespace cctorch

#endif // TENSOR_H
