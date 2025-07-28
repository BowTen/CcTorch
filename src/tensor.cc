#include "../include/tensor.h"
#include <queue>
#include <iostream>
#include <cmath>

namespace cctorch
{

	// Constructor implementations
	Tensor::Tensor(float value) : data(std::make_shared<tensor_data>(value)) {}

	Tensor::Tensor(float value, Tensor par1, Tensor par2, back_type back)
		: data(std::make_shared<tensor_data>(value, par1, par2, back)) {}

	tensor_data::tensor_data(float value)
		: value(value), grad(0.0f), sons(0), back(Tensor::back_type::NONE) {}

	tensor_data::tensor_data(float value, Tensor par1, Tensor par2, Tensor::back_type back)
		: value(value), grad(0.0f), par1(par1), par2(par2), sons(0), back(back) {}

	float Tensor::value() const
	{
		return data ? data->value : 0.0f; // Return 0 if data is null
	}

	float Tensor::grad() const
	{
		return data ? data->grad : 0.0f; // Return 0 if data is null
	}

	bool Tensor::topo_decent() const
	{
		return data != nullptr && --data->sons == 0;
	}

	// Backward propagation implementation
	void Tensor::_backward() const
	{
		switch (this->data->back) // Using switch-case for clarity
		{
		case back_type::ADD:
			add_backward();
			break;

		case back_type::SUB:
			sub_backward();
			break;

		case back_type::MUL:
			mul_backward();
			break;

		case back_type::DIV:
			div_backward();
			break;

		case back_type::RELU:
			relu_backward();
			break;

		case back_type::EXP:
			exp_backward();
			break;

		case back_type::LOG:
			log_backward();
			break;

		case back_type::NONE:
			return;
		}
	}

	void Tensor::backward()
	{
		std::queue<Tensor> que;
		que.push(*this);
		this->data->grad = 1.0f; // Initialize the gradient for the root tensor
		while (!que.empty())
		{
			Tensor current = que.front();
			que.pop();
			current._backward();

			if (current.data->par1.topo_decent())
			{
				que.push(current.data->par1);
			}
			if (current.data->par2.topo_decent())
			{
				que.push(current.data->par2);
			}
			current.drop_par(1); // Clear the reference to avoid dangling pointers
			current.drop_par(2); // Clear the reference to avoid dangling pointers
		}
	}

	// Operator implementations
	Tensor Tensor::operator+(const Tensor &other) const
	{
		this->data->sons++;
		other.data->sons++;
		return Tensor(this->data->value + other.data->value, *this, other, back_type::ADD);
	}

	// Operator implementations
	Tensor Tensor::operator-(const Tensor &other) const
	{
		this->data->sons++;
		other.data->sons++;
		return Tensor(this->data->value - other.data->value, *this, other, back_type::SUB);
	}

	Tensor Tensor::operator*(const Tensor &other) const
	{
		this->data->sons++;
		other.data->sons++;
		return Tensor(this->data->value * other.data->value, *this, other, back_type::MUL);
	}

	Tensor Tensor::operator/(const Tensor &other) const
	{
		this->data->sons++;
		other.data->sons++;
		return Tensor(this->data->value / other.data->value, *this, other, back_type::DIV);
	}

	Tensor Tensor::relu() const
	{
		this->data->sons++;
		return Tensor((this->data->value > 0) ? this->data->value : 0, *this, Tensor(), back_type::RELU);
	}

	Tensor Tensor::exp() const
	{
		this->data->sons++;
		return Tensor(std::exp(this->data->value), *this, Tensor(), back_type::EXP);
	}

	Tensor Tensor::log() const
	{
		this->data->sons++;
		return Tensor(std::log(this->data->value), *this, Tensor(), back_type::LOG);
	}

	void Tensor::zero_grad()
	{
		if (data)
		{
			data->grad = 0.0f;
		}
	}

	void Tensor::drop_par(int i)
	{
		if (data)
		{
			if (i == 1 && data->par1.data)
			{
				data->par1.data = nullptr;
			}
			else if (i == 2 && data->par2.data)
			{
				data->par2.data = nullptr;
			}
		}
	}

	// Private backward function implementations
	void Tensor::add_backward() const
	{
		data->par1.data->grad += data->grad;
		data->par2.data->grad += data->grad;
	}

	void Tensor::sub_backward() const
	{
		data->par1.data->grad += data->grad;
		data->par2.data->grad -= data->grad;
	}

	void Tensor::mul_backward() const
	{
		data->par1.data->grad += data->par2.data->value * data->grad;
		data->par2.data->grad += data->par1.data->value * data->grad;
	}

	void Tensor::div_backward() const
	{
		data->par1.data->grad += data->grad / data->par2.data->value;
		data->par2.data->grad -= (data->par1.data->value * data->grad) / (data->par2.data->value * data->par2.data->value);
	}

	void Tensor::relu_backward() const
	{
		if (data->par1.data->value > 0) // 基于输入值判断，而不是输出值
		{
			data->par1.data->grad += data->grad;
		}
	}

	void Tensor::exp_backward() const
	{
		data->par1.data->grad += data->grad * data->value;
	}

	void Tensor::log_backward() const
	{
		data->par1.data->grad += data->grad / data->par1.value();
	}

	std::vector<cctorch::Tensor> flatten(const std::vector<std::vector<cctorch::Tensor>> &inputs)
	{
		std::vector<cctorch::Tensor> flat;
		for (const auto &vec : inputs)
		{
			for (const auto &tensor : vec)
			{
				flat.push_back(tensor);
			}
		}
		return flat;
	}

	Tensor random_tensor(float min, float max)
	{
		float value = min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
		return Tensor(value);
	}

	std::vector<Tensor> to_tensor(const std::vector<float> &vec)
	{
		std::vector<Tensor> tensors;
		tensors.reserve(vec.size());
		for (float v : vec)
		{
			tensors.emplace_back(v);
		}
		return tensors;
	}

	std::vector<std::vector<Tensor>> to_tensor(const std::vector<std::vector<float>> &vec)
	{
		std::vector<std::vector<Tensor>> tensors;
		tensors.reserve(vec.size());
		for (const auto &v : vec)
		{
			tensors.emplace_back(to_tensor(v));
		}
		return tensors;
	}
} // namespace cctorch