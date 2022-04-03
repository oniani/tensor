/// A zero-dependency tensor implementation based on C++ STL.

#ifndef TENSOR_H
#define TENSOR_H

// `#include` directives {{{

#include <array>
#include <cmath>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

// }}}

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// core {{{

namespace core {

    template <std::size_t Rank, Arithmetic T>
    class tensor {
        static_assert(Rank > 0, "Rank must be a positive integer.");

       private:
        T* m_data;
        std::array<std::size_t, Rank> m_dims;
        std::size_t m_size;
        std::array<std::size_t, Rank> m_strides;

       public:
        // Core {{{

        /// Constructs an empty tensor.
        constexpr tensor() noexcept {
            m_data = nullptr;
            m_dims = std::array<std::size_t, Rank>{};
            m_size = 0;
            m_strides = std::array<std::size_t, Rank>{};
        }

        /// Constructs a one-dimensional tensor.
        constexpr tensor(std::initializer_list<T> v_list) noexcept {
            m_data = nullptr;
            m_dims = std::array<std::size_t, Rank>{};
            m_size = 0;
            m_strides = std::array<std::size_t, Rank>{};

            if (v_list.size() == 0) {
                return;
            }

            std::size_t size = v_list.size();

            m_data = new T[size]{};
            m_dims[0] = size;
            m_size = size;
            m_strides[0] = 1;

            std::copy(v_list.begin(), v_list.end(), m_data);
        }

        /// Constructs an arbitrary-dimensional tensor.
        constexpr tensor(std::initializer_list<tensor<Rank, T>> t_list) noexcept {
            if (t_list.size() == 0) {
                m_data = nullptr;
                m_dims = std::array<std::size_t, Rank>{};
                m_size = 0;
                m_strides = std::array<std::size_t, Rank>{};
                return;
            }

            std::size_t size = 0;
            for (const tensor<Rank, T>& t : t_list) {
                if (size == 0) {
                    m_dims[0] = t_list.size();
                    std::copy(t.m_dims.begin(), t.m_dims.begin() + Rank, m_dims.begin() + 1);
                }
                size += t.size();
            }
            m_data = new T[size]{};
            m_size = size;

            std::size_t acc_idx = 0;
            for (const tensor<Rank, T>& t : t_list) {
                for (std::size_t idx = 0; idx < t.size(); ++idx) {
                    m_data[acc_idx++] = t.m_data[idx];
                }
            }

            auto prod = static_cast<float>(m_size);
            for (std::size_t idx = 0; idx < Rank; ++idx) {
                prod /= m_dims[idx];
                m_strides[idx] = static_cast<std::size_t>(prod);
            }
        }

        /// Constructs a tensor with the provided dimensions.
        constexpr tensor(const std::array<std::size_t, Rank> dims) noexcept {
            m_dims = dims;
            m_size = std::accumulate(begin(dims), end(dims), 1, std::multiplies<std::size_t>());
            m_data = new T[m_size]{};

            auto prod = static_cast<float>(m_size);
            for (std::size_t idx = 0; idx < Rank; ++idx) {
                prod /= m_dims[idx];
                m_strides[idx] = static_cast<std::size_t>(prod);
            }
        }

        /// Defines a copy constructor.
        constexpr tensor(const tensor& rhs) noexcept {
            m_data = new T[rhs.m_size]{};
            std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);

            m_dims = rhs.m_dims;
            m_size = rhs.m_size;
            m_strides = rhs.m_strides;
        }

        /// Overloads the copy assignment operator.
        constexpr auto& operator=(const tensor& rhs) noexcept {
            if (this != &rhs) {
                m_data = new T[rhs.m_size]{};
                std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);

                m_dims = rhs.m_dims;
                m_size = rhs.m_size;
                m_strides = rhs.m_strides;
            }
            return *this;
        }

        /// Defines a move constructor.
        constexpr tensor(tensor&& rhs) noexcept
            : m_data(rhs.m_data), m_dims(rhs.m_dims), m_size(rhs.m_size), m_strides(rhs.m_strides) {
            rhs.m_data = nullptr;
        }

        /// Overloads the move assignment operator.
        constexpr auto& operator=(const tensor&& rhs) noexcept {
            if (this != &rhs) {
                m_data = rhs.m_data;
                m_dims = rhs.m_dims;
                m_size = rhs.m_size;
                m_strides = rhs.m_strides;

                rhs.m_data = nullptr;
            }
            return *this;
        }

        /// Frees the memory and points the dangling pointer to `nullptr`.
        constexpr ~tensor() noexcept {
            delete[] m_data;
            m_data = nullptr;
        }

        // }}}

        // Convenience {{{

        /// Helps priting a tensor.
        constexpr const T* __print(const T* data, const std::size_t* dims, const std::size_t rank) {
            const char* p_sep = "";
            std::cout << '{';
            if (rank > 1) {
                for (std::size_t idx = 0; idx < dims[0]; ++idx) {
                    std::cout << p_sep;
                    data = __print(data, &dims[1], rank - 1);
                    p_sep = ", ";
                }
            } else {
                for (std::size_t idx = 0; idx < dims[0]; ++idx) {
                    std::cout << p_sep << *data++;
                    p_sep = ", ";
                }
            }
            std::cout << '}';
            return data;
        }

        /// Prints the tensor.
        constexpr void print() {
            (void)__print(m_data, m_dims.data(), Rank);
            std::cout << '\n';
        }

        /// Prints a flat representation of the tensor.
        constexpr auto flat_print() const {
            std::cout << '{' << ' ';
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                std::cout << m_data[idx] << ' ';
            }
            std::cout << '}' << std::endl;
        }

        // }}}

        // Core utilities {{{

        /// Getter methods for member variables.
        [[nodiscard]] constexpr auto data() const noexcept { return m_data; }
        [[nodiscard]] constexpr auto dims() const noexcept { return m_dims; }
        [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

        /// Gets the value by specified indices.
        [[nodiscard]] constexpr auto get(const std::array<std::size_t, Rank> dims) const {
            std::size_t flat_idx = 0;
            for (std::size_t idx = 0; idx < Rank; ++idx) {
                flat_idx += dims[idx] * m_strides[idx];
            }
            return m_data[flat_idx];
        }

        /// Operator for getting data.
        [[nodiscard]] constexpr auto& operator[](const std::size_t idx) const {
            if (idx < 0 or idx > m_size - 1) {
                throw std::out_of_range("Index out of bounds.");
            }
            return m_data[idx];
        }

        // }}}

        // Basic arithmetic operators {{{

        /// Adds the tensor to the other tensor.
        [[nodiscard]] constexpr auto operator+(const tensor& other) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] += other[idx];
            }
            return tensor;
        }

        /// Subtracts the tensor from the other tensor.
        [[nodiscard]] constexpr auto operator-(const tensor& other) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] -= other[idx];
            }
            return tensor;
        }

        /// Multiplies the tensor by the other tensor.
        [[nodiscard]] constexpr auto operator*(const tensor& other) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] *= other[idx];
            }
            return tensor;
        }

        /// Divides the tensor by the other tensor.
        [[nodiscard]] constexpr auto operator/(const tensor& other) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                if (other[idx] == 0) {
                    throw std::domain_error("Division by zero.");
                }
                tensor[idx] /= other[idx];
            }
            return tensor;
        }

        // }}}

        // Basic arithmetic broadcasting {{{

        /// Broadcasts addition via the specified value.
        template <Arithmetic U>
        [[nodiscard]] constexpr auto operator+(const U& val) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] += val;
            }
            return tensor;
        }

        /// Broadcasts subtraction via the specified value.
        template <Arithmetic U>
        [[nodiscard]] constexpr auto operator-(const U& val) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] -= val;
            }
            return tensor;
        }

        /// Broadcasts multiplication via the specified value.
        template <Arithmetic U>
        [[nodiscard]] constexpr auto operator*(const U& val) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] *= val;
            }
            return tensor;
        }

        /// Broadcasts divison via the specified value.
        template <Arithmetic U>
        [[nodiscard]] constexpr auto operator/(const U& val) const {
            if (val == 0) {
                throw std::domain_error("Division by zero.");
            }
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] /= val;
            }
            return tensor;
        }

        // }}}

        // Comparsion operators {{{

        /// Returns true if the tensor is equal to the other tensor, false otherwise.
        [[nodiscard]] constexpr auto operator==(const tensor& other) const {
            if (m_size != other.size()) {
                return false;
            }
            if (m_dims != other.dims()) {
                return false;
            }
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                if (m_data[idx] != other[idx]) {
                    return false;
                }
            }
            return true;
        }

        /// Returns true if the tensor is not equal to the other tensor, false otherwise.
        [[nodiscard]] constexpr auto operator!=(const tensor& other) const {
            if (m_size != other.size()) {
                return true;
            }
            if (m_dims != other.dims()) {
                return true;
            }
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                if (m_data[idx] == other[idx]) {
                    return false;
                }
            }
            return true;
        }

        /// Returns true if the tensor is greater than the other tensor in all element-wise
        /// comparisons.
        [[nodiscard]] constexpr auto operator>(const tensor& other) const {
            if (m_size != other.size()) {
                throw std::runtime_error("Tensor size mismatch.");
            }
            if (m_dims != other.dims()) {
                throw std::runtime_error("Tensor dimension mismatch.");
            }
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                if (m_data[idx] <= other[idx]) {
                    return false;
                }
            }
            return true;
        }

        /// Returns true if the tensor is greater than or equal the other tensor in all element-wise
        /// comparisons.
        [[nodiscard]] constexpr auto operator>=(const tensor& other) const {
            if (m_size != other.size()) {
                throw std::runtime_error("Tensor size mismatch.");
            }
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                if (m_data[idx] < other[idx]) {
                    return false;
                }
            }
            return true;
        }

        /// Returns true if the tensor is less than the other tensor in all element-wise
        /// comparisons.
        [[nodiscard]] constexpr auto operator<(const tensor& other) const {
            if (m_size != other.size()) {
                throw std::runtime_error("Tensor size mismatch.");
            }
            if (m_dims != other.dims()) {
                throw std::runtime_error("Tensor dimension mismatch.");
            }
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                if (m_data[idx] >= other[idx]) {
                    return false;
                }
            }
            return true;
        }

        /// Returns true if the tensor is less than or equal the other tensor in all element-wise
        /// comparisons.
        [[nodiscard]] constexpr auto operator<=(const tensor& other) const {
            if (m_size != other.size()) {
                throw std::runtime_error("Tensor size mismatch.");
            }
            if (m_dims != other.dims()) {
                throw std::runtime_error("Tensor dimension mismatch.");
            }
            for (std::size_t idx = 0; idx < m_size; ++idx) {
                if (m_data[idx] > other[idx]) {
                    return false;
                }
            }
            return true;
        }

        // }}}

        // Handy broadcasting operations {{{

        /// Broadcasts the power operation.
        template <Arithmetic U>
        [[nodiscard]] constexpr auto pow(U exp) const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] = std::pow(tensor[idx], exp);
            }
            return tensor;
        }

        /// Broadcasts the square operation.
        [[nodiscard]] constexpr auto square() const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] = std::pow(tensor[idx], 2);
            }
            return tensor;
        }

        /// Broadcasts the square root operation.
        [[nodiscard]] constexpr auto sqrt() const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] = std::sqrt(tensor[idx]);
            }
            return tensor;
        }

        /// Broadcasts the sin operation.
        [[nodiscard]] constexpr auto sin() const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] = std::sin(tensor[idx]);
            }
            return tensor;
        }

        /// Broadcasts the cos operation.
        [[nodiscard]] constexpr auto cos() const {
            auto tensor = *this;
            for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
                tensor[idx] = std::cos(tensor[idx]);
            }
            return tensor;
        }

        // }}}
    };

}  // namespace core

// }}}

// tensor type {{{

namespace type {

template <Arithmetic T>
using tensor1d = core::tensor<1, T>;

template <Arithmetic T>
using tensor2d = core::tensor<2, T>;

template <Arithmetic T>
using tensor3d = core::tensor<3, T>;

template <Arithmetic T>
using tensor4d = core::tensor<4, T>;

template <Arithmetic T>
using tensor5d = core::tensor<5, T>;

}  // namespace type

// }}}

// builder {{{

namespace builder {

/// Constructs a tensor of zeros with the provided dimensions.
template <std::size_t Rank, Arithmetic T>
[[nodiscard]] constexpr auto zeros(std::array<std::size_t, Rank> dims) {
    auto tensor = core::tensor<Rank, T>(dims);
    for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
        tensor[idx] = static_cast<T>(0);
    }
    return tensor;
}

/// Constructs a tensor of ones with the provided dimensions.
template <std::size_t Rank, Arithmetic T>
[[nodiscard]] constexpr auto ones(std::array<std::size_t, Rank> dims) {
    auto tensor = core::tensor<Rank, T>(dims);
    for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
        tensor[idx] = static_cast<T>(1);
    }
    return tensor;
}

/// Constructs a tensor of `x`s with the provided dimensions.
template <std::size_t Rank, Arithmetic T>
[[nodiscard]] constexpr auto xs(std::array<std::size_t, Rank> dims, const T x) {
    auto tensor = core::tensor<Rank, T>(dims);
    for (std::size_t idx = 0; idx < tensor.size(); ++idx) {
        tensor[idx] = static_cast<T>(x);
    }
    return tensor;
}

/// Constructs a tensor of zeros with the shape that matches that of the provided tensor.
template <std::size_t Rank, Arithmetic T>
[[nodiscard]] constexpr auto zeros_like(const core::tensor<Rank, T>& t) {
    auto tensor = core::tensor<Rank, T>(t.dims());
    for (std::size_t idx = 0; idx < t.size(); ++idx) {
        tensor[idx] = static_cast<T>(0);
    }
    return tensor;
}

/// Constructs a tensor of ones with the shape that matches that of the provided tensor.
template <std::size_t Rank, Arithmetic T>
[[nodiscard]] constexpr auto ones_like(const core::tensor<Rank, T>& t) {
    auto tensor = core::tensor<Rank, T>(t.dims());
    for (std::size_t idx = 0; idx < t.size(); ++idx) {
        tensor[idx] = static_cast<T>(1);
    }
    return tensor;
}

/// Constructs a tensor of `x`s with the shape that matches that of the provided tensor.
template <std::size_t Rank, Arithmetic T>
[[nodiscard]] constexpr auto xs_like(const core::tensor<Rank, T>& t, const T x) {
    auto tensor = core::tensor(t.dims());
    for (std::size_t idx = 0; idx < t.size(); ++idx) {
        tensor[idx] = static_cast<T>(x);
    }
    return tensor;
}

/// Constructs a one-dimensional tensor including values from begin to end with the given stride.
template <Arithmetic T>
[[nodiscard]] constexpr auto range1d(const T begin, const T end, const T stride) {
    auto dims = std::array<std::size_t, 1>{static_cast<std::size_t>((end - begin) / stride)};
    auto tensor = core::tensor<1, T>(dims);
    std::size_t idx = 0;
    for (T val = begin; val < end; val += stride) {
        tensor[idx++] = val;
    }
    return tensor;
}

}  // namespace builder

// }}}

#endif  // BF_H
