/// A zero-dependency tensor implementation based on C++ STL.

#ifndef TENSOR_H
#define TENSOR_H

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor {

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <std::uint32_t Rank, Arithmetic T>
class Tensor {
    static_assert(Rank > 0, "Rank must be a positive integer.");

    using Dims = std::array<std::uint32_t, Rank>;

   private:
    Dims m_dims;
    T* m_data;
    std::size_t m_size;

    /// Verifying that all of the numbers representing dimensions are positive.
    [[nodiscard]] constexpr auto dimcheck(const Dims& dims) const {
        if (std::find(dims.begin(), dims.end(), 0) != dims.end()) {
            throw std::domain_error("Zero dimension not allowed.");
        }
    }

   public:
    /// Constructs a tensor via provided dimensions and a vector.
    explicit constexpr Tensor(Dims dims, std::vector<T> data) {
        dimcheck(dims);
        m_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint32_t>());
        m_data = new T[m_size]{};
        std::copy(data.begin(), data.end(), m_data);
        m_dims = dims;
    }

    /// Constructs a tensor via pointer.
    explicit constexpr Tensor(Dims dims, T* data) {
        dimcheck(dims);
        m_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint32_t>());
        m_data = new T[m_size]{};
        if (data) std::memcpy(m_data, data, m_size * sizeof(T));
        m_dims = dims;
    }

    /// Defines a copy constructor.
    explicit constexpr Tensor(const Tensor& rhs) {
        m_dims = rhs.m_dims;
        m_data = new T[rhs.m_size]{};
        for (std::size_t idx = 0; idx < rhs.m_size; idx++) {
            m_data[idx] = rhs.m_data[idx];
        }
        m_size = rhs.m_size;
    }

    /// Defines a move constructor.
    constexpr Tensor(Tensor&& rhs) noexcept
        : m_dims(rhs.m_dims), m_data(rhs.m_data), m_size(rhs.m_size) {
        rhs.m_data = nullptr;
    }

    /// Frees the memory and points the dangling pointer to `nullptr`.
    ~Tensor() noexcept {
        delete[] m_data;
        m_data = nullptr;
    }

    /// Creates a copy constructor for the tensor.
    [[nodiscard]] constexpr auto copy() const noexcept { return Tensor{m_dims, m_data}; }

    /// Getter methods for member variables.
    [[nodiscard]] constexpr auto dims() const noexcept { return m_dims; }
    [[nodiscard]] constexpr auto data() const noexcept { return m_data; }
    [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

    /// Overrides `<<` to be able to output the tensor.
    ///
    /// friend std::ostream& operator<<(std::ostream& stream, const Tensor& t) {
    ///    auto data = t.data();
    ///    for (int idx = N - 1; idx >= 0; idx--) {
    ///        stream << "{";
    ///        for (int d = 0; d < t.dims()[idx]; d++) {
    ///            stream << *data++ << ", ";
    ///        }
    ///        stream << "}";
    ///    }
    ///    return stream;
    ///}
    ///

    [[nodiscard]] constexpr auto& operator[](const std::size_t idx) const {
        if (idx < 0 or idx > m_size - 1) {
            throw std::out_of_range("Index out of bounds.");
        }
        return m_data[idx];
    }

    constexpr auto flat_print() const {
        std::cout << "flat_data { ";
        for (std::size_t idx = 0; idx < m_size - 1; idx++) {
            std::cout << m_data[idx] << " ";
        }
        std::cout << "}" << std::endl;
    }

    /// Gets the value by specified indices.
    [[nodiscard]] constexpr auto flat_get(const std::array<std::size_t, Rank> dims) const {
        int idx = 0;
        auto prod = m_size;
        for (std::size_t i = 0; i < Rank; i++) {
            prod /= m_dims[i];
            idx += dims[i] * prod;
        }
        return m_data[idx];
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Basic arithmetic
    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// Adds the tensor to the other tensor.
    [[nodiscard]] constexpr auto operator+(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] += other[idx];
        }
        return tensor;
    }

    /// Subtracts the tensor from the other tensor.
    [[nodiscard]] constexpr auto operator-(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] -= other[idx];
        }
        return tensor;
    }

    /// Multiplies the tensor by the other tensor.
    [[nodiscard]] constexpr auto operator*(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] *= other[idx];
        }
        return tensor;
    }

    /// Divides the tensor by the other tensor.
    [[nodiscard]] constexpr auto operator/(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            if (other[idx] == 0) {
                throw std::domain_error("Division by zero.");
            }
            tensor[idx] /= other[idx];
        }
        return tensor;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Basic arithmetic broadcasting
    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// Broadcasts addition via the specified value.
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator+(const U& val) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] += val;
        }
        return tensor;
    }

    /// Broadcasts subtraction via the specified value.
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator-(const U& val) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] -= val;
        }
        return tensor;
    }

    /// Broadcasts multiplication via the specified value.
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator*(const U& val) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
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
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] /= val;
        }
        return tensor;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Handy broadcasting operations
    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// Broadcasts the power operation.
    template <Arithmetic U>
    [[nodiscard]] constexpr auto pow(U exp) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] = std::pow(tensor[idx], exp);
        }
        return tensor;
    }

    /// Broadcasts the square operation.
    [[nodiscard]] constexpr auto square() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] = std::pow(tensor[idx], 2);
        }
        return tensor;
    }

    /// Broadcasts the square root operation.
    [[nodiscard]] constexpr auto sqrt() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] = std::sqrt(tensor[idx]);
        }
        return tensor;
    }

    /// Broadcasts the sin operation.
    [[nodiscard]] constexpr auto sin() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] = std::sin(tensor[idx]);
        }
        return tensor;
    }

    /// Broadcasts the cos operation.
    [[nodiscard]] constexpr auto cos() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] = std::cos(tensor[idx]);
        }
        return tensor;
    }
};
}  // namespace tensor

#endif  // BF_H
