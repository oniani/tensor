/// A zero-dependency tensor implementation based on C++ STL.

#ifndef TENSOR_H
#define TENSOR_H

#include <array>
#include <cmath>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace tensor {

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <std::uint32_t Rank, Arithmetic T>
class Tensor {
    static_assert(Rank > 0, "Rank must be a positive integer.");

   private:
    T* m_data;
    std::array<std::uint32_t, Rank> m_dims;
    std::size_t m_size;
    std::array<std::uint32_t, Rank> m_strides;

   public:
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Core
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Defines a constructor for an empty tensor.
    explicit constexpr Tensor() {
        m_data = nullptr;
        m_dims = std::array<std::uint32_t, Rank>{};
        m_size = 0;
        m_strides = std::array<std::uint32_t, Rank>{};
    }

    /// Defines a constructor for a tensor using an initializer list.
    constexpr Tensor(std::initializer_list<T> v_list) {
        m_data = nullptr;
        m_dims = std::array<std::uint32_t, Rank>{};
        m_size = 0;
        m_strides = std::array<std::uint32_t, Rank>{};

        if (v_list.size() == 0) {
            return;
        }

        auto size = v_list.size();

        m_data = new T[size];
        m_dims[0] = size;
        m_size = size;
        m_strides[0] = 1;

        std::copy(v_list.begin(), v_list.end(), m_data);
    }

    /// Defines a constructor for an initializer list of tensors.
    constexpr Tensor(std::initializer_list<Tensor<Rank, T>> t_list) {
        if (t_list.size() == 0) {
            m_data = nullptr;
            m_dims = std::array<std::uint32_t, Rank>{};
            m_size = 0;
            m_strides = std::array<std::uint32_t, Rank>{};
            return;
        }

        std::size_t size = 0;
        for (const Tensor<Rank, T>& t : t_list) {
            if (size == 0) {
                m_dims[0] = t_list.size();
                auto t_dims = t.dims();
                std::copy(t.m_dims.begin(), t.m_dims.begin() + Rank, m_dims.begin() + 1);
            }
            size += t.size();
        }
        m_data = new T[size];
        m_size = size;

        std::size_t acc_idx = 0;
        for (const Tensor<Rank, T>& t : t_list) {
            for (std::size_t idx = 0; idx < t.size(); idx++) {
                m_data[acc_idx++] = t.m_data[idx];
            }
        }

        auto prod = static_cast<float>(m_size);
        for (std::size_t idx = 0; idx < Rank; idx++) {
            prod /= m_dims[idx];
            m_strides[idx] = static_cast<std::size_t>(prod);
        }
    }

    /// Defines a copy constructor.
    explicit constexpr Tensor(const Tensor& rhs) {
        m_data = new T[rhs.m_size]{};
        m_dims = rhs.m_dims;
        m_size = rhs.m_size;
        m_strides = rhs.m_strides;
        std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);
    }

    // TODO: Not implemented, yet.
    constexpr Tensor& operator=(const Tensor& rhs){};

    /// Defines a move constructor.
    constexpr Tensor(Tensor&& rhs) noexcept
        : m_data(rhs.m_data), m_dims(rhs.m_dims), m_size(rhs.m_size), m_strides(rhs.m_strides) {
        rhs.m_data = nullptr;
    }

    /// Creates a copy constructor for the tensor.
    /// TODO: Copy constructor + copy assignment is the way to go.
    [[nodiscard]] constexpr auto copy() const noexcept {
        Tensor t{};
        t.m_data = new T[m_size]{};
        t.m_dims = m_dims;
        t.m_size = m_size;
        t.m_strides = m_strides;
        std::copy(m_data, m_data + m_size, t.m_data);
        return t;
    }

    /// Frees the memory and points the dangling pointer to `nullptr`.
    ~Tensor() noexcept {
        delete[] m_data;
        m_data = nullptr;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Convenience
    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// Getter methods for member variables.
    [[nodiscard]] constexpr auto data() const noexcept { return m_data; }
    [[nodiscard]] constexpr auto dims() const noexcept { return m_dims; }
    [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

    /// Overrides `<<` to be able to output the tensor.
    // friend std::ostream& operator<<(std::ostream& os, const Tensor& a) {}

    [[nodiscard]] constexpr auto& operator[](const std::size_t idx) const {
        if (idx < 0 or idx > m_size - 1) {
            throw std::out_of_range("Index out of bounds.");
        }
        return m_data[idx];
    }

    constexpr auto flat_print() const {
        std::cout << "flat_data { ";
        for (std::size_t idx = 0; idx < m_size; idx++) {
            std::cout << m_data[idx] << " ";
        }
        std::cout << "}" << std::endl;
    }

    /// Gets the value by specified indices.
    [[nodiscard]] constexpr auto get(const std::array<std::size_t, Rank> dims) const {
        std::size_t flat_idx = 0;
        for (std::size_t idx = 0; idx < Rank; idx++) {
            flat_idx += dims[idx] * m_strides[idx];
        }
        return m_data[flat_idx];
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Useful constructors
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // TODO: Need to first write a function that allows for initialization via dimensions.
    template <std::size_t Size>
    [[nodiscard]] constexpr auto zeros(std::array<std::uint32_t, Size> dims) {}

    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Basic arithmetic operators
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
    /// Comparsion operators
    /////////////////////////////////////////////////////////////////////////////////////////////////

    /// Returns true if the tensor is equal to the other tensor, false otherwise.
    [[nodiscard]] constexpr auto operator==(const Tensor& other) const {
        if (m_size != other.size()) {
            return false;
        }
        if (m_dims != other.dims()) {
            return false;
        }
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (this->operator[](idx) != other[idx]) {
                return false;
            }
        }
        return true;
    }

    /// Returns true if the tensor is not equal to the other tensor, false otherwise.
    [[nodiscard]] constexpr auto operator!=(const Tensor& other) const {
        if (m_size != other.size()) {
            return true;
        }
        if (m_dims != other.dims()) {
            return true;
        }
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (this->operator[](idx) == other[idx]) {
                return false;
            }
        }
        return true;
    }

    /// Returns true if the tensor is greater than the other tensor in all element-wise comparisons.
    [[nodiscard]] constexpr auto operator>(const Tensor& other) const {
        if (m_size != other.size()) {
            throw std::runtime_error("Tensor size mismatch.");
        }
        if (m_dims != other.dims()) {
            throw std::runtime_error("Tensor dimension mismatch.");
        }
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (this->operator[](idx) <= other[idx]) {
                return false;
            }
        }
        return true;
    }

    /// Returns true if the tensor is greater than or equal the other tensor in all element-wise
    /// comparisons.
    [[nodiscard]] constexpr auto operator>=(const Tensor& other) const {
        if (m_size != other.size()) {
            throw std::runtime_error("Tensor size mismatch.");
        }
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (this->operator[](idx) < other[idx]) {
                return false;
            }
        }
        return true;
    }

    /// Returns true if the tensor is less than the other tensor in all element-wise comparisons.
    [[nodiscard]] constexpr auto operator<(const Tensor& other) const {
        if (m_size != other.size()) {
            throw std::runtime_error("Tensor size mismatch.");
        }
        if (m_dims != other.dims()) {
            throw std::runtime_error("Tensor dimension mismatch.");
        }
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (this->operator[](idx) >= other[idx]) {
                return false;
            }
        }
        return true;
    }

    /// Returns true if the tensor is less than or equal the other tensor in all element-wise
    /// comparisons.
    [[nodiscard]] constexpr auto operator<=(const Tensor& other) const {
        if (m_size != other.size()) {
            throw std::runtime_error("Tensor size mismatch.");
        }
        if (m_dims != other.dims()) {
            throw std::runtime_error("Tensor dimension mismatch.");
        }
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (this->operator[](idx) > other[idx]) {
                return false;
            }
        }
        return true;
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
