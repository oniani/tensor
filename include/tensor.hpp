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
    static_assert(Rank >= 0, "Rank must be a non-negative integer.");

   private:
    friend class Tensor<Rank + 1, T>;
    friend class Tensor<Rank - 1, T>;

    T* m_data;
    std::array<std::uint32_t, Rank> m_dims;
    std::size_t m_size;

    /// Verifying that all of the numbers representing dimensions are positive.
    [[nodiscard]] constexpr auto dimcheck(const std::array<std::uint32_t, Rank>& dims) const {
        if (std::find(dims.begin(), dims.end(), 0) != dims.end()) {
            throw std::domain_error("Zero dimension not allowed.");
        }
    }

   public:
    /////////////////////////////////////////////////////////////////////////////////////////////////
    /// Core
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Defines a constructor for an empty tensor.
    Tensor() {
        m_size = 0;
        m_data = nullptr;
    }

    /// Defines a constructor for a tensor using an initializer list.
    Tensor(std::initializer_list<T> list) {
        m_data = nullptr;
        m_size = 0;

        if (list.size() == 0) {
            return;
        }

        m_size = list.size();
        m_dims[0] = m_size;
        m_data = new T[m_size];

        std::size_t idx = 0;
        for (const T& val : list) {
            m_data[idx++] = val;
        }
    }

    /// Defines a constructor for an initializer list of tensors.
    Tensor(std::initializer_list<Tensor<Rank - 1, T>> list) {
        if (list.size() == 0) {
            m_size = 0;
            m_data = nullptr;
            return;
        }

        std::size_t list_size = list.size();
        std::size_t counter = 0;
        for (const Tensor<Rank - 1, T>& t : list) {
            if (counter == 0) {
                m_dims[0] = list_size;
                std::copy(t.m_dims.begin(), t.m_dims.begin() + Rank - 1, m_dims.begin() + 1);
            }
            counter += t.m_size;
        }

        m_size = counter;
        m_data = new T[m_size];

        counter = 0;
        for (const Tensor<Rank - 1, T>& t : list) {
            for (std::size_t idx = 0; idx < t.m_size; idx++) {
                m_data[counter++] = t.m_data[idx];
            }
        }
    }

    /// Defines a copy constructor.
    explicit constexpr Tensor(const Tensor& rhs) {
        m_dims = rhs.m_dims;
        m_data = new T[rhs.m_size]{};
        m_size = rhs.m_size;
        std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);
    }

    // TODO: Not implemented, yet.
    constexpr Tensor& operator=(const Tensor& rhs){};

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
    /// TODO: Copy constructor + copy assignment is the way to go.
    [[nodiscard]] constexpr auto copy() const noexcept {
        Tensor t{};
        t.m_dims = m_dims;
        t.m_data = new T[m_size]{};
        t.m_size = m_size;
        std::copy(m_data, m_data + m_size, t.m_data);
        return t;
    }

    /// Getter methods for member variables.
    [[nodiscard]] constexpr auto data() const noexcept { return m_data; }
    [[nodiscard]] constexpr auto dims() const noexcept { return m_dims; }
    [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

    /// Overrides `<<` to be able to output the tensor.
    friend std::ostream& operator<<(std::ostream& os, const Tensor& a) {}

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
