#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tensor {

template <std::uint32_t N, typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>, T> >
class Tensor {
   private:
    std::array<std::uint32_t, N> m_dims;
    T* m_data;
    std::size_t m_size;

    // Verifying that all of the numbers representing dimensions are positive.
    constexpr void dimcheck(const std::array<std::uint32_t, N>& dims) const {
        for (const auto& dim : dims) {
            if (dim == 0) {
                throw std::domain_error("Zero dimension not allowed.");
            }
        }
    }

   public:
    // Constructs a tensor via provided dimensions.
    constexpr Tensor(std::array<std::uint32_t, N> dims) {
        dimcheck(dims);
        m_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint32_t>());
        m_data = new T[m_size]{};
        m_dims = dims;
    }

    // Constructs a tensor via provided dimensions and a vector.
    constexpr Tensor(std::array<std::uint32_t, N> dims, std::vector<T> v) {
        dimcheck(dims);
        m_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint32_t>());
        m_data = new T[m_size]{};
        std::copy(v.begin(), v.end(), m_data);
        m_dims = dims;
    }

    // Constructs a tensor via pointer.
    constexpr Tensor(std::array<std::uint32_t, N> dims, T* data) {
        dimcheck(dims);
        m_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::uint32_t>());
        m_data = new T[m_size]{};
        if (data) std::memcpy(m_data, data, m_size * sizeof(T));
        m_dims = dims;
    }

    // Constructs a copy of the tensor.
    constexpr Tensor(const Tensor& tensor) = default;

    // Frees the memory and points the dangling pointer to `nullptr`.
    ~Tensor() {
        delete m_data;
        m_data = nullptr;
    }

    // Getter methods for member variables.
    [[nodiscard]] constexpr auto dims() const { return m_dims; }
    [[nodiscard]] constexpr auto data() const { return m_data; }
    [[nodiscard]] constexpr auto size() const { return m_size; }

    // Creates a copy of the tensor.
    [[nodiscard]] constexpr auto copy() const { return Tensor{m_dims, m_data}; }

    // Overrides `<<` to be able to output the tensor.
    //
    // friend std::ostream& operator<<(std::ostream& stream, const Tensor& t) {
    //    auto data = t.data();
    //    for (int idx = N - 1; idx >= 0; idx--) {
    //        stream << "{";
    //        for (int d = 0; d < t.dims()[idx]; d++) {
    //            stream << *data++ << ", ";
    //        }
    //        stream << "}";
    //    }
    //    return stream;
    //}

    constexpr void flat_print() const {
        std::cout << "flat_data { ";
        for (std::size_t idx = 0; idx < m_size - 1; idx++) {
            std::cout << m_data[idx] << " ";
        }
        std::cout << "}" << std::endl;
    }

    [[nodiscard]] constexpr auto& operator[](const size_t idx) const {
        if (idx < 0 or idx > m_size - 1) {
            throw std::out_of_range("Index out of bounds.");
        }
        return m_data[idx];
    }

    // Gets the value by specified indices.
    [[nodiscard]] constexpr auto flat_get(const std::array<std::size_t, N> dims) const {
        int idx = 0;
        auto prod = m_size;
        for (std::size_t i = 0; i < N; i++) {
            prod /= m_dims[i];
            idx += dims[i] * prod;
        }
        return m_data[idx];
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Basic arithmetic
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Adds the tensor to the other tensor.
    [[nodiscard]] constexpr auto operator+(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < tensor.size(); idx++) {
            tensor[idx] += other[idx];
        }
        return tensor;
    }

    // Subtracts the tensor from the other tensor.
    [[nodiscard]] constexpr auto operator-(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] -= other[idx];
        }
        return tensor;
    }

    // Multiplies the tensor by the other tensor.
    [[nodiscard]] constexpr auto operator*(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] *= other[idx];
        }
        return tensor;
    }

    // Divides the tensor by the other tensor.
    [[nodiscard]] constexpr auto operator/(const Tensor& other) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            if (other.m_data[idx] == 0) {
                throw "Division by zero.";
            }
            tensor[idx] /= other[idx];
        }
        return tensor;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Basic arithmetic broadcasting
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Broadcasts addition via the specified value.
    template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>, U> >
    [[nodiscard]] constexpr auto operator+(const U& val) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] += val;
        }
        return tensor;
    }

    // Broadcasts subtraction via the specified value.
    template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>, U> >
    [[nodiscard]] constexpr auto operator-(const U& val) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] -= val;
        }
        return tensor;
    }

    // Broadcasts multiplication via the specified value.
    template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>, U> >
    [[nodiscard]] constexpr auto operator*(const U& val) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] *= val;
        }
        return tensor;
    }

    // Broadcasts divison via the specified value.
    template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>, U> >
    [[nodiscard]] constexpr auto operator/(const U& val) const {
        if (val == 0) {
            throw "Division by zero.";
        }
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] /= val;
        }
        return tensor;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // Handy broadcasting operations
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Broadcasts the power operation.
    template <typename U, typename = std::enable_if_t<std::is_arithmetic_v<U>, U> >
    [[nodiscard]] constexpr auto pow(U exp) const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] = std::pow(tensor[idx], exp);
        }
        return tensor;
    }

    // Broadcasts the square operation.
    [[nodiscard]] constexpr auto square() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] = std::pow(tensor[idx], 2);
        }
        return tensor;
    }

    // Broadcasts the square root operation.
    [[nodiscard]] constexpr auto sqrt() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] = std::sqrt(tensor[idx]);
        }
        return tensor;
    }

    // Broadcasts the sin operation.
    [[nodiscard]] constexpr auto sin() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] = std::sin(tensor[idx]);
        }
        return tensor;
    }

    // Broadcasts the cos operation.
    [[nodiscard]] constexpr auto cos() const {
        auto tensor = this->copy();
        for (std::size_t idx = 0; idx < m_size; idx++) {
            tensor[idx] = std::cos(tensor[idx]);
        }
        return tensor;
    }
};
}  // namespace tensor
