/*
 * MIT License
 *
 * Copyright (c) 2022 David Oniani
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef CORE_HPP
#define CORE_HPP

#include <array>
#include <cmath>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

using size_type = std::size_t;

template <size_type N>
using array = std::array<size_type, N>;

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

namespace core {

  /**
   * Defines the representation of a tensor.
   *
   * NOTE: Prefer "order" to "rank," as its unambiguous and order 0 tensors exist (they are scalars)
   *
   * @tparam T An arithmetic type representing the type of each element in tensor.
   * @tparam Order The NTTP representing the order of a tensor.
   */
  template <Arithmetic T, size_type Order>
  class tensor {
   private:
    T* m_data;
    array<Order> m_extents;
    size_type m_size;
    array<Order> m_strides;

   public:
    // Core {{{

    /**
     * @brief Constructs an empty tensor.
     */
    constexpr tensor() noexcept : m_data{nullptr}, m_extents{}, m_size{0}, m_strides{} {}

    /**
     * @brief Constructs an order one tensor.
     * @param values Initializer list holding values of the order one tensor.
     */
    constexpr tensor(std::initializer_list<T> values) noexcept {
      m_size = values.size();
      m_extents = {m_size};
      m_strides = {1};

      m_data = new T[m_size];
      std::copy(values.begin(), values.end(), m_data);
    }

    /**
     * @brief Constructs a tensor of an arbitrary order.
     * @param t_list Initializer list holding tensors.
     */
    constexpr tensor(std::initializer_list<tensor<T, Order> > t_list) {
      size_type size{0};
      size_type check_size{0};
      for (const tensor<T, Order>& t : t_list) {
        if (size == 0) {
          m_extents[0] = t_list.size();
          std::copy(t.m_extents.begin(), t.m_extents.begin() + Order, m_extents.begin() + 1);
          check_size = t.size();
        } else if (check_size != t.size()) {
          throw std::runtime_error("Tensor dimension mismatch.");
        }
        size += t.size();
      }
      m_data = new T[size];
      m_size = size;

      size_type acc_idx = 0;
      for (const tensor<T, Order>& t : t_list) {
        for (size_type idx = 0; idx < t.size(); ++idx) {
          m_data[acc_idx++] = t.m_data[idx];
        }
      }

      auto prod = static_cast<float>(m_size);
      for (size_type idx = 0; idx < Order; ++idx) {
        prod /= m_extents[idx];
        m_strides[idx] = static_cast<size_type>(prod);
      }
    }

    /**
     * @brief Constructs a tensor from the provided extents.
     * @param extents Extents for constructing a tensor.
     */
    constexpr tensor(const array<Order> extents) {
      m_extents = extents;
      m_size = std::reduce(extents.begin(), extents.end(), 1, std::multiplies<size_type>());
      m_data = new T[m_size];

      auto prod = static_cast<float>(m_size);
      for (size_type idx = 0; idx < Order; ++idx) {
        prod /= m_extents[idx];
        m_strides[idx] = static_cast<size_type>(prod);
      }
    }

    /**
     * @brief Defines a copy constructor.
     * @param rhs Right-hand side of the assignment.
     */
    constexpr tensor(const tensor& rhs) {
      m_data = new T[rhs.m_size];
      std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);

      m_extents = rhs.m_extents;
      m_size = rhs.m_size;
      m_strides = rhs.m_strides;
    }

    /**
     * @brief Overloads the copy assignment operator.
     * @param rhs Right-hand side of the assignment.
     */
    constexpr auto& operator=(const tensor& rhs) {
      if (this != &rhs) {
        m_data = new T[rhs.m_size];
        std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);

        m_extents = rhs.m_extents;
        m_size = rhs.m_size;
        m_strides = rhs.m_strides;
      }
      return *this;
    }

    /**
     * @brief Defines a move constructor.
     * @param rhs Right-hand side of the assignment.
     */
    constexpr tensor(tensor&& rhs) noexcept
        : m_data{rhs.m_data},
          m_extents{rhs.m_extents},
          m_size{rhs.m_size},
          m_strides{rhs.m_strides} {
      rhs.m_data = nullptr;
    }

    /**
     * @brief Overloads the move assignment operator.
     * @param rhs Right-hand side of the assignment.
     */
    constexpr auto& operator=(const tensor&& rhs) noexcept {
      if (this != &rhs) {
        m_data = rhs.m_data;
        m_extents = rhs.m_extents;
        m_size = rhs.m_size;
        m_strides = rhs.m_strides;

        rhs.m_data = nullptr;
      }
      return *this;
    }

    /**
     * @brief Defines the destructor. Frees the memory and points the dangling pointer to `nullptr`.
     * @param rhs Right-hand side of the assignment.
     */
    constexpr ~tensor() {
      delete[] m_data;
      m_data = nullptr;
    }

    // }}}

    // Core utilities {{{

    /**
     * @brief Defines an operator for getting raw data.
     * @param idx Index for obtaining a value.
     */
    [[nodiscard]] constexpr auto& operator[](const size_type idx) const {
      if (idx > m_size - 1) {
        throw std::out_of_range("Index out of bounds.");
      }
      return m_data[idx];
    }

    /**
     * @brief Returns a pointer to the underlying pointer.
     * @return A pointer to the underlying pointer.
     */
    [[nodiscard]] constexpr auto data() const noexcept { return m_data; }

    /**
     * @brief Returns the extents.
     * @return Extents.
     */
    [[nodiscard]] constexpr auto extents() const noexcept { return m_extents; }

    /**
     * @brief Returns the size of the tensor.
     * @return Size of the tensor.
     */
    [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

    /**
     * @brief Getter methods for member variables.
     * @param idxs Array of indices for obtaining data.
     */
    template <size_type U>
    [[nodiscard]] constexpr auto get(const std::array<size_type, U> idxs) const {
      size_type flat_idx = 0;
      for (size_type idx = 0; idx < U; ++idx) {
        flat_idx += idxs[idx] * m_strides[idx];
      }

      if constexpr (Order == U) {
        return m_data[flat_idx];
      }

      if constexpr (Order != U) {
        std::array<size_type, Order - U> extents;
        std::copy(m_extents.begin() + U, m_extents.end(), extents.begin());

        auto offset = std::reduce(m_extents.begin() + idxs.size(), m_extents.end(), 1,
                                  std::multiplies<int>());

        auto result = tensor<T, Order - U>(extents);
        for (size_type idx = flat_idx; idx < flat_idx + offset; ++idx) {
          result[idx - flat_idx] = m_data[idx];
        }

        return result;
      }
    }

    // }}}

    // Printing {{{

    /**
     * @brief Helper method for printing a tensor.
     * @param data Data representation.
     * @param Extents C-style array representing extents.
     * @param order Order of a tensor.
     */
    constexpr const T* __print(const T* data, const size_type* extents, const size_type order) {
      const char* p_sep = "";
      std::cout << '{';
      if (order > 1) {
        for (size_type idx = 0; idx < extents[0]; ++idx) {
          std::cout << p_sep;
          data = __print(data, &extents[1], order - 1);
          p_sep = ", ";
        }
      } else {
        for (size_type idx = 0; idx < extents[0]; ++idx) {
          std::cout << p_sep << *data++;
          p_sep = ", ";
        }
      }
      std::cout << '}';
      return data;
    }

    /**
     * @brief Prints the tensor via the helper method.
     */
    constexpr void print() {
      std::cout << "tensor ";
      (void)__print(m_data, m_extents.data(), Order);
      std::cout << '\n';

      std::cout << "shape (";
      if (m_extents.size() == 1) {
        std::cout << m_extents[0];
      } else {
        for (size_type idx = 0; idx < m_extents.size() - 1; ++idx) {
          std::cout << m_extents[idx] << ", ";
        }
        std::cout << m_extents.back();
      }
      std::cout << ')' << '\n';

      std::cout << "size " << m_size << '\n';
    }

    /**
     * @brief Prints a flat representation of the tensor.
     */
    constexpr void flat_print() const {
      std::cout << '{' << ' ';
      for (size_type idx = 0; idx < m_size; ++idx) {
        std::cout << m_data[idx] << ' ';
      }
      std::cout << '}' << std::endl;
    }

    // }}}

    // Basic arithmetic operators {{{

    /**
     * @brief Adds the other tensor to the tensor.
     * @param other Other tensor.
     * @return New tensor representing the result of the addition.
     */
    [[nodiscard]] constexpr auto operator+(const tensor& other) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] += other[idx];
      }
      return result;
    }

    /**
     * @brief Subtracts the other tensor from the tensor.
     * @param other Other tensor.
     * @return New tensor representing the result of the subtraction.
     */
    [[nodiscard]] constexpr auto operator-(const tensor& other) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] -= other[idx];
      }
      return result;
    }

    /**
     * @brief Multiplies the tensor by the other tensor.
     * @param other Other tensor.
     * @return New tensor representing the result of the multiplication.
     */
    [[nodiscard]] constexpr auto operator*(const tensor& other) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] *= other[idx];
      }
      return result;
    }

    /**
     * @brief Divides the tensor by the other tensor.
     * @param other Other tensor.
     * @return New tensor representing the result of the division.
     */
    [[nodiscard]] constexpr auto operator/(const tensor& other) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        if (other[idx] == 0) {
          throw std::domain_error("Division by zero.");
        }
        result[idx] /= other[idx];
      }
      return result;
    }

    // }}}

    // Basic arithmetic broadcasting {{{

    /**
     * @brief Broadcasts addition via the specified value.
     * @tparam U Arithmetic type specifying the type of the addition value.
     * @param val Value to be added to every element of the tensor.
     * @return Result tensor with every value incremented by `val`.
     */
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator+(const U& val) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] += val;
      }
      return result;
    }

    /**
     * @brief Broadcasts subtraction via the specified value.
     * @tparam U Arithmetic type specifying the type of the subtraction value.
     * @param val Value to be subtracted from every element of the tensor.
     * @return Result tensor with every value decremented by `val`.
     */
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator-(const U& val) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] -= val;
      }
      return result;
    }

    /**
     * @brief Broadcasts multiplication via the specified value.
     * @tparam U Arithmetic type specifying the type of the multiplication value.
     * @param val Value to be multiplied by every element of the tensor.
     * @return Result tensor with every value multiplied by `val`.
     */
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator*(const U& val) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] *= val;
      }
      return result;
    }

    /**
     * @brief Broadcasts division via the specified value.
     * @tparam U Arithmetic type specifying the type of the division value.
     * @param val Value to be divided by every element of the tensor.
     * @return Result tensor with every value divided by `val`.
     */
    template <Arithmetic U>
    [[nodiscard]] constexpr auto operator/(const U& val) const {
      if (val == 0) {
        throw std::domain_error("Division by zero.");
      }
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] /= val;
      }
      return result;
    }

    // }}}

    // Comparsion operators {{{

    /**
     * @brief Returns true if the tensor is equal to the other tensor, false otherwise.
     * @param other Other tensor.
     * @return `true` if the comparison holds, `false` otherwise.
     */
    [[nodiscard]] constexpr auto operator==(const tensor& other) const {
      if (m_size != other.size()) {
        return false;
      }
      if (m_extents != other.extents()) {
        return false;
      }
      for (size_type idx = 0; idx < m_size; ++idx) {
        if (m_data[idx] != other[idx]) {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Returns true if the tensor is not equal to the other tensor, false otherwise.
     * @param other Other tensor.
     * @return `true` if the comparison holds, `false` otherwise.
     */
    [[nodiscard]] constexpr auto operator!=(const tensor& other) const {
      if (m_size != other.size()) {
        return true;
      }
      if (m_extents != other.extents()) {
        return true;
      }
      for (size_type idx = 0; idx < m_size; ++idx) {
        if (m_data[idx] == other[idx]) {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Returns true if tensor is greater in all element-wise comparisons.
     * @param other Other tensor.
     * @return `true` if the comparison holds, `false` otherwise.
     */
    [[nodiscard]] constexpr auto operator>(const tensor& other) const {
      if (m_size != other.size()) {
        throw std::runtime_error("Tensor size mismatch.");
      }
      if (m_extents != other.extents()) {
        throw std::runtime_error("Tensor dimension mismatch.");
      }
      for (size_type idx = 0; idx < m_size; ++idx) {
        if (m_data[idx] <= other[idx]) {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Returns true if tensor is greater than or equal in all element-wise comparisons.
     * @param other Other tensor.
     * @return `true` if the comparison holds, `false` otherwise.
     */
    [[nodiscard]] constexpr auto operator>=(const tensor& other) const {
      if (m_size != other.size()) {
        throw std::runtime_error("Tensor size mismatch.");
      }
      for (size_type idx = 0; idx < m_size; ++idx) {
        if (m_data[idx] < other[idx]) {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Returns true if tensor is less than in all element-wise comparisons.
     * @param other Other tensor.
     * @return `true` if the comparison holds, `false` otherwise.
     */
    [[nodiscard]] constexpr auto operator<(const tensor& other) const {
      if (m_size != other.size()) {
        throw std::runtime_error("Tensor size mismatch.");
      }
      if (m_extents != other.extents()) {
        throw std::runtime_error("Tensor dimension mismatch.");
      }
      for (size_type idx = 0; idx < m_size; ++idx) {
        if (m_data[idx] >= other[idx]) {
          return false;
        }
      }
      return true;
    }

    /**
     * @brief Returns true if tensor is less than or equal in all element-wise comparisons.
     * @param other Other tensor.
     * @return `true` if the comparison holds, `false` otherwise.
     */
    [[nodiscard]] constexpr auto operator<=(const tensor& other) const {
      if (m_size != other.size()) {
        throw std::runtime_error("Tensor size mismatch.");
      }
      if (m_extents != other.extents()) {
        throw std::runtime_error("Tensor dimension mismatch.");
      }
      for (size_type idx = 0; idx < m_size; ++idx) {
        if (m_data[idx] > other[idx]) {
          return false;
        }
      }
      return true;
    }

    // }}}

    // Handy broadcasting operations {{{

    /**
     * @brief Broadcasts the power operation across the tensor.
     * @param exp Exponent.
     * @return Tensor with every value transformed via the power function.
     */
    template <Arithmetic U>
    [[nodiscard]] constexpr auto pow(U exp) const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::pow(result[idx], exp);
      }
      return result;
    }

    /**
     * @brief Broadcasts the square operation across the tensor.
     * @return Tensor with every value transformed via the square function.
     */
    [[nodiscard]] constexpr auto square() const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::pow(result[idx], 2);
      }
      return result;
    }

    /**
     * @brief Broadcasts the square root operation across the tensor.
     * @return Tensor with every value transformed via the square root function.
     */
    [[nodiscard]] constexpr auto sqrt() const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::sqrt(result[idx]);
      }
      return result;
    }

    /**
     * @brief Broadcasts the sine operation across the tensor.
     * @return Tensor with every value transformed via the sine function.
     */
    [[nodiscard]] constexpr auto sin() const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::sin(result[idx]);
      }
      return result;
    }

    /**
     * @brief Broadcasts the cosine operation across the tensor.
     * @return Tensor with every value transformed via the cosine function.
     */
    [[nodiscard]] constexpr auto cos() const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::cos(result[idx]);
      }
      return result;
    }

    /**
     * @brief Broadcasts the tangent operation across the tensor.
     * @return Tensor with every value transformed via the tangent function.
     */
    [[nodiscard]] constexpr auto tan() const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::tan(result[idx]);
      }
      return result;
    }

    /**
     * @brief Broadcasts the round operation across the tensor.
     * @return Tensor with every value transformed via the round function.
     */
    [[nodiscard]] constexpr auto round() const {
      auto result = *this;
      for (size_type idx = 0; idx < result.size(); ++idx) {
        result[idx] = std::round(result[idx]);
      }
      return result;
    }

    // }}}
  };

}  // namespace core

#endif  // CORE_HPP
