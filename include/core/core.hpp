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

#ifndef CORE_H
#define CORE_H

#include <array>
#include <cmath>
#include <concepts>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

namespace core {

/**
 * Defines the representation of a tensor.
 *
 * @tparam T An arithmetic type representing the type of each element in tensor.
 * @tparam Order The NTTP representing the order of a tensor.
 */
template <Arithmetic T, std::size_t Order>
class tensor {
  // NOTE: Prefer "order" to "rank," as its unambiguous and order 0 tensors exist (they're scalars!)
  static_assert(Order >= 0, "Order must be a non-negative integer.");

 private:
  T *m_data;
  std::array<std::size_t, Order> m_dims;
  std::size_t m_size;
  std::array<std::size_t, Order> m_strides;

 public:
  // Core {{{

  /**
   * Constructs an empty tensor.
   */
  constexpr tensor() : m_data{nullptr}, m_dims{}, m_size{0}, m_strides{} {}

  /**
   * Constructs a one-dimensional tensor.
   *
   * @param values An initializer list holding values of the one-dimensional tensor.
   */
  constexpr tensor(std::initializer_list<T> values) {
    if (values.size() == 0) {
      return;
    }

    std::size_t size = values.size();

    m_data = new T[size];
    std::copy(values.begin(), values.end(), m_data);

    m_dims[0] = size;
    m_size = size;
    m_strides[0] = 1;
  }

  /**
   * Constructs an arbitrary-dimensional tensor.
   *
   * @param t_list An initializer list holding tensors.
   */
  constexpr tensor(std::initializer_list<tensor<T, Order> > t_list) {
    if (t_list.size() == 0) {
      m_data = nullptr;
      m_dims = std::array<std::size_t, Order>{};
      m_size = 0;
      m_strides = std::array<std::size_t, Order>{};
      return;
    }

    std::size_t size = 0;
    for (const tensor<T, Order> &t : t_list) {
      if (size == 0) {
        m_dims[0] = t_list.size();
        std::copy(t.m_dims.begin(), t.m_dims.begin() + Order, m_dims.begin() + 1);
      }
      size += t.size();
    }
    m_data = new T[size];
    m_size = size;

    std::size_t acc_idx = 0;
    for (const tensor<T, Order> &t : t_list) {
      for (std::size_t idx = 0; idx < t.size(); ++idx) {
        m_data[acc_idx++] = t.m_data[idx];
      }
    }

    auto prod = static_cast<float>(m_size);
    for (std::size_t idx = 0; idx < Order; ++idx) {
      prod /= m_dims[idx];
      m_strides[idx] = static_cast<std::size_t>(prod);
    }
  }

  /**
   * Constructs a tensor from the provided dimensions.
   *
   * @param dims Dimensions for constructing a tensor.
   */
  constexpr tensor(const std::array<std::size_t, Order> dims) {
    m_dims = dims;
    m_size = std::reduce(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>());
    m_data = new T[m_size];

    auto prod = static_cast<float>(m_size);
    for (std::size_t idx = 0; idx < Order; ++idx) {
      prod /= m_dims[idx];
      m_strides[idx] = static_cast<std::size_t>(prod);
    }
  }

  /**
   * Defines a copy constructor.
   *
   * @param rhs A right-hand side of the assignment.
   */
  constexpr tensor(const tensor &rhs) {
    m_data = new T[rhs.m_size];
    std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);

    m_dims = rhs.m_dims;
    m_size = rhs.m_size;
    m_strides = rhs.m_strides;
  }

  /**
   * Overloads the copy assignment operator.
   *
   * @param rhs A right-hand side of the assignment.
   */
  constexpr auto &operator=(const tensor &rhs) {
    if (this != &rhs) {
      m_data = new T[rhs.m_size];
      std::copy(rhs.m_data, rhs.m_data + rhs.m_size, m_data);

      m_dims = rhs.m_dims;
      m_size = rhs.m_size;
      m_strides = rhs.m_strides;
    }
    return *this;
  }

  /**
   * Defines a move constructor.
   *
   * @param rhs A right-hand side of the assignment.
   */
  constexpr tensor(tensor &&rhs) noexcept
      : m_data{rhs.m_data}, m_dims{rhs.m_dims}, m_size{rhs.m_size}, m_strides{rhs.m_strides} {
    rhs.m_data = nullptr;
  }

  /**
   * Overloads the move assignment operator.
   *
   * @param rhs A right-hand side of the assignment.
   */
  constexpr auto &operator=(const tensor &&rhs) noexcept {
    if (this != &rhs) {
      m_data = rhs.m_data;
      m_dims = rhs.m_dims;
      m_size = rhs.m_size;
      m_strides = rhs.m_strides;

      rhs.m_data = nullptr;
    }
    return *this;
  }

  /**
   * Defines the destructor.
   * Frees the memory and points the dangling pointer to `nullptr`.
   *
   * @param rhs A right-hand side of the assignment.
   */
  constexpr ~tensor() {
    delete[] m_data;
    m_data = nullptr;
  }

  // }}}

  // Core utilities {{{

  /**
   * Defines an operator for getting raw data.
   *
   * @param idx An index for obtaining a value.
   * @throws `std::out_of_range`
   */
  [[nodiscard]] constexpr auto &operator[](const std::size_t idx) const {
    if (idx > m_size - 1) {
      throw std::out_of_range("Index out of bounds.");
    }
    return m_data[idx];
  }

  /**
   * Getter methods for member variables.
   */
  [[nodiscard]] constexpr auto data() const noexcept { return m_data; }
  [[nodiscard]] constexpr auto dims() const noexcept { return m_dims; }
  [[nodiscard]] constexpr auto size() const noexcept { return m_size; }

  /**
   * Getter methods for member variables.
   *
   * @param idxs An array of indices for obtaining data.
   */
  template <std::size_t U>
  [[nodiscard]] constexpr auto get(const std::array<std::size_t, U> idxs) const {
    std::size_t flat_idx = 0;
    for (std::size_t idx = 0; idx < U; ++idx) {
      flat_idx += idxs[idx] * m_strides[idx];
    }

    if constexpr (Order == U) {
      return m_data[flat_idx];
    }

    if constexpr (Order != U) {
      std::array<std::size_t, Order - U> dims;
      std::copy(m_dims.begin() + U, m_dims.end(), dims.begin());

      auto offset =
          std::reduce(m_dims.begin() + idxs.size(), m_dims.end(), 1, std::multiplies<int>());

      auto result = tensor<T, Order - U>(dims);
      for (std::size_t idx = flat_idx; idx < flat_idx + offset; ++idx) {
        result[idx - flat_idx] = m_data[idx];
      }

      return result;
    }
  }

  // }}}

  // Printing {{{

  /**
   * A helper method for printing a tensor.
   *
   * @param data Data representation.
   * @param dims A C-style array representing dimensions.
   * @param order Order of a tensor.
   */
  constexpr const T *__print(const T *data, const std::size_t *dims, const std::size_t order) {
    const char *p_sep = "";
    std::cout << '{';
    if (order > 1) {
      for (std::size_t idx = 0; idx < dims[0]; ++idx) {
        std::cout << p_sep;
        data = __print(data, &dims[1], order - 1);
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

  /**
   * Prints the tensor via helper method.
   */
  constexpr void print() {
    std::cout << "tensor ";
    (void)__print(m_data, m_dims.data(), Order);
    std::cout << '\n';

    std::cout << "shape (";
    if (m_dims.size() == 1) {
      std::cout << m_dims[0];
    } else {
      for (std::size_t idx = 0; idx < m_dims.size() - 1; ++idx) {
        std::cout << m_dims[idx] << ", ";
      }
      std::cout << m_dims.back();
    }
    std::cout << ')' << '\n';

    std::cout << "size " << m_size << '\n';
  }

  /**
   * Prints a flat representation of the tensor.
   */
  constexpr auto flat_print() const {
    std::cout << '{' << ' ';
    for (std::size_t idx = 0; idx < m_size; ++idx) {
      std::cout << m_data[idx] << ' ';
    }
    std::cout << '}' << std::endl;
  }

  // }}}

  // Basic arithmetic operators {{{

  /**
   * Add the other tensor to the tensor.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator+(const tensor &other) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] += other[idx];
    }
    return result;
  }

  /**
   * Subtracts the other tensor from the tensor.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator-(const tensor &other) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] -= other[idx];
    }
    return result;
  }

  /**
   * Multiplies the tensor by the other tensor.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator*(const tensor &other) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] *= other[idx];
    }
    return result;
  }

  /**
   * Divides the tensor by the other tensor.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator/(const tensor &other) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
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
   * Broadcasts addition via the specified value.
   *
   * @tparam U An arithmetic type specifying the type of the addition value.
   * @param val The value to be added to every element of the tensor.
   */
  template <Arithmetic U>
  [[nodiscard]] constexpr auto operator+(const U &val) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] += val;
    }
    return result;
  }

  /**
   * Broadcasts subtraction via the specified value.
   *
   * @tparam U An arithmetic type specifying the type of the subtraction value.
   * @param val The value to be subtracted from every element of the tensor.
   */
  template <Arithmetic U>
  [[nodiscard]] constexpr auto operator-(const U &val) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] -= val;
    }
    return result;
  }

  /**
   * Broadcasts multiplication via the specified value.
   *
   * @tparam U An arithmetic type specifying the type of the multiplication value.
   * @param val The value to be multiplied by every element of the tensor.
   */
  template <Arithmetic U>
  [[nodiscard]] constexpr auto operator*(const U &val) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] *= val;
    }
    return result;
  }

  /**
   * Broadcasts division via the specified value.
   *
   * @tparam U An arithmetic type specifying the type of the division value.
   * @param val The value to be divided by every element of the tensor.
   */
  template <Arithmetic U>
  [[nodiscard]] constexpr auto operator/(const U &val) const {
    if (val == 0) {
      throw std::domain_error("Division by zero.");
    }
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] /= val;
    }
    return result;
  }

  // }}}

  // Comparsion operators {{{

  /**
   * Returns true if the tensor is equal to the other tensor, false otherwise.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator==(const tensor &other) const {
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

  /**
   * Returns true if the tensor is not equal to the other tensor, false otherwise.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator!=(const tensor &other) const {
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

  /**
   * Returns true if the tensor is greater than the other tensor in all element-wise comparisons.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator>(const tensor &other) const {
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

  /**
   * Returns true if the tensor is greater than or equal the other tensor in all element-wise
   * comparisons.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator>=(const tensor &other) const {
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

  /**
   * Returns true if the tensor is less than the other tensor in all element-wise comparisons.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator<(const tensor &other) const {
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

  /**
   * Returns true if the tensor is less than or equal the other tensor in all element-wise
   * comparisons.
   *
   * @param other The other tensor.
   */
  [[nodiscard]] constexpr auto operator<=(const tensor &other) const {
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

  /**
   * Broadcasts the power operation across the tensor.
   *
   * @param exp The exponent.
   */
  template <Arithmetic U>
  [[nodiscard]] constexpr auto pow(U exp) const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::pow(result[idx], exp);
    }
    return result;
  }

  /**
   * Broadcasts the square operation across the tensor.
   */
  [[nodiscard]] constexpr auto square() const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::pow(result[idx], 2);
    }
    return result;
  }

  /**
   * Broadcasts the square root operation across the tensor.
   */
  [[nodiscard]] constexpr auto sqrt() const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::sqrt(result[idx]);
    }
    return result;
  }

  /**
   * Broadcasts the sine operation across the tensor.
   */
  [[nodiscard]] constexpr auto sin() const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::sin(result[idx]);
    }
    return result;
  }

  /**
   * Broadcasts the cosine operation across the tensor.
   */
  [[nodiscard]] constexpr auto cos() const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::cos(result[idx]);
    }
    return result;
  }

  /**
   * Broadcasts the tangent operation across the tensor.
   */
  [[nodiscard]] constexpr auto tan() const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::tan(result[idx]);
    }
    return result;
  }

  /**
   * Broadcasts the round operation across the tensor.
   */
  [[nodiscard]] constexpr auto round() const {
    auto result = *this;
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = std::round(result[idx]);
    }
    return result;
  }

  // }}}
};

}  // namespace core

#endif  // CORE_H
