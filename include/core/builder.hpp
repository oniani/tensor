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

#ifndef BUILDER_HPP
#define BUILDER_HPP

#include "core.hpp"

namespace builder {

  /**
   * @brief Constructs a tensor of zeros from the provided extents.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Order of the tensor.
   * @param extents Extents for constructing a tensor of zeros.
   * @return A tensor of zeros with the provided extents.
   */
  template <Arithmetic T, size_type Order>
  [[nodiscard]] constexpr auto zeros(const array<Order>& extents) {
    auto result = core::tensor<T, Order>(extents);
    for (size_type idx = 0; idx < result.size(); ++idx) {
      result[idx] = static_cast<T>(0);
    }
    return result;
  }

  /**
   * @brief Constructs a tensor of ones from the provided extents.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Order of the tensor.
   * @param extents Extents for constructing a tensor of ones.
   * @return A tensor of ones with the provided extents.
   */
  template <Arithmetic T, size_type Order>
  [[nodiscard]] constexpr auto ones(const array<Order>& extents) {
    auto result = core::tensor<T, Order>(extents);
    for (size_type idx = 0; idx < result.size(); ++idx) {
      result[idx] = static_cast<T>(1);
    }
    return result;
  }

  /**
   * @brief Constructs a tensor of specified values from the provided extents.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Order of the tensor.
   * @param extents Extents for constructing a tensor of specified values.
   * @return A tensor of specified values with the provided extents.
   */
  template <Arithmetic T, size_type Order>
  [[nodiscard]] constexpr auto xs(const array<Order>& extents, const T x) {
    auto result = core::tensor<T, Order>(extents);
    for (size_type idx = 0; idx < result.size(); ++idx) {
      result[idx] = x;
    }
    return result;
  }

  /**
   * @brief Constructs a tensor of zeros from the extents of the provided tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Order of the tensor.
   * @param t Tensor to match the extents against.
   * @return A tensor of zeros with the extents of the provided tensor.
   */
  template <Arithmetic T, size_type Order>
  [[nodiscard]] constexpr auto zeros_like(const core::tensor<T, Order>& t) {
    auto result = core::tensor<T, Order>(t.extents());
    for (size_type idx = 0; idx < t.size(); ++idx) {
      result[idx] = static_cast<T>(0);
    }
    return result;
  }

  /**
   * @brief Constructs a tensor of ones from the extents of the provided tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Order of the tensor.
   * @param t Tensor to match the extents against.
   * @return A tensor of ones with the extents of the provided tensor.
   */
  template <Arithmetic T, size_type Order>
  [[nodiscard]] constexpr auto ones_like(const core::tensor<T, Order>& t) {
    auto result = core::tensor<T, Order>(t.extents());
    for (size_type idx = 0; idx < t.size(); ++idx) {
      result[idx] = static_cast<T>(1);
    }
    return result;
  }

  /**
   * @brief Constructs a tensor of specified values from the extents of the provided tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Order of the tensor.
   * @param t Tensor to match the extents against.
   * @return A tensor of specified values with the extents of the provided tensor.
   */
  template <Arithmetic T, size_type Order>
  [[nodiscard]] constexpr auto xs_like(const core::tensor<T, Order>& t, const T x) {
    auto result = core::tensor(t.extents());
    for (size_type idx = 0; idx < t.size(); ++idx) {
      result[idx] = x;
    }
    return result;
  }

  /**
   * @brief Constructs an order one tensor including values from begin to end with the given stride.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   * @param begin Start of the range.
   * @param end End of the range.
   * @param stride Stride for the range.
   * @return An order one tensor of values specified by the provided range and stride.
   */
  template <Arithmetic T>
  [[nodiscard]] constexpr auto range1(const T begin, const T end, const T stride) {
    auto extents = std::array<size_type, 1>{static_cast<size_type>((end - begin) / stride)};
    auto result = core::tensor<T, 1>(extents);
    size_type idx = 0;
    for (T val = begin; val < end; val += stride) {
      result[idx++] = val;
    }
    return result;
  }

}  // namespace builder

#endif  // BUILDER_HPP
