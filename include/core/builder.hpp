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
   * Constructs a tensor of zeros from the provided dimensions.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Represents the order of the tensor.
   * @param dims Dimensions for constructing a tensor of zeros.
   * @return A tensor of zeros with the provided dimensions.
   */
  template <Arithmetic T, std::size_t Order>
  [[nodiscard]] constexpr auto zeros(const std::array<std::size_t, Order>& dims) {
    auto result = core::tensor<T, Order>(dims);
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = static_cast<T>(0);
    }
    return result;
  }

  /**
   * Constructs a tensor of ones from the provided dimensions.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Represents the order of the tensor.
   * @param dims Dimensions for constructing a tensor of ones.
   * @return A tensor of ones with the provided dimensions.
   */
  template <Arithmetic T, std::size_t Order>
  [[nodiscard]] constexpr auto ones(const std::array<std::size_t, Order>& dims) {
    auto result = core::tensor<T, Order>(dims);
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = static_cast<T>(1);
    }
    return result;
  }

  /**
   * Constructs a tensor of specified values from the provided dimensions.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Represents the order of the tensor.
   * @param dims Dimensions for constructing a tensor of specified values.
   * @return A tensor of specified values with the provided dimensions.
   */
  template <Arithmetic T, std::size_t Order>
  [[nodiscard]] constexpr auto xs(const std::array<std::size_t, Order>& dims, const T x) {
    auto result = core::tensor<T, Order>(dims);
    for (std::size_t idx = 0; idx < result.size(); ++idx) {
      result[idx] = static_cast<T>(x);
    }
    return result;
  }

  /**
   * Constructs a tensor of zeros from the dimensions of the provided tensor.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Represents the order of the tensor.
   * @param t A tensor to match the dimensions against.
   * @return A tensor of zeros with the dimensions of the provided tensor.
   */
  template <Arithmetic T, std::size_t Order>
  [[nodiscard]] constexpr auto zeros_like(const core::tensor<T, Order>& t) {
    auto result = core::tensor<T, Order>(t.dims());
    for (std::size_t idx = 0; idx < t.size(); ++idx) {
      result[idx] = static_cast<T>(0);
    }
    return result;
  }

  /**
   * Constructs a tensor of ones from the dimensions of the provided tensor.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Represents the order of the tensor.
   * @param t A tensor to match the dimensions against.
   * @return A tensor of ones with the dimensions of the provided tensor.
   */
  template <Arithmetic T, std::size_t Order>
  [[nodiscard]] constexpr auto ones_like(const core::tensor<T, Order>& t) {
    auto result = core::tensor<T, Order>(t.dims());
    for (std::size_t idx = 0; idx < t.size(); ++idx) {
      result[idx] = static_cast<T>(1);
    }
    return result;
  }

  /**
   * Constructs a tensor of specified values from the dimensions of the provided tensor.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @tparam Order Represents the order of the tensor.
   * @param t A tensor to match the dimensions against.
   * @return A tensor of specified values with the dimensions of the provided tensor.
   */
  template <Arithmetic T, std::size_t Order>
  [[nodiscard]] constexpr auto xs_like(const core::tensor<T, Order>& t, const T x) {
    auto result = core::tensor(t.dims());
    for (std::size_t idx = 0; idx < t.size(); ++idx) {
      result[idx] = static_cast<T>(x);
    }
    return result;
  }

  /**
   * Constructs a one-dimensional tensor including values from begin to end with the given stride.
   *
   * @tparam T An arithmetic type representing the type of every element in the returned tensor.
   * @param begin The start of the range.
   * @param end The end of the range.
   * @param stride A stride for the range.
   * @return A one-dimensional tensor of values specified by the provided range and stride.
   */
  template <Arithmetic T>
  [[nodiscard]] constexpr auto range1d(const T begin, const T end, const T stride) {
    auto dims = std::array<std::size_t, 1>{static_cast<std::size_t>((end - begin) / stride)};
    auto result = core::tensor<T, 1>(dims);
    std::size_t idx = 0;
    for (T val = begin; val < end; val += stride) {
      result[idx++] = val;
    }
    return result;
  }

}  // namespace builder

#endif  // BUILDER_HPP
