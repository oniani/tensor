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

#ifndef TYPE_HPP
#define TYPE_HPP

#include "core.hpp"

namespace type {

  /**
   * @brief Constructs a type representing an order one tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor1 = core::tensor<T, 1>;

  /**
   * @brief Constructs a type representing an order two tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor2 = core::tensor<T, 2>;

  /**
   * @brief Constructs a type representing an order three tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor3 = core::tensor<T, 3>;

  /**
   * @brief Constructs a type representing an order four tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor4 = core::tensor<T, 4>;

  /**
   * @brief Constructs a type representing an order five tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor5 = core::tensor<T, 5>;

  /**
   * @brief Constructs a type representing an order six tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor6 = core::tensor<T, 6>;

  /**
   * @brief Constructs a type representing an order seven tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor7 = core::tensor<T, 7>;

  /**
   * @brief Constructs a type representing an order eight tensor.
   * @tparam T Arithmetic type representing the type of every element in the returned tensor.
   */
  template <Arithmetic T>
  using tensor8 = core::tensor<T, 8>;

}  // namespace type

#endif  // TYPE_HPP
