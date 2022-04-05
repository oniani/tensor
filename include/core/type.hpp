#include "core.hpp"

namespace type {

/**
 * Constructs a type representing a one-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor1d = core::tensor<1, T>;

/**
 * Constructs a type representing a two-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor2d = core::tensor<2, T>;

/**
 * Constructs a type representing a three-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor3d = core::tensor<3, T>;

/**
 * Constructs a type representing a four-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor4d = core::tensor<4, T>;

/**
 * Constructs a type representing a five-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor5d = core::tensor<5, T>;

/**
 * Constructs a type representing a six-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor6d = core::tensor<6, T>;

/**
 * Constructs a type representing a seven-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor7d = core::tensor<7, T>;

/**
 * Constructs a type representing an eight-dimensional tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor8d = core::tensor<8, T>;

}  // namespace type
