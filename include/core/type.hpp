#include "core.hpp"

namespace type {

/**
 * Constructs a type representing an order one tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor1 = core::tensor<T, 1>;

/**
 * Constructs a type representing an order two tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor2 = core::tensor<T, 2>;

/**
 * Constructs a type representing an order three tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor3 = core::tensor<T, 3>;

/**
 * Constructs a type representing an order four tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor4 = core::tensor<T, 4>;

/**
 * Constructs a type representing an order five tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor5 = core::tensor<T, 5>;

/**
 * Constructs a type representing an order six tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor6 = core::tensor<T, 6>;

/**
 * Constructs a type representing an order seven tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor7 = core::tensor<T, 7>;

/**
 * Constructs a type representing an order eight tensor.
 *
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 */
template <Arithmetic T>
using tensor8 = core::tensor<T, 8>;

}  // namespace type
