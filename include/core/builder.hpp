#ifndef BUILDER_HPP
#define BUILDER_HPP

#include "core.hpp"

namespace builder {

/**
 * Constructs a tensor of zeros from the provided dimensions.
 *
 * @tparam Order Represents the order of the tensor.
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 * @param dims Dimensions for constructing a tensor of zeros.
 * @return A tensor of zeros with the provided dimensions.
 */
template <std::size_t Order, Arithmetic T>
[[nodiscard]] constexpr auto zeros(std::array<std::size_t, Order> dims) {
  auto result = core::tensor<Order, T>(dims);
  for (std::size_t idx = 0; idx < result.size(); ++idx) {
    result[idx] = static_cast<T>(0);
  }
  return result;
}

/**
 * Constructs a tensor of ones from the provided dimensions.
 *
 * @tparam Order Represents the order of the tensor.
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 * @param dims Dimensions for constructing a tensor of ones.
 * @return A tensor of ones with the provided dimensions.
 */
template <std::size_t Order, Arithmetic T>
[[nodiscard]] constexpr auto ones(std::array<std::size_t, Order> dims) {
  auto result = core::tensor<Order, T>(dims);
  for (std::size_t idx = 0; idx < result.size(); ++idx) {
    result[idx] = static_cast<T>(1);
  }
  return result;
}

/**
 * Constructs a tensor of specified values from the provided dimensions.
 *
 * @tparam Order Represents the order of the tensor.
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 * @param dims Dimensions for constructing a tensor of specified values.
 * @return A tensor of specified values with the provided dimensions.
 */
template <std::size_t Order, Arithmetic T>
[[nodiscard]] constexpr auto xs(std::array<std::size_t, Order> dims, const T x) {
  auto result = core::tensor<Order, T>(dims);
  for (std::size_t idx = 0; idx < result.size(); ++idx) {
    result[idx] = static_cast<T>(x);
  }
  return result;
}

/**
 * Constructs a tensor of zeros from the dimensions of the provided tensor.
 *
 * @tparam Order Represents the order of the tensor.
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 * @param t A tensor to match the dimensions against.
 * @return A tensor of zeros with the dimensions of the provided tensor.
 */
template <std::size_t Order, Arithmetic T>
[[nodiscard]] constexpr auto zeros_like(const core::tensor<Order, T> &t) {
  auto result = core::tensor<Order, T>(t.dims());
  for (std::size_t idx = 0; idx < t.size(); ++idx) {
    result[idx] = static_cast<T>(0);
  }
  return result;
}

/**
 * Constructs a tensor of ones from the dimensions of the provided tensor.
 *
 * @tparam Order Represents the order of the tensor.
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 * @param t A tensor to match the dimensions against.
 * @return A tensor of ones with the dimensions of the provided tensor.
 */
template <std::size_t Order, Arithmetic T>
[[nodiscard]] constexpr auto ones_like(const core::tensor<Order, T> &t) {
  auto result = core::tensor<Order, T>(t.dims());
  for (std::size_t idx = 0; idx < t.size(); ++idx) {
    result[idx] = static_cast<T>(1);
  }
  return result;
}

/**
 * Constructs a tensor of specified values from the dimensions of the provided tensor.
 *
 * @tparam Order Represents the order of the tensor.
 * @tparam T An arithmetic type representing the type of every element in the returned tensor.
 * @param t A tensor to match the dimensions against.
 * @return A tensor of specified values with the dimensions of the provided tensor.
 */
template <std::size_t Order, Arithmetic T>
[[nodiscard]] constexpr auto xs_like(const core::tensor<Order, T> &t, const T x) {
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
 * @param begin A starting number.
 * @param end An end number.
 * @param stride A stride for generating the one-dimensional tensor.
 * @return A one-dimensional tensor of values specified by the provided range and stride.
 */
template <Arithmetic T>
[[nodiscard]] constexpr auto range1d(const T begin, const T end, const T stride) {
  auto dims = std::array<std::size_t, 1>{static_cast<std::size_t>((end - begin) / stride)};
  auto result = core::tensor<1, T>(dims);
  std::size_t idx = 0;
  for (T val = begin; val < end; val += stride) {
    result[idx++] = val;
  }
  return result;
}

}  // namespace builder

#endif  // BUILDER_HPP
