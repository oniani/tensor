#define CATCH_CONFIG_MAIN

#include "../include/tensor.hpp"
#include "../third_party/catch.hpp"

using namespace core;

TEST_CASE("Tensor 1D - Core utilities", "[1D][arithmetic][data][dims][size][get]") {
  const tensor<float, 1> t1 = {0, 1, 2, 3, 4};
  const tensor<float, 1> t2 = {5, 6, 7, 8, 9};

  for (std::size_t idx = 0; idx < t1.size(); ++idx) {
    REQUIRE(t1.data()[idx] == t1[idx]);
  }
  for (std::size_t idx = 0; idx < t2.size(); ++idx) {
    REQUIRE(t2.data()[idx] == t2[idx]);
  }

  REQUIRE(t1.dims() == std::array<std::size_t, 1>{5});
  REQUIRE(t2.dims() == std::array<std::size_t, 1>{5});

  REQUIRE(t1.size() == 5);
  REQUIRE(t2.size() == 5);

  REQUIRE(t1.get<1>({0}) == 0);
  REQUIRE(t1.get<1>({1}) == 1);
  REQUIRE(t1.get<1>({2}) == 2);
  REQUIRE(t1.get<1>({3}) == 3);
  REQUIRE(t1.get<1>({4}) == 4);

  REQUIRE(t2.get<1>({0}) == 5);
  REQUIRE(t2.get<1>({1}) == 6);
  REQUIRE(t2.get<1>({2}) == 7);
  REQUIRE(t2.get<1>({3}) == 8);
  REQUIRE(t2.get<1>({4}) == 9);
}

TEST_CASE("Tensor 1D - Basic arithmetic operators", "[1D][add][sub][mul][div]") {
  tensor<float, 1> t1 = {0, 1, 2, 3, 4};
  tensor<float, 1> t2 = {5, 6, 7, 8, 9};

  const auto t3 = t1 + t2;
  REQUIRE(t3.get<1>({0}) == 5);
  REQUIRE(t3.get<1>({1}) == 7);
  REQUIRE(t3.get<1>({2}) == 9);
  REQUIRE(t3.get<1>({3}) == 11);
  REQUIRE(t3.get<1>({4}) == 13);

  const auto t4 = t1 - t2;
  REQUIRE(t4.get<1>({0}) == -5);
  REQUIRE(t4.get<1>({1}) == -5);
  REQUIRE(t4.get<1>({2}) == -5);
  REQUIRE(t4.get<1>({3}) == -5);
  REQUIRE(t4.get<1>({4}) == -5);

  const auto t5 = t1 * t2;
  REQUIRE(t5.get<1>({0}) == 0);
  REQUIRE(t5.get<1>({1}) == 6);
  REQUIRE(t5.get<1>({2}) == 14);
  REQUIRE(t5.get<1>({3}) == 24);
  REQUIRE(t5.get<1>({4}) == 36);

  const auto t6 = t1 / t2;
  REQUIRE(std::round(t6.get<1>({0}) * 10e5) / 10e5 == std::round(0.0 / 5 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<1>({1}) * 10e5) / 10e5 == std::round(1.0 / 6 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<1>({2}) * 10e5) / 10e5 == std::round(2.0 / 7 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<1>({3}) * 10e5) / 10e5 == std::round(3.0 / 8 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<1>({4}) * 10e5) / 10e5 == std::round(4.0 / 9 * 10e5) / 10e5);
}

TEST_CASE("Tensor 1D - Basic arithmetic broadcasting", "[1D][add][sub][mul][div]") {
  const tensor<float, 1> t1 = {0, 1, 2, 3, 4};
  const tensor<float, 1> t2 = {5, 6, 7, 8, 9};

  REQUIRE(t1 + 0 == t1);
  REQUIRE(t1 + 1 == tensor<float, 1>{1, 2, 3, 4, 5});
  REQUIRE(t1 + 2 == tensor<float, 1>{2, 3, 4, 5, 6});
  REQUIRE(t1 + 5 == t2);

  REQUIRE(t2 - 0 == t2);
  REQUIRE(t2 - 1 == tensor<float, 1>{4, 5, 6, 7, 8});
  REQUIRE(t2 - 2 == tensor<float, 1>{3, 4, 5, 6, 7});
  REQUIRE(t2 - 5 == t1);

  REQUIRE(t1 * 0 == tensor<float, 1>{0, 0, 0, 0, 0});
  REQUIRE(t1 * 1 == t1);
  REQUIRE(t1 * 2 == tensor<float, 1>{0, 2, 4, 6, 8});
  REQUIRE(t1 * 5 == tensor<float, 1>{0, 5, 10, 15, 20});

  REQUIRE(t2 / 1 == t2);
  REQUIRE(t2 / 2 == tensor<float, 1>{2.5, 3, 3.5, 4, 4.5});
  REQUIRE(t2 / 4 == tensor<float, 1>{1.25, 1.5, 1.75, 2, 2.25});
  REQUIRE(t2 / 8 == tensor<float, 1>{0.625, 0.75, 0.875, 1, 1.125});
}

TEST_CASE("Tensor 1D - Comparison operators", "[1D][eq][neq][gt][geq][lt][ltq]") {
  const tensor<float, 1> t1 = {0, 1, 2, 3, 4};
  const tensor<float, 1> t2 = {5, 6, 7, 8, 9};

  REQUIRE(t1 == t1);
  REQUIRE(t2 == t2);
  REQUIRE(t1 != t2);
  REQUIRE(t2 != t1);

  REQUIRE(!(t1 > t2));
  REQUIRE(t2 > t1);
  REQUIRE(!(t1 >= t2));
  REQUIRE(t2 >= t1);

  REQUIRE(t1 < t2);
  REQUIRE(!(t2 < t1));
  REQUIRE(t1 <= t2);
  REQUIRE(!(t2 <= t1));
}

TEST_CASE("Tensor 1D - Handy broadcasting operations", "[1D][pow][square][sqrt][sin][cos]") {
  const tensor<float, 1> t1 = {0, 1, 2, 3, 4};
  const tensor<float, 1> t2 = {5, 6, 7, 8, 9};

  REQUIRE(t1.pow(1) == tensor<float, 1>{0, 1, 2, 3, 4});
  REQUIRE(t1.pow(2) == tensor<float, 1>{0, 1, 4, 9, 16});
  REQUIRE(t2.pow(1) == tensor<float, 1>{5, 6, 7, 8, 9});
  REQUIRE(t2.pow(2) == tensor<float, 1>{25, 36, 49, 64, 81});

  REQUIRE(t1.square() == tensor<float, 1>{0, 1, 4, 9, 16});
  REQUIRE(t2.square() == tensor<float, 1>{25, 36, 49, 64, 81});

  REQUIRE(t1.sqrt() == tensor<float, 1>{0, 1, 1.4142135f, 1.7320508f, 2});
  REQUIRE(t2.sqrt() == tensor<float, 1>{2.2360679f, 2.44948974f, 2.6457513f, 2.828427f, 3});

  REQUIRE(t1.sin() == tensor<float, 1>{0, 0.84147098f, 0.9092974f, 0.141120f, -0.75680249f});
  REQUIRE(t2.sin() ==
          tensor<float, 1>{-0.95892427f, -0.27941549f, 0.656986598f, 0.989358246f, 0.412118485f});

  REQUIRE(t1.cos() ==
          tensor<float, 1>{1.0, 0.5403023058f, -0.416146836f, -0.989992496f, -0.653643620f});
  REQUIRE(t2.cos() == tensor<float, 1>{0.2836621854f, 0.9601702866f, 0.7539022543f, -0.145500033f,
                                       -0.911130261f});
}
