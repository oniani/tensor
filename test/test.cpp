#define CATCH_CONFIG_MAIN

#include "../include/tensor.hpp"
#include "../third_party/catch.hpp"

using namespace type;

// tensor1 {{{

TEST_CASE("tensor1 - Core utilities", "[tensor1][arithmetic][data][dims][size][get]") {
  const tensor1<float> t1{0, 1, 2, 3, 4};
  const tensor1<float> t2{5, 6, 7, 8, 9};

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

TEST_CASE("tensor1 - Basic arithmetic operators", "[tensor1][add][sub][mul][div]") {
  const tensor1<float> t1{0, 1, 2, 3, 4};
  const tensor1<float> t2{5, 6, 7, 8, 9};

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

TEST_CASE("tensor1 - Basic arithmetic broadcasting", "[tensor1][add][sub][mul][div]") {
  const tensor1<float> t1{0, 1, 2, 3, 4};
  const tensor1<float> t2{5, 6, 7, 8, 9};

  REQUIRE(t1 + 0 == t1);
  REQUIRE(t1 + 1 == tensor1<float>{1, 2, 3, 4, 5});
  REQUIRE(t1 + 2 == tensor1<float>{2, 3, 4, 5, 6});
  REQUIRE(t1 + 5 == t2);

  REQUIRE(t2 - 0 == t2);
  REQUIRE(t2 - 1 == tensor1<float>{4, 5, 6, 7, 8});
  REQUIRE(t2 - 2 == tensor1<float>{3, 4, 5, 6, 7});
  REQUIRE(t2 - 5 == t1);

  REQUIRE(t1 * 0 == tensor1<float>{0, 0, 0, 0, 0});
  REQUIRE(t1 * 1 == t1);
  REQUIRE(t1 * 2 == tensor1<float>{0, 2, 4, 6, 8});
  REQUIRE(t1 * 5 == tensor1<float>{0, 5, 10, 15, 20});

  REQUIRE(t2 / 1 == t2);
  REQUIRE(t2 / 2 == tensor1<float>{2.5, 3, 3.5, 4, 4.5});
  REQUIRE(t2 / 4 == tensor1<float>{1.25, 1.5, 1.75, 2, 2.25});
  REQUIRE(t2 / 8 == tensor1<float>{0.625, 0.75, 0.875, 1, 1.125});
}

TEST_CASE("tensor1 - Comparison operators", "[tensor1][eq][neq][gt][geq][lt][ltq]") {
  const tensor1<float> t1{0, 1, 2, 3, 4};
  const tensor1<float> t2{5, 6, 7, 8, 9};

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

TEST_CASE("tensor1 - Handy broadcasting operations", "[tensor1][pow][square][sqrt][sin][cos]") {
  const tensor1<float> t1{0, 1, 2, 3, 4};
  const tensor1<float> t2{5, 6, 7, 8, 9};

  REQUIRE(t1.pow(1) == tensor1<float>{0, 1, 2, 3, 4});
  REQUIRE(t1.pow(2) == tensor1<float>{0, 1, 4, 9, 16});
  REQUIRE(t2.pow(1) == tensor1<float>{5, 6, 7, 8, 9});
  REQUIRE(t2.pow(2) == tensor1<float>{25, 36, 49, 64, 81});

  REQUIRE(t1.square() == tensor1<float>{0, 1, 4, 9, 16});
  REQUIRE(t2.square() == tensor1<float>{25, 36, 49, 64, 81});

  REQUIRE(t1.sqrt() == tensor1<float>{0, 1, 1.4142135f, 1.7320508f, 2});
  REQUIRE(t2.sqrt() == tensor1<float>{2.2360679f, 2.44948974f, 2.6457513f, 2.828427f, 3});

  REQUIRE(t1.sin() == tensor1<float>{0, 0.84147098f, 0.9092974f, 0.141120f, -0.75680249f});
  REQUIRE(t2.sin() ==
          tensor1<float>{-0.95892427f, -0.27941549f, 0.656986598f, 0.989358246f, 0.412118485f});

  REQUIRE(t1.cos() ==
          tensor1<float>{1.0, 0.5403023058f, -0.416146836f, -0.989992496f, -0.653643620f});
  REQUIRE(t2.cos() == tensor1<float>{0.2836621854f, 0.9601702866f, 0.7539022543f, -0.145500033f,
                                     -0.911130261f});
}

// }}}

// tensor2 {{{

TEST_CASE("tensor2 - Core utilities", "[tensor2][arithmetic][data][dims][size][get]") {
  const tensor2<float> t1{{0, 1}, {2, 3}, {4, 4}};
  const tensor2<float> t2{{5, 6}, {7, 8}, {9, 9}};

  for (std::size_t idx = 0; idx < t1.size(); ++idx) {
    REQUIRE(t1.data()[idx] == t1[idx]);
  }
  for (std::size_t idx = 0; idx < t2.size(); ++idx) {
    REQUIRE(t2.data()[idx] == t2[idx]);
  }

  REQUIRE(t1.dims() == std::array<std::size_t, 2>{3, 2});
  REQUIRE(t2.dims() == std::array<std::size_t, 2>{3, 2});

  REQUIRE(t1.size() == 6);
  REQUIRE(t2.size() == 6);

  REQUIRE(t1.get<1>({0}) == tensor1<float>{0, 1});
  REQUIRE(t1.get<1>({1}) == tensor1<float>{2, 3});
  REQUIRE(t1.get<1>({2}) == tensor1<float>{4, 4});

  REQUIRE(t2.get<1>({0}) == tensor1<float>{5, 6});
  REQUIRE(t2.get<1>({1}) == tensor1<float>{7, 8});
  REQUIRE(t2.get<1>({2}) == tensor1<float>{9, 9});

  REQUIRE(t1.get<2>({0, 0}) == 0);
  REQUIRE(t1.get<2>({0, 1}) == 1);
  REQUIRE(t1.get<2>({1, 0}) == 2);
  REQUIRE(t1.get<2>({1, 1}) == 3);
  REQUIRE(t1.get<2>({2, 0}) == 4);
  REQUIRE(t1.get<2>({2, 1}) == 4);

  REQUIRE(t2.get<2>({0, 0}) == 5);
  REQUIRE(t2.get<2>({0, 1}) == 6);
  REQUIRE(t2.get<2>({1, 0}) == 7);
  REQUIRE(t2.get<2>({1, 1}) == 8);
  REQUIRE(t2.get<2>({2, 0}) == 9);
  REQUIRE(t2.get<2>({2, 1}) == 9);
}

TEST_CASE("tensor2 - Basic arithmetic operators", "[tensor][add][sub][mul][div]") {
  const tensor2<float> t1{{0, 1}, {2, 3}, {4, 4}};
  const tensor2<float> t2{{5, 6}, {7, 8}, {9, 9}};

  const auto t3 = t1 + t2;
  REQUIRE(t3.get<2>({0, 0}) == 5);
  REQUIRE(t3.get<2>({0, 1}) == 7);
  REQUIRE(t3.get<2>({1, 0}) == 9);
  REQUIRE(t3.get<2>({1, 1}) == 11);
  REQUIRE(t3.get<2>({2, 0}) == 13);
  REQUIRE(t3.get<2>({2, 1}) == 13);

  const auto t4 = t1 - t2;
  REQUIRE(t4.get<2>({0, 0}) == -5);
  REQUIRE(t4.get<2>({0, 1}) == -5);
  REQUIRE(t4.get<2>({1, 0}) == -5);
  REQUIRE(t4.get<2>({1, 1}) == -5);
  REQUIRE(t4.get<2>({2, 0}) == -5);
  REQUIRE(t4.get<2>({2, 1}) == -5);

  const auto t5 = t1 * t2;
  REQUIRE(t5.get<2>({0, 0}) == 0);
  REQUIRE(t5.get<2>({0, 1}) == 6);
  REQUIRE(t5.get<2>({1, 0}) == 14);
  REQUIRE(t5.get<2>({1, 1}) == 24);
  REQUIRE(t5.get<2>({2, 0}) == 36);
  REQUIRE(t5.get<2>({2, 1}) == 36);

  const auto t6 = t1 / t2;
  REQUIRE(std::round(t6.get<2>({0, 0}) * 10e5) / 10e5 == std::round(0.0 / 5 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<2>({0, 1}) * 10e5) / 10e5 == std::round(1.0 / 6 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<2>({1, 0}) * 10e5) / 10e5 == std::round(2.0 / 7 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<2>({1, 1}) * 10e5) / 10e5 == std::round(3.0 / 8 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<2>({2, 0}) * 10e5) / 10e5 == std::round(4.0 / 9 * 10e5) / 10e5);
  REQUIRE(std::round(t6.get<2>({2, 1}) * 10e5) / 10e5 == std::round(4.0 / 9 * 10e5) / 10e5);
}

TEST_CASE("tensor2 - Basic arithmetic broadcasting", "[1D][add][sub][mul][div]") {
  const tensor2<float> t1{{0, 1}, {2, 3}, {4, 4}};
  const tensor2<float> t2{{5, 6}, {7, 8}, {9, 9}};

  REQUIRE(t1 + 0 == t1);
  REQUIRE(t1 + 1 == tensor2<float>{{1, 2}, {3, 4}, {5, 5}});
  REQUIRE(t1 + 2 == tensor2<float>{{2, 3}, {4, 5}, {6, 6}});
  REQUIRE(t1 + 5 == t2);

  REQUIRE(t2 - 0 == t2);
  REQUIRE(t2 - 1 == tensor2<float>{{4, 5}, {6, 7}, {8, 8}});
  REQUIRE(t2 - 2 == tensor2<float>{{3, 4}, {5, 6}, {7, 7}});
  REQUIRE(t2 - 5 == t1);

  REQUIRE(t1 * 0 == tensor2<float>{{0, 0}, {0, 0}, {0, 0}});
  REQUIRE(t1 * 1 == t1);
  REQUIRE(t1 * 2 == tensor2<float>{{0, 2}, {4, 6}, {8, 8}});
  REQUIRE(t1 * 5 == tensor2<float>{{0, 5}, {10, 15}, {20, 20}});

  REQUIRE(t2 / 1 == t2);
  REQUIRE(t2 / 2 == tensor2<float>{{2.5, 3}, {3.5, 4}, {4.5, 4.5}});
  REQUIRE(t2 / 4 == tensor2<float>{{1.25, 1.5}, {1.75, 2}, {2.25, 2.25}});
  REQUIRE(t2 / 8 == tensor2<float>{{0.625, 0.75}, {0.875, 1}, {1.125, 1.125}});
}

TEST_CASE("tensor2 - Comparison operators", "[1D][eq][neq][gt][geq][lt][ltq]") {
  const tensor2<float> t1{{0, 1}, {2, 3}, {4, 4}};
  const tensor2<float> t2{{5, 6}, {7, 8}, {9, 9}};

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

TEST_CASE("tensor2 - Handy broadcasting operations", "[1D][pow][square][sqrt][sin][cos]") {
  const tensor2<float> t1{{0, 1}, {2, 3}, {4, 4}};
  const tensor2<float> t2{{5, 6}, {7, 8}, {9, 9}};

  REQUIRE(t1.pow(1) == tensor2<float>{{0, 1}, {2, 3}, {4, 4}});
  REQUIRE(t1.pow(2) == tensor2<float>{{0, 1}, {4, 9}, {16, 16}});
  REQUIRE(t2.pow(1) == tensor2<float>{{5, 6}, {7, 8}, {9, 9}});
  REQUIRE(t2.pow(2) == tensor2<float>{{25, 36}, {49, 64}, {81, 81}});

  REQUIRE(t1.square() == tensor2<float>{{0, 1}, {4, 9}, {16, 16}});
  REQUIRE(t2.square() == tensor2<float>{{25, 36}, {49, 64}, {81, 81}});

  REQUIRE(t1.sqrt() == tensor2<float>{{0, 1}, {1.4142135f, 1.7320508f}, {2, 2}});
  REQUIRE(t2.sqrt() == tensor2<float>{{2.2360679f, 2.44948974f}, {2.6457513f, 2.828427f}, {3, 3}});

  REQUIRE(t1.sin() ==
          tensor2<float>{{0, 0.84147098f}, {0.9092974f, 0.141120f}, {-0.75680249f, -0.75680249f}});
  REQUIRE(t2.sin() == tensor2<float>{{-0.95892427f, -0.27941549f},
                                     {0.656986598f, 0.989358246f},
                                     {0.412118485f, 0.412118485f}});

  REQUIRE(t1.cos() == tensor2<float>{{1.0, 0.5403023058f},
                                     {-0.416146836f, -0.989992496f},
                                     {-0.653643620f, -0.653643620f}});
  REQUIRE(t2.cos() == tensor2<float>{{0.2836621854f, 0.9601702866f},
                                     {0.7539022543f, -0.145500033f},
                                     {-0.911130261f, -0.911130261f}});
}

// }}}
