#define CATCH_CONFIG_MAIN

#include "../include/tensor.hpp"
#include "../third_party/catch.hpp"

TEST_CASE("Tensor 1D - basic arithmetic", "[1D][arithmetic][get]") {
    auto t1 = tensor::Tensor<1, float>{0, 1, 2, 3, 4};
    auto t2 = tensor::Tensor<1, float>{5, 6, 7, 8, 9};

    REQUIRE(t1.get({0}) == 0);
    REQUIRE(t1.get({1}) == 1);
    REQUIRE(t1.get({2}) == 2);
    REQUIRE(t1.get({3}) == 3);
    REQUIRE(t1.get({4}) == 4);

    REQUIRE(t2.get({0}) == 5);
    REQUIRE(t2.get({1}) == 6);
    REQUIRE(t2.get({2}) == 7);
    REQUIRE(t2.get({3}) == 8);
    REQUIRE(t2.get({4}) == 9);

    auto t3 = t1 + t2;
    REQUIRE(t3.get({0}) == 5);
    REQUIRE(t3.get({1}) == 7);
    REQUIRE(t3.get({2}) == 9);
    REQUIRE(t3.get({3}) == 11);
    REQUIRE(t3.get({4}) == 13);

    auto t4 = t1 - t2;
    REQUIRE(t4.get({0}) == -5);
    REQUIRE(t4.get({1}) == -5);
    REQUIRE(t4.get({2}) == -5);
    REQUIRE(t4.get({3}) == -5);
    REQUIRE(t4.get({4}) == -5);

    auto t5 = t1 * t2;
    REQUIRE(t5.get({0}) == 0);
    REQUIRE(t5.get({1}) == 6);
    REQUIRE(t5.get({2}) == 14);
    REQUIRE(t5.get({3}) == 24);
    REQUIRE(t5.get({4}) == 36);

    auto t6 = t1 / t2;
    REQUIRE(t6.get({0}) == 0.0);
    REQUIRE(round(t6.get({1}) * 10e5) / 10e5 == round(1.0 / 6 * 10e5) / 10e5);
    REQUIRE(round(t6.get({2}) * 10e5) / 10e5 == round(2.0 / 7 * 10e5) / 10e5);
    REQUIRE(round(t6.get({3}) * 10e5) / 10e5 == round(3.0 / 8 * 10e5) / 10e5);
    REQUIRE(round(t6.get({4}) * 10e5) / 10e5 == round(4.0 / 9 * 10e5) / 10e5);
}

TEST_CASE("Tensor 1D - basic arithmetic broadcasting", "[1D][arithmetic][broadcast]") {
    auto t1 = tensor::Tensor<1, float>{0, 1, 2, 3, 4};
    auto t2 = tensor::Tensor<1, float>{5, 6, 7, 8, 9};

    REQUIRE(t1 + 0 == t1);
    REQUIRE(t1 + 1 == tensor::Tensor<1, float>{1, 2, 3, 4, 5});
    REQUIRE(t1 + 2 == tensor::Tensor<1, float>{2, 3, 4, 5, 6});
    REQUIRE(t1 + 5 == t2);

    REQUIRE(t2 - 0 == t2);
    REQUIRE(t2 - 1 == tensor::Tensor<1, float>{4, 5, 6, 7, 8});
    REQUIRE(t2 - 2 == tensor::Tensor<1, float>{3, 4, 5, 6, 7});
    REQUIRE(t2 - 5 == t1);

    REQUIRE(t1 * 0 == tensor::Tensor<1, float>{0, 0, 0, 0, 0});
    REQUIRE(t1 * 1 == t1);
    REQUIRE(t1 * 2 == tensor::Tensor<1, float>{0, 2, 4, 6, 8});
    REQUIRE(t1 * 5 == tensor::Tensor<1, float>{0, 5, 10, 15, 20});

    REQUIRE(t2 / 1 == t2);
    REQUIRE(t2 / 2 == tensor::Tensor<1, float>{2.5, 3, 3.5, 4, 4.5});
    REQUIRE(t2 / 4 == tensor::Tensor<1, float>{1.25, 1.5, 1.75, 2, 2.25});
    REQUIRE(t2 / 8 == tensor::Tensor<1, float>{0.625, 0.75, 0.875, 1, 1.125});
}

TEST_CASE("Tensor 1D - comparison", "[1D][comp]") {
    auto t1 = tensor::Tensor<1, float>{0, 1, 2, 3, 4};
    auto t2 = tensor::Tensor<1, float>{5, 6, 7, 8, 9};

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
