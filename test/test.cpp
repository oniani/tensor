#define CATCH_CONFIG_MAIN
#include "../include/tensor.hpp"
#include "../third_party/catch.hpp"

using std::operator""s;

/*
TEST_CASE("Search and insert strings", "[insert][search][string]") {
    auto bf = bf::BloomFilter{3, 1e-2};

    bf.insert("hello"s);
    bf.insert("world"s);
    bf.insert(""s);

    REQUIRE(bf.search("hello"s));
    REQUIRE(bf.search("world"s));
    REQUIRE(bf.search(""s));

    bf.clear();

    REQUIRE(!bf.search("hello"s));
    REQUIRE(!bf.search("world"s));
    REQUIRE(!bf.search(""s));
}

TEST_CASE("Search and insert strings from a vector", "[insert][search][string]") {
    auto bf = bf::BloomFilter(5, 1e-3);
    auto words = std::vector<std::string>{"afopsiv"s, "coxpz"s, "pqeacxnvzm"s, "zm"s, "acxk"s};

    for (const auto &word : words) {
        bf.insert(word);
        REQUIRE(bf.search(word));
    }
}

TEST_CASE("Insert and search many", "[insert_many][search_many][int]") {
    auto bf = bf::BloomFilter(12, 1e-4);
    auto nums = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    bf.insert_many(nums);
    for (const auto &num : nums) {
        REQUIRE(bf.search(num));
    }

    auto vals = bf.search_many(nums);
    for (const auto &val : vals) {
        REQUIRE(val);
    }
}
*/
