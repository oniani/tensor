#include <cassert>

#include "include/tensor.hxx"

// TODO:
//   1. use Catch2 for testing.
//   2. ZOMBIES strategy - Zero, One, Many, Boundaries, Interface, Exceptions, and Simple Scenarios.

int main() {
    // {{ {1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
    //  {{13, 14, 15, 16}, {17, 18, 19, 20}, {21, 22, 23, 24}}}
    //
    //  NOTE: Tensor<float, 4>{{2, 3, 2}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    //  12}}; still works
    auto t1 = tensor::Tensor<3, float>{{2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};

    auto t2 = tensor::Tensor<3, float>{{2, 3, 4}, {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24}};

    auto t3 = tensor::Tensor<2, float>{{2, 3}, {1, 2, 3, 4, 5, 6}};

    // Testing
    assert(t1.flat_get({1, 1, 0}) == 17);
    assert(t1.flat_get({0, 1, 0}) == 5);

    assert(t3.flat_get({0, 1}) == 2);
    assert(t3.flat_get({1, 0}) == 4);

    // Basic tensor operations
    assert((t1 + t2).flat_get({1, 0, 0}) == 26);
    assert((t1 + t2).data()[12] == 26);

    assert((t1 - t2).flat_get({1, 0, 0}) == 0);
    assert((t1 - t2).data()[12] == 0);

    assert((t1 * t2).flat_get({1, 0, 0}) == 169);
    assert((t1 * t2).data()[12] == 169);

    assert((t1 / t2).flat_get({1, 0, 0}) == 1);
    assert((t1 / t2).data()[12] == 1);

    // Basic broadcasting by value
    assert((t1 + 10).flat_get({1, 0, 0}) == 23);
    assert((t1 + 10).data()[12] == 23);

    assert((t1 - 10).flat_get({1, 0, 0}) == 3);
    assert((t1 - 10).data()[12] == 3);

    assert((t1 * 10).flat_get({1, 0, 0}) == 130);
    assert((t1 * 10).data()[12] == 130);

    assert((t1 / 10).flat_get({1, 0, 0}) == 1.3F);
    assert((t1 / 10).data()[12] == 1.3F);

    // Flat print the contents of the data pointer
    t1.flat_print();
    t2.flat_print();
    (t1 + t2).flat_print();
    (t1 - t2).flat_print();
    (t1 * t2).flat_print();
    (t1 / t2).flat_print();
    (t1.pow(3)).flat_print();
    (t1.square()).flat_print();
    (t1.sqrt()).flat_print();
    (t1.sin()).flat_print();
    (t1.cos()).flat_print();
}
