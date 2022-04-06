# tensor

A fast, zero-dependency tensor library in C++.

## API

```cpp
#include "tensor.hpp"

using namespace type;

int main() {
  tensor2<float> t{{0, 1}, {2, 3}, {4, 5}};
  t.square().print();
}
```

## Testing

```console
$ mkdir build
$ cd build
$ cmake -DCMAKE_CXX_COMPILER=clang++ .. && cmake --build . && ./test/tests
```

## References

- [Tensor](https://en.wikipedia.org/wiki/Tensor)

## License

[MIT License](LICENSE)
