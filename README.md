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
$ cmake -DCOMPILER=clang -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=VCPKG_TOOLCHAIN_FILE ..
$ cmake --build .
```

## References

- [Tensor][tensor]

## License

[MIT License][license]

[tensor]: https://en.wikipedia.org/wiki/Tensor
[license]: LICENSE
