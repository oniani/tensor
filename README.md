# tensor

A tensor library.

## API

```cpp
#include "tensor.hpp"

using namespace core;

int main() {
    tensor<2, float> t = {{0, 1}, {2, 3}, {4, 5}};
    t.square().print();
}
```

## Testing

```console
$ mkdir build
$ cd build
$ cmake -DCMAKE_CXX_COMPILER=clang++ .. && cmake --build . --config Release -- -j 4 && ./test/tests
```

## References

- [Tensor](https://en.wikipedia.org/wiki/Tensor)

## License

[MIT License](LICENSE)
