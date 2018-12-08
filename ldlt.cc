#include <iostream>
#include "src/ldlt.h"

int main()
{
    // 6 by 6 sparse symmetric pos def matrix
    ldlt::SparseMatrixCSC<double,uint64_t> A{
        6, 6,
        {0, 4, 8, 11, 15, 19, 24},
        {0, 1, 4, 5, 0, 1, 2, 3, 1, 2, 5, 1, 3, 4, 5, 0, 3, 4, 5, 0, 2, 3, 4, 5},
        {20.0, 0.56, 0.26, 0.71, 0.56, 20.0, 0.28, 0.26, 0.28, 20.0, 0.69, 0.26, 20.0, 0.61, 0.59, 0.26, 0.61, 20.0, 0.22, 0.71, 0.69, 0.59, 0.22, 21.17}
    };

    // Allocate the etree, visited and count vecs
    auto const factorization = ([&](){
        auto symbolic = ldlt::SymbolicLDLT<double,uint64_t>::fromSparseMatrixCSC(A);
        return symbolic.numeric().apply(A).extract();
    })();

    std::cout << factorization.L << '\n';
}