#ifndef SPARSE_MATRIX_CSC_
#define SPARSE_MATRIX_CSC_

#include <iostream>
#include <vector>

namespace ldlt {

template <class Tv, class Ti>
struct SparseMatrixCSC
{
    Ti n;
    Ti m;
    std::vector<Ti> colptr;
    std::vector<Ti> rowval;
    std::vector<Tv> nzval;

    SparseMatrixCSC(Ti n, Ti m, std::vector<Ti> && colptr, std::vector<Ti> && rowval, std::vector<Tv> && nzval)
    : n(n), m(m), colptr(std::forward<std::vector<Ti>>(colptr)), rowval(std::forward<std::vector<Ti>>(rowval)), nzval(std::forward<std::vector<Tv>>(nzval))
    {}
};

template<class Tv, class Ti>
::std::ostream &operator<<(::std::ostream &os, SparseMatrixCSC<Tv,Ti> const &A)
{
    for (size_t j = 0; j < A.m; ++j)
        for (size_t k = A.colptr[j]; k != A.colptr[j + 1]; ++k)
            os << '[' << A.rowval[k] << ',' << j << "] = " << A.nzval[k] << '\n';
    
    return os;
}

};

#endif