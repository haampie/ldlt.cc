#ifndef LDLT_
#define LDLT_

#include <vector>
#include <numeric>
#include <iterator>
#include <algorithm>

#include "sparse_matrix_csc.h"

namespace ldlt {

/**********************
 * [A11 A12]   [L11  ]   [D1   ]   [L11' L21']
 * [A21 A22] = [L21 I] * [   D2] * [     I   ]
 * 
 * Assuming we already have factorized A11 = L11 * D1 * L11', we only have to
 * find L21 and D2. Equating gives: A12 = L11 * D1 * L21', so if we define
 * y := L11 \ A12, then L21 = (D1 \ y)'. Finally A22 = L21 * D1 * L21' + D2,
 * so D2 = A22 - L21 * y.
 * The algorithm then builds the full L and D step by step with A12 and L21 
 * vectors and A22 and D2 scalar. Computing y requires a sparse-sparse solve.
 * 
 * - The symbolic factorization will determine the sparsity pattern of L
 * - The numeric factorization will do the flops to compute L and D.
 * - The symbolic factorization can be reused for multiple numeric factorizations
 *   of different matrices with identical sparsity pattern
 * - The effective factors can be extracted out of the numeric factorization
 *   without copy.
 */

template <class Tv,class Ti>
struct FactorLDLT {
    SparseMatrixCSC<Tv,Ti> L;
    std::vector<Tv> D;
    Ti rank;

    FactorLDLT(SparseMatrixCSC<Tv,Ti> && L, std::vector<Tv> && D, Ti rank)
        : L(std::forward<SparseMatrixCSC<Tv,Ti>>(L)), D(std::forward<std::vector<Tv>>(D)), rank(rank)
    {}
};

template<class Tv,class Ti>
class SymbolicLDLT;

template <class Tv,class Ti>
class NumericLDLT
{
private:
    SymbolicLDLT<Tv,Ti> &s;
    SparseMatrixCSC<Tv,Ti> L;
    std::vector<Tv> D;
    std::vector<Tv> y;
    std::vector<Ti> pattern;
    Ti rank;
    NumericLDLT(SymbolicLDLT<Tv,Ti> &s, SparseMatrixCSC<Tv,Ti> && L)
        : s(s), L(std::forward<SparseMatrixCSC<Tv,Ti>>(L)), D(s.n), y(s.n), pattern(s.n), rank(0)
    {}

public:
    static NumericLDLT<Tv,Ti> fromSymbolic(SymbolicLDLT<Tv,Ti> &s)
    {
        // Buliding the colptr should probably be done once in the symbolic factorization
        std::vector<Ti> colptr(s.n + 1); colptr[0] = 0;
        for (size_t i = 0; i < s.n; colptr[i + 1] = colptr[i] + s.nzcount[i], ++i);
        auto nnz = colptr.back();
        return {s, {s.n, s.n, std::move(colptr), std::vector<Ti>(nnz), std::vector<Tv>(nnz)}};
    }

    NumericLDLT &apply(SparseMatrixCSC<Tv,Ti> const &A) {
        rank = A.n;
        std::fill(s.nzcount.begin(), s.nzcount.end(), Ti{0});
        std::fill(y.begin(), y.end(), Tv{0.0});
        std::iota(s.visited.begin(), s.visited.end(), Ti{0});

        for (size_t k = 0; k < A.m; ++k)
        {
            auto top = A.m;

            // Loop over the nonzeros in A12 (below the diagonal)
            for (auto j = A.colptr[k]; j != A.colptr[k + 1] && A.rowval[j] <= k; ++j)
            {
                auto i = A.rowval[j];
                y[i] = A.nzval[j];
                size_t length = 0;

                // Follow the etree to the root
                while (s.visited[i] != k)
                {
                    pattern[length++] = i;
                    s.visited[i] = k;
                    i = s.parent[i];
                }

                // Prepend this path to the rest.
                while (length > 0) pattern[--top] = pattern[--length];
            }

            D[k] = y[k];
            y[k] = 0.0;

            // Compute L21 = D1 \ y and D2 = A22 - L21 * y.
            for (size_t l = top; l < A.n; ++l)
            {
                auto i = pattern[l];
                auto yi = y[i];
                y[i] = 0.0;
                auto j = L.colptr[i];

                for (; j < L.colptr[i] + s.nzcount[i]; ++j)
                    y[L.rowval[j]] -= L.nzval[j] * yi;
                
                auto Lki = yi / D[i];
                D[k] -= Lki * yi;
                L.rowval[j] = k;
                L.nzval[j] = Lki;
                s.nzcount[i]++;
            }

            if (D[k] == 0.0)
            {
                rank = k;
                break;
            }
        }

        return *this;
    }

    FactorLDLT<Tv,Ti> extract() {
        return {std::move(L), std::move(D), rank};
    }
};

template <class Tv, class Ti>
struct SymbolicLDLT
{
    friend class NumericLDLT<Tv,Ti>;
    // We use parent[i] = -1 as a `null` value, so make sure we
    // make sure we have signed integers for that vector.
    using SignedTi = typename std::make_signed<Ti>::type;
private:
    Ti n;
    std::vector<SignedTi> parent;
    std::vector<Ti> visited;
    std::vector<Ti> nzcount;
    SymbolicLDLT(SparseMatrixCSC<Tv, Ti> const &A) : n(A.n), parent(A.n), visited(A.n), nzcount(A.n)
    {}

public:
    static SymbolicLDLT fromSparseMatrixCSC(SparseMatrixCSC<Tv,Ti> const &A)
    {
        SymbolicLDLT s{A};
        // Set counts to zero, parents to none nd visited to the idx.
        std::fill(s.nzcount.begin(), s.nzcount.end(), 0);
        std::fill(s.parent.begin(), s.parent.end(), -1);
        std::iota(s.visited.begin(), s.visited.end(), 0);

        for (size_t k = 0; k < A.m; ++k)
        {
            // Loop over the nonzeros in A12 below the diagonal
            for (size_t j = A.colptr[k]; j != A.colptr[k + 1] && A.rowval[j] < k; ++j)
            {
                auto i = A.rowval[j];

                // Follow the etree to the root
                while (s.visited[i] != k)
                {
                    // If column 'i' does not have a parent
                    // it will be the current 'fill in' in row 'k'.
                    if (s.parent[i] == -1) s.parent[i] = k;
                    ++s.nzcount[i];
                    s.visited[i] = k;
                    i = s.parent[i];
                }
            }
        }

        return std::move(s);
    }

    NumericLDLT<Tv,Ti> numeric()
    {
        return NumericLDLT<Tv,Ti>::fromSymbolic(*this);
    }
};


} // namespace chol

#endif