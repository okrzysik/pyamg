// DO NOT EDIT: this file is generated

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>

#include "adaptive_relaxation.h"

namespace py = pybind11;

template<class I, class T>
void _optimal_smoother(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
py::array_t<I> & smoother_ID,
py::array_t<T> & smoothing_factor,
   py::array_t<T> & modes,
            I bndry_strat
                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_smoother_ID = smoother_ID.mutable_unchecked();
    auto py_smoothing_factor = smoothing_factor.mutable_unchecked();
    auto py_modes = modes.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    I *_smoother_ID = py_smoother_ID.mutable_data();
    T *_smoothing_factor = py_smoothing_factor.mutable_data();
    const T *_modes = py_modes.data();

    return optimal_smoother<I, T>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
             _smoother_ID, smoother_ID.shape(0),
        _smoothing_factor, smoothing_factor.shape(0),
                   _modes, modes.shape(0),
              bndry_strat
                                  );
}

template<class I, class T>
void _optimal_smoother_adjacency(
py::array_t<I> & smoother_ID,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj
                                 )
{
    auto py_smoother_ID = smoother_ID.unchecked();
    auto py_Sp = Sp.mutable_unchecked();
    auto py_Sj = Sj.mutable_unchecked();
    const I *_smoother_ID = py_smoother_ID.data();
    I *_Sp = py_Sp.mutable_data();
    I *_Sj = py_Sj.mutable_data();

    return optimal_smoother_adjacency<I, T>(
             _smoother_ID, smoother_ID.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0)
                                            );
}

template<class I, class T>
void _operator_dependent_interpolation(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
      py::array_t<I> & Pp,
      py::array_t<I> & Pj,
      py::array_t<T> & Px,
              const int n
                                       )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.mutable_unchecked();
    auto py_Pp = Pp.unchecked();
    auto py_Pj = Pj.mutable_unchecked();
    auto py_Px = Px.mutable_unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    T *_Ax = py_Ax.mutable_data();
    const I *_Pp = py_Pp.data();
    I *_Pj = py_Pj.mutable_data();
    T *_Px = py_Px.mutable_data();

    return operator_dependent_interpolation<I, T>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                      _Pp, Pp.shape(0),
                      _Pj, Pj.shape(0),
                      _Px, Px.shape(0),
                        n
                                                  );
}

template<class I, class T>
void _lsa_splitting(
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
     const T alpha_thresh,
py::array_t<I> & splitting,
py::array_t<I> & num_blocks
                    )
{
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    auto py_splitting = splitting.mutable_unchecked();
    auto py_num_blocks = num_blocks.mutable_unchecked();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();
    I *_splitting = py_splitting.mutable_data();
    I *_num_blocks = py_num_blocks.mutable_data();

    return lsa_splitting<I, T>(
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
             alpha_thresh,
               _splitting, splitting.shape(0),
              _num_blocks, num_blocks.shape(0)
                               );
}

template<class I, class T>
void _lsa_strength(
   py::array_t<I> & Ahatp,
   py::array_t<T> & Ahatx,
      py::array_t<I> & Sp,
      py::array_t<I> & Sj,
      py::array_t<T> & Sx,
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax
                   )
{
    auto py_Ahatp = Ahatp.unchecked();
    auto py_Ahatx = Ahatx.mutable_unchecked();
    auto py_Sp = Sp.unchecked();
    auto py_Sj = Sj.unchecked();
    auto py_Sx = Sx.mutable_unchecked();
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    const I *_Ahatp = py_Ahatp.data();
    T *_Ahatx = py_Ahatx.mutable_data();
    const I *_Sp = py_Sp.data();
    const I *_Sj = py_Sj.data();
    T *_Sx = py_Sx.mutable_data();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();

    return lsa_strength<I, T>(
                   _Ahatp, Ahatp.shape(0),
                   _Ahatx, Ahatx.shape(0),
                      _Sp, Sp.shape(0),
                      _Sj, Sj.shape(0),
                      _Sx, Sx.shape(0),
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0)
                              );
}

template<class I, class T, class F>
void _block_gauss_seidel_gen_inv(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<I> & Bp,
      py::array_t<I> & Mp,
      py::array_t<T> & Mx,
  const I block_row_start,
   const I block_row_stop,
   const I block_row_step
                                 )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Bp = Bp.unchecked();
    auto py_Mp = Mp.unchecked();
    auto py_Mx = Mx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const I *_Bp = py_Bp.data();
    const I *_Mp = py_Mp.data();
    const T *_Mx = py_Mx.data();

    return block_gauss_seidel_gen_inv<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Bp, Bp.shape(0),
                      _Mp, Mp.shape(0),
                      _Mx, Mx.shape(0),
          block_row_start,
           block_row_stop,
           block_row_step
                                               );
}

template<class I, class T, class F>
void _block_gauss_seidel_gen_splu(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<I> & Bp,
     py::array_t<I> & Lpp,
     py::array_t<I> & Lpj,
      py::array_t<I> & Lp,
      py::array_t<I> & Lj,
      py::array_t<T> & Lx,
     py::array_t<I> & Upp,
     py::array_t<I> & Upj,
      py::array_t<I> & Up,
      py::array_t<I> & Uj,
      py::array_t<T> & Ux,
      py::array_t<I> & Pc,
      py::array_t<I> & Pr,
  const I block_row_start,
   const I block_row_stop,
   const I block_row_step
                                  )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Bp = Bp.unchecked();
    auto py_Lpp = Lpp.unchecked();
    auto py_Lpj = Lpj.unchecked();
    auto py_Lp = Lp.unchecked();
    auto py_Lj = Lj.unchecked();
    auto py_Lx = Lx.unchecked();
    auto py_Upp = Upp.unchecked();
    auto py_Upj = Upj.unchecked();
    auto py_Up = Up.unchecked();
    auto py_Uj = Uj.unchecked();
    auto py_Ux = Ux.unchecked();
    auto py_Pc = Pc.unchecked();
    auto py_Pr = Pr.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const I *_Bp = py_Bp.data();
    const I *_Lpp = py_Lpp.data();
    const I *_Lpj = py_Lpj.data();
    const I *_Lp = py_Lp.data();
    const I *_Lj = py_Lj.data();
    const T *_Lx = py_Lx.data();
    const I *_Upp = py_Upp.data();
    const I *_Upj = py_Upj.data();
    const I *_Up = py_Up.data();
    const I *_Uj = py_Uj.data();
    const T *_Ux = py_Ux.data();
    const I *_Pc = py_Pc.data();
    const I *_Pr = py_Pr.data();

    return block_gauss_seidel_gen_splu<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Bp, Bp.shape(0),
                     _Lpp, Lpp.shape(0),
                     _Lpj, Lpj.shape(0),
                      _Lp, Lp.shape(0),
                      _Lj, Lj.shape(0),
                      _Lx, Lx.shape(0),
                     _Upp, Upp.shape(0),
                     _Upj, Upj.shape(0),
                      _Up, Up.shape(0),
                      _Uj, Uj.shape(0),
                      _Ux, Ux.shape(0),
                      _Pc, Pc.shape(0),
                      _Pr, Pr.shape(0),
          block_row_start,
           block_row_stop,
           block_row_step
                                                );
}

template<class I, class T, class F>
void _block_gauss_seidel_gen_inexact(
      py::array_t<I> & Ap,
      py::array_t<I> & Aj,
      py::array_t<T> & Ax,
       py::array_t<T> & x,
       py::array_t<T> & b,
      py::array_t<I> & Bp,
     py::array_t<I> & Dpp,
     py::array_t<I> & Dpj,
      py::array_t<I> & Dp,
      py::array_t<I> & Dj,
      py::array_t<T> & Dx,
 const I inner_iterations,
  const I block_row_start,
   const I block_row_stop,
   const I block_row_step
                                     )
{
    auto py_Ap = Ap.unchecked();
    auto py_Aj = Aj.unchecked();
    auto py_Ax = Ax.unchecked();
    auto py_x = x.mutable_unchecked();
    auto py_b = b.unchecked();
    auto py_Bp = Bp.unchecked();
    auto py_Dpp = Dpp.unchecked();
    auto py_Dpj = Dpj.unchecked();
    auto py_Dp = Dp.unchecked();
    auto py_Dj = Dj.unchecked();
    auto py_Dx = Dx.unchecked();
    const I *_Ap = py_Ap.data();
    const I *_Aj = py_Aj.data();
    const T *_Ax = py_Ax.data();
    T *_x = py_x.mutable_data();
    const T *_b = py_b.data();
    const I *_Bp = py_Bp.data();
    const I *_Dpp = py_Dpp.data();
    const I *_Dpj = py_Dpj.data();
    const I *_Dp = py_Dp.data();
    const I *_Dj = py_Dj.data();
    const T *_Dx = py_Dx.data();

    return block_gauss_seidel_gen_inexact<I, T, F>(
                      _Ap, Ap.shape(0),
                      _Aj, Aj.shape(0),
                      _Ax, Ax.shape(0),
                       _x, x.shape(0),
                       _b, b.shape(0),
                      _Bp, Bp.shape(0),
                     _Dpp, Dpp.shape(0),
                     _Dpj, Dpj.shape(0),
                      _Dp, Dp.shape(0),
                      _Dj, Dj.shape(0),
                      _Dx, Dx.shape(0),
         inner_iterations,
          block_row_start,
           block_row_stop,
           block_row_step
                                                   );
}

PYBIND11_MODULE(adaptive_relaxation, m) {
    m.doc() = R"pbdoc(
    Pybind11 bindings for adaptive_relaxation.h

    Methods
    -------
    optimal_smoother
    optimal_smoother_adjacency
    operator_dependent_interpolation
    lsa_splitting
    lsa_strength
    block_gauss_seidel_gen_inv
    block_gauss_seidel_gen_splu
    block_gauss_seidel_gen_inexact
    )pbdoc";

    py::options options;
    options.disable_function_signatures();

    m.def("optimal_smoother", &_optimal_smoother<int, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("smoother_ID").noconvert(), py::arg("smoothing_factor").noconvert(), py::arg("modes").noconvert(), py::arg("bndry_strat"));
    m.def("optimal_smoother", &_optimal_smoother<int, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("smoother_ID").noconvert(), py::arg("smoothing_factor").noconvert(), py::arg("modes").noconvert(), py::arg("bndry_strat"),
R"pbdoc(
)pbdoc");

    m.def("optimal_smoother_adjacency", &_optimal_smoother_adjacency<int, float>,
        py::arg("smoother_ID").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert());
    m.def("optimal_smoother_adjacency", &_optimal_smoother_adjacency<int, double>,
        py::arg("smoother_ID").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(),
R"pbdoc(
Build adjacency matrix for blocks in optimal smoothers
smoother_ID[] : An array holding the smoother type for every DOF
Sp[] and Sj[] are the row pointer and column index array for the adjacency matrix we construct here.
To build row i of this adjacency matrix, we simpy look at the type of smoother used at that point to discern which of its distance-1 neighbors it connects to, and these connections go into the adjacency matrix.)pbdoc");

    m.def("operator_dependent_interpolation", &_operator_dependent_interpolation<int, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(), py::arg("n"));
    m.def("operator_dependent_interpolation", &_operator_dependent_interpolation<int, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("Pp").noconvert(), py::arg("Pj").noconvert(), py::arg("Px").noconvert(), py::arg("n"),
R"pbdoc(
)pbdoc");

    m.def("lsa_splitting", &_lsa_splitting<int, float>,
        py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("alpha_thresh"), py::arg("splitting").noconvert(), py::arg("num_blocks").noconvert());
    m.def("lsa_splitting", &_lsa_splitting<int, double>,
        py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("alpha_thresh"), py::arg("splitting").noconvert(), py::arg("num_blocks").noconvert(),
R"pbdoc(
Given the symmetric strength-of-connection matrix S, split DOFs into disjoint blocks that are
strongly connected to one another.

 Parameters
----------
Sp : array
    CSR row pointer
Sj : array
    CSR index array
Sx : array
    CSR data array
alpha_thresh : float
     threshold for determing strong connections
splitting : array
    index array, with splitting[i] being the block index of DOF i
num_blocks : array
     integer holding the number of

Returns
-------
Nothing, splitting and num_blocks will be modified inplace

Notes
-----
The SOC matrix S must be symmetric! (The implementation uses symmetric assumptions; unclear what
a non-symmetric algorithm would look like.) All of its entries are also assumed non-negative.)pbdoc");

    m.def("lsa_strength", &_lsa_strength<int, float>,
        py::arg("Ahatp").noconvert(), py::arg("Ahatx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert());
    m.def("lsa_strength", &_lsa_strength<int, double>,
        py::arg("Ahatp").noconvert(), py::arg("Ahatx").noconvert(), py::arg("Sp").noconvert(), py::arg("Sj").noconvert(), py::arg("Sx").noconvert(), py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(),
R"pbdoc(
Build the strength matrix for LSA

Given the CSR matrix A, for every row i, we compute the matrix Ahati that is the restriction of A to the distance 1 neighbors of i. We then invert each of these matrices and compute the strength measure.)pbdoc");

    m.def("block_gauss_seidel_gen_inv", &_block_gauss_seidel_gen_inv<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Mp").noconvert(), py::arg("Mx").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_inv", &_block_gauss_seidel_gen_inv<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Mp").noconvert(), py::arg("Mx").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_inv", &_block_gauss_seidel_gen_inv<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Mp").noconvert(), py::arg("Mx").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_inv", &_block_gauss_seidel_gen_inv<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Mp").noconvert(), py::arg("Mx").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"),
R"pbdoc(
)pbdoc");

    m.def("block_gauss_seidel_gen_splu", &_block_gauss_seidel_gen_splu<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Lpp").noconvert(), py::arg("Lpj").noconvert(), py::arg("Lp").noconvert(), py::arg("Lj").noconvert(), py::arg("Lx").noconvert(), py::arg("Upp").noconvert(), py::arg("Upj").noconvert(), py::arg("Up").noconvert(), py::arg("Uj").noconvert(), py::arg("Ux").noconvert(), py::arg("Pc").noconvert(), py::arg("Pr").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_splu", &_block_gauss_seidel_gen_splu<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Lpp").noconvert(), py::arg("Lpj").noconvert(), py::arg("Lp").noconvert(), py::arg("Lj").noconvert(), py::arg("Lx").noconvert(), py::arg("Upp").noconvert(), py::arg("Upj").noconvert(), py::arg("Up").noconvert(), py::arg("Uj").noconvert(), py::arg("Ux").noconvert(), py::arg("Pc").noconvert(), py::arg("Pr").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_splu", &_block_gauss_seidel_gen_splu<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Lpp").noconvert(), py::arg("Lpj").noconvert(), py::arg("Lp").noconvert(), py::arg("Lj").noconvert(), py::arg("Lx").noconvert(), py::arg("Upp").noconvert(), py::arg("Upj").noconvert(), py::arg("Up").noconvert(), py::arg("Uj").noconvert(), py::arg("Ux").noconvert(), py::arg("Pc").noconvert(), py::arg("Pr").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_splu", &_block_gauss_seidel_gen_splu<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Lpp").noconvert(), py::arg("Lpj").noconvert(), py::arg("Lp").noconvert(), py::arg("Lj").noconvert(), py::arg("Lx").noconvert(), py::arg("Upp").noconvert(), py::arg("Upj").noconvert(), py::arg("Up").noconvert(), py::arg("Uj").noconvert(), py::arg("Ux").noconvert(), py::arg("Pc").noconvert(), py::arg("Pr").noconvert(), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"),
R"pbdoc(
Block Gauss--Seidel where the diagonal blocks are inverted directly using precomputed SPLU
decompositions.
The L and U factors are inverted by calling forward and backward sweeps of gauss_seidel(),
respectively.

NOTE: L and U factors have to be in CSR format, not CSC!
NOTE: This function uses gauss_seidel(), so it has to be placed after the definition of
gauss_seidel())pbdoc");

    m.def("block_gauss_seidel_gen_inexact", &_block_gauss_seidel_gen_inexact<int, float, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Dpp").noconvert(), py::arg("Dpj").noconvert(), py::arg("Dp").noconvert(), py::arg("Dj").noconvert(), py::arg("Dx").noconvert(), py::arg("inner_iterations"), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_inexact", &_block_gauss_seidel_gen_inexact<int, double, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Dpp").noconvert(), py::arg("Dpj").noconvert(), py::arg("Dp").noconvert(), py::arg("Dj").noconvert(), py::arg("Dx").noconvert(), py::arg("inner_iterations"), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_inexact", &_block_gauss_seidel_gen_inexact<int, std::complex<float>, float>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Dpp").noconvert(), py::arg("Dpj").noconvert(), py::arg("Dp").noconvert(), py::arg("Dj").noconvert(), py::arg("Dx").noconvert(), py::arg("inner_iterations"), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"));
    m.def("block_gauss_seidel_gen_inexact", &_block_gauss_seidel_gen_inexact<int, std::complex<double>, double>,
        py::arg("Ap").noconvert(), py::arg("Aj").noconvert(), py::arg("Ax").noconvert(), py::arg("x").noconvert(), py::arg("b").noconvert(), py::arg("Bp").noconvert(), py::arg("Dpp").noconvert(), py::arg("Dpj").noconvert(), py::arg("Dp").noconvert(), py::arg("Dj").noconvert(), py::arg("Dx").noconvert(), py::arg("inner_iterations"), py::arg("block_row_start"), py::arg("block_row_stop"), py::arg("block_row_step"),
R"pbdoc(
As above, but not the diagonal blocks are inverted inexactly via some fixed number of pointwise Gauss--Seidel iterations
See the "get_diag_blocks" function for how the diagonal block data structures are setup.


NOTE: This function uses gauss_seidel(), so it has to be placed after the definition of gauss_seidel())pbdoc");

}

