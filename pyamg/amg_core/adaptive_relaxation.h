#ifndef ADAPTIVE_RELAXATION_H
#define ADAPTIVE_RELAXATION_H

#include "relaxation.h" // Need access to gauss_seidel()
#include "linalg.h"
#include <list>     // Need lists for the splitting algorithm.

#include <math.h>
#include <complex> // Needed for LFA stuff


template<class I>
inline I index_from_grid_inds(const I i, const I j, const I n) {
    return i + j * (n - 1);
}


template<class I, class T>
void nine_point_prec_stencil(T M_stencil[], const I smoother_ID, const T A_stencil[])
{
    // Initialize M to zero
    memset(M_stencil, 0.0, 9 * sizeof(M_stencil[0]));
    switch (smoother_ID) {
        // point
        case 0:
            M_stencil[0] = A_stencil[0];
            M_stencil[1] = A_stencil[1];
            M_stencil[2] = A_stencil[2];
            M_stencil[3] = A_stencil[3];
            M_stencil[4] = A_stencil[4];
            break;
        // x-line
        case 1:
            M_stencil[0] = A_stencil[0];
            M_stencil[1] = A_stencil[1];
            M_stencil[2] = A_stencil[2];
            M_stencil[3] = A_stencil[3];
            M_stencil[4] = A_stencil[4];
            M_stencil[5] = A_stencil[5];
            break;
        // 45-line
        case 2:
            M_stencil[0] = A_stencil[0];
            M_stencil[1] = A_stencil[1];
            M_stencil[2] = A_stencil[2];
            M_stencil[4] = A_stencil[4];
            M_stencil[5] = A_stencil[5];
            M_stencil[8] = A_stencil[8];
            break;
        // y-line
        case 3:
            M_stencil[0] = A_stencil[0];
            M_stencil[1] = A_stencil[1];
            M_stencil[3] = A_stencil[3];
            M_stencil[4] = A_stencil[4];
            M_stencil[6] = A_stencil[6];
            M_stencil[7] = A_stencil[7];
            break;
        // 135-line
        case 4:
            M_stencil[0] = A_stencil[0];
            M_stencil[1] = A_stencil[1];
            M_stencil[2] = A_stencil[2];
            M_stencil[3] = A_stencil[3];
            M_stencil[4] = A_stencil[4];
            M_stencil[6] = A_stencil[6];
            break;
        default:
            std::cout << "error... smoother_ID not recognised...\n";
            break;
    }
}

/* Given offsets i and j, return index into 9-point stencil. Note that here i and j are offsets about zero in the hortizontal and vertical directions respectively. */
template<class I>
I nine_point_stencil_index(const I i, const I j)
{
    if (abs(i) > 1 || abs(j) > 1) {
        std::cout << "Error: stencil isn't 9 point.";
        std::cout << "  (i,j)=(" << i << "," << j << ")";

    }

    I index = 0;
    switch (j) {
        case -1:
            switch (i) {
                case -1: index = 0; break;
                case  0: index = 1; break;
                case  1: index = 2; break;
                default: index = -1; break;
            }
            break;
        case 0:
            switch (i) {
                case -1: index = 3; break;
                case  0: index = 4; break;
                case  1: index = 5; break;
                default: index = -1; break;
            }
            break;
        case 1:
            switch (i) {
                case -1: index = 6; break;
                case  0: index = 7; break;
                case  1: index = 8; break;
                default: index = -1; break;
            }
            break;
        default: index = -1; break;
    }
    return index;
}

// Assumed stencil is ordered west to east and south to north
template<class T>
std::complex<T> nine_point_fourier_symbol(const T A[], const T omega1, const T omega2)
{
    // imaginary unit
    const std::complex<T> i(0.0,1.0); 
    // Compute symbol
    std::complex<T> A_tilde =     A[6] * std::exp(i*(-omega1 + omega2))  // NW
                                + A[7] * std::exp(i*(        + omega2))  // N
                                + A[8] * std::exp(i*(+omega1 + omega2))  // NE
                                + A[3] * std::exp(i*(-omega1         ))  // W
                                + A[4]                                    // C
                                + A[5] * std::exp(i*(+omega1         ))  // E
                                + A[0] * std::exp(i*(-omega1 - omega2))  // SW
                                + A[1] * std::exp(i*(        - omega2))  // S
                                + A[2] * std::exp(i*(+omega1 - omega2));  // SE
    return A_tilde;
}

/* Extract the 9-point stencil from the given CSR matrix A for the given DOF. A is assumed to apply to a grid with n-1 interior DOFs in each direction. 
    Any DOF that doesn't have a full 9-point stencil just has those missing entries padded with zeros 
*/
template<class I, class T>
void nine_point_stencil_get(const I Ap[], 
                            const I Aj[],
                            const T Ax[],
                            const I dof,
                            const I n,
                                T A_stencil[]) 
{
    // Zero out all entries in the stencil
    memset(A_stencil, 0.0, 9 * sizeof(A_stencil[0]));
    
    // Grid indices of current dof
    I j = dof / (n-1); // Note the integer division
    I i = dof - j * (n-1); 

    // Get 9-point stencil for this DOF
    // Loop over all columns connected to DOF
    for (I col_idx = Ap[dof]; col_idx < Ap[dof+1]; col_idx++) {

        // global index of connection
        I col = Aj[col_idx];
        T Aij = Ax[col_idx];

        // Grid indices of connection
        I j_col = col / (n-1);
        I i_col = col - j_col * (n-1);

        // if (dof == 0) {
        //     std::cout << col << ", " << Aij << ", " << i_col << ", " << j_col << ", " << nine_point_stencil_index(i_col - i, j_col - j) << "\n"; 
        // }

        // Instert Aij into appropriate place based on grid index offsets
        A_stencil[nine_point_stencil_index(i_col - i, j_col - j)] = Aij;
    }
}


template<class T>
void nine_point_stencil_print(const T A_stencil[])
{
    std::cout << A_stencil[nine_point_stencil_index(-1, 1)] << " "; 
    std::cout << A_stencil[nine_point_stencil_index( 0, 1)] << " "; 
    std::cout << A_stencil[nine_point_stencil_index( 1, 1)] << " "; 
    std::cout << "\n";
    std::cout << A_stencil[nine_point_stencil_index(-1, 0)] << " "; 
    std::cout << A_stencil[nine_point_stencil_index( 0, 0)] << " "; 
    std::cout << A_stencil[nine_point_stencil_index( 1, 0)] << " "; 
    std::cout << "\n"; 
    std::cout << A_stencil[nine_point_stencil_index(-1,-1)] << " "; 
    std::cout << A_stencil[nine_point_stencil_index( 0,-1)] << " "; 
    std::cout << A_stencil[nine_point_stencil_index( 1,-1)] << " "; 
    std::cout << "\n"; 
}

/* Compute the optimal smoother for a single stencil, optimizing it over the give modes. This is a helper function for "optimal_smoother" below. */
template<class I, class T>
I optimal_smoother_local(const T  A_stencil[],
                               T  M_stencil[], // This is to be populated; is just a place holder
                         const T  modes[], 
                         const I  num_modes,
                               T& smoothing_factor,
                         const T  point_thresh) 
{
    I num_smoothers = 5; // point (==0), x (==1), 45 (==2), y (==3), 135 (==4)

    // Test A on RHFM
    std::complex<T> A_action[num_modes];
    for (I mode = 0; mode < num_modes; mode++) {
        A_action[mode] = nine_point_fourier_symbol(A_stencil, modes[2*mode], modes[2*mode+1]);

        // // There should be a HF mode on which the local action of A is zero, no?
        // if (mode == 0) {
        //     std::cout << "|A action| = " << std::abs(A_action[mode]) << "\n";
        // }
    }

    // Test each of the preconditioners on the RHFM
    const std::complex<T> one(1.0,0.0); 
    T MU[num_smoothers] = { 0.0 };
    std::complex<T> M_action[num_modes];
    for (I smoother = 0; smoother < num_smoothers; smoother++) {
        nine_point_prec_stencil(M_stencil, smoother, A_stencil);

        // Test M on RHFM
        for (I mode = 0; mode < num_modes; mode++) {
            M_action[mode] = nine_point_fourier_symbol(M_stencil, modes[2*mode], modes[2*mode+1]);
        }

        T S_max = 0.0;
        // Compute approximate smoothing factor by maximizing action of smoother on RHFM
        for (I mode = 0; mode < num_modes; mode++) {
            T S_temp = std::abs( one - A_action[mode]/M_action[mode] );
            if (S_temp > S_max) {
                S_max = S_temp;
            }
        }

        MU[smoother] = S_max;
    }

    // Find smoother with minimum approximate smoothing factor
    I min_smoother = 0;
    T mu_min       = MU[0];
    for (I smoother = 1; smoother < num_smoothers; smoother++) {
        if (MU[smoother] < mu_min) {
            mu_min       = MU[smoother]; 
            min_smoother = smoother;
        }
    }

    // To actually select this smoother, make sure it's at least a factor of X improvement over point smoothing.
    //std::cout << "mu_min = " << mu_min << ", MU[0] = " << MU[0] << ", point_thresh = " << point_thresh << "\n";
    if (MU[0] / mu_min < point_thresh) {
        mu_min       = MU[0];
        min_smoother = 0;
    }

    // Store the minimized smoothing factor
    smoothing_factor = mu_min;

    return min_smoother;
}


/* For matrix A using a 9-point stencil on a 2D grid, compute approximately at every DOF what is 
 *the optimal smoother over a set of predefined smoothers.

A is assumed ordered row-wise lexicographically with n-1 interior DOFs in either direction 

Parameters
 * ----------
 * Ap : array
 *     CSR row pointer
 * Aj : array
 *     CSR index array
 * Ax : array
 *     CSR data array
 * smoother_ID : array
 *     Integer array holding the type of smoother for each DOF. This is what is populated in this   function. It needs to be pre-allocated with a size equal to the number of DOFs.
 * smoothing_factor : array
 *     Array holding the approximately optimal smoothing factor at each grid point.
 * modes : array
 *      These are the frequencies of modes to optimize over. They are blocked in x and y pairs
 * bndry_strat : int
 *      ID for strategy to emply at boundary-adjacent points since these DOFs do not have a full 9-point stencil.

 */
template<class I, class T>
void optimal_smoother(const I Ap[],               const int Ap_size, 
                      const I Aj[],               const int Aj_size,
                      const T Ax[],               const int Ax_size,
                            I smoother_ID[],      const int smoother_ID_size, 
                            T smoothing_factor[], const int smoothing_factor_size,
                      const T modes[],            const int modes_size,
                            I bndry_strat,
                            T point_thresh) 
{

    I num_modes = modes_size / 2;

    I num_smoothers = 5; // point (==0), x (==1), 45 (==2), y (==3), 135 (==4) 

    I N = Ap_size - 1; // Total number of DOFs.
    I n = sqrt(N) + 1; // N = (n-1)^2 for n-1 DOFs in each direction

    // 9-point stencil for each DOF
    T A_stencil[9];
    // 9-point stencil for associated preconditioner
    T M_stencil[9];

    // Loop over all DOFs
    // For each DOF, test the corresponding preconditioned 9-point stencils against the given modes
    // Any DOF that doesn't have a full 9-point stencil just has those missing entries padded with zeros 
    for (I dof = 0; dof < N; dof++) {

        // Get 9-point stencil for given DOF
        nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil); 

        // Compute the associated optimal smoother
        smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);

    }
    // Finished processing all DOFs


    // Deal with boundary-adjacent DOFs
    // --------------------------------

    // Strategy 1 is what's implemented above, so don't do anything
    if (bndry_strat == 1) {

    // Strategy 0 forces point smoothers at all of these DOFs
    } else if (bndry_strat == 0) {

        // Corner points
        smoother_ID[index_from_grid_inds(0,   0,   n)] = 0; // SW
        smoother_ID[index_from_grid_inds(n-2, 0,   n)] = 0; // SE
        smoother_ID[index_from_grid_inds(0,   n-2, n)] = 0; // NW
        smoother_ID[index_from_grid_inds(n-2, n-2, n)] = 0; // NE

        // West boundary. //I i = 0;
        for (I j = 1; j < n-2; j++) {
            I ind = index_from_grid_inds(0, j, n);
            smoother_ID[ind] = 0;
        }
        // East boundary. //I i = n-2;
        for (I j = 1; j < n-2; j++) {
            I ind = index_from_grid_inds(n-2, j, n);
            smoother_ID[ind] = 0;
        }
        // South boundary. //I j = 0;
        for (I i = 1; i < n-2; i++) {
            I ind = index_from_grid_inds(i, 0, n);
            smoother_ID[ind] = 0;
        }
        // North boundary. // I j = n-2;
        for (I i = 1; i < n-2; i++) {
            I ind = index_from_grid_inds(i, n-2, n);
            smoother_ID[ind] = 0;
        }

    // Strategy 2 re-computes the smoothers for boundary-adjacent DOFs assuming a symmetric stencil
    } else if (bndry_strat == 2) {

        // A symmetric stencil is of the form
        // NW N NE
        // W  O W
        // NE N NW

        // Initialize to dummy values
        I dof = -1;
        T SW = -1.0, S = -1.0, SE = -1.0, W = -1.0, O = -1.0, E = -1.0, NW = -1.0, N = -1.0, NE = -1.0;

        // -------------
        // Corner points
        // -------------

        // ---------------------
        // SW
        // x  N  NE      NW N NE
        // x  O  W   --> W  O  W
        // x  x  x       NE N NW
        dof = index_from_grid_inds(0, 0, n);
        nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil); 
        // Extract known entries
        O  = A_stencil[nine_point_stencil_index( 0, 0)]; 
        W  = A_stencil[nine_point_stencil_index( 1, 0)]; 
        N  = A_stencil[nine_point_stencil_index( 0, 1)]; 
        NE = A_stencil[nine_point_stencil_index( 1, 1)]; 
        // Compute missing stencil entry
        NW = -(0.5*O + W + N + NE); 
        // Insert missing entries into stencil of A
        A_stencil[nine_point_stencil_index(-1,-1)] = NE;
        A_stencil[nine_point_stencil_index( 0,-1)] = N;
        A_stencil[nine_point_stencil_index( 1,-1)] = NW;
        A_stencil[nine_point_stencil_index(-1, 0)] = W;
        A_stencil[nine_point_stencil_index(-1, 1)] = NW;
        // Get optimal smoother on updated stencil
        smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);

        // ---------------------
        // SE
        // NW N  x      NW N NE
        // W  O  x  --> W  O  W
        // x  x  x      NE N NW
        dof = index_from_grid_inds(n-2, 0, n);
        nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil); 
        // Extract known entries
        W  = A_stencil[nine_point_stencil_index(-1, 0)]; 
        O  = A_stencil[nine_point_stencil_index( 0, 0)]; 
        NW = A_stencil[nine_point_stencil_index(-1, 1)]; 
        N  = A_stencil[nine_point_stencil_index( 0, 1)]; 
        // Compute missing stencil entry
        NE = -(W + 0.5*O + NW + N); 
        // Insert missing entries into stencil of A
        A_stencil[nine_point_stencil_index(-1,-1)] = NE;
        A_stencil[nine_point_stencil_index( 0,-1)] = N;
        A_stencil[nine_point_stencil_index( 1,-1)] = NW;
        A_stencil[nine_point_stencil_index( 1, 0)] = W;
        A_stencil[nine_point_stencil_index( 1, 1)] = NE;
        // Get optimal smoother on updated stencil
        smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);

        // ---------------------
        // NW
        // x  x  x       NW N NE
        // x  O  W   --> W  O  W
        // x  N  NW      NE N NW
        dof = index_from_grid_inds(0, n-2, n);
        nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil); 
        // Extract known entries
        N  = A_stencil[nine_point_stencil_index( 0,-1)];
        NW = A_stencil[nine_point_stencil_index( 1,-1)]; 
        O  = A_stencil[nine_point_stencil_index( 0, 0)]; 
        W  = A_stencil[nine_point_stencil_index( 1, 0)]; 
        // Compute missing stencil entry
        NE = -(N + NW + 0.5*O + W); 
        // Insert missing entries into stencil of A
        A_stencil[nine_point_stencil_index(-1,-1)] = NE;
        A_stencil[nine_point_stencil_index(-1, 0)] = W;
        A_stencil[nine_point_stencil_index(-1, 1)] = NW;
        A_stencil[nine_point_stencil_index( 0, 1)] = N;
        A_stencil[nine_point_stencil_index( 1, 1)] = NE;
        // Get optimal smoother on updated stencil
        smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);

        // --------------------
        // NE
        // x  x  x     NW N NE
        // W  O  x --> W  O  W
        // NE N  x     NE N NW
        dof = index_from_grid_inds(n-2, n-2, n);
        nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil); 
        // Extract known entries
        NE = A_stencil[nine_point_stencil_index(-1,-1)];
        N  = A_stencil[nine_point_stencil_index( 0,-1)]; 
        W  = A_stencil[nine_point_stencil_index(-1, 0)]; 
        O  = A_stencil[nine_point_stencil_index( 0, 0)]; 
        // Compute missing stencil entry
        NW = -(NE + N + W + 0.5*O); 
        // Insert missing entries into stencil of A
        A_stencil[nine_point_stencil_index( 1,-1)] = NW;
        A_stencil[nine_point_stencil_index( 1, 0)] = W;
        A_stencil[nine_point_stencil_index(-1, 1)] = NW;
        A_stencil[nine_point_stencil_index( 0, 1)] = N;
        A_stencil[nine_point_stencil_index( 1, 1)] = NE;
        // Get optimal smoother on updated stencil
        smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);

        // Non-corner points

        // West boundary. //I i = 0;
        // x N NE     NW N NE
        // x O  W --> W  O  W
        // x N NW     NE N NW
        for (I j = 1; j < n-2; j++) {
            dof = index_from_grid_inds(0, j, n);
            nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil); 
            // Extract known entries
            NW = A_stencil[nine_point_stencil_index( 1,-1)];
            W  = A_stencil[nine_point_stencil_index( 1, 0)];
            NE = A_stencil[nine_point_stencil_index( 1, 1)];
            // Insert missing entries into stencil of A
            A_stencil[nine_point_stencil_index(-1,-1)] = NE;
            A_stencil[nine_point_stencil_index(-1, 0)] = W; 
            A_stencil[nine_point_stencil_index(-1, 1)] = NW;
            // Get optimal smoother on updated stencil
            smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);
        }
        // East boundary. //I i = n-2;
        // NW N  x     NW N NE
        // W  O  x --> W  O  W
        // NE N  x     NE S NW
        for (I j = 1; j < n-2; j++) {
            dof = index_from_grid_inds(n-2, j, n);
            nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil);
            // Extract known entries
            NE = A_stencil[nine_point_stencil_index(-1,-1)];
            W  = A_stencil[nine_point_stencil_index(-1, 0)]; 
            NW = A_stencil[nine_point_stencil_index(-1, 1)];
            // Insert missing entries into stencil of A
            A_stencil[nine_point_stencil_index( 1,-1)] = NW;
            A_stencil[nine_point_stencil_index( 1, 0)] = W; 
            A_stencil[nine_point_stencil_index( 1, 1)] = NE;
            // Get optimal smoother on updated stencil
            smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);
        }
        // South boundary. //I j = 0;
        // NW N NE     NW N NE
        // W  O  W --> W  O  W
        // x  x  x     NE N NW
        for (I i = 1; i < n-2; i++) {
            dof = index_from_grid_inds(i, 0, n);
            nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil);
            // Extract known entries
            NW = A_stencil[nine_point_stencil_index(-1, 1)]; 
            N  = A_stencil[nine_point_stencil_index( 0, 1)];
            NE = A_stencil[nine_point_stencil_index( 1, 1)];
            // Insert missing entries into stencil of A
            A_stencil[nine_point_stencil_index(-1,-1)] = NE;
            A_stencil[nine_point_stencil_index( 0,-1)] = N; 
            A_stencil[nine_point_stencil_index( 1,-1)] = NW;
            // Get optimal smoother on updated stencil
            smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);

            // std::cout << "south boundary i = " << i << "\n";
            // nine_point_stencil_print(A_stencil);
            // std::cout << "\n";
        }
        // North boundary. // I j = n-2;
        // x  x  x     NW N NE
        // W  O  W --> W  O  W
        // NE N NW     NE N NW
        for (I i = 1; i < n-2; i++) {
            dof = index_from_grid_inds(i, n-2, n);
            nine_point_stencil_get(Ap, Aj, Ax, dof, n, A_stencil);
            // Extract known entries
            NE = A_stencil[nine_point_stencil_index(-1,-1)];
            N  = A_stencil[nine_point_stencil_index( 0,-1)]; 
            NW = A_stencil[nine_point_stencil_index( 1,-1)];
            // Insert missing entries into stencil of A
            A_stencil[nine_point_stencil_index(-1, 1)] = NW;
            A_stencil[nine_point_stencil_index( 0, 1)] = N; 
            A_stencil[nine_point_stencil_index( 1, 1)] = NE;
            // Get optimal smoother on updated stencil
            smoother_ID[dof] = optimal_smoother_local(A_stencil, M_stencil, modes, num_modes, smoothing_factor[dof], point_thresh);
        }
    } 
    // Finished dealing with boundary DOFs    
}



/* For the given INTERIOR DOF, find its distance 1 neighbours (including itself) on the mesh based on the type of smoother it uses.
 * Sj is an array in which the indicies of the connections are placed 
 * count is the number of connections added
 */
template<class I>
void smoother_dist1_con_interior(const I i,
                                 const I j,
                                 const I n,
                                 const I dof,
                                 const I smoother_ID,
                                       I Sj[],
                                       I &count)
{
    switch (smoother_ID)
    {
    // point
    case 0:
        Sj[0] = dof;
        count = 1;
        break;

    // x-line. W and E neighbors
    case 1:
        Sj[0] = index_from_grid_inds(i-1, j, n);
        Sj[1] = dof;
        Sj[2] = index_from_grid_inds(i+1, j, n);
        count = 3;
        /* code */
        break;

    // 45-line. SW and NE neighbors
    case 2:
        Sj[0] = index_from_grid_inds(i-1, j-1, n);
        Sj[1] = dof;
        Sj[2] = index_from_grid_inds(i+1, j+1, n);
        count = 3;
        break;

    // y-line. S and N neighbors
    case 3:
        Sj[0] = index_from_grid_inds(i, j-1, n);
        Sj[1] = dof;
        Sj[2] = index_from_grid_inds(i, j+1, n);
        count = 3;
        break;

    // 135-line. SE and NW neighbors
    case 4:
        Sj[0] = index_from_grid_inds(i+1, j-1, n);
        Sj[1] = dof;
        Sj[2] = index_from_grid_inds(i-1, j+1, n);
        count = 3;
        break;
    
    default:
        std::cout << "error smoother_ID not recognised";
        break;
    }
}


/* Almost identical to the above, except the DOF in question is adjacent to the boundary and so doesn't have a full set of distance-1 neighbors. As such, this function just employs some additonal checks that the distance-1 neighbors exist on the mesh before they are added into Sj (there's no point on doing all of these time-consuming checks for all of the interior DOF which have a full set of neighbors).
 */
template<class I>
void smoother_dist1_con_bndry_adj(const I i,
                                  const I j,
                                  const I n,
                                  const I dof,
                                  const I smoother_ID,
                                        I Sj[],
                                        I &count)
{
    I count_loc = 0; // Counter for number of elements added for this DOF
    switch (smoother_ID)
    {
    // point
    case 0:
        Sj[0] = dof;
        count = 1;
        break;

    // x-line. W and E neighbors. Ensure they exist.
    case 1:
        if (i != 0) { // West neighbor: I cannot be on the west boundary
            Sj[count]  = index_from_grid_inds(i-1, j, n);
            count_loc += 1;
        }
        Sj[count_loc]  = dof;
        count_loc     += 1;
        if (i != n-2) { // East neighbor: I cannot be on the east boundary
            Sj[count_loc]  = index_from_grid_inds(i+1, j, n);
            count_loc     += 1;
        }
        count = count_loc;
        break;

    // 45-line. SW and NE neighbors. Ensure they exist.
    case 2:
        if (i != 0 && j != 0) { // SW neighbor: I cannot be on the west or south boundaries
            Sj[0]      = index_from_grid_inds(i-1, j-1, n);
            count_loc += 1;
        }
        Sj[count_loc]  = dof;
        count_loc     += 1;
        if (j != n-2 && i != n-2) { // NE neighbor: I cannot be on the north or east boundaries
            Sj[count_loc]  = index_from_grid_inds(i+1, j+1, n);
            count_loc     += 1;
        }
        count = count_loc;
        break;

    // y-line. S and N neighbors. Ensure they exist.
    case 3:
        if (j != 0) { // S neighbor: I cannot be on the south boundary
            Sj[0]      = index_from_grid_inds(i, j-1, n);
            count_loc += 1;
        }
        Sj[count_loc]  = dof;
        count_loc     += 1;
        if (j != n-2) { // N neighbor: I cannot be on the north boundary
            Sj[count_loc]  = index_from_grid_inds(i, j+1, n);
            count_loc     += 1;
        }
        count = count_loc;
        break;

    // 135-line. SE and NW neighbors. Ensure they exist.
    case 4:
        // SE neighbor: I cannot be on the south or east boundaries
        if (j != 0 && i != n-2) { 
            Sj[0]      = index_from_grid_inds(i+1, j-1, n);
            count_loc += 1;
        }
        Sj[count_loc]  = dof;
        count_loc     += 1;
        // NW neighbor: I cannot be on the north or west boundaries
        if (j != n-2 && i != 0) { 
            Sj[count_loc]  = index_from_grid_inds(i-1, j+1, n);
            count_loc     += 1;
        }
        count = count_loc;
        break;
    
    default:
        std::cout << "error smoother_ID not recognised";
        break;
    }
}

/* Build adjacency matrix for blocks in optimal smoothers
 * smoother_ID[] : An array holding the smoother type for every DOF
 * Sp[] and Sj[] are the row pointer and column index array for the adjacency matrix we construct here.  
 * To build row i of this adjacency matrix, we simpy look at the type of smoother used at that point to discern which of its distance-1 neighbors it connects to, and these connections go into the adjacency matrix. 
 */
template<class I, class T>
void optimal_smoother_adjacency(const I smoother_ID[], const int smoother_ID_size,
                                      I Sp[],          const int Sp_size,
                                      I Sj[],          const int Sj_size)
{
    
    I N = smoother_ID_size; // Total number of DOFs.
    I n = sqrt(N) + 1; // N = (n-1)^2 for n-1 DOFs in each direction

    // Build adjacency matrix
    // Each DOF can be directly connected to at most 3 DOFs including itself. This gives us an upper bound on allocating the NZ for the adjacency matrix. 
    Sp[0]   = 0;
    I count = 0; // counter for the cumulative number of elements added to Sj
    I count_loc; // counter for the number of elements added to Sj for the given DOF
    I dof;       // The given dof

    // Rather than looping from DOF = 0 up to DOF = N-1, we iterate through the DOFs based on the decomposition of the grid into boundary-adjacent and interior points. This saves a lot of checking whether a DOF is boundary-adjacent or not.

    // DOFs along south boundary
    {
        I j = 0;
        for (I i = 0; i < n-1; i++) {   
            count_loc = 0;
            dof = index_from_grid_inds(i, j, n);
            smoother_dist1_con_bndry_adj(i, j, n, dof, smoother_ID[dof], &(Sj[count]), count_loc);
            count += count_loc;
            Sp[dof+1] = count;
        }
    }

    // All DOFs between south and north boundaries
    for (I j = 1; j < n-2; j++) {

        // Single west boundary DOF
        {
            I i = 0;
            count_loc = 0;
            dof = index_from_grid_inds(i, j, n);
            smoother_dist1_con_bndry_adj(i, j, n, dof, smoother_ID[dof], &(Sj[count]), count_loc);
            count += count_loc;
            Sp[dof+1] = count;
        }

        // Interior DOFs
        for (I i = 1; i < n-2; i++) {
            count_loc = 0;
            dof = index_from_grid_inds(i, j, n);
            smoother_dist1_con_interior(i, j, n, dof, smoother_ID[dof], &(Sj[count]), count_loc);
            count += count_loc;
            Sp[dof+1] = count;
        }

        // Single east boundary DOF
        {
            I i = n-2;
            count_loc = 0;
            dof = index_from_grid_inds(i, j, n);
            smoother_dist1_con_bndry_adj(i, j, n, dof, smoother_ID[dof], &(Sj[count]), count_loc);
            count += count_loc;
            Sp[dof+1] = count;
        }

    }

    // DOFs along north boundary
    {
        I j = n-2;
        for (I i = 0; i < n-1; i++) {
            count_loc = 0;
            dof = index_from_grid_inds(i, j, n);
            smoother_dist1_con_bndry_adj(i, j, n, dof, smoother_ID[dof], &(Sj[count]), count_loc);
            count += count_loc;
            Sp[dof+1] = count;
        }
    }
}



// Get coarse-grid index for a point on both fine and coarse grids given its fine grid indices
template<class I>
inline I coarse_index_from_fine_grid_inds(const I i_f, const I j_f, const I n_f) {
    I i_c = (i_f - 1) / 2, j_c = (j_f - 1) / 2, n_c = n_f / 2;
    return index_from_grid_inds(i_c, j_c, n_c);
}

// Fine-grid point at (i_f,j_f). Note this function makes no sense if the point immediately to its east is not on the coarse grid. That is, this should be a diamond.
template<class I>
inline I east_coarse_neighbour_of_fine_point(const I i_f, const I j_f, const I n_f) {
    I i_c = i_f / 2 - 1, j_c = (j_f - 1) / 2, n_c = n_f / 2;
    return index_from_grid_inds(i_c, j_c, n_c);
}

// Fine-grid point at (i_f,j_f). Note this function makes no sense if the point immediately to its west is not on the coarse grid. That is, this should be a diamond.
template<class I>
inline I west_coarse_neighbour_of_fine_point(const I i_f, const I j_f, const I n_f) {
    I i_c = i_f / 2, j_c = (j_f - 1) / 2, n_c = n_f / 2;
    return index_from_grid_inds(i_c, j_c, n_c);
}

// Fine-grid point at (i_f,j_f). Note this function makes no sense if the point immediately to its east is not on the coarse grid. That is, this should be a square.
template<class I>
inline I north_coarse_neighbour_of_fine_point(const I i_f, const I j_f, const I n_f) {
    I i_c = (i_f - 1) / 2, j_c = j_f / 2, n_c = n_f / 2;
    return index_from_grid_inds(i_c, j_c, n_c);
}

// Fine-grid point at (i_f,j_f). Note this function makes no sense if the point immediately to its east is not on the coarse grid. That is, this should be a square.
template<class I>
inline I south_coarse_neighbour_of_fine_point(const I i_f, const I j_f, const I n_f) {
    I i_c = (i_f - 1) / 2, j_c = j_f / 2 - 1, n_c = n_f / 2;
    return index_from_grid_inds(i_c, j_c, n_c);
}

// Fine-grid point at (i_f,j_f). Note this function makes no sense if the point immediately to its corner is not on the coarse grid. That is, this should be an open circle
template<class I>
inline I cornering_coarse_neighbour_of_fine_point(const I i_f, const I j_f, const I n_f, const I corner) {
    // nw = 0, ne = 1, sw = 2, sw = 3
    I i_c = 0, j_c = 0, n_c = n_f / 2;
    // nw
    if (corner == 0) { 
        i_c = i_f / 2 - 1, j_c = j_f / 2;
    // ne
    } else if (corner == 1) { 
        i_c = i_f / 2    , j_c = j_f / 2;
    // sw
    } else if (corner == 2) { 
        i_c = i_f / 2 - 1, j_c = j_f / 2 - 1;
    // se
    } else { 
        i_c = i_f / 2    , j_c = j_f / 2 - 1;
    }

    return index_from_grid_inds(i_c, j_c, n_c);
}


// Compute y = P*x. Note that this is not optimal storage wise, but who cares.
// P is a list of ints such that x[i] -> x[p[i]]
template<class I, class T>
inline void row_permute_vector(T y[], const I p[], const T x[], const int y_size) {
    for (I i = 0; i < y_size; i++) {
        y[p[i]] = x[i];
    }
}

// P is a list of ints such that x[p[i]] -> x[i]
template<class I, class T>
inline void col_permute_vector(T y[], const I p[], const T x[], const int y_size) {
    for (I i = 0; i < y_size; i++) {
        y[i] = x[p[i]];
    }
}




template<class I, class T>
void operator_dependent_interpolation(const I Ap[], const int Ap_size, 
                                      const I Aj[], const int Aj_size,
                                            T Ax[], const int Ax_size,
                                      const I Pp[], const int Pp_size, 
                                            I Pj[], const int Pj_size,
                                            T Px[], const int Px_size,
                                      const int n)                                  
{

    I n_f = I(n);
    //std::cout << "n_f = " << n_f << "\n";

    // Build rows of P corresponding to odd-numbered x-lines
    // These lines have closed circles and diamonds
    // The j/vertical index
    for (I j_fine = 1; j_fine < n_f-1; j_fine += 2) {
        
        // Loop over points corresponding to diamonds
        for (I i_fine = 0; i_fine < n_f-1; i_fine += 2) {

            // Index on fine grid of point we're building row of P for
            I index = index_from_grid_inds(i_fine, j_fine, n_f);

            // Collapse vertical dependence of DOF in questions stencil.
            T S_w = 0.0, S_c = 0.0, S_e = 0.0;

            // Loop over all entries in the stencil of A.
            for (I j = Ap[index]; j < Ap[index+1]; j++) {
                // Index of fine grid point we connect to 
                I con_ind = Aj[j];

                // Horizontal index of the connection, since con_ind = con_i + con_j * (n_f - 1)
                I con_i = con_ind % (n_f - 1);

                // Hortizontal offset of -1, 0 or +1 tells us how to lump.
                I horz_offset = i_fine - con_i;
                if (horz_offset == -1) {
                    S_w += Ax[j]; // Connection is to our west
                } else if (horz_offset == 0) {
                    S_c += Ax[j]; // Connection is aligned with us
                } else if (horz_offset == 1) {
                    S_e += Ax[j]; // Connection is to our east
                } else {
                    std::cout << "error... this is wrong....\n";
                }
            }

            I count = 0;
            // Set east connection
            // The coarse point to the west of i_fine == 0 is on boundary, so no connection needed
            if (i_fine > 0) {
                Pj[Pp[index]+count] = east_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f);
                Px[Pp[index]+count] = -S_e / S_c;
                count++;
            }

            // Set west connection
            // The coarse point to the east of i_fine == n_f - 2 is on boundary, so no connection needed
            if (i_fine < n_f-2) {
                Pj[Pp[index]+count] = west_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f);
                Px[Pp[index]+count] = -S_w / S_c;
            }
        }
        // Finished rows for all diamonds

        // Rows corresponding to filled circles, interpolate by value
        for (I i_fine = 1; i_fine < n_f - 1; i_fine += 2) {
            // Index on fine grid of point we're building row of P for
            I index = index_from_grid_inds(i_fine, j_fine, n_f);

            Pj[Pp[index]] = coarse_index_from_fine_grid_inds(i_fine, j_fine, n_f);
            Px[Pp[index]] = 1.0;
        }
        // Finished rows for all filled circles
    }


    // Build rows of P corresponding to even-numbered x-lines
    // These lines have open circles and squares
    // The j/vertical index
    for (I j_fine = 0; j_fine < n_f-1; j_fine += 2) {
        
        // Loop over points corresponding to squares
        for (I i_fine = 1; i_fine < n_f-1; i_fine += 2) {

            // Index on fine grid of point we're building row of P for
            I index = i_fine + j_fine * (n_f - 1);

            // Collapse horizontal dependence of DOF in questions stencil.
            T S_n = 0.0, S_c = 0.0, S_s = 0.0;

            // Loop over all entries in the stencil of A.
            for (I j = Ap[index]; j < Ap[index+1]; j++) {
                // Index of fine grid point we connect to 
                I con_ind = Aj[j];

                // Vertical index of the connection, since con_ind = con_i + con_j * (n_f - 1)
                I con_i = con_ind % (n_f - 1);
                I con_j = ( con_ind - con_i ) / ( n_f - 1 );

                // Vertical offset of -1, 0 or +1 tells us how to lump.
                I vert_offset = j_fine - con_j;
                if (vert_offset == -1) {
                    S_n += Ax[j]; // Connection is to our north
                } else if (vert_offset == 0) {
                    S_c += Ax[j]; // Connection is aligned with us
                } else if (vert_offset == 1) {
                    S_s += Ax[j]; // Connection is to our south
                } else {
                    std::cout << "error... this is wrong....\n";
                }
            }

            I count = 0;
            // Set south connection
            // The coarse point to the south of j_fine == 0 is on boundary, so no connection needed
            if (j_fine > 0) {
                Pj[Pp[index]+count] = south_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f);
                Px[Pp[index]+count] = -S_s / S_c;
                count++;
            }

            // Set north connection
            // The coarse point to the north of j_fine == n_f - 2 is on boundary, so no connection needed
            if (j_fine < n_f-2) {
                Pj[Pp[index]+count] = north_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f);
                Px[Pp[index]+count] = -S_n / S_c;
            }
        }
        // Finished square points

        // Points corresponding to open circles
        for (I i_fine = 0; i_fine < n_f - 1; i_fine += 2) {
            // Index on fine grid of point we're building row of P for
            I index = index_from_grid_inds(i_fine, j_fine, n_f);

            // Get entries in the stencil of current point
            T S_nw = 0.0, S_n = 0.0, S_ne = 0.0, 
              S_w  = 0.0, S_c = 0.0, S_e  = 0.0, 
              S_sw = 0.0, S_s = 0.0, S_se = 0.0; 
 
            for (I j = Ap[index]; j < Ap[index+1]; j++) {

                // Index of fine grid point we connect to 
                I con_ind = Aj[j];

                // Vertical and horizontal grid indices of the connection, since con_ind = con_i + con_j * (n_f - 1)
                I con_i = con_ind % (n_f - 1);
                I con_j = ( con_ind - con_i ) / ( n_f - 1 );

                // Vertical and horizontal offsets.
                I horz_offset = con_i - i_fine;
                I vert_offset = con_j - j_fine;

                // if (index == 0) {
                //     std::cout << con_ind << ", " << con_i << ", " << con_j << ", " << horz_offset << ", " << vert_offset << "\n";   
                // }
                
                if        (horz_offset == -1 && vert_offset ==  1) {
                    S_nw = Ax[j];
                } else if (horz_offset ==  0 && vert_offset ==  1) {
                    S_n  = Ax[j];
                } else if (horz_offset ==  1 && vert_offset ==  1) {
                    S_ne = Ax[j];
                } else if (horz_offset == -1 && vert_offset ==  0) {
                    S_w  = Ax[j];
                } else if (horz_offset ==  0 && vert_offset ==  0) {
                    S_c  = Ax[j];
                } else if (horz_offset ==  1 && vert_offset ==  0) {
                    S_e  = Ax[j];
                } else if (horz_offset == -1 && vert_offset == -1) {
                    S_sw = Ax[j];
                } else if (horz_offset ==  0 && vert_offset == -1) {
                    S_s  = Ax[j];
                } else if (horz_offset ==  1 && vert_offset == -1) {
                    S_se = Ax[j];
                }
            }

            I count = 0; // Counter for inserting into current row of P.

            // -------------------------------------------
            // NW corner; check it's not a boundary point
            // -------------------------------------------
            if (i_fine > 0 && j_fine < n_f - 2) {
                
                // Index on coarse grid of NW corner
                I c_index_nw = cornering_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f, 0);

                // Need to find alpha_z, which is the interpolation weight from point z to the NW cornering point.  // Only west and north neighbors connect to the NW corner.
                T alpha_nw = 1.0, alpha_w = 0.0, alpha_n = 0.0;

                I index_w = index_from_grid_inds(i_fine - 1, j_fine,     n_f);
                I index_n = index_from_grid_inds(i_fine,     j_fine + 1, n_f);

                // Get west neighbor interpolation weight to NW corner
                for (I k = Pp[index_w]; k < Pp[index_w+1]; k++) {
                    if (Pj[k] == c_index_nw) {
                        alpha_w = Px[k];
                        break; 
                    }
                }

                // Get north neighbor interpolation weight to NW corner
                for (I k = Pp[index_n]; k < Pp[index_n+1]; k++) {
                    if (Pj[k] == c_index_nw) {
                        alpha_n = Px[k];
                        break; 
                    }
                }

                // Interpolation weight to NW corner
                Pj[Pp[index] + count] = c_index_nw;
                Px[Pp[index] + count] = -(S_nw * alpha_nw + S_n * alpha_n + S_w * alpha_w) / S_c;
                count++;
            }

            // -------------------------------------------
            // NE corner; check it's not a boundary point
            // -------------------------------------------
            if (i_fine < n_f - 2 && j_fine < n_f - 2) {
                // Index on coarse grid of NE corner
                I c_index_ne = cornering_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f, 1);

                // Need to find beta_z, which is the interpolation weight from point z to the NE cornering point. // Only north and east neighbors connect to the NW corner.
                T beta_ne = 1.0, beta_n = 0.0, beta_e = 0.0; // Trivially so

                I index_n = index_from_grid_inds(i_fine,     j_fine + 1, n_f);
                I index_e = index_from_grid_inds(i_fine + 1, j_fine,     n_f);

                // Get north neighbor interpolation weight to NE corner
                for (I k = Pp[index_n]; k < Pp[index_n+1]; k++) {
                    if (Pj[k] == c_index_ne) {
                        beta_n = Px[k];
                        break; 
                    }
                }

                // Get east neighbor interpolation weight to NE corner
                for (I k = Pp[index_e]; k < Pp[index_e+1]; k++) {
                    if (Pj[k] == c_index_ne) {
                        beta_e = Px[k];
                        break; 
                    }
                }

                // Interpolation weight to NE corner
                Pj[Pp[index] + count] = c_index_ne;
                Px[Pp[index] + count] = -(S_n * beta_n + S_ne * beta_ne + S_e * beta_e) / S_c;
                count++;

                // if (i_fine == 0 && j_fine == 0) {
                //     std::cout << c_index_ne << ", " << beta_e << ", "  << index_e << ", "  << S_e << ", "  <<  beta_n << ", "  << index_n << ", "  << S_n << "\n" ;
                // }
            }

            // -------------------------------------------
            // SW corner; check it's not a boundary point
            // -------------------------------------------
            if (i_fine > 0 && j_fine > 0) {

                // Index on coarse grid of SW corner
                I c_index_sw = cornering_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f, 2);

                // Need to find gamma_z, which is the interpolation weight from point z to the SW cornering point. // Only west and south neighbors connect to the SW corner.
                T gamma_sw = 1.0, gamma_w = 0.0, gamma_s = 0.0; 
                
                I index_w = index_from_grid_inds(i_fine - 1, j_fine,     n_f);
                I index_s = index_from_grid_inds(i_fine,     j_fine - 1, n_f);

                // Get west neighbor interpolation weight to SW corner
                for (I k = Pp[index_w]; k < Pp[index_w+1]; k++) {
                    if (Pj[k] == c_index_sw) {
                        gamma_w = Px[k];
                        break; 
                    }
                }

                // Get south neighbor interpolation weight to NW corner
                for (I k = Pp[index_s]; k < Pp[index_s+1]; k++) {
                    if (Pj[k] == c_index_sw) {
                        gamma_s = Px[k];
                        break; 
                    }
                }

                // Interpolation weight to SW corner
                Pj[Pp[index] + count] = c_index_sw;
                Px[Pp[index] + count] = -(S_w * gamma_w + S_sw * gamma_sw + S_s * gamma_s) / S_c;
                count++;
            }

            // -------------------------------------------
            // SE corner; check it's not a boundary point
            // -------------------------------------------
            if (i_fine < n_f - 2 && j_fine > 0) {

                // Index on coarse grid of SE corner
                I c_index_se = cornering_coarse_neighbour_of_fine_point(i_fine, j_fine, n_f, 3);

                // Need to find alpha_z, which is the interpolation weight from point z to the NW cornering point. Only east and south neighbors connect to the SE corner. 
                T delta_se = 1.0, delta_e = 0.0, delta_s = 0.0;

                I index_e = index_from_grid_inds(i_fine + 1, j_fine,     n_f);
                I index_s = index_from_grid_inds(i_fine,     j_fine - 1, n_f);

                // Get east neighbor interpolation weight to SE corner
                for (I k = Pp[index_e]; k < Pp[index_e+1]; k++) {
                    if (Pj[k] == c_index_se) {
                        delta_e = Px[k];
                        break; 
                    }
                }

                // Get south neighbor interpolation weight to NW corner
                for (I k = Pp[index_s]; k < Pp[index_s+1]; k++) {
                    if (Pj[k] == c_index_se) {
                        delta_s = Px[k];
                        break; 
                    }
                }

                // Interpolation weight to SE corner
                Pj[Pp[index] + count] = c_index_se;
                Px[Pp[index] + count] = -(S_e * delta_e + S_s * delta_s + S_se * delta_se) / S_c;
                count++;
            }            
        }
        // Finished all open circle points
    }
    // Finished all even x-lines

}


// // Create required lumped entries
// // Eliminate vertical direction
// T St_w  = S_nw + S_w + S_sw;
// T St_cx = S_n  + S_c + S_s;
// T St_e  = S_ne + S_e + S_se; 
// // Eliminate hortizontal direction
// T St_n  = S_nw + S_n + S_ne;
// T St_cy = S_w  + S_c + S_e;
// T St_s  = S_sw + S_s + S_se;

// // Compute interpolation coefficients
// T a = S_n * St_w / St_cx + S_w * St_n / St_cy - S_nw; 
// T b = S_n * St_e / St_cx + S_e * St_n / St_cy - S_ne;
// T c = S_s * St_w / St_cx + S_w * St_s / St_cy - S_sw;
// T d = S_s * St_e / St_cx + S_e * St_s / St_cy - S_se;


/*
 * Given the symmetric strength-of-connection matrix S, split DOFs into disjoint blocks that are 
 * strongly connected to one another.
 * 
 *  Parameters
 * ----------
 * Sp : array
 *     CSR row pointer
 * Sj : array
 *     CSR index array
 * Sx : array
 *     CSR data array
 * alpha_thresh : float
 *      threshold for determing strong connections 
 * splitting : array
 *     index array, with splitting[i] being the block index of DOF i
 * num_blocks : array
 *      integer holding the number of 
 *
 * Returns
 * -------
 * Nothing, splitting and num_blocks will be modified inplace
 *
 * Notes
 * -----
 * The SOC matrix S must be symmetric! (The implementation uses symmetric assumptions; unclear what
 * a non-symmetric algorithm would look like.) All of its entries are also assumed non-negative.
 * 
 */
template<class I, class T>
void lsa_splitting(const I Sp[], const int Sp_size, 
                   const I Sj[], const int Sj_size,
                         T Sx[], const int Sx_size,
                   const T alpha_thresh,
                         I splitting[], const int splitting_size, // splitting needs to be pre-allocated
                         I num_blocks[], const int num_blocks_size) // Number of blocks found. TODO: Does this really need to be an array? I couldn't work out how to modify it in place otherwise...
{

    // Initialize splitting: If splitting[i] = -1, then DOF i has not been put into a block
    for (I i = 0; i < splitting_size; i++) {
        splitting[i] = -1;
    }

    std::list<I> just_blocked;        // DOFs put into a block on the previous itertaion
    std::list<I> just_blocked_str_cn; // Strongly connected DOFs to those in just_blocked 

    I current_block = 0; // Index of block being assembled.
    I DOFs_blocked  = 0; // Counter for the number of DOFs processed 
    I U_head        = 0; // Head of unprocessed DOFs.

    // Add DOF 0 to the first block
    splitting[0] = current_block;
    just_blocked.push_back(0); 
    DOFs_blocked++; 
    
    // Continue processing so long as all DOFs not yet processed
    while (DOFs_blocked < splitting_size) {

        // Iterate over all DOFs i in just_blocked, processing their strong connections
        for (auto const& i : just_blocked) {

            // Compute max of S in row i
            T Si_max = 0.0;
            for (I j_ind = Sp[i]; j_ind < Sp[i+1]; j_ind++) {
                if (Sx[j_ind] > Si_max) {
                    Si_max = Sx[j_ind];
                }
            }
            Si_max *= alpha_thresh; // All elements in row i are compared to this

            // Loop over all columns j in row i, checking if they're strongly connected to i
            for (I j_ind = Sp[i]; j_ind < Sp[i+1]; j_ind++) {
                I j = Sj[j_ind]; // the column itself

                // Add DOF j to current block if strongly connected to i and not already in block
                // (NOTE: We don't need to check j != i, since splitting[i] != -1, so i would fail the second test anyway)
                if (Sx[j_ind] >= Si_max && splitting[j] == -1) {
                    just_blocked_str_cn.push_back(j); // Add j to list of DOFs to be processed next iteration
                    splitting[j] = current_block; // j now blocked
                    DOFs_blocked++;
                }
            } 
        } // All connections to just_blocked have been processed


        // There are DOFs in just_blocked_str_cn, which have just been added. These become just_blocked, and on the next iter we process their strong connections.
        if (!(just_blocked_str_cn.empty())) {

            just_blocked = std::list(just_blocked_str_cn); // just_blocked becomes a copy of the just_blocked_str_cn list
            just_blocked_str_cn.clear();                   // Reset list that holds strong connections to just_blocked

        // Finalize construction of current block since no more DOFs exist that are strongly connected to any DOFs in it.
        } else {

            // Drop out if no more DOFs to be processed
            if (DOFs_blocked == splitting_size) {
                break; 
            }

            // Reset lists
            just_blocked_str_cn.clear();
            just_blocked.clear();

            // Find next unprocessed DOF. Starting from U_head (the previous head of U that the prior block was initialized from), sequentially iterate through the remaining DOFs until we find one that's not processed.
            for (I head = U_head+1; head < splitting_size; head++) {
                if (splitting[head] == -1) {
                    U_head = head;
                    break;
                }
            }

            // Create a new block consisting of U_head
            just_blocked.push_back(U_head);
            current_block++; // Increment block index
            splitting[U_head] = current_block;
            DOFs_blocked++; 

        } // 
        
    } // All DOFs have been processed

    // Number of disjoint blocks found
    num_blocks[0] = current_block+1; // Blocks indexed from 0: total is 1 more the current index   
}

/*
 * Build the strength matrix for LSA
 *
 * Given the CSR matrix A, for every row i, we compute the matrix Ahati that is the restriction of A to the distance 1 neighbors of i. We then invert each of these matrices and compute the strength measure.
 * 
 * 
 * 
 */
template<class I, class T>
void lsa_strength(const I Ahatp[], const int Ahatp_size, // Pointer into the below array
                      T Ahatx[], const int Ahatx_size, // Empty, but pre-allocated to hold concatenation of inv(Ahati) for all i, stored in row-major format.
                      const I Sp[], const int Sp_size, // S is the strength of connection matrix. Should be allocated with same non-zero structure as A.
                      const I Sj[], const int Sj_size,
                            T Sx[], const int Sx_size,
                      const I Ap[], const int Ap_size,
                      const I Aj[], const int Aj_size,
                      const T Ax[], const int Ax_size)
{

    // Build 1-hood matrix for every DOF i. Ahati = A(Ni, Ni)
    for ( I i = 0; i < Ap_size - 1; i++ ) {

        I Ahati_size = I ( sqrt(Ahatp[i+1] - Ahatp[i]) ); 

        // Counter for instering elements into Ahatx
        I count = Ahatp[i];

        // Loop over all columns in row i, i.e., 1-hood(i)
        for ( I k = Ap[i]; k < Ap[i+1]; k++ ) {

            // DOF of 1-hood(i) we're building the in Ahati of 
            I row = Aj[k];

            // Loop over elements in 1-hood(i) and extract "row"'s connection to them (this ordering of steps is necessary to ensure Ahati is ordered as Ni)
            for ( I j = Ap[i]; j < Ap[i+1]; j++ ) {

                // We want to find connection to this column
                I col = Aj[j];
                I col_connection = 0;
                // Loop through columns in given row of A and see if a connection to "col" exists
                for ( I m = Ap[row]; m < Ap[row+1]; m++ ) {
                    if ( Aj[m] == col ) {
                        Ahatx[count] = Ax[m];
                        col_connection = 1;
                        count += 1;
                        break; // Stop searching for connection now
                    }
                }

                // If indice not found, set element to zero (no connection)
                if (col_connection == 0) {
                    Ahatx[count] = 0.0;
                    count += 1;
                }
            }
        }
        // Finalized matrix Ahati

        // Invert Ahati in place.
        std::vector<T> Ainv = matInverse(&Ahatx[Ahatp[i]], Ahati_size);

        // Confused why I need to copy out the data of Ainv since the A above is overwritten. Oh, maybe it's overwritten but not by the inverse itself?
        for (I count = 0; count < Ahatp[i+1] - Ahatp[i]; count++ ) {
            Ahatx[Ahatp[i]+count] = Ainv[count];
        }
    }
    // Finished inverting all PCM

    // Populate SOC matrix S (we could actually do this just after each PCM has been formed. Note also that we don't really need to store each PCM at the moment...)
    for ( I row = 0; row < Sp_size-1; row++ ) {

        // Loop over columns in current row to get relative index of diagonal entry
        I diag_offset = -1;
        I count = 0;
        for ( I col_idx = Sp[row]; col_idx < Sp[row+1]; col_idx++ ) {
            if ( Sj[col_idx] == row ) {
                diag_offset = count;
                break;
            }
            count++;
        }
        if (diag_offset == -1) {
            std::cerr << "Error: there is no diagonal connection...\n";
        }
        // Get Ahat(i,i)
        I Ahati_size = I ( sqrt(Ahatp[row+1] - Ahatp[row]) ); // Number of rows/cols of PCM 
        T Ahatii = std::abs(Ahatx[Ahatp[row] + Ahati_size * diag_offset + diag_offset]); // ith row, ith column

        // Loop over columns in current row and populate with the measure
        I col_count = 0;
        for ( I col_idx = Sp[row]; col_idx < Sp[row+1]; col_idx++ ) {
            Sx[col_idx] = std::abs(Ahatx[Ahatp[row] + Ahati_size * diag_offset + col_count]) / Ahatii; // ith row, jth column. recall stored in row-major ordering
            col_count++;
        }
    }
    // Finished construcing S.
}


/*
 * Perform one iteration of block Gauss-Seidel relaxation on the linear
 * system Ax = b, where A is stored in CSR format and x and b
 * are column vectors.
 * 
 * A is a block matrix, not necessarily having constant block size.
 * 
 * Doesn't return anything, x is modified in place.
 * 
 * Dense precomputed inverses of diagonal blocks of A are used.
 * 
 * Mx is a concatenation of the inverses of diagonal blocks of A stored in row major format
 * Mp[i] is the pointer into Mx at which the data for the ith block starts. The number of elements in the ith inverse diagonal block is Mp[i+1]-Mp[i]
 * 
 * NOTES
 * -----
 * 
*/
template<class I, class T, class F>
void block_gauss_seidel_gen_inv(const I Ap[], const int Ap_size,
                  const I Aj[], const int Aj_size,
                  const T Ax[], const int Ax_size,
                        T  x[], const int  x_size,
                  const T  b[], const int  b_size,
                  const I  Bp[], const int Bp_size, // Bp is block indptr specifying block rows of A
                  const I Mp[], const int Mp_size,  // 
                  const T Mx[], const int Mx_size,
                  const I block_row_start,
                  const I block_row_stop, // Index of the final block to update (not 1 past it!)
                  const I block_row_step) 
{

    // Loop over all block rows dictated by the parameters.
    for (I block = block_row_start; block <= block_row_stop; block += block_row_step) {

        I row_start = Bp[block];   // First row of current block
        I row_stop  = Bp[block+1]; // 1 more than last row of current block

        // Number of rows in block
        I block_size = row_stop - row_start;

        // std::cout << " " << block << " " << row_start << " " << row_stop << " " << block_size << "\n"; 

        // copy local block from b
        T * r_local = new T[block_size];
        for(I count = 0; count < block_size; count++) {
            r_local[count] = b[count + row_start];
        }

        // Subtract out components of current block row of A*x except for the block diagonal
        // Loop over rows in current block row
        for (I count = 0, row = row_start; count < block_size && row < row_stop; count ++, row++) {

            // Loop through columns in the current row
            for (I col_idx = Ap[row]; col_idx < Ap[row+1]; col_idx++) {

                // Current column
                I col = Aj[col_idx];

                // Check column is not in the diagonal block, then subtract it out
                if (col < row_start || col >= row_stop) {
                    r_local[count] -= Ax[col_idx] * x[col];
                }
            }
        }

        // The exact solution should be a fixed-point of this inversion, so uncomment this block and the one below to check this is the case.
        // // Print block to be updated.
        // std::cout << "pre-update block " << block << ": ";
        // for (I count = 0; count < block_size; count++) {
        //    std::cout << x[count + row_start] << ", ";
        // }
        // std::cout << "\n";

        // Now invert diagonal block. Basic error check
        if (Mp[block+1] - Mp[block] != block_size * block_size) {
            std::cerr << "Error: diagonal block size is inconsistent.\n";
        }

        /* Compute x_local = M*r_local for dense M and store in block of x. */
        I is_col_major = false;
        matvec(&Mx[Mp[block]], block_size, block_size,
               &r_local[0], &x[row_start], is_col_major);

        // Done with r_local (next block is potentially different size)
        delete[] r_local;

        // // Print block that has been updated.
        // std::cout << "pst-update block " << block << ": ";
        // for (I count = 0; count < block_size; count++) {
        //    std::cout << x[count + row_start] << ", ";
        // }
        // std::cout << "\n";
    }
}



/* Block Gauss--Seidel where the diagonal blocks are inverted directly using precomputed SPLU 
 * decompositions.
 * The L and U factors are inverted by calling forward and backward sweeps of gauss_seidel(),
 * respectively.
 * 
 * NOTE: L and U factors have to be in CSR format, not CSC!
 * NOTE: This function uses gauss_seidel(), so it has to be placed after the definition of
 * gauss_seidel()
 */
template<class I, class T, class F>
void block_gauss_seidel_gen_splu(const I Ap[], const int Ap_size,
                  const I  Aj[], const int Aj_size,
                  const T  Ax[], const int Ax_size,
                        T   x[], const int  x_size,
                  const T   b[], const int  b_size,
                  const I  Bp[], const int Bp_size, // Bp is block indptr specifying block rows of A
                  const I Lpp[], const int Lpp_size,
                  const I Lpj[], const int Lpj_size,
                  const I  Lp[], const int Lp_size,  // 
                  const I  Lj[], const int Lj_size,
                  const T  Lx[], const int Lx_size,
                  const I Upp[], const int Upp_size,
                  const I Upj[], const int Upj_size,
                  const I  Up[], const int Up_size,  // 
                  const I  Uj[], const int Uj_size,
                  const T  Ux[], const int Ux_size,
                  const I  Pc[], const int Pc_size, 
                  const I  Pr[], const int Pr_size, 
                  const I block_row_start,
                  const I block_row_stop, // Index of the final block to update (not 1 past it!)
                  const I block_row_step) 
{

    // Loop over all block rows dictated by the parameters.
    for (I block = block_row_start; block <= block_row_stop; block += block_row_step) {

        I row_start = Bp[block];   // First row of current block
        I row_stop  = Bp[block+1]; // 1 more than last row of current block

        // Number of rows in block
        I block_size = row_stop - row_start;

        // std::cout << " " << block << " " << row_start << " " << row_stop << " " << block_size << "\n"; 

        // copy local block from b
        T * r_local = new T[block_size];
        for(I count = 0; count < block_size; count++) {
            r_local[count] = b[count + row_start];
        }
        // auxillary vector
        T * aux     = new T[block_size];

        // Subtract out components of current block row of A*x except for the block diagonal
        // Loop over rows in current block row
        for (I count = 0, row = row_start; count < block_size && row < row_stop; count ++, row++) {

            // Loop through columns in the current row
            for (I col_idx = Ap[row]; col_idx < Ap[row+1]; col_idx++) {

                // Current column
                I col = Aj[col_idx];

                // Check column is not in the diagonal block, then subtract it out
                if (col < row_start || col >= row_stop) {
                    r_local[count] -= Ax[col_idx] * x[col];
                }
            }
        }


        // The exact solution should be a fixed-point of this inversion, so uncomment this block and the one below to check this is the case.
        // // Print block to be updated.
        // std::cout << "pre-update block " << block << ": ";
        // for (I count = 0; count < block_size; count++) {
        //    std::cout << x[count + row_start] << ", ";
        // }
        // std::cout << "\n";
        
        // Now invert diagonal block using SPLU 
        // Row data for current block in Xp, starts at index Xpp[block], and ends at one less than Xpp[block+1]
        // Column and non-zero data for current block in Xj and Xx starts at index Xpj[block], and ends at one less than Xpp[block+1]
        I Lp_start = Lpp[block];
        I Lp_stop  = Lpp[block+1]; 
        I Lp_block_size = Lp_stop - Lp_start; // Number of rows in current block
        I Lj_start = Lpj[block];
        I Lj_stop  = Lpj[block+1];
        I Lj_block_size = Lj_stop - Lj_start; // nnz in current block

        I Up_start = Upp[block];
        I Up_stop  = Upp[block+1];
        I Up_block_size = Up_stop - Up_start; // Number of rows in current block
        I Uj_start = Upj[block];
        I Uj_stop  = Upj[block+1];
        I Uj_block_size = Uj_stop - Uj_start; // nnz in current block

        /* See docs for scipy.sparse.linalg.SuperLU for how exactly the sparse-aware LU of A is done. I'm confused about the implementation of Pr and Pc. They are stored as lists, but the lists are interpreted differently in each case. This is why I have separate row_permute_vector and a col_permute_vector functions. */
        /* Solve A_local * x_local = r_local == A*x = b
         * Have: Pr @ A @ Pc = L @ U and A @ x = b
         *
         * Hence Pr.T @ L @ U @ Pc.T @ x = b
         * Hence        L @ U        @ y = f, with y := Pc.T @ x, and f =: Pr @ b
         * Hence        L            @ z = f, with z := U @ y
         * Hence                       z = L^{-1} @ f
         * Hence            U        @ y = z, 
         * Hence                       y = U^{-1} @ z
         * Hence                       x = Pc @ y
         */ 

        // if (block == 3) {
        //     std::cout << "Pr = ";
        //     for (I count = 0; count < block_size; count++) {
        //         std::cout <<  Pr[row_start + count] << " ,";
        //     }
        //     std::cout << "\nPc = ";
        //     for (I count = 0; count < block_size; count++) {
        //         std::cout <<  Pc[row_start + count] << " ,";
        //     }
        //     std::cout << "\nLp = ";
        //     for (I count = 0; count < Lp_block_size; count++) {
        //         std::cout <<  Lp[Lp_start + count] << " ,";
        //     }
        //     std::cout << "\nLj = ";
        //     for (I count = 0; count < Lj_block_size; count++) {
        //         std::cout <<  Lj[Lj_start + count] << " ,";
        //     }
        //     std::cout << "\nLx = ";
        //     for (I count = 0; count < Lj_block_size; count++) {
        //         std::cout <<  Lx[Lj_start + count] << " ,";
        //     }
        // }

        // Compute f = Pr * b
        // Use f == aux and b == r_local 
        row_permute_vector<I, T>(aux, &Pr[row_start], r_local, block_size);
        // Don't need r_local any longer

        // Solve for z s.t. L * z = f with a forward sweep of GS on L
        // Use z == r_local and f == aux
        gauss_seidel<I, T, F>(&Lp[Lp_start], Lp_block_size, 
                              &Lj[Lj_start], Lj_block_size, 
                              &Lx[Lj_start], Lj_block_size, 
                               r_local,      block_size,
                               aux,          block_size,
                               0, block_size, 1);
        // Don't need aux any longer

        // Solve for y s.t. U * y = z with a backward sweep of GS on U
        // Use y == aux and z == r_local
        gauss_seidel<I, T, F>(&Up[Up_start], Up_block_size, 
                              &Uj[Uj_start], Uj_block_size, 
                              &Ux[Uj_start], Uj_block_size, 
                               aux,          block_size,
                               r_local,      block_size,
                               block_size-1, -1, -1); 
        // Don't need r_local any longer

        // Compute x = Pc * y, with y == aux
        col_permute_vector<I, T>(&x[row_start], &Pc[row_start], aux, block_size);
        // Don't need aux any longer

        // Done with r_local (next block is potentially different size)
        delete[] r_local;
        delete[] aux;


        // // Print block that has been updated.
        // std::cout << "pst-update block " << block << ": ";
        // for (I count = 0; count < block_size; count++) {
        //    std::cout << x[count + row_start] << ", ";
        // }
        // std::cout << "\n";
    }
}


/*
 * As above, but not the diagonal blocks are inverted inexactly via some fixed number of pointwise Gauss--Seidel iterations 
 * See the "get_diag_blocks" function for how the diagonal block data structures are setup.
 * 
 * 
 * NOTE: This function uses gauss_seidel(), so it has to be placed after the definition of gauss_seidel()
 */
template<class I, class T, class F>
void block_gauss_seidel_gen_inexact(const I Ap[], const int Ap_size,
                  const I  Aj[], const int Aj_size,
                  const T  Ax[], const int Ax_size,
                        T   x[], const int  x_size,
                  const T   b[], const int  b_size,
                  const I  Bp[], const int Bp_size, // Bp is block indptr specifying block rows of A
                  const I Dpp[], const int Dpp_size,
                  const I Dpj[], const int Dpj_size,
                  const I  Dp[], const int Dp_size,
                  const I  Dj[], const int Dj_size,
                  const T  Dx[], const int Dx_size,
                  const I inner_iterations, // Number of pointwise GS iterations for inexactly solving diagonal systems
                  const I block_row_start,
                  const I block_row_stop, // Index of the final block to update (not 1 past it!)
                  const I block_row_step) 
{

    // Loop over all block rows dictated by the parameters.
    for (I block = block_row_start; block <= block_row_stop; block += block_row_step) {

        I row_start = Bp[block];   // First row of current block
        I row_stop  = Bp[block+1]; // 1 more than last row of current block

        // Number of rows in block
        I block_size = row_stop - row_start;

        // std::cout << " " << block << " " << row_start << " " << row_stop << " " << block_size << "\n"; 

        // copy local block from b
        T * r_local = new T[block_size];
        for(I count = 0; count < block_size; count++) {
            r_local[count] = b[count + row_start];
        }

        // Subtract out components of current block row of A*x except for the block diagonal
        // Loop over rows in current block row
        for (I count = 0, row = row_start; count < block_size && row < row_stop; count ++, row++) {

            // Loop through columns in the current row
            for (I col_idx = Ap[row]; col_idx < Ap[row+1]; col_idx++) {

                // Current column
                I col = Aj[col_idx];

                // Check column is not in the diagonal block, then subtract it out
                if (col < row_start || col >= row_stop) {
                    r_local[count] -= Ax[col_idx] * x[col];
                }
            }
        }

        // Now inexactly invert diagonal block using point-wise GS iterations
        // Row data for current block in Dp starts at index Dpp[block], and ends at one less than Dpp[block+1]
        // Column and non-zero data for current block in Dj and Dx starts at index Dpj[block], and ends at one less than Dpp[block+1]
        I Dp_start       = Dpp[block];
        I Dp_stop        = Dpp[block+1]-1;
        I Dp_block_size  = Dp_stop - Dp_start; // Number of rows in current block

        // Now invert diagonal block. Basic error check
        if (Dp_block_size != block_size) {
            std::cerr << "Error: diagonal block size is inconsistent.\n";
        }

        I Dj_start      = Dpj[block];
        I Dj_stop       = Dpj[block+1]-1;
        I Dj_block_size = Dj_stop - Dj_start; // nnz in current block

        // These define a standard foward sweep of GS
        I local_row_start = 0;
        I local_row_stop  = block_size;
        I local_row_step  = 1; 

        for (I iter = 0; iter < inner_iterations; iter++) {
            
            /* The unknowns are swept through according to the slice defined
            * by row_start, row_end, and row_step.  These options are used
            * to implement standard forward and backward sweeps, or sweeping
            * only a subset of the unknowns.  A forward sweep is implemented
            * with gauss_seidel(Ap, Aj, Ax, x, b, 0, N, 1) where N is the
            * number of rows in matrix A.  Similarly, a backward sweep is
            * implemented with gauss_seidel(Ap, Aj, Ax, x, b, N, -1, -1).
            */
            // Solve D*x = r. x is modified in place
            // Need to call function w/ template parameters otherwise it complains.
            gauss_seidel<I, T, F>(&Dp[Dp_start], Dp_block_size,
                         &Dj[Dj_start], Dj_block_size,
                         &Dx[Dj_start], Dj_block_size,
                         &x[row_start], block_size,
                         &r_local[0],   block_size,
                         local_row_start, local_row_stop, local_row_step);
        }

        // Done with r_local (next block is potentially different size)
        delete[] r_local;
    }
}


#endif