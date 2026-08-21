/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _UTILS_
#define _UTILS_
#include <assert.h>
#include <math.h>
#include <vector>
#include <cmath>

#include "common.h"
#include "Vec.h"

namespace rnn {

  // utils:
  void hardclipping( Vec& v, my_int b = -1, my_int e = -1){
    if(b == -1 ) b = 0;
    if(e == -1 ) e = v.size();
    for(my_int i = b; i < e; i++){
      if( v[i] < -15) v[i] = -15;
      if( v[i] > 15) v[i] = 15;
    }
  }

  /* uniform distribution, (0..1] */
  my_real drand()
  {
    return (rand()+1.0)/(RAND_MAX+1.0);
  }

  /* normal distribution, centered on 0, std dev 1 */
  my_real random_normal()
  {
    return sqrt(-2*log(drand())) * cos(2*M_PI*drand());
  }

  my_real random(my_real min, my_real max){
    return rand()/(my_real)RAND_MAX*(max-min)+min;
  }

  /* Estimate spectral radius of a square matrix via power iteration. */
  my_real estimate_spectral_radius(const Vec2D& m, my_int niter = 200){
    assert(m.nrow() == m.ncol());
    my_int n = m.nrow();
    if(n == 0) return 0;
    // Power iteration on m and m^T; take max (helps for non-symmetric reservoirs).
    my_real radius = 0;
    for(my_int pass = 0; pass < 2; pass++){
      Vec v(n, 0), w(n, 0);
      for(my_int i = 0; i < n; i++) v[i] = random(-1.0, 1.0);
      my_real norm = 0;
      for(my_int i = 0; i < n; i++) norm += v[i] * v[i];
      norm = sqrt(norm);
      if(norm < 1e-12) {
        for(my_int i = 0; i < n; i++) v[i] = 0;
        v[0] = 1;
        norm = 1;
      }
      for(my_int i = 0; i < n; i++) v[i] /= norm;

      my_real rpass = 0;
      for(my_int it = 0; it < niter; it++){
        for(my_int i = 0; i < n; i++){
          my_real s = 0;
          if(pass == 0){
            for(my_int j = 0; j < n; j++) s += m(i, j) * v[j];
          } else {
            for(my_int j = 0; j < n; j++) s += m(j, i) * v[j];
          }
          w[i] = s;
        }
        norm = 0;
        for(my_int i = 0; i < n; i++) norm += w[i] * w[i];
        norm = sqrt(norm);
        if(norm < 1e-16){ rpass = 0; break; }
        rpass = norm;
        for(my_int i = 0; i < n; i++) v[i] = w[i] / norm;
      }
      if(rpass > radius) radius = rpass;
    }
    return radius;
  }

  void scale_matrix(Vec2D& m, my_real factor){
    for(my_int i = 0; i < m.size(); i++)
      m[i] *= factor;
  }

  /* Solve A x = b for symmetric positive-definite A via Gaussian elimination
     with partial pivoting. A is n x n (row-major in Vec2D), destroyed in place.
     b is length n; solution written to x. Returns false on failure. */
  bool solve_linear_system(Vec2D& A, const Vec& b, Vec& x){
    my_int n = A.nrow();
    assert(A.ncol() == n);
    assert(b.size() == n);
    assert(x.size() == n);

    // Augment A with b in a local copy of RHS
    std::vector<my_real> rhs(n);
    for(my_int i = 0; i < n; i++) rhs[i] = b[i];

    for(my_int k = 0; k < n; k++){
      // Partial pivot
      my_int piv = k;
      my_real best = fabs(A(k, k));
      for(my_int i = k + 1; i < n; i++){
        my_real val = fabs(A(i, k));
        if(val > best){ best = val; piv = i; }
      }
      if(best < 1e-14) return false;
      if(piv != k){
        for(my_int j = k; j < n; j++){
          my_real tmp = A(k, j); A(k, j) = A(piv, j); A(piv, j) = tmp;
        }
        my_real tmp = rhs[k]; rhs[k] = rhs[piv]; rhs[piv] = tmp;
      }
      my_real diag = A(k, k);
      for(my_int i = k + 1; i < n; i++){
        my_real f = A(i, k) / diag;
        if(f == 0) continue;
        for(my_int j = k; j < n; j++)
          A(i, j) -= f * A(k, j);
        rhs[i] -= f * rhs[k];
      }
    }

    // Back substitution
    for(my_int i = n - 1; i >= 0; i--){
      my_real s = rhs[i];
      for(my_int j = i + 1; j < n; j++)
        s -= A(i, j) * x[j];
      if(fabs(A(i, i)) < 1e-14) return false;
      x[i] = s / A(i, i);
    }
    return true;
  }

  /* Sparse random init: each entry nonzero with probability sparsity in [-1,1]. */
  void sparse_random_init(Vec2D& m, my_real sparsity){
    for(my_int i = 0; i < m.size(); i++){
      if(drand() <= sparsity)
        m[i] = random(-1.0, 1.0);
      else
        m[i] = 0;
    }
  }

  void matrixXvector(Vec& dest, const Vec& srcvec, const Vec2D& srcmatrix,
      const my_int& obegin, const my_int& oend,
      const my_int& ibegin, const my_int& iend, const my_int& type)
  {
    // type = 0 -> srcmatrix * srcvec
    // type = 1 -> srcmatrix^T * srcvec

    assert(srcmatrix.nrow() >= oend);
    assert(srcmatrix.ncol() >= iend);

    my_int a, b;
    my_real val1, val2, val3, val4;
    my_real val5, val6, val7, val8;

    my_int matrix_width = srcmatrix.ncol();


    if (type==0) {		//ac mod
      assert(dest.size() >= oend);
      assert(srcvec.size() >= iend);
      for (b=0; b<(oend-obegin)/8; b++) {
        val1=0;
        val2=0;
        val3=0;
        val4=0;

        val5=0;
        val6=0;
        val7=0;
        val8=0;

        for (a=ibegin; a<iend; a++) {
          val1 += srcvec[a] * srcmatrix[a+(b*8+obegin+0)*matrix_width];
          val2 += srcvec[a] * srcmatrix[a+(b*8+obegin+1)*matrix_width];
          val3 += srcvec[a] * srcmatrix[a+(b*8+obegin+2)*matrix_width];
          val4 += srcvec[a] * srcmatrix[a+(b*8+obegin+3)*matrix_width];

          val5 += srcvec[a] * srcmatrix[a+(b*8+obegin+4)*matrix_width];
          val6 += srcvec[a] * srcmatrix[a+(b*8+obegin+5)*matrix_width];
          val7 += srcvec[a] * srcmatrix[a+(b*8+obegin+6)*matrix_width];
          val8 += srcvec[a] * srcmatrix[a+(b*8+obegin+7)*matrix_width];
        }
        dest[b*8+obegin+0] += val1;
        dest[b*8+obegin+1] += val2;
        dest[b*8+obegin+2] += val3;
        dest[b*8+obegin+3] += val4;

        dest[b*8+obegin+4] += val5;
        dest[b*8+obegin+5] += val6;
        dest[b*8+obegin+6] += val7;
        dest[b*8+obegin+7] += val8;
      }

      for (b=b*8; b<oend-obegin; b++) {
        for (a=ibegin; a<iend; a++) {
          dest[b+obegin] += srcvec[a] * srcmatrix[a+(b+obegin)*matrix_width];
        }
      }
    }
    else {		//er mod
      assert(dest.size() >= iend);
      assert(srcvec.size() >= oend);
      for (a=0; a<(iend-ibegin)/8; a++) {
        val1=0;
        val2=0;
        val3=0;
        val4=0;

        val5=0;
        val6=0;
        val7=0;
        val8=0;

        for (b=obegin; b<oend; b++) {
          val1 += srcvec[b] * srcmatrix[a*8+ibegin+0+b*matrix_width];
          val2 += srcvec[b] * srcmatrix[a*8+ibegin+1+b*matrix_width];
          val3 += srcvec[b] * srcmatrix[a*8+ibegin+2+b*matrix_width];
          val4 += srcvec[b] * srcmatrix[a*8+ibegin+3+b*matrix_width];

          val5 += srcvec[b] * srcmatrix[a*8+ibegin+4+b*matrix_width];
          val6 += srcvec[b] * srcmatrix[a*8+ibegin+5+b*matrix_width];
          val7 += srcvec[b] * srcmatrix[a*8+ibegin+6+b*matrix_width];
          val8 += srcvec[b] * srcmatrix[a*8+ibegin+7+b*matrix_width];
        }
        dest[a*8+ibegin+0] += val1;
        dest[a*8+ibegin+1] += val2;
        dest[a*8+ibegin+2] += val3;
        dest[a*8+ibegin+3] += val4;

        dest[a*8+ibegin+4] += val5;
        dest[a*8+ibegin+5] += val6;
        dest[a*8+ibegin+6] += val7;
        dest[a*8+ibegin+7] += val8;
      }

      for (a=a*8; a<iend-ibegin; a++) {
        for (b=obegin; b<oend; b++) {
          dest[a+ibegin] += srcvec[b] * srcmatrix[a+ibegin+b*matrix_width];
        }
      }
    }
  }


} // end of namespace


#endif
