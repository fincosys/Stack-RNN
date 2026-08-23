/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Butcher B-series features indexed by unordered rooted trees (OEIS A000081).
 *
 *  For a smooth vector field f, the elementary differential F(τ) attached to
 *  an unordered rooted tree τ satisfies:
 *    F(•)(y) = f(y)
 *    F([τ1,...,τk])(y) = f^{(k)}(y)( F(τ1)(y), ..., F(τk)(y) )
 *
 *  Here we use a frozen random reservoir vector field
 *    f(y) = φ( W_f y )
 *  with componentwise φ ∈ {tanh, id}, and a diagonal / Hadamard approximation
 *  of higher derivatives that stays in R^H:
 *    F(•) = f(y)
 *    F([τ1,...,τk]) = Dφ(W_f y) ⊙ ( W_f ( F(τ1) ⊙ ... ⊙ F(τk) ) )
 *  (for k=1 this matches the true Jacobian action of tanh/Id; for k>1 it is
 *  a cheap multilinear surrogate suitable as a fixed feature map).
 *
 *  Readout features for ridge regression (per tree τ of order ≤ q):
 *    φ_τ = ( h^{|τ|} / σ(τ) ) * F(τ)   optionally projected / summarized
 *  plus optional bias. Only the linear readout is trained.
 */
#ifndef _BSERIES_
#define _BSERIES_
#include <vector>
#include <string>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "common.h"
#include "Vec.h"
#include "Linear.h"
#include "Nonlinearity.h"
#include "utils.h"
#include "RootedTree.h"

namespace rnn
{

#ifndef RNN_ESN_NONLIN_DEFINED
#define RNN_ESN_NONLIN_DEFINED
  enum ESNNonlin {
    NL_TANH = 0,
    NL_SIGMOID = 1
  };
#endif

  // How to collapse each elementary differential F(τ) ∈ R^H into features
  enum BSeriesFeatMode {
    BS_FEAT_MEAN = 0,     // one scalar mean(F(τ)) per tree
    BS_FEAT_FULL = 1,     // full H dims per tree (large)
    BS_FEAT_PROJ = 2,     // random projection to bdim dims per tree
    BS_FEAT_MEAN_ROOT = 3 // mean(F(τ)) for all τ, plus full y (base state)
  };

  struct BSeriesReadout
  {
    public:

      BSeriesReadout() :
        _HIDDEN(0), _OUT(0), _FEAT(0), _order(0), _bdim(1),
        _feat_mode(BS_FEAT_MEAN_ROOT), _nonlin(NL_TANH),
        _ridge(1e-4), _step(1.0), _ncollect(0), _last_target(0),
        _catalog(), _Wf(), _proj(), _res2out(1,1),
        _out(), _err_out(), _feat(), _gram(), _B(),
        _Fcache(), _y(), _pre()
      {}

      BSeriesReadout(my_int hidden,
                     my_int so,
                     my_int order = 4,
                     my_int feat_mode = BS_FEAT_MEAN_ROOT,
                     my_int bdim = 2,
                     my_real ridge = 1e-4,
                     my_int nonlin = NL_TANH,
                     my_real step = 1.0,
                     my_real wf_scale = 0.5,
                     my_real sparsity = 0.2) :
        _HIDDEN(hidden),
        _OUT(so),
        _FEAT(0),
        _order(order < 1 ? 1 : order),
        _bdim(bdim < 1 ? 1 : bdim),
        _feat_mode(feat_mode),
        _nonlin(nonlin),
        _ridge(ridge),
        _step(step),
        _ncollect(0),
        _last_target(0),
        _catalog(),
        _Wf(_HIDDEN, _HIDDEN),
        _proj(),
        _res2out(1, 1),
        _out(_OUT, 0),
        _err_out(_OUT, 0),
        _feat(),
        _gram(),
        _B(),
        _Fcache(),
        _y(_HIDDEN, 0),
        _pre(_HIDDEN, 0)
      {
        if(_order > 8) _order = 8;
        _catalog.build(_order);
        _FEAT = computeFeatSize();
        _res2out = Linear(_FEAT, _OUT);
        _feat = Vec(_FEAT, 0);
        _gram = Vec2D(_FEAT, _FEAT, 0);
        _B = Vec2D(_FEAT, _OUT, 0);
        _Fcache = std::vector<Vec>(_catalog.size() + 1, Vec(_HIDDEN, 0));

        // Frozen random field Jacobian generator W_f
        sparse_random_init(_Wf._data, sparsity);
        my_real rho = estimate_spectral_radius(_Wf._data);
        if(rho > 1e-12)
          scale_matrix(_Wf._data, wf_scale / rho);
        else
          _Wf.zeros();

        // Random projections for BS_FEAT_PROJ: one (bdim x H) per tree
        if(_feat_mode == BS_FEAT_PROJ){
          _proj = std::vector<Linear>(_catalog.size() + 1, Linear(_HIDDEN, _bdim));
          for(my_int t = 1; t <= _catalog.size(); t++){
            for(my_int i = 0; i < _proj[t].size(); i++)
              _proj[t]._data[i] = random(-1.0, 1.0) / sqrt((my_real)_HIDDEN);
          }
        }

        _res2out.zeros();
        clearCollectors();
      }

      my_int ntrees() const { return _catalog.size(); }

      my_int computeFeatSize() const {
        my_int nt = _catalog.size();
        my_int f = 1; // bias
        if(_feat_mode == BS_FEAT_MEAN)
          f += nt;
        else if(_feat_mode == BS_FEAT_FULL)
          f += nt * _HIDDEN;
        else if(_feat_mode == BS_FEAT_PROJ)
          f += nt * _bdim;
        else { // MEAN_ROOT
          f += nt + _HIDDEN;
        }
        return f;
      }

      void clearCollectors(){
        _gram.zeros();
        _B.zeros();
        _ncollect = 0;
      }

      void apply_phi(Vec& v){
        if(_nonlin == NL_SIGMOID)
          Sigmoid::forward(v);
        else if(_nonlin == NL_TANH)
          Tanh::forward(v);
        // else identity
      }

      // Dφ(pre)_i diagonal factor
      my_real dphi(my_real pre_i, my_real phi_i) const {
        if(_nonlin == NL_TANH)
          return 1.0 - phi_i * phi_i;
        if(_nonlin == NL_SIGMOID)
          return phi_i * (1.0 - phi_i);
        return 1.0;
      }

      // f(y) = φ(W_f y); writes pre = W_f y and returns f in out
      void eval_f(const Vec& y, Vec& out, Vec& pre){
        pre.zeros();
        _Wf.forward(y, pre);
        out = pre;
        apply_phi(out);
      }

      // Elementary differentials for all trees, bottom-up by increasing order.
      void computeDifferentials(const Vec& y){
        assert(y.size() == _HIDDEN);
        _y = y;
        // pre and f(y) for Jacobian diagonal
        eval_f(y, _Fcache[0], _pre); // use slot 0 as scratch for f(y)
        // For B-series some authors set F(•)=f(y); we store f(y) as tree id of •
        // Tree id 1 is always •
        if(_catalog.size() >= 1)
          _Fcache[1] = _Fcache[0];

        for(my_int ord = 2; ord <= _order; ord++){
          const std::vector<my_int>& ids = _catalog.by_order[ord];
          for(size_t ii = 0; ii < ids.size(); ii++){
            my_int id = ids[ii];
            const UTreeShape& sh = _catalog.get(id);
            // Hadamard product of child differentials
            Vec prod(_HIDDEN, 0);
            if(sh.child_ids.empty()){
              prod = _Fcache[1];
            } else {
              prod = _Fcache[sh.child_ids[0]];
              for(size_t c = 1; c < sh.child_ids.size(); c++){
                const Vec& fc = _Fcache[sh.child_ids[c]];
                for(my_int j = 0; j < _HIDDEN; j++)
                  prod[j] *= fc[j];
              }
            }
            // W_f * prod
            Vec Wp(_HIDDEN, 0);
            _Wf.forward(prod, Wp);
            // Dφ(pre) ⊙ Wp
            Vec& F = _Fcache[id];
            for(my_int j = 0; j < _HIDDEN; j++){
              my_real phi_j = _Fcache[1][j]; // f(y)_j
              F[j] = dphi(_pre[j], phi_j) * Wp[j];
            }
          }
        }
      }

      // Build readout feature vector from current _Fcache / _y
      void buildFeatures(){
        _feat.zeros();
        my_int off = 0;
        my_int nt = _catalog.size();

        if(_feat_mode == BS_FEAT_MEAN_ROOT){
          for(my_int j = 0; j < _HIDDEN; j++)
            _feat[off++] = _y[j];
        }

        for(my_int t = 1; t <= nt; t++){
          const UTreeShape& sh = _catalog.get(t);
          // Butcher weight factor h^{|τ|} / σ(τ)
          my_real w = 1.0;
          for(my_int p = 0; p < sh.order; p++) w *= _step;
          if(sh.symmetry > 0) w /= (my_real)sh.symmetry;

          const Vec& F = _Fcache[t];
          if(_feat_mode == BS_FEAT_MEAN || _feat_mode == BS_FEAT_MEAN_ROOT){
            my_real m = 0;
            for(my_int j = 0; j < _HIDDEN; j++) m += F[j];
            m /= (my_real)_HIDDEN;
            _feat[off++] = w * m;
          } else if(_feat_mode == BS_FEAT_FULL){
            for(my_int j = 0; j < _HIDDEN; j++)
              _feat[off++] = w * F[j];
          } else { // PROJ
            Vec tmp(_bdim, 0);
            _proj[t].forward(F, tmp);
            for(my_int j = 0; j < _bdim; j++)
              _feat[off++] = w * tmp[j];
          }
        }
        _feat[off++] = 1.0; // bias
        assert(off == _FEAT);
      }

      void forwardFromState(const Vec& y, my_int target = -1){
        _last_target = target;
        computeDifferentials(y);
        buildFeatures();
        _out.zeros();
        _res2out.forward(_feat, _out);
        Softmax::forward(_out);
      }

      void collectFromState(const Vec& y, my_int target){
        forwardFromState(y, target);
        for(my_int i = 0; i < _FEAT; i++){
          my_real fi = _feat[i];
          if(fi == 0) continue;
          for(my_int j = 0; j < _FEAT; j++)
            _gram(i, j) += fi * _feat[j];
        }
        if(target >= 0 && target < _OUT){
          for(my_int i = 0; i < _FEAT; i++)
            _B(i, target) += _feat[i];
        }
        _ncollect++;
      }

      bool fitReadout(){
        if(_ncollect == 0) return false;
        Vec2D A(_FEAT, _FEAT, 0);
        Vec bcol(_FEAT, 0);
        Vec xcol(_FEAT, 0);
        for(my_int o = 0; o < _OUT; o++){
          for(my_int i = 0; i < _FEAT; i++){
            for(my_int j = 0; j < _FEAT; j++)
              A(i, j) = _gram(i, j);
            A(i, i) += _ridge;
            bcol[i] = _B(i, o);
          }
          if(!solve_linear_system(A, bcol, xcol)){
            for(my_int i = 0; i < _FEAT; i++){
              my_real d = _gram(i, i) + _ridge;
              xcol[i] = (fabs(d) > 1e-14) ? (_B(i, o) / d) : 0;
            }
          }
          for(my_int i = 0; i < _FEAT; i++)
            _res2out._data(o, i) = xcol[i];
        }
        return true;
      }

      void update(const my_real& lr){
        for(my_int i = 0; i < _OUT; i++)
          _err_out[i] = -_out[i];
        if(_last_target >= 0 && _last_target < _OUT)
          _err_out[_last_target] += 1;
        buildFeatures();
        _res2out.resetGradient();
        _res2out.computeGradient(_feat, _err_out);
        _res2out.update(lr);
      }

      my_real eval(const my_int& target) const { return _out[target]; }

      my_int pred() const {
        my_int p = 0;
        my_real pv = _out[0];
        for(my_int i = 1; i < _OUT; i++){
          if(pv < _out[i]){ p = i; pv = _out[i]; }
        }
        return p;
      }

      void copy(const BSeriesReadout& other){
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);
        assert(_FEAT == other._FEAT);
        _order = other._order;
        _bdim = other._bdim;
        _feat_mode = other._feat_mode;
        _nonlin = other._nonlin;
        _ridge = other._ridge;
        _step = other._step;
        _ncollect = other._ncollect;
        _last_target = other._last_target;
        _catalog = other._catalog;
        _Wf._data = other._Wf._data;
        _proj = other._proj;
        _res2out._data = other._res2out._data;
        _out = other._out;
        _feat = other._feat;
        _gram = other._gram;
        _B = other._B;
        _Fcache = other._Fcache;
        _y = other._y;
        _pre = other._pre;
      }

      my_int _HIDDEN;
      my_int _OUT;
      my_int _FEAT;
      my_int _order;
      my_int _bdim;
      my_int _feat_mode;
      my_int _nonlin;
      my_real _ridge;
      my_real _step;
      my_int _ncollect;
      my_int _last_target;

      UTreeCatalog _catalog;
      Linear _Wf;
      std::vector<Linear> _proj;
      Linear _res2out;
      Vec _out;
      Vec _err_out;
      Vec _feat;
      Vec2D _gram;
      Vec2D _B;
      std::vector<Vec> _Fcache;
      Vec _y;
      Vec _pre;
  };

} // namespace rnn
#endif
