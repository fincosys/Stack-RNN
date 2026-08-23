/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Deep Echo State Network (DeepESN / DESN)
 *  Hierarchical stack of leaky reservoir layers (Gallicchio & Micheli).
 *  Layer 1 is driven by the external input; layer i>1 is driven by layer i-1
 *  at the same time step. Only the linear readout is trained.
 */
#ifndef _DESN_
#define _DESN_
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "common.h"
#include "Vec.h"
#include "Linear.h"
#include "Nonlinearity.h"
#include "utils.h"

namespace rnn
{

  // Reuse ESN feature / nonlinearity enums when available
#ifndef _ESN_
  enum ESNFeat {
    FEAT_RES = 0,
    FEAT_RES_IN = 1,
    FEAT_RES_STACK = 2,
    FEAT_RES_IN_STACK = 3
  };
#endif
#ifndef RNN_ESN_NONLIN_DEFINED
#define RNN_ESN_NONLIN_DEFINED
  enum ESNNonlin {
    NL_TANH = 0,
    NL_SIGMOID = 1
  };
#endif

  // DESN-specific feature modes (stack features not used; keep names for API parity)
  enum DESNFeat {
    DESN_FEAT_ALL = 0,   // concat all layer states + bias
    DESN_FEAT_LAST = 1,  // top layer only + bias
    DESN_FEAT_ALL_IN = 2 // all layers + one-hot input + bias
  };

  struct DESN
  {
    public:

      DESN(const std::string& filename)
      {
        load(filename);
        resetState();
      }

      DESN(my_int si,
          my_int sh,
          my_int so,
          my_int nlayers = 2,
          my_int mem = 50,
          my_real spectral_radius = 0.9,
          my_real input_scaling = 0.5,
          my_real leaking_rate = 1.0,
          my_real sparsity = 0.1,
          my_real ridge = 1e-4,
          my_int feat_mode = DESN_FEAT_ALL,
          my_int nonlin = NL_TANH,
          my_real inter_scaling = 0.5) :
        _count(0),
        _steps_since_reset(0),
        _HIDDEN(sh),
        _NLAYERS(nlayers < 1 ? 1 : nlayers),
        _BPTT(mem > 1 ? mem : 2),
        _IN(si),
        _OUT(so),
        _it_mem(_BPTT - 1),
        _spectral_radius(spectral_radius),
        _input_scaling(input_scaling),
        _leaking_rate(leaking_rate),
        _sparsity(sparsity),
        _ridge(ridge),
        _feat_mode(feat_mode),
        _nonlin(nonlin),
        _inter_scaling(inter_scaling),
        _ncollect(0),
        _in2res(_HIDDEN, _IN),
        _inter2res(),
        _res2res(),
        _res2out(1, 1),
        _in(_BPTT, 0),
        _res(),
        _out(_BPTT, Vec(_OUT, 0)),
        _targets(_BPTT, 0),
        _err_out(_OUT, 0),
        _feat(),
        _gram(),
        _B()
        {
          _inter2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
          _res2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
          // _res[layer][time]
          _res = std::vector<std::vector<Vec> >(
              _NLAYERS, std::vector<Vec>(_BPTT, Vec(_HIDDEN, 0)));
          _FEAT = computeFeatSize();
          _res2out = Linear(_FEAT, _OUT);
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
          this->initializeReservoir();
        }

      my_int computeFeatSize() const {
        my_int f = 1; // bias
        if(_feat_mode == DESN_FEAT_LAST)
          f += _HIDDEN;
        else
          f += _HIDDEN * _NLAYERS;
        if(_feat_mode == DESN_FEAT_ALL_IN)
          f += _IN;
        return f;
      }

      void initializeReservoir()
      {
        // Layer 0 input weights
        for(my_int i = 0; i < _in2res.size(); i++)
          _in2res._data[i] = random(-_input_scaling, _input_scaling);

        for(my_int l = 0; l < _NLAYERS; l++){
          // Inter-layer (from previous layer state). Layer 0 uses _in2res instead.
          if(l > 0){
            for(my_int i = 0; i < _inter2res[l].size(); i++)
              _inter2res[l]._data[i] = random(-_inter_scaling, _inter_scaling);
          } else {
            _inter2res[l].zeros();
          }

          // Recurrent weights + spectral radius scaling
          sparse_random_init(_res2res[l]._data, _sparsity);
          my_real rho = estimate_spectral_radius(_res2res[l]._data);
          if(rho > 1e-12)
            scale_matrix(_res2res[l]._data, _spectral_radius / rho);
          else
            _res2res[l].zeros();
        }

        _res2out.zeros();
        clearCollectors();
        resetState();
      }

      void clearCollectors()
      {
        _gram.zeros();
        _B.zeros();
        _ncollect = 0;
      }

      void resetState()
      {
        _count = 0;
        _steps_since_reset = 0;
        my_int m = _it_mem;
        for(my_int l = 0; l < _NLAYERS; l++)
          _res[l][m].zeros();
      }

      void emptyStacks() { resetState(); }

      void applyNonlin(Vec& v)
      {
        if(_nonlin == NL_SIGMOID)
          Sigmoid::forward(v);
        else
          Tanh::forward(v);
      }

      void buildFeatures()
      {
        _feat.zeros();
        my_int off = 0;
        if(_feat_mode == DESN_FEAT_LAST){
          for(my_int i = 0; i < _HIDDEN; i++)
            _feat[off++] = _res[_NLAYERS - 1][_it_mem][i];
        } else {
          for(my_int l = 0; l < _NLAYERS; l++)
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _res[l][_it_mem][i];
        }
        _feat[off++] = 1.0; // bias

        if(_feat_mode == DESN_FEAT_ALL_IN){
          my_int cur = _in[_it_mem];
          if(cur >= 0 && cur < _IN)
            _feat[off + cur] = 1.0;
        }
      }

      void forward(const my_int& cur, const my_int& target, bool /*ishard*/ = false)
      {
        my_int old_it = _it_mem;
        _it_mem = (_it_mem + 1) % _in.size();

        _out[_it_mem].zeros();
        for(my_int l = 0; l < _NLAYERS; l++)
          _res[l][_it_mem].zeros();

        _targets[_it_mem] = target;
        _in[_it_mem] = cur;

        my_real a = _leaking_rate;
        my_real oma = 1.0 - a;

        for(my_int l = 0; l < _NLAYERS; l++){
          Vec pre(_HIDDEN, 0);

          if(l == 0)
            _in2res.forward_transpose(cur, pre);
          else
            _inter2res[l].forward(_res[l - 1][_it_mem], pre);

          _res2res[l].forward(_res[l][old_it], pre);
          applyNonlin(pre);

          for(my_int i = 0; i < _HIDDEN; i++)
            _res[l][_it_mem][i] = oma * _res[l][old_it][i] + a * pre[i];
        }

        buildFeatures();
        _res2out.forward(_feat, _out[_it_mem]);
        Softmax::forward(_out[_it_mem]);

        _count++;
        _steps_since_reset++;
      }

      void collect(const my_int& cur, const my_int& target, my_int washout = 0)
      {
        forward(cur, target, false);
        if(_steps_since_reset <= washout) return;

        buildFeatures();
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

      bool fitReadout()
      {
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

      void update(const my_real& lr)
      {
        for(my_int i = 0; i < _OUT; i++)
          _err_out[i] = -_out[_it_mem][i];
        _err_out[_targets[_it_mem]] += 1;

        buildFeatures();
        _res2out.resetGradient();
        _res2out.computeGradient(_feat, _err_out);
        _res2out.update(lr);
      }

      my_real eval(const my_int& target) const {
        return _out[_it_mem][target];
      }

      my_int pred() const {
        my_int p = 0;
        my_real pv = _out[_it_mem][0];
        for(my_int i = 1; i < _OUT; i++){
          if(pv < _out[_it_mem][i]){
            p = i;
            pv = _out[_it_mem][i];
          }
        }
        return p;
      }

      void copy(const DESN& other)
      {
        assert(_IN == other._IN);
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);
        assert(_FEAT == other._FEAT);
        assert(_NLAYERS == other._NLAYERS);
        assert(_BPTT == other._BPTT);

        _it_mem = other._it_mem;
        _spectral_radius = other._spectral_radius;
        _input_scaling = other._input_scaling;
        _leaking_rate = other._leaking_rate;
        _sparsity = other._sparsity;
        _ridge = other._ridge;
        _feat_mode = other._feat_mode;
        _nonlin = other._nonlin;
        _inter_scaling = other._inter_scaling;
        _count = other._count;
        _steps_since_reset = other._steps_since_reset;
        _ncollect = other._ncollect;

        _in2res._data = other._in2res._data;
        _res2out._data = other._res2out._data;
        for(my_int l = 0; l < _NLAYERS; l++){
          _inter2res[l]._data = other._inter2res[l]._data;
          _res2res[l]._data = other._res2res[l]._data;
          for(my_int m = 0; m < _BPTT; m++)
            _res[l][m] = other._res[l][m];
        }
        for(my_int m = 0; m < _BPTT; m++){
          _out[m] = other._out[m];
          _in[m] = other._in[m];
          _targets[m] = other._targets[m];
        }
        _gram = other._gram;
        _B = other._B;
      }

      void save(std::string filename)
      {
        FILE* f = fopen(filename.c_str(), "w");
        if(!f) return;
        fprintf(f, "%d %d %d %d %d %d %d %d\n",
            _IN, _HIDDEN, _OUT, _NLAYERS, _BPTT, _FEAT, _feat_mode, _nonlin);
        fprintf(f, "%.10f %.10f %.10f %.10f %.10f %.10f\n",
            _spectral_radius, _input_scaling, _leaking_rate,
            _sparsity, _ridge, _inter_scaling);

        for(my_int i = 0; i < _in2res.size(); i++) fprintf(f, "%f,", _in2res._data[i]);
        for(my_int l = 0; l < _NLAYERS; l++){
          for(my_int i = 0; i < _inter2res[l].size(); i++)
            fprintf(f, "%f,", _inter2res[l]._data[i]);
          for(my_int i = 0; i < _res2res[l].size(); i++)
            fprintf(f, "%f,", _res2res[l]._data[i]);
        }
        for(my_int i = 0; i < _res2out.size(); i++) fprintf(f, "%f,", _res2out._data[i]);
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f = fopen(filename.c_str(), "r");
        if(!f) return;
        fscanf(f, "%d %d %d %d %d %d %d %d\n",
            &_IN, &_HIDDEN, &_OUT, &_NLAYERS, &_BPTT, &_FEAT, &_feat_mode, &_nonlin);
        fscanf(f, "%lf %lf %lf %lf %lf %lf\n",
            &_spectral_radius, &_input_scaling, &_leaking_rate,
            &_sparsity, &_ridge, &_inter_scaling);

        _in2res = Linear(_HIDDEN, _IN);
        _inter2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
        _res2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
        _res2out = Linear(_FEAT, _OUT);
        _res = std::vector<std::vector<Vec> >(
            _NLAYERS, std::vector<Vec>(_BPTT, Vec(_HIDDEN, 0)));

        for(my_int i = 0; i < _in2res.size(); i++) fscanf(f, "%lf,", &_in2res._data[i]);
        for(my_int l = 0; l < _NLAYERS; l++){
          for(my_int i = 0; i < _inter2res[l].size(); i++)
            fscanf(f, "%lf,", &_inter2res[l]._data[i]);
          for(my_int i = 0; i < _res2res[l].size(); i++)
            fscanf(f, "%lf,", &_res2res[l]._data[i]);
        }
        for(my_int i = 0; i < _res2out.size(); i++) fscanf(f, "%lf,", &_res2out._data[i]);
        fclose(f);

        _it_mem = _BPTT - 1;
        _in = std::vector<my_int>(_BPTT, 0);
        _out = std::vector<Vec>(_BPTT, Vec(_OUT, 0));
        _targets = std::vector<my_int>(_BPTT, 0);
        _err_out = Vec(_OUT, 0);
        _feat = Vec(_FEAT, 0);
        _gram = Vec2D(_FEAT, _FEAT, 0);
        _B = Vec2D(_FEAT, _OUT, 0);
        _count = 0;
        _steps_since_reset = 0;
        _ncollect = 0;
      }

      my_int _count;
      my_int _steps_since_reset;
      my_int _HIDDEN;
      my_int _NLAYERS;
      my_int _BPTT;
      my_int _IN;
      my_int _OUT;
      my_int _FEAT;
      my_int _it_mem;
      my_real _spectral_radius;
      my_real _input_scaling;
      my_real _leaking_rate;
      my_real _sparsity;
      my_real _ridge;
      my_int _feat_mode;
      my_int _nonlin;
      my_real _inter_scaling;
      my_int _ncollect;

      Linear _in2res;
      std::vector<Linear> _inter2res;
      std::vector<Linear> _res2res;
      Linear _res2out;

      std::vector<my_int> _in;
      std::vector<std::vector<Vec> > _res; // [layer][time]
      std::vector<Vec> _out;
      std::vector<my_int> _targets;

      Vec _err_out;
      Vec _feat;
      Vec2D _gram;
      Vec2D _B;
  };

} // namespace rnn
#endif
