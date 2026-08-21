/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#ifndef _ESN_
#define _ESN_
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

#ifndef EMPTY_STACK_VALUE
#define EMPTY_STACK_VALUE -1
#endif

namespace rnn
{

  // Feature modes for readout phi(x)
  enum ESNFeat {
    FEAT_RES = 0,
    FEAT_RES_IN = 1,
    FEAT_RES_STACK = 2,
    FEAT_RES_IN_STACK = 3
  };

  // Nonlinearity choice for reservoir
  enum ESNNonlin {
    NL_TANH = 0,
    NL_SIGMOID = 1
  };

#ifndef RNN_STACK_ACTIONS_DEFINED
#define RNN_STACK_ACTIONS_DEFINED
  enum {push, pop, noop};
#endif

  struct ESN
  {
    public:

      ESN(const std::string& filename)
      {
        load(filename);
        resetState();
      }

      ESN(my_int si,
          my_int sh,
          my_int so,
          my_int nstack = 0,
          my_int stack_capacity = 200,
          my_int mem = 50,
          my_int mod = 2,
          bool isnoop = false,
          my_int depth = 1,
          my_real spectral_radius = 0.9,
          my_real input_scaling = 0.5,
          my_real leaking_rate = 1.0,
          my_real sparsity = 0.1,
          my_real ridge = 1e-4,
          my_int feat_mode = FEAT_RES,
          my_int nonlin = NL_TANH) :
        _count(0),
        _steps_since_reset(0),
        _HIDDEN(sh),
        _NB_STACK(nstack),
        _STACK_SIZE(stack_capacity),
        _ACTION(2 + ((isnoop) ? 1 : 0)),
        _TOP_OF_STACK(0),
        _BPTT(mem > 1 ? mem : 2),
        _IN(si),
        _OUT(so),
        _it_mem(_BPTT - 1),
        _mod(mod),
        _DEPTH(depth),
        _spectral_radius(spectral_radius),
        _input_scaling(input_scaling),
        _leaking_rate(leaking_rate),
        _sparsity(sparsity),
        _ridge(ridge),
        _feat_mode(feat_mode),
        _nonlin(nonlin),
        _ncollect(0),
        _in2res(_HIDDEN, _IN),
        _res2res(_HIDDEN, _HIDDEN),
        _hid2act(_NB_STACK, Linear(_HIDDEN, _ACTION)),
        _hid2stack(_NB_STACK, Linear(_HIDDEN, _STACK_SIZE)),
        _stack2hid(_NB_STACK, Linear(_STACK_SIZE, _HIDDEN)),
        _res2out(1, 1), // resized after feat size known
        _in(_BPTT, 0),
        _res(_BPTT, Vec(_HIDDEN, 0)),
        _act(_NB_STACK, std::vector<Vec>(_BPTT, Vec(_ACTION, 0))),
        _stack(_NB_STACK, std::vector<Vec>(_BPTT, Vec(_STACK_SIZE, 0))),
        _out(_BPTT, Vec(_OUT, 0)),
        _targets(_BPTT, 0),
        _err_out(_OUT, 0),
        _feat(),
        _isemptied(_BPTT, false),
        _gram(),
        _B()
        {
          _FEAT = computeFeatSize();
          // Linear(si, so) stores matrix (so x si) = (_OUT x _FEAT)
          _res2out = Linear(_FEAT, _OUT);
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
          this->initializeReservoir();
        }

      my_int computeFeatSize() const {
        // +1 for constant bias feature (standard ESN readout)
        my_int f = _HIDDEN + 1;
        if(_feat_mode == FEAT_RES_IN || _feat_mode == FEAT_RES_IN_STACK)
          f += _IN;
        if(_feat_mode == FEAT_RES_STACK || _feat_mode == FEAT_RES_IN_STACK)
          f += _NB_STACK * _DEPTH;
        return f;
      }

      void initializeReservoir()
      {
        // Input weights: dense random, scaled by input_scaling
        for(my_int i = 0; i < _in2res.size(); i++)
          _in2res._data[i] = random(-_input_scaling, _input_scaling);

        // Recurrent reservoir: sparse + spectral radius scaling (mod 2 / full recurrence)
        if(_mod == 2 || _NB_STACK == 0){
          sparse_random_init(_res2res._data, _sparsity);
          my_real rho = estimate_spectral_radius(_res2res._data);
          if(rho > 1e-12)
            scale_matrix(_res2res._data, _spectral_radius / rho);
          else
            _res2res.zeros();
        } else {
          // stack-only recurrence (mod 0/1): no dense W_res
          _res2res.zeros();
        }

        // Stack-related maps: random then freeze unused depth rows/cols like StackRNN
        for(my_int s = 0; s < _NB_STACK; s++){
          _hid2act[s].initialize();
          _hid2stack[s].initialize();
          _stack2hid[s].initialize();
          for(my_int j = 0; j < _HIDDEN; j++)
            for(my_int i = _TOP_OF_STACK + _DEPTH; i < _TOP_OF_STACK + _STACK_SIZE; i++)
              _stack2hid[s]._data(j, i) = 0;
          for(my_int i = _TOP_OF_STACK + 1; i < _TOP_OF_STACK + _STACK_SIZE; i++)
            for(my_int j = 0; j < _HIDDEN; j++)
              _hid2stack[s]._data(i, j) = 0;
        }

        // Readout starts at zero
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
        _isemptied[m] = true;
        _res[m].zeros();
        for(my_int s = 0; s < _NB_STACK; s++){
          _act[s][m].zeros();
          for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE; i++)
            _stack[s][m][i] = EMPTY_STACK_VALUE;
        }
      }

      void emptyStacks()
      {
        resetState();
      }

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
        for(my_int i = 0; i < _HIDDEN; i++)
          _feat[off++] = _res[_it_mem][i];
        _feat[off++] = 1.0; // bias

        if(_feat_mode == FEAT_RES_IN || _feat_mode == FEAT_RES_IN_STACK){
          // one-hot current input
          my_int cur = _in[_it_mem];
          if(cur >= 0 && cur < _IN)
            _feat[off + cur] = 1.0;
          off += _IN;
        }

        if(_feat_mode == FEAT_RES_STACK || _feat_mode == FEAT_RES_IN_STACK){
          for(my_int s = 0; s < _NB_STACK; s++){
            for(my_int d = 0; d < _DEPTH; d++)
              _feat[off++] = _stack[s][_it_mem][_TOP_OF_STACK + d];
          }
        }
      }

      void forward(const my_int& cur, const my_int& target, bool ishard = false)
      {
        my_int old_it = _it_mem;
        _it_mem = (_it_mem + 1) % _in.size();
        _isemptied[_it_mem] = false;

        _out[_it_mem].zeros();
        _res[_it_mem].zeros();
        for(my_int s = 0; s < _NB_STACK; s++){
          _act[s][_it_mem].zeros();
          _stack[s][_it_mem].zeros();
        }

        _targets[_it_mem] = target;
        _in[_it_mem] = cur;

        // pre-activation buffer
        Vec pre(_HIDDEN, 0);

        // W_in * u(t) via column of transpose-style embedding
        _in2res.forward_transpose(cur, pre);

        // recurrent: W_res * x(t-1) when mod==2 or no stacks
        if(_mod == 2 || _NB_STACK == 0){
          _res2res.forward(_res[old_it], pre);
        }

        // stack tops (t-1) -> reservoir when mod != 0
        if(_mod != 0 && _NB_STACK > 0){
          for(my_int s = 0; s < _NB_STACK; s++){
            _stack2hid[s].forward(_stack[s][old_it], pre,
                _TOP_OF_STACK, _TOP_OF_STACK + _DEPTH, 0, _HIDDEN);
          }
        }

        applyNonlin(pre);

        // leaking integration: x(t) = (1-a) x(t-1) + a f(...)
        my_real a = _leaking_rate;
        my_real oma = 1.0 - a;
        for(my_int i = 0; i < _HIDDEN; i++)
          _res[_it_mem][i] = oma * _res[old_it][i] + a * pre[i];

        // stack dynamics (frozen random controllers)
        for(my_int s = 0; s < _NB_STACK; s++){
          _hid2act[s].forward(_res[_it_mem], _act[s][_it_mem]);
          Softmax::forward(_act[s][_it_mem]);
          if(ishard){
            my_int im = 0; my_real pm = _act[s][_it_mem][0];
            _act[s][_it_mem][0] = 0;
            for(my_int i = 1; i < _ACTION; i++){
              if(pm < _act[s][_it_mem][i]){
                im = i;
                pm = _act[s][_it_mem][i];
              }
              _act[s][_it_mem][i] = 0;
            }
            _act[s][_it_mem][im] = 1;
          }

          my_real pop_weight = _act[s][_it_mem][pop];
          my_real push_weight = _act[s][_it_mem][push];

          for(my_int i = _TOP_OF_STACK + 1; i < _STACK_SIZE; i++)
            _stack[s][_it_mem][i] += _stack[s][old_it][i - 1] * push_weight;

          _stack[s][_it_mem][_TOP_OF_STACK] = 0;
          for(my_int i = 0; i < _HIDDEN; i++)
            _stack[s][_it_mem][_TOP_OF_STACK] +=
              _hid2stack[s]._data(_TOP_OF_STACK, i) * _res[_it_mem][i];
          if(_stack[s][_it_mem][_TOP_OF_STACK] > 50)
            _stack[s][_it_mem][_TOP_OF_STACK] = 50;
          if(_stack[s][_it_mem][_TOP_OF_STACK] < -50)
            _stack[s][_it_mem][_TOP_OF_STACK] = -50;
          _stack[s][_it_mem][_TOP_OF_STACK] =
            1 / (1 + exp(-_stack[s][_it_mem][_TOP_OF_STACK]));
          _stack[s][_it_mem][_TOP_OF_STACK] *= push_weight;

          for(my_int i = _TOP_OF_STACK; i < _STACK_SIZE - 1; i++)
            _stack[s][_it_mem][i] += _stack[s][old_it][i + 1] * pop_weight;
          _stack[s][_it_mem][_STACK_SIZE - 1] += EMPTY_STACK_VALUE * pop_weight;

          if(_ACTION == 3){
            my_real noop_weight = _act[s][_it_mem][noop];
            for(my_int i = _TOP_OF_STACK; i < _TOP_OF_STACK + _STACK_SIZE; i++)
              _stack[s][_it_mem][i] += _stack[s][old_it][i] * noop_weight;
          }
        }

        // readout
        buildFeatures();
        _res2out.forward(_feat, _out[_it_mem]);
        Softmax::forward(_out[_it_mem]);

        _count++;
        _steps_since_reset++;
      }

      // Forward and accumulate Gram matrix G = Phi^T Phi and B = Phi^T Y
      void collect(const my_int& cur, const my_int& target, my_int washout = 0)
      {
        forward(cur, target, false);
        if(_steps_since_reset <= washout) return;

        buildFeatures(); // features already match current state
        // G += phi * phi^T
        for(my_int i = 0; i < _FEAT; i++){
          my_real fi = _feat[i];
          if(fi == 0) continue;
          for(my_int j = 0; j < _FEAT; j++)
            _gram(i, j) += fi * _feat[j];
        }
        // B += phi * onehot(target)^T  stored as (feat x out)
        if(target >= 0 && target < _OUT){
          for(my_int i = 0; i < _FEAT; i++)
            _B(i, target) += _feat[i];
        }
        _ncollect++;
      }

      bool fitReadout()
      {
        if(_ncollect == 0) return false;

        // Solve (G + lambda I) W^T = B  for each output column
        // W is (_OUT x _FEAT) in Linear layout: _res2out._data(o, f)
        Vec2D A(_FEAT, _FEAT, 0);
        Vec bcol(_FEAT, 0);
        Vec xcol(_FEAT, 0);

        for(my_int o = 0; o < _OUT; o++){
          // copy G + lambda I
          for(my_int i = 0; i < _FEAT; i++){
            for(my_int j = 0; j < _FEAT; j++)
              A(i, j) = _gram(i, j);
            A(i, i) += _ridge;
            bcol[i] = _B(i, o);
          }
          if(!solve_linear_system(A, bcol, xcol)){
            // fallback: keep previous / use pseudo via diagonal only
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

      // Online SGD on readout only (softmax cross-entropy)
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

      void copy(const ESN& other)
      {
        assert(_IN == other._IN);
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);
        assert(_FEAT == other._FEAT);
        assert(_NB_STACK == other._NB_STACK);
        assert(_STACK_SIZE == other._STACK_SIZE);
        assert(_ACTION == other._ACTION);
        assert(_DEPTH == other._DEPTH);
        assert(_BPTT == other._BPTT);

        _it_mem = other._it_mem;
        _mod = other._mod;
        _spectral_radius = other._spectral_radius;
        _input_scaling = other._input_scaling;
        _leaking_rate = other._leaking_rate;
        _sparsity = other._sparsity;
        _ridge = other._ridge;
        _feat_mode = other._feat_mode;
        _nonlin = other._nonlin;
        _count = other._count;
        _steps_since_reset = other._steps_since_reset;
        _ncollect = other._ncollect;

        _in2res._data = other._in2res._data;
        _res2res._data = other._res2res._data;
        _res2out._data = other._res2out._data;

        for(my_int s = 0; s < _NB_STACK; s++){
          _hid2act[s]._data = other._hid2act[s]._data;
          _hid2stack[s]._data = other._hid2stack[s]._data;
          _stack2hid[s]._data = other._stack2hid[s]._data;
          for(my_int m = 0; m < _BPTT; m++){
            _act[s][m] = other._act[s][m];
            _stack[s][m] = other._stack[s][m];
          }
        }

        for(my_int m = 0; m < _BPTT; m++){
          _out[m] = other._out[m];
          _in[m] = other._in[m];
          _res[m] = other._res[m];
          _targets[m] = other._targets[m];
          _isemptied[m] = other._isemptied[m];
        }

        _gram = other._gram;
        _B = other._B;
      }

      void save(std::string filename)
      {
        FILE* f = fopen(filename.c_str(), "w");
        if(!f) return;
        fprintf(f, "%d %d %d %d %d %d %d %d %d %d %d %d %d\n",
            _IN, _ACTION, _HIDDEN, _NB_STACK, _STACK_SIZE, _OUT,
            _BPTT, _mod, _DEPTH, _FEAT, _feat_mode, _nonlin, (int)_ncollect);
        fprintf(f, "%.10f %.10f %.10f %.10f %.10f\n",
            _spectral_radius, _input_scaling, _leaking_rate, _sparsity, _ridge);

        for(my_int i = 0; i < _in2res.size(); i++) fprintf(f, "%f,", _in2res._data[i]);
        for(my_int i = 0; i < _res2res.size(); i++) fprintf(f, "%f,", _res2res._data[i]);
        for(my_int s = 0; s < _NB_STACK; s++){
          for(my_int i = 0; i < _hid2act[s].size(); i++) fprintf(f, "%f,", _hid2act[s]._data[i]);
          for(my_int i = 0; i < _hid2stack[s].size(); i++) fprintf(f, "%f,", _hid2stack[s]._data[i]);
          for(my_int i = 0; i < _stack2hid[s].size(); i++) fprintf(f, "%f,", _stack2hid[s]._data[i]);
        }
        for(my_int i = 0; i < _res2out.size(); i++) fprintf(f, "%f,", _res2out._data[i]);
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f = fopen(filename.c_str(), "r");
        if(!f) return;
        int ncollect_tmp = 0;
        fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d %d %d\n",
            &_IN, &_ACTION, &_HIDDEN, &_NB_STACK, &_STACK_SIZE, &_OUT,
            &_BPTT, &_mod, &_DEPTH, &_FEAT, &_feat_mode, &_nonlin, &ncollect_tmp);
        fscanf(f, "%lf %lf %lf %lf %lf\n",
            &_spectral_radius, &_input_scaling, &_leaking_rate, &_sparsity, &_ridge);

        _TOP_OF_STACK = 0;
        _ncollect = ncollect_tmp;
        _in2res = Linear(_HIDDEN, _IN);
        _res2res = Linear(_HIDDEN, _HIDDEN);
        _hid2act = std::vector<Linear>(_NB_STACK, Linear(_HIDDEN, _ACTION));
        _hid2stack = std::vector<Linear>(_NB_STACK, Linear(_HIDDEN, _STACK_SIZE));
        _stack2hid = std::vector<Linear>(_NB_STACK, Linear(_STACK_SIZE, _HIDDEN));
        _res2out = Linear(_FEAT, _OUT);

        for(my_int i = 0; i < _in2res.size(); i++) fscanf(f, "%lf,", &_in2res._data[i]);
        for(my_int i = 0; i < _res2res.size(); i++) fscanf(f, "%lf,", &_res2res._data[i]);
        for(my_int s = 0; s < _NB_STACK; s++){
          for(my_int i = 0; i < _hid2act[s].size(); i++) fscanf(f, "%lf,", &_hid2act[s]._data[i]);
          for(my_int i = 0; i < _hid2stack[s].size(); i++) fscanf(f, "%lf,", &_hid2stack[s]._data[i]);
          for(my_int i = 0; i < _stack2hid[s].size(); i++) fscanf(f, "%lf,", &_stack2hid[s]._data[i]);
        }
        for(my_int i = 0; i < _res2out.size(); i++) fscanf(f, "%lf,", &_res2out._data[i]);
        fclose(f);

        _isemptied = std::vector<bool>(_BPTT, false);
        _it_mem = _BPTT - 1;
        _in = std::vector<my_int>(_BPTT, 0);
        _res = std::vector<Vec>(_BPTT, Vec(_HIDDEN, 0));
        _act = std::vector<std::vector<Vec> >(
            _NB_STACK, std::vector<Vec>(_BPTT, Vec(_ACTION, 0)));
        _stack = std::vector<std::vector<Vec> >(
            _NB_STACK, std::vector<Vec>(_BPTT, Vec(_STACK_SIZE, 0)));
        _out = std::vector<Vec>(_BPTT, Vec(_OUT, 0));
        _targets = std::vector<my_int>(_BPTT, 0);
        _err_out = Vec(_OUT, 0);
        _feat = Vec(_FEAT, 0);
        _gram = Vec2D(_FEAT, _FEAT, 0);
        _B = Vec2D(_FEAT, _OUT, 0);
        _count = 0;
        _steps_since_reset = 0;
      }

      // Public fields (trainer / logging access, StackRNN style)
      my_int _count;
      my_int _steps_since_reset;
      my_int _HIDDEN;
      my_int _NB_STACK;
      my_int _STACK_SIZE;
      my_int _ACTION;
      my_int _TOP_OF_STACK;
      my_int _BPTT;
      my_int _IN;
      my_int _OUT;
      my_int _FEAT;
      my_int _it_mem;
      my_int _mod;
      my_int _DEPTH;
      my_real _spectral_radius;
      my_real _input_scaling;
      my_real _leaking_rate;
      my_real _sparsity;
      my_real _ridge;
      my_int _feat_mode;
      my_int _nonlin;
      my_int _ncollect;

      Linear _in2res;
      Linear _res2res;
      std::vector<Linear> _hid2act;
      std::vector<Linear> _hid2stack;
      std::vector<Linear> _stack2hid;
      Linear _res2out;

      std::vector<my_int> _in;
      std::vector<Vec> _res;
      std::vector<std::vector<Vec> > _act;
      std::vector<std::vector<Vec> > _stack;
      std::vector<Vec> _out;
      std::vector<my_int> _targets;

      Vec _err_out;
      Vec _feat;
      std::vector<bool> _isemptied;

      Vec2D _gram;
      Vec2D _B;
  };

} // namespace rnn
#endif
