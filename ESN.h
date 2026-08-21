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
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "common.h"
#include "Vec.h"
#include "Linear.h"
#include "Nonlinearity.h"
#include "utils.h"

#define ESN_EMPTY_STACK_VALUE -1

namespace rnn {

  // Local action indices (same order as StackRNN: push, pop, noop).
  enum ESNActionIdx { ESN_PUSH = 0, ESN_POP = 1, ESN_NOOP = 2 };

  // Echo-State Network with optional frozen stack controllers (Stack-ESN B1).
  // Reservoir and input weights are fixed after initialization; only the
  // readout is trained (ridge regression and/or SGD).
  struct ESN
  {
    public:

      enum ActType { ACT_TANH = 0, ACT_SIGMOID = 1 };

      ESN(const std::string& filename)
      {
        load(filename);
        reset();
      }

      ESN(my_int si,
          my_int sh,
          my_int nstack,
          my_int stack_capacity,
          my_int so,
          my_real spectral_radius = 0.9,
          my_real input_scaling = 0.5,
          my_real sparsity = 0.1,
          my_real leak_rate = 1.0,
          my_int washout = 0,
          my_real ridge_lambda = 1e-6,
          bool isnoop = false,
          my_int depth = 1,
          my_int mod = 1,
          my_int act_type = ACT_TANH,
          bool use_bias = true) :
        _spectral_radius(spectral_radius),
        _input_scaling(input_scaling),
        _sparsity(sparsity),
        _leak_rate(leak_rate),
        _washout(washout),
        _ridge_lambda(ridge_lambda),
        _act_type(act_type),
        _use_bias(use_bias),
        _step_in_seq(0),
        _HIDDEN(sh),
        _NB_STACK(nstack),
        _STACK_SIZE(stack_capacity > 0 ? stack_capacity : 1),
        _ACTION(2 + (isnoop ? 1 : 0)),
        _TOP_OF_STACK(0),
        _IN(si),
        _OUT(so),
        _mod(mod),
        _DEPTH(depth < 1 ? 1 : depth),
        _feat_dim(0),
        _W_in(sh, si),
        _W_res(sh, sh, 0),
        _bias(sh, 0),
        _hid2act(nstack, Linear(sh, 2 + (isnoop ? 1 : 0))),
        _hid2stack(nstack, Linear(sh, stack_capacity > 0 ? stack_capacity : 1)),
        _stack2hid(nstack, Linear(stack_capacity > 0 ? stack_capacity : 1, sh)),
        _x(sh, 0),
        _x_prev(sh, 0),
        _pre(sh, 0),
        _act(nstack, Vec(2 + (isnoop ? 1 : 0), 0)),
        _stack(nstack, Vec(stack_capacity > 0 ? stack_capacity : 1, 0)),
        _stack_prev(nstack, Vec(stack_capacity > 0 ? stack_capacity : 1, 0)),
        _out(so, 0),
        _features(),
        _hid2out(),
        _err_out(so, 0),
        _target(0),
        _collected_states(),
        _collected_targets()
      {
        _feat_dim = _HIDDEN + (_NB_STACK > 0 ? _NB_STACK * _DEPTH : 0) + (_use_bias ? 1 : 0);
        _features = Vec(_feat_dim, 0);
        _hid2out = Linear(_feat_dim, _OUT);
        initialize();
      }

      my_int featureDim() const { return _feat_dim; }

      void initialize()
      {
        // Fixed random input weights.
        for(my_int i = 0; i < _W_in.size(); i++)
          _W_in._data[i] = random(-_input_scaling, _input_scaling);

        // Sparse random reservoir, then scale to target spectral radius.
        for(my_int i = 0; i < _HIDDEN; i++){
          for(my_int j = 0; j < _HIDDEN; j++){
            if(drand() <= _sparsity)
              _W_res(i, j) = random_normal();
            else
              _W_res(i, j) = 0;
          }
        }
        my_real rho = power_iteration_spectral_radius(_W_res);
        if(rho > 1e-12)
          scale_matrix(_W_res, _spectral_radius / rho);
        else{
          // Degenerate empty reservoir: put small diagonal recurrence.
          for(my_int i = 0; i < _HIDDEN; i++)
            _W_res(i, i) = _spectral_radius;
        }

        for(my_int i = 0; i < _bias.size(); i++)
          _bias[i] = random(-_input_scaling, _input_scaling);

        // Frozen random stack controllers (Stack-ESN B1).
        for(my_int s = 0; s < _NB_STACK; s++){
          _hid2act[s].initialize();
          _hid2stack[s].initialize();
          _stack2hid[s].initialize();
          // Only top-_DEPTH stack cells couple into the reservoir / readout.
          for(my_int j = 0; j < _HIDDEN; j++)
            for(my_int i = _TOP_OF_STACK + _DEPTH; i < _STACK_SIZE; i++)
              _stack2hid[s]._data(j, i) = 0;
          for(my_int i = _TOP_OF_STACK + 1; i < _STACK_SIZE; i++)
            for(my_int j = 0; j < _HIDDEN; j++)
              _hid2stack[s]._data(i, j) = 0;
        }

        // Trainable readout starts at zero.
        _hid2out.zeros();
        reset();
        clearCollected();
      }

      void reset()
      {
        _x.zeros();
        _x_prev.zeros();
        _pre.zeros();
        _out.zeros();
        _features.zeros();
        _step_in_seq = 0;
        emptyStacks();
      }

      void emptyStacks()
      {
        for(my_int s = 0; s < _NB_STACK; s++){
          _act[s].zeros();
          for(my_int i = 0; i < _STACK_SIZE; i++){
            _stack[s][i] = ESN_EMPTY_STACK_VALUE;
            _stack_prev[s][i] = ESN_EMPTY_STACK_VALUE;
          }
        }
      }

      // Build extended readout feature vector [x ; stack_tops ; bias].
      void buildFeatures()
      {
        my_int k = 0;
        for(my_int i = 0; i < _HIDDEN; i++)
          _features[k++] = _x[i];
        for(my_int s = 0; s < _NB_STACK; s++){
          for(my_int d = 0; d < _DEPTH; d++){
            my_int idx = _TOP_OF_STACK + d;
            _features[k++] = (idx < _STACK_SIZE) ? _stack[s][idx] : 0;
          }
        }
        if(_use_bias) _features[k++] = 1.0;
        assert(k == _feat_dim);
      }

      const Vec& features() const { return _features; }

      void collectState()
      {
        if(_step_in_seq <= _washout) return;
        buildFeatures();
        _collected_states.push_back(_features);
        _collected_targets.push_back(_target);
      }

      // Always record the current (state, target), ignoring washout.
      // Useful when the caller already applies washout externally.
      void collectStateForce()
      {
        buildFeatures();
        _collected_states.push_back(_features);
        _collected_targets.push_back(_target);
      }

      void clearCollected()
      {
        _collected_states.clear();
        _collected_targets.clear();
      }

      my_int numCollected() const { return (my_int)_collected_states.size(); }

      // Closed-form ridge regression on collected (state, target) pairs.
      // Targets are class indices; solved against one-hot labels (MSE on one-hot).
      bool fitReadoutRidge()
      {
        my_int N = (my_int)_collected_states.size();
        if(N == 0) return false;
        my_int F = _feat_dim;
        my_int O = _OUT;

        // A = X^T X + λ I  (F x F)
        // B = X^T T        (F x O)
        Vec2D A(F, F, 0);
        Vec2D B(F, O, 0);

        for(my_int n = 0; n < N; n++){
          const Vec& x = _collected_states[n];
          my_int t = _collected_targets[n];
          assert(t >= 0 && t < O);
          for(my_int i = 0; i < F; i++){
            for(my_int j = 0; j < F; j++)
              A(i, j) += x[i] * x[j];
            B(i, t) += x[i];
          }
        }
        for(my_int i = 0; i < F; i++)
          A(i, i) += _ridge_lambda;

        // Solve A W_col = B_col for each output column; store as Linear (O x F).
        _hid2out.zeros();
        for(my_int o = 0; o < O; o++){
          Vec2D Acopy(F, F, 0);
          for(my_int i = 0; i < F; i++)
            for(my_int j = 0; j < F; j++)
              Acopy(i, j) = A(i, j);
          Vec b(F, 0), w(F, 0);
          for(my_int i = 0; i < F; i++) b[i] = B(i, o);
          if(!solve_linear_system(Acopy, b, w)){
            // Fall back: keep previous readout (zeros if first fit).
            return false;
          }
          for(my_int i = 0; i < F; i++)
            _hid2out._data(o, i) = w[i];
        }
        return true;
      }

      bool fitReadoutRidge(const std::vector<Vec>& states,
                           const std::vector<my_int>& targets)
      {
        _collected_states = states;
        _collected_targets = targets;
        return fitReadoutRidge();
      }

      void applyActivation(Vec& v)
      {
        if(_act_type == ACT_SIGMOID) Sigmoid::forward(v);
        else Tanh::forward(v);
      }

      void forward(const my_int& cur, const my_int& target, bool ishard = false)
      {
        assert(cur >= 0 && cur < _IN);
        _target = target;
        _step_in_seq++;

        // Save previous reservoir and stacks.
        for(my_int i = 0; i < _HIDDEN; i++) _x_prev[i] = _x[i];
        for(my_int s = 0; s < _NB_STACK; s++)
          for(my_int i = 0; i < _STACK_SIZE; i++)
            _stack_prev[s][i] = _stack[s][i];

        // pre = W_in u + W_res x_prev + bias + stack tops
        _pre.zeros();
        _W_in.forward_transpose(cur, _pre);
        // W_res * x_prev
        for(my_int i = 0; i < _HIDDEN; i++){
          my_real s = 0;
          for(my_int j = 0; j < _HIDDEN; j++)
            s += _W_res(i, j) * _x_prev[j];
          _pre[i] += s;
          _pre[i] += _bias[i];
        }

        // Optional stack tops -> reservoir (mod != 0), matching StackRNN.
        if(_mod != 0 && _NB_STACK > 0){
          for(my_int s = 0; s < _NB_STACK; s++){
            _stack2hid[s].forward(_stack_prev[s], _pre,
                _TOP_OF_STACK, _TOP_OF_STACK + _DEPTH, 0, _HIDDEN);
          }
        }

        applyActivation(_pre);

        // Leaky integration: x = (1-α) x_prev + α f(pre)
        my_real a = _leak_rate;
        my_real oma = 1.0 - a;
        for(my_int i = 0; i < _HIDDEN; i++)
          _x[i] = oma * _x_prev[i] + a * _pre[i];

        // Stack controller driven by current reservoir (frozen weights).
        for(my_int s = 0; s < _NB_STACK; s++){
          _act[s].zeros();
          _stack[s].zeros();
          _hid2act[s].forward(_x, _act[s]);
          Softmax::forward(_act[s]);
          if(ishard){
            my_int im = 0; my_real pm = _act[s][0];
            _act[s][0] = 0;
            for(my_int i = 1; i < _ACTION; i++){
              if(pm < _act[s][i]){ im = i; pm = _act[s][i]; }
              _act[s][i] = 0;
            }
            _act[s][im] = 1;
          }

          my_real pop_weight = _act[s][ESN_POP];
          my_real push_weight = _act[s][ESN_PUSH];

          // Push: shift down, write top from reservoir.
          for(my_int i = _TOP_OF_STACK + 1; i < _STACK_SIZE; i++)
            _stack[s][i] += _stack_prev[s][i - 1] * push_weight;

          _stack[s][_TOP_OF_STACK] = 0;
          for(my_int i = 0; i < _HIDDEN; i++)
            _stack[s][_TOP_OF_STACK] += _hid2stack[s]._data(_TOP_OF_STACK, i) * _x[i];
          if(_stack[s][_TOP_OF_STACK] > 50) _stack[s][_TOP_OF_STACK] = 50;
          if(_stack[s][_TOP_OF_STACK] < -50) _stack[s][_TOP_OF_STACK] = -50;
          _stack[s][_TOP_OF_STACK] = 1.0 / (1.0 + exp(-_stack[s][_TOP_OF_STACK]));
          _stack[s][_TOP_OF_STACK] *= push_weight;

          // Pop: shift up.
          for(my_int i = _TOP_OF_STACK; i < _STACK_SIZE - 1; i++)
            _stack[s][i] += _stack_prev[s][i + 1] * pop_weight;
          _stack[s][_STACK_SIZE - 1] += ESN_EMPTY_STACK_VALUE * pop_weight;

          // No-op.
          if(_ACTION == 3){
            my_real noop_weight = _act[s][ESN_NOOP];
            for(my_int i = _TOP_OF_STACK; i < _STACK_SIZE; i++)
              _stack[s][i] += _stack_prev[s][i] * noop_weight;
          }
        }

        // Readout.
        buildFeatures();
        _out.zeros();
        _hid2out.forward(_features, _out);
        Softmax::forward(_out);
      }

      // SGD on readout only (softmax NLL).
      void backwardReadoutOnly()
      {
        for(my_int i = 0; i < _OUT; i++) _err_out[i] = -_out[i];
        _err_out[_target] += 1;
        _hid2out.resetGradient();
        _hid2out.computeGradient(_features, _err_out);
      }

      void updateReadout(const my_real& lr)
      {
        _hid2out.update(lr);
      }

      my_real eval(const my_int& target) const {
        return _out[target];
      }

      my_int pred() const {
        my_int p = 0;
        my_real pv = _out[0];
        for(my_int i = 1; i < _OUT; i++){
          if(pv < _out[i]){ p = i; pv = _out[i]; }
        }
        return p;
      }

      void copy(const ESN& other)
      {
        assert(_IN == other._IN);
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);
        assert(_NB_STACK == other._NB_STACK);
        assert(_STACK_SIZE == other._STACK_SIZE);
        assert(_DEPTH == other._DEPTH);
        assert(_feat_dim == other._feat_dim);

        _spectral_radius = other._spectral_radius;
        _input_scaling = other._input_scaling;
        _sparsity = other._sparsity;
        _leak_rate = other._leak_rate;
        _washout = other._washout;
        _ridge_lambda = other._ridge_lambda;
        _act_type = other._act_type;
        _use_bias = other._use_bias;
        _mod = other._mod;
        _ACTION = other._ACTION;
        _step_in_seq = other._step_in_seq;
        _target = other._target;

        _W_in._data = other._W_in._data;
        _W_res = other._W_res;
        // copy bias
        for(my_int i = 0; i < _bias.size(); i++) _bias[i] = other._bias[i];
        _hid2out._data = other._hid2out._data;

        for(my_int s = 0; s < _NB_STACK; s++){
          _hid2act[s]._data = other._hid2act[s]._data;
          _hid2stack[s]._data = other._hid2stack[s]._data;
          _stack2hid[s]._data = other._stack2hid[s]._data;
          _act[s] = other._act[s];
          _stack[s] = other._stack[s];
          _stack_prev[s] = other._stack_prev[s];
        }
        _x = other._x;
        _x_prev = other._x_prev;
        _pre = other._pre;
        _out = other._out;
        _features = other._features;
      }

      void save(std::string filename)
      {
        FILE* f = fopen(filename.c_str(), "w");
        fprintf(f, "%d %d %d %d %d %d %d %d %d %d %d\n",
            _IN, _ACTION, _HIDDEN, _NB_STACK, _STACK_SIZE, _OUT,
            _mod, _DEPTH, _act_type, (int)_use_bias, _feat_dim);
        fprintf(f, "%.10f %.10f %.10f %.10f %d %.10f\n",
            _spectral_radius, _input_scaling, _sparsity, _leak_rate,
            _washout, _ridge_lambda);
        for(my_int i = 0; i < _W_in.size(); i++) fprintf(f, "%f,", _W_in._data[i]);
        fprintf(f, "\n");
        for(my_int i = 0; i < _W_res.size(); i++) fprintf(f, "%f,", _W_res[i]);
        fprintf(f, "\n");
        for(my_int i = 0; i < _bias.size(); i++) fprintf(f, "%f,", _bias[i]);
        fprintf(f, "\n");
        for(my_int s = 0; s < _NB_STACK; s++){
          for(my_int i = 0; i < _hid2act[s].size(); i++) fprintf(f, "%f,", _hid2act[s]._data[i]);
          for(my_int i = 0; i < _hid2stack[s].size(); i++) fprintf(f, "%f,", _hid2stack[s]._data[i]);
          for(my_int i = 0; i < _stack2hid[s].size(); i++) fprintf(f, "%f,", _stack2hid[s]._data[i]);
        }
        fprintf(f, "\n");
        for(my_int i = 0; i < _hid2out.size(); i++) fprintf(f, "%f,", _hid2out._data[i]);
        fprintf(f, "\n");
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f = fopen(filename.c_str(), "r");
        int use_bias_i = 1;
        fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d\n",
            &_IN, &_ACTION, &_HIDDEN, &_NB_STACK, &_STACK_SIZE, &_OUT,
            &_mod, &_DEPTH, &_act_type, &use_bias_i, &_feat_dim);
        _use_bias = use_bias_i != 0;
        fscanf(f, "%lf %lf %lf %lf %d %lf\n",
            &_spectral_radius, &_input_scaling, &_sparsity, &_leak_rate,
            &_washout, &_ridge_lambda);
        _TOP_OF_STACK = 0;
        _W_in = Linear(_HIDDEN, _IN);
        _W_res = Vec2D(_HIDDEN, _HIDDEN, 0);
        _bias = Vec(_HIDDEN, 0);
        _hid2act = std::vector<Linear>(_NB_STACK, Linear(_HIDDEN, _ACTION));
        _hid2stack = std::vector<Linear>(_NB_STACK, Linear(_HIDDEN, _STACK_SIZE));
        _stack2hid = std::vector<Linear>(_NB_STACK, Linear(_STACK_SIZE, _HIDDEN));
        _hid2out = Linear(_feat_dim, _OUT);
        _x = Vec(_HIDDEN, 0);
        _x_prev = Vec(_HIDDEN, 0);
        _pre = Vec(_HIDDEN, 0);
        _act = std::vector<Vec>(_NB_STACK, Vec(_ACTION, 0));
        _stack = std::vector<Vec>(_NB_STACK, Vec(_STACK_SIZE, 0));
        _stack_prev = std::vector<Vec>(_NB_STACK, Vec(_STACK_SIZE, 0));
        _out = Vec(_OUT, 0);
        _features = Vec(_feat_dim, 0);
        _err_out = Vec(_OUT, 0);

        for(my_int i = 0; i < _W_in.size(); i++) fscanf(f, "%lf,", &_W_in._data[i]);
        for(my_int i = 0; i < _W_res.size(); i++) fscanf(f, "%lf,", &_W_res[i]);
        for(my_int i = 0; i < _bias.size(); i++) fscanf(f, "%lf,", &_bias[i]);
        for(my_int s = 0; s < _NB_STACK; s++){
          for(my_int i = 0; i < _hid2act[s].size(); i++) fscanf(f, "%lf,", &_hid2act[s]._data[i]);
          for(my_int i = 0; i < _hid2stack[s].size(); i++) fscanf(f, "%lf,", &_hid2stack[s]._data[i]);
          for(my_int i = 0; i < _stack2hid[s].size(); i++) fscanf(f, "%lf,", &_stack2hid[s]._data[i]);
        }
        for(my_int i = 0; i < _hid2out.size(); i++) fscanf(f, "%lf,", &_hid2out._data[i]);
        fclose(f);
        _step_in_seq = 0;
        _target = 0;
        clearCollected();
      }

      // ---- public configuration / state (mirrors StackRNN style) ----
      my_real _spectral_radius;
      my_real _input_scaling;
      my_real _sparsity;
      my_real _leak_rate;
      my_int  _washout;
      my_real _ridge_lambda;
      my_int  _act_type;
      bool    _use_bias;
      my_int  _step_in_seq;

      my_int _HIDDEN;
      my_int _NB_STACK;
      my_int _STACK_SIZE;
      my_int _ACTION;
      my_int _TOP_OF_STACK;
      my_int _IN;
      my_int _OUT;
      my_int _mod;
      my_int _DEPTH;
      my_int _feat_dim;

      Linear _W_in;
      Vec2D  _W_res;
      Vec    _bias;

      std::vector<Linear> _hid2act;
      std::vector<Linear> _hid2stack;
      std::vector<Linear> _stack2hid;

      Vec _x;
      Vec _x_prev;
      Vec _pre;
      std::vector<Vec> _act;
      std::vector<Vec> _stack;
      std::vector<Vec> _stack_prev;
      Vec _out;
      Vec _features;
      Linear _hid2out;
      Vec _err_out;
      my_int _target;

      std::vector<Vec> _collected_states;
      std::vector<my_int> _collected_targets;
  };

} // namespace rnn
#endif
