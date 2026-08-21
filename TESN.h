/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Tree Echo State Network (TreeESN / TESN)
 *  Recursive reservoir computing on ordered trees (Gallicchio & Micheli).
 *
 *  For a node v with label u(v) and children ch_1..ch_k:
 *    x(v) = f( W_in u(v) + sum_j W_hat x(ch_j) )
 *  (optional leak toward zero / contractive scaling via spectral radius).
 *  State mapping aggregates node states into a fixed-size feature vector;
 *  only the linear readout is trained (ridge or SGD).
 */
#ifndef _TESN_
#define _TESN_
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <math.h>
#include <assert.h>
#include <stdio.h>

#include "common.h"
#include "Vec.h"
#include "Linear.h"
#include "Nonlinearity.h"
#include "utils.h"
#include "Tree.h"

namespace rnn
{

#ifndef _ESN_
  enum ESNNonlin {
    NL_TANH = 0,
    NL_SIGMOID = 1
  };
#endif

  // How to map the multiset of node states to a fixed feature vector
  enum TESNMap {
    TESN_MAP_ROOT = 0,   // root state only
    TESN_MAP_MEAN = 1,   // mean over all nodes
    TESN_MAP_SUM  = 2,   // sum over all nodes
    TESN_MAP_LEAVES = 3, // mean over leaves
    TESN_MAP_ROOT_MEAN = 4 // concat(root, mean)
  };

  struct TESN
  {
    public:

      TESN(const std::string& filename)
      {
        load(filename);
      }

      TESN(my_int nlabels,
          my_int sh,
          my_int so,
          my_int max_children = 8,
          my_real spectral_radius = 0.9,
          my_real input_scaling = 0.5,
          my_real sparsity = 0.1,
          my_real ridge = 1e-4,
          my_int map_mode = TESN_MAP_ROOT_MEAN,
          my_int nonlin = NL_TANH,
          my_real child_scaling = 1.0) :
        _HIDDEN(sh),
        _IN(nlabels),
        _OUT(so),
        _MAX_CHILDREN(max_children < 1 ? 1 : max_children),
        _spectral_radius(spectral_radius),
        _input_scaling(input_scaling),
        _sparsity(sparsity),
        _ridge(ridge),
        _map_mode(map_mode),
        _nonlin(nonlin),
        _child_scaling(child_scaling),
        _ncollect(0),
        _last_target(0),
        _in2res(_HIDDEN, _IN),
        _child2res(),
        _res2out(1, 1),
        _out(_OUT, 0),
        _err_out(_OUT, 0),
        _feat(),
        _root_state(_HIDDEN, 0),
        _mean_state(_HIDDEN, 0),
        _sum_state(_HIDDEN, 0),
        _leaf_mean(_HIDDEN, 0),
        _gram(),
        _B()
        {
          // Shared recursive reservoir matrix applied to each child state
          _child2res = Linear(_HIDDEN, _HIDDEN);
          _FEAT = computeFeatSize();
          _res2out = Linear(_FEAT, _OUT);
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
          this->initializeReservoir();
        }

      my_int computeFeatSize() const {
        my_int f = 1; // bias
        if(_map_mode == TESN_MAP_ROOT_MEAN)
          f += 2 * _HIDDEN;
        else
          f += _HIDDEN;
        return f;
      }

      void initializeReservoir()
      {
        for(my_int i = 0; i < _in2res.size(); i++)
          _in2res._data[i] = random(-_input_scaling, _input_scaling);

        sparse_random_init(_child2res._data, _sparsity);
        my_real rho = estimate_spectral_radius(_child2res._data);
        // Contractivity for tree ESP: scale so spectral radius * max_children * child_scaling is controlled
        my_real target = _spectral_radius;
        if(_MAX_CHILDREN > 1)
          target = _spectral_radius / (my_real)_MAX_CHILDREN;
        target *= _child_scaling;
        if(rho > 1e-12)
          scale_matrix(_child2res._data, target / rho);
        else
          _child2res.zeros();

        _res2out.zeros();
        clearCollectors();
      }

      void clearCollectors()
      {
        _gram.zeros();
        _B.zeros();
        _ncollect = 0;
      }

      void applyNonlin(Vec& v)
      {
        if(_nonlin == NL_SIGMOID)
          Sigmoid::forward(v);
        else
          Tanh::forward(v);
      }

      // Bottom-up encoding via post-order traversal (avoids deep C++ recursion).
      void encodeTree(TreeNode* root, std::map<TreeNode*, Vec>& state_map)
      {
        state_map.clear();
        if(!root) return;
        std::vector<TreeNode*> nodes;
        root->postorder(nodes);
        for(size_t ni = 0; ni < nodes.size(); ni++){
          TreeNode* node = nodes[ni];
          Vec pre(_HIDDEN, 0);

          my_int lab = node->label;
          if(lab < 0) lab = 0;
          if(lab >= _IN) lab = lab % _IN;
          _in2res.forward_transpose(lab, pre);

          my_int nc = node->nChildren();
          if(nc > _MAX_CHILDREN) nc = _MAX_CHILDREN;
          for(my_int c = 0; c < nc; c++){
            TreeNode* ch = node->children[c];
            _child2res.forward(state_map[ch], pre);
          }

          applyNonlin(pre);
          state_map[node] = pre;
        }
      }

      void aggregateStates(TreeNode* root, std::map<TreeNode*, Vec>& state_map)
      {
        _root_state.zeros();
        _mean_state.zeros();
        _sum_state.zeros();
        _leaf_mean.zeros();
        if(!root) return;

        _root_state = state_map[root];

        std::vector<TreeNode*> nodes;
        root->postorder(nodes);
        my_int n = 0, nleaf = 0;
        for(size_t i = 0; i < nodes.size(); i++){
          const Vec& s = state_map[nodes[i]];
          for(my_int j = 0; j < _HIDDEN; j++)
            _sum_state[j] += s[j];
          n++;
          if(nodes[i]->isLeaf()){
            for(my_int j = 0; j < _HIDDEN; j++)
              _leaf_mean[j] += s[j];
            nleaf++;
          }
        }
        if(n > 0){
          for(my_int j = 0; j < _HIDDEN; j++)
            _mean_state[j] = _sum_state[j] / (my_real)n;
        }
        if(nleaf > 0){
          for(my_int j = 0; j < _HIDDEN; j++)
            _leaf_mean[j] /= (my_real)nleaf;
        }
      }

      void buildFeatures()
      {
        _feat.zeros();
        my_int off = 0;
        const Vec* a = &_root_state;
        if(_map_mode == TESN_MAP_MEAN) a = &_mean_state;
        else if(_map_mode == TESN_MAP_SUM) a = &_sum_state;
        else if(_map_mode == TESN_MAP_LEAVES) a = &_leaf_mean;

        if(_map_mode == TESN_MAP_ROOT_MEAN){
          for(my_int i = 0; i < _HIDDEN; i++)
            _feat[off++] = _root_state[i];
          for(my_int i = 0; i < _HIDDEN; i++)
            _feat[off++] = _mean_state[i];
        } else {
          for(my_int i = 0; i < _HIDDEN; i++)
            _feat[off++] = (*a)[i];
        }
        _feat[off++] = 1.0;
      }

      // Encode a whole tree and produce softmax output for tree.target (or given target)
      void forward(Tree& tree, my_int target = -1)
      {
        if(target < 0) target = tree.target;
        _last_target = target;
        _out.zeros();

        if(!tree.root){
          _root_state.zeros();
          _mean_state.zeros();
          _sum_state.zeros();
          _leaf_mean.zeros();
          buildFeatures();
          _res2out.forward(_feat, _out);
          Softmax::forward(_out);
          return;
        }

        std::map<TreeNode*, Vec> state_map;
        encodeTree(tree.root, state_map);
        aggregateStates(tree.root, state_map);
        buildFeatures();
        _res2out.forward(_feat, _out);
        Softmax::forward(_out);
      }

      // Also works on a symbol sequence encoded as a left-branching chain.
      void forwardSequence(const std::vector<my_int>& seq, my_int target)
      {
        Tree tmp;
        tmp.root = sequence_to_chain(seq, true);
        tmp.target = target;
        forward(tmp, target);
        // Tree destructor frees root
      }

      void collect(Tree& tree, my_int target = -1)
      {
        if(target < 0) target = tree.target;
        forward(tree, target);
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
          _err_out[i] = -_out[i];
        if(_last_target >= 0 && _last_target < _OUT)
          _err_out[_last_target] += 1;
        buildFeatures();
        _res2out.resetGradient();
        _res2out.computeGradient(_feat, _err_out);
        _res2out.update(lr);
      }

      my_real eval(const my_int& target) const {
        return _out[target];
      }

      my_int pred() const {
        my_int p = 0;
        my_real pv = _out[0];
        for(my_int i = 1; i < _OUT; i++){
          if(pv < _out[i]){
            p = i;
            pv = _out[i];
          }
        }
        return p;
      }

      void copy(const TESN& other)
      {
        assert(_IN == other._IN);
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);
        assert(_FEAT == other._FEAT);

        _MAX_CHILDREN = other._MAX_CHILDREN;
        _spectral_radius = other._spectral_radius;
        _input_scaling = other._input_scaling;
        _sparsity = other._sparsity;
        _ridge = other._ridge;
        _map_mode = other._map_mode;
        _nonlin = other._nonlin;
        _child_scaling = other._child_scaling;
        _ncollect = other._ncollect;
        _last_target = other._last_target;

        _in2res._data = other._in2res._data;
        _child2res._data = other._child2res._data;
        _res2out._data = other._res2out._data;
        _out = other._out;
        _feat = other._feat;
        _root_state = other._root_state;
        _mean_state = other._mean_state;
        _sum_state = other._sum_state;
        _leaf_mean = other._leaf_mean;
        _gram = other._gram;
        _B = other._B;
      }

      void save(std::string filename)
      {
        FILE* f = fopen(filename.c_str(), "w");
        if(!f) return;
        fprintf(f, "%d %d %d %d %d %d %d\n",
            _IN, _HIDDEN, _OUT, _MAX_CHILDREN, _FEAT, _map_mode, _nonlin);
        fprintf(f, "%.10f %.10f %.10f %.10f %.10f\n",
            _spectral_radius, _input_scaling, _sparsity, _ridge, _child_scaling);
        for(my_int i = 0; i < _in2res.size(); i++) fprintf(f, "%f,", _in2res._data[i]);
        for(my_int i = 0; i < _child2res.size(); i++) fprintf(f, "%f,", _child2res._data[i]);
        for(my_int i = 0; i < _res2out.size(); i++) fprintf(f, "%f,", _res2out._data[i]);
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f = fopen(filename.c_str(), "r");
        if(!f) return;
        fscanf(f, "%d %d %d %d %d %d %d\n",
            &_IN, &_HIDDEN, &_OUT, &_MAX_CHILDREN, &_FEAT, &_map_mode, &_nonlin);
        fscanf(f, "%lf %lf %lf %lf %lf\n",
            &_spectral_radius, &_input_scaling, &_sparsity, &_ridge, &_child_scaling);
        _in2res = Linear(_HIDDEN, _IN);
        _child2res = Linear(_HIDDEN, _HIDDEN);
        _res2out = Linear(_FEAT, _OUT);
        for(my_int i = 0; i < _in2res.size(); i++) fscanf(f, "%lf,", &_in2res._data[i]);
        for(my_int i = 0; i < _child2res.size(); i++) fscanf(f, "%lf,", &_child2res._data[i]);
        for(my_int i = 0; i < _res2out.size(); i++) fscanf(f, "%lf,", &_res2out._data[i]);
        fclose(f);
        _out = Vec(_OUT, 0);
        _err_out = Vec(_OUT, 0);
        _feat = Vec(_FEAT, 0);
        _root_state = Vec(_HIDDEN, 0);
        _mean_state = Vec(_HIDDEN, 0);
        _sum_state = Vec(_HIDDEN, 0);
        _leaf_mean = Vec(_HIDDEN, 0);
        _gram = Vec2D(_FEAT, _FEAT, 0);
        _B = Vec2D(_FEAT, _OUT, 0);
        _ncollect = 0;
        _last_target = 0;
      }

      my_int _HIDDEN;
      my_int _IN;
      my_int _OUT;
      my_int _FEAT;
      my_int _MAX_CHILDREN;
      my_real _spectral_radius;
      my_real _input_scaling;
      my_real _sparsity;
      my_real _ridge;
      my_int _map_mode;
      my_int _nonlin;
      my_real _child_scaling;
      my_int _ncollect;
      my_int _last_target;

      Linear _in2res;
      Linear _child2res;
      Linear _res2out;

      Vec _out;
      Vec _err_out;
      Vec _feat;
      Vec _root_state;
      Vec _mean_state;
      Vec _sum_state;
      Vec _leaf_mean;

      Vec2D _gram;
      Vec2D _B;
  };

} // namespace rnn
#endif
