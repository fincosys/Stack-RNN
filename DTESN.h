/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Deep Tree Echo State Network (DeepTESN / DTESN)
 *  Stack of recursive reservoirs on ordered trees.
 *
 *  Layer 1:
 *    x^(1)(v) = f( W_in u(v) + sum_j W_hat^(1) x^(1)(ch_j) )
 *  Layer i>1:
 *    x^(i)(v) = f( W^(i) x^(i-1)(v) + sum_j W_hat^(i) x^(i)(ch_j) )
 *
 *  Features concatenate (mapped) states across layers; only readout is trained.
 */
#ifndef _DTESN_
#define _DTESN_
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
#include "TESN.h" // TESNMap enum

namespace rnn
{

  struct DTESN
  {
    public:

      DTESN(const std::string& filename)
      {
        load(filename);
      }

      DTESN(my_int nlabels,
          my_int sh,
          my_int so,
          my_int nlayers = 2,
          my_int max_children = 8,
          my_real spectral_radius = 0.9,
          my_real input_scaling = 0.5,
          my_real sparsity = 0.1,
          my_real ridge = 1e-4,
          my_int map_mode = TESN_MAP_ROOT_MEAN,
          my_int nonlin = NL_TANH,
          my_real child_scaling = 1.0,
          my_real inter_scaling = 0.5) :
        _HIDDEN(sh),
        _IN(nlabels),
        _OUT(so),
        _NLAYERS(nlayers < 1 ? 1 : nlayers),
        _MAX_CHILDREN(max_children < 1 ? 1 : max_children),
        _spectral_radius(spectral_radius),
        _input_scaling(input_scaling),
        _sparsity(sparsity),
        _ridge(ridge),
        _map_mode(map_mode),
        _nonlin(nonlin),
        _child_scaling(child_scaling),
        _inter_scaling(inter_scaling),
        _ncollect(0),
        _last_target(0),
        _in2res(_HIDDEN, _IN),
        _inter2res(),
        _child2res(),
        _res2out(1, 1),
        _out(_OUT, 0),
        _err_out(_OUT, 0),
        _feat(),
        _root_state(),
        _mean_state(),
        _gram(),
        _B()
        {
          _inter2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
          _child2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
          _root_state = std::vector<Vec>(_NLAYERS, Vec(_HIDDEN, 0));
          _mean_state = std::vector<Vec>(_NLAYERS, Vec(_HIDDEN, 0));
          _FEAT = computeFeatSize();
          _res2out = Linear(_FEAT, _OUT);
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
          this->initializeReservoir();
        }

      my_int computeFeatSize() const {
        my_int per = _HIDDEN;
        if(_map_mode == TESN_MAP_ROOT_MEAN) per = 2 * _HIDDEN;
        return per * _NLAYERS + 1; // + bias
      }

      void initializeReservoir()
      {
        for(my_int i = 0; i < _in2res.size(); i++)
          _in2res._data[i] = random(-_input_scaling, _input_scaling);

        my_real target = _spectral_radius;
        if(_MAX_CHILDREN > 1)
          target = _spectral_radius / (my_real)_MAX_CHILDREN;
        target *= _child_scaling;

        for(my_int l = 0; l < _NLAYERS; l++){
          if(l == 0)
            _inter2res[l].zeros();
          else {
            for(my_int i = 0; i < _inter2res[l].size(); i++)
              _inter2res[l]._data[i] = random(-_inter_scaling, _inter_scaling);
          }

          sparse_random_init(_child2res[l]._data, _sparsity);
          my_real rho = estimate_spectral_radius(_child2res[l]._data);
          if(rho > 1e-12)
            scale_matrix(_child2res[l]._data, target / rho);
          else
            _child2res[l].zeros();
        }

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

      // Encode all layers for the whole tree. state_map[l][node] = x^(l)(node)
      void encodeTree(TreeNode* root,
                     std::vector<std::map<TreeNode*, Vec> >& state_maps)
      {
        state_maps.assign(_NLAYERS, std::map<TreeNode*, Vec>());
        if(!root) return;

        std::vector<TreeNode*> nodes;
        root->postorder(nodes);

        for(my_int l = 0; l < _NLAYERS; l++){
          for(size_t ni = 0; ni < nodes.size(); ni++){
            TreeNode* node = nodes[ni];
            Vec pre(_HIDDEN, 0);

            if(l == 0){
              my_int lab = node->label;
              if(lab < 0) lab = 0;
              if(lab >= _IN) lab = lab % _IN;
              _in2res.forward_transpose(lab, pre);
            } else {
              // previous layer state at same node (already computed: postorder + layer loop)
              _inter2res[l].forward(state_maps[l - 1][node], pre);
            }

            my_int nc = node->nChildren();
            if(nc > _MAX_CHILDREN) nc = _MAX_CHILDREN;
            for(my_int c = 0; c < nc; c++){
              TreeNode* ch = node->children[c];
              _child2res[l].forward(state_maps[l][ch], pre);
            }

            applyNonlin(pre);
            state_maps[l][node] = pre;
          }
        }
      }

      void aggregateStates(TreeNode* root,
                           std::vector<std::map<TreeNode*, Vec> >& state_maps)
      {
        for(my_int l = 0; l < _NLAYERS; l++){
          _root_state[l].zeros();
          _mean_state[l].zeros();
        }
        if(!root) return;

        std::vector<TreeNode*> nodes;
        root->postorder(nodes);
        my_int n = (my_int)nodes.size();
        if(n == 0) return;

        for(my_int l = 0; l < _NLAYERS; l++){
          _root_state[l] = state_maps[l][root];
          for(size_t i = 0; i < nodes.size(); i++){
            const Vec& s = state_maps[l][nodes[i]];
            for(my_int j = 0; j < _HIDDEN; j++)
              _mean_state[l][j] += s[j];
          }
          for(my_int j = 0; j < _HIDDEN; j++)
            _mean_state[l][j] /= (my_real)n;
        }
      }

      void buildFeatures()
      {
        _feat.zeros();
        my_int off = 0;
        for(my_int l = 0; l < _NLAYERS; l++){
          if(_map_mode == TESN_MAP_MEAN){
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _mean_state[l][i];
          } else if(_map_mode == TESN_MAP_ROOT){
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _root_state[l][i];
          } else if(_map_mode == TESN_MAP_SUM){
            // reuse mean * n not stored; use mean as proxy scaled — store mean
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _mean_state[l][i];
          } else if(_map_mode == TESN_MAP_LEAVES){
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _mean_state[l][i];
          } else {
            // ROOT_MEAN default
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _root_state[l][i];
            for(my_int i = 0; i < _HIDDEN; i++)
              _feat[off++] = _mean_state[l][i];
          }
        }
        _feat[off++] = 1.0;
      }

      void forward(Tree& tree, my_int target = -1)
      {
        if(target < 0) target = tree.target;
        _last_target = target;
        _out.zeros();

        std::vector<std::map<TreeNode*, Vec> > state_maps;
        encodeTree(tree.root, state_maps);
        aggregateStates(tree.root, state_maps);
        buildFeatures();
        _res2out.forward(_feat, _out);
        Softmax::forward(_out);
      }

      void forwardSequence(const std::vector<my_int>& seq, my_int target)
      {
        Tree tmp;
        tmp.root = sequence_to_chain(seq, true);
        tmp.target = target;
        forward(tmp, target);
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

      void copy(const DTESN& other)
      {
        assert(_IN == other._IN);
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);
        assert(_FEAT == other._FEAT);
        assert(_NLAYERS == other._NLAYERS);

        _MAX_CHILDREN = other._MAX_CHILDREN;
        _spectral_radius = other._spectral_radius;
        _input_scaling = other._input_scaling;
        _sparsity = other._sparsity;
        _ridge = other._ridge;
        _map_mode = other._map_mode;
        _nonlin = other._nonlin;
        _child_scaling = other._child_scaling;
        _inter_scaling = other._inter_scaling;
        _ncollect = other._ncollect;
        _last_target = other._last_target;

        _in2res._data = other._in2res._data;
        _res2out._data = other._res2out._data;
        for(my_int l = 0; l < _NLAYERS; l++){
          _inter2res[l]._data = other._inter2res[l]._data;
          _child2res[l]._data = other._child2res[l]._data;
          _root_state[l] = other._root_state[l];
          _mean_state[l] = other._mean_state[l];
        }
        _out = other._out;
        _feat = other._feat;
        _gram = other._gram;
        _B = other._B;
      }

      void save(std::string filename)
      {
        FILE* f = fopen(filename.c_str(), "w");
        if(!f) return;
        fprintf(f, "%d %d %d %d %d %d %d %d\n",
            _IN, _HIDDEN, _OUT, _NLAYERS, _MAX_CHILDREN, _FEAT, _map_mode, _nonlin);
        fprintf(f, "%.10f %.10f %.10f %.10f %.10f %.10f\n",
            _spectral_radius, _input_scaling, _sparsity, _ridge,
            _child_scaling, _inter_scaling);
        for(my_int i = 0; i < _in2res.size(); i++) fprintf(f, "%f,", _in2res._data[i]);
        for(my_int l = 0; l < _NLAYERS; l++){
          for(my_int i = 0; i < _inter2res[l].size(); i++)
            fprintf(f, "%f,", _inter2res[l]._data[i]);
          for(my_int i = 0; i < _child2res[l].size(); i++)
            fprintf(f, "%f,", _child2res[l]._data[i]);
        }
        for(my_int i = 0; i < _res2out.size(); i++) fprintf(f, "%f,", _res2out._data[i]);
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f = fopen(filename.c_str(), "r");
        if(!f) return;
        fscanf(f, "%d %d %d %d %d %d %d %d\n",
            &_IN, &_HIDDEN, &_OUT, &_NLAYERS, &_MAX_CHILDREN, &_FEAT, &_map_mode, &_nonlin);
        fscanf(f, "%lf %lf %lf %lf %lf %lf\n",
            &_spectral_radius, &_input_scaling, &_sparsity, &_ridge,
            &_child_scaling, &_inter_scaling);
        _in2res = Linear(_HIDDEN, _IN);
        _inter2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
        _child2res = std::vector<Linear>(_NLAYERS, Linear(_HIDDEN, _HIDDEN));
        _res2out = Linear(_FEAT, _OUT);
        _root_state = std::vector<Vec>(_NLAYERS, Vec(_HIDDEN, 0));
        _mean_state = std::vector<Vec>(_NLAYERS, Vec(_HIDDEN, 0));
        for(my_int i = 0; i < _in2res.size(); i++) fscanf(f, "%lf,", &_in2res._data[i]);
        for(my_int l = 0; l < _NLAYERS; l++){
          for(my_int i = 0; i < _inter2res[l].size(); i++)
            fscanf(f, "%lf,", &_inter2res[l]._data[i]);
          for(my_int i = 0; i < _child2res[l].size(); i++)
            fscanf(f, "%lf,", &_child2res[l]._data[i]);
        }
        for(my_int i = 0; i < _res2out.size(); i++) fscanf(f, "%lf,", &_res2out._data[i]);
        fclose(f);
        _out = Vec(_OUT, 0);
        _err_out = Vec(_OUT, 0);
        _feat = Vec(_FEAT, 0);
        _gram = Vec2D(_FEAT, _FEAT, 0);
        _B = Vec2D(_FEAT, _OUT, 0);
        _ncollect = 0;
        _last_target = 0;
      }

      my_int _HIDDEN;
      my_int _IN;
      my_int _OUT;
      my_int _FEAT;
      my_int _NLAYERS;
      my_int _MAX_CHILDREN;
      my_real _spectral_radius;
      my_real _input_scaling;
      my_real _sparsity;
      my_real _ridge;
      my_int _map_mode;
      my_int _nonlin;
      my_real _child_scaling;
      my_real _inter_scaling;
      my_int _ncollect;
      my_int _last_target;

      Linear _in2res;
      std::vector<Linear> _inter2res;
      std::vector<Linear> _child2res;
      Linear _res2out;

      Vec _out;
      Vec _err_out;
      Vec _feat;
      std::vector<Vec> _root_state;
      std::vector<Vec> _mean_state;

      Vec2D _gram;
      Vec2D _B;
  };

} // namespace rnn
#endif
