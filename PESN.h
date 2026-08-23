/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Membrane P-system Echo State Network (PESN)
 *
 *  Cell-like P-systems have a membrane structure that is an unordered rooted
 *  tree (OEIS A000081). Each membrane holds a multiset of objects (here: a
 *  discrete label / symbol) and communicates with its inner membranes.
 *
 *  Recursive reservoir (bottom-up send-out / dissolution toward the skin):
 *    x(m) = f( W_in u(m) + W_mem * ⊕_{m' inner m} x(m') )
 *  where ⊕ is a commutative aggregation (sum), so child order is irrelevant —
 *  matching unordered membranes / A000081 trees (contrast ordered Tree-ESN).
 *
 *  Readout options:
 *    - standard state mapping (skin / mean / …) + linear ridge  (like TESN)
 *    - Butcher B-series ridge features on the skin state (BSeries.h), also
 *      indexed by A000081 unordered rooted trees
 *
 *  Only the readout is trained; membrane reservoir weights stay frozen.
 */
#ifndef _PESN_
#define _PESN_
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
#include "RootedTree.h"
#include "BSeries.h"
#include "TESN.h" // TESNMap, ESNNonlin

namespace rnn
{

  enum PESNReadout {
    PESN_READOUT_MAP = 0,     // classical root/mean/... map + ridge
    PESN_READOUT_BSERIES = 1, // Butcher B-series features + ridge
    PESN_READOUT_BOTH = 2     // concat map features and B-series scalars
  };

  struct PESN
  {
    public:

      PESN(const std::string& filename)
      {
        load(filename);
      }

      PESN(my_int nlabels,
          my_int sh,
          my_int so,
          my_int max_children = 8,
          my_real spectral_radius = 0.9,
          my_real input_scaling = 0.5,
          my_real sparsity = 0.1,
          my_real ridge = 1e-4,
          my_int map_mode = TESN_MAP_ROOT_MEAN,
          my_int nonlin = NL_TANH,
          my_real child_scaling = 1.0,
          my_int readout_mode = PESN_READOUT_BSERIES,
          my_int bs_order = 4,
          my_int bs_feat = BS_FEAT_MEAN_ROOT,
          my_int bs_dim = 2,
          my_real bs_step = 1.0) :
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
        _readout_mode(readout_mode),
        _bs_order(bs_order),
        _bs_feat(bs_feat),
        _bs_dim(bs_dim),
        _bs_step(bs_step),
        _ncollect(0),
        _last_target(0),
        _in2res(_HIDDEN, _IN),
        _mem2res(_HIDDEN, _HIDDEN),
        _res2out(1, 1),
        _out(_OUT, 0),
        _err_out(_OUT, 0),
        _feat(),
        _skin_state(_HIDDEN, 0),
        _mean_state(_HIDDEN, 0),
        _sum_state(_HIDDEN, 0),
        _leaf_mean(_HIDDEN, 0),
        _gram(),
        _B(),
        _bseries()
        {
          _FEAT = 0;
          this->initializeReservoir();
        }

      my_int computeMapFeatSize() const {
        my_int f = 1;
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

        sparse_random_init(_mem2res._data, _sparsity);
        my_real rho = estimate_spectral_radius(_mem2res._data);
        // Contractivity: control rho * max_children for ESP on trees
        my_real target = _spectral_radius;
        if(_MAX_CHILDREN > 1)
          target = _spectral_radius / (my_real)_MAX_CHILDREN;
        target *= _child_scaling;
        if(rho > 1e-12)
          scale_matrix(_mem2res._data, target / rho);
        else
          _mem2res.zeros();

        // B-series readout block (always constructed; used when mode needs it)
        if(_readout_mode == PESN_READOUT_BSERIES ||
           _readout_mode == PESN_READOUT_BOTH){
          _bseries = BSeriesReadout(_HIDDEN, _OUT, _bs_order, _bs_feat,
                                    _bs_dim, _ridge, _nonlin, _bs_step,
                                    0.5, _sparsity);
        }

        // Feature size / linear head
        if(_readout_mode == PESN_READOUT_BSERIES){
          _FEAT = _bseries._FEAT;
          // Use BSeries' own head; keep a stub _res2out
          _res2out = Linear(1, _OUT);
          _res2out.zeros();
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(1, 1, 0);
          _B = Vec2D(1, 1, 0);
        } else if(_readout_mode == PESN_READOUT_MAP){
          _FEAT = computeMapFeatSize();
          _res2out = Linear(_FEAT, _OUT);
          _res2out.zeros();
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
        } else {
          // BOTH: map features (no bias) + B-series features (includes bias)
          my_int mapf = computeMapFeatSize() - 1; // drop map bias; keep BS bias
          _FEAT = mapf + _bseries._FEAT;
          _res2out = Linear(_FEAT, _OUT);
          _res2out.zeros();
          _feat = Vec(_FEAT, 0);
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
        }

        clearCollectors();
      }

      void clearCollectors()
      {
        if(_readout_mode == PESN_READOUT_BSERIES)
          _bseries.clearCollectors();
        else {
          _gram.zeros();
          _B.zeros();
        }
        _ncollect = 0;
      }

      void applyNonlin(Vec& v)
      {
        if(_nonlin == NL_SIGMOID)
          Sigmoid::forward(v);
        else
          Tanh::forward(v);
      }

      // Bottom-up membrane encoding. Children aggregated commutatively (sum),
      // after canonicalizing the unordered membrane tree.
      void encodeMembranes(TreeNode* skin, std::map<TreeNode*, Vec>& state_map)
      {
        state_map.clear();
        if(!skin) return;
        canonicalize_unordered(skin);

        std::vector<TreeNode*> nodes;
        skin->postorder(nodes);
        for(size_t ni = 0; ni < nodes.size(); ni++){
          TreeNode* node = nodes[ni];
          Vec pre(_HIDDEN, 0);

          my_int lab = node->label;
          if(lab < 0) lab = 0;
          if(lab >= _IN) lab = lab % _IN;
          _in2res.forward_transpose(lab, pre);

          // Commutative communication from inner membranes
          my_int nc = node->nChildren();
          if(nc > _MAX_CHILDREN) nc = _MAX_CHILDREN;
          // After canonicalize, any order is fine; still sum for unordered
          for(my_int c = 0; c < nc; c++){
            TreeNode* ch = node->children[c];
            _mem2res.forward(state_map[ch], pre);
          }

          applyNonlin(pre);
          state_map[node] = pre;
        }
      }

      void aggregateStates(TreeNode* skin, std::map<TreeNode*, Vec>& state_map)
      {
        _skin_state.zeros();
        _mean_state.zeros();
        _sum_state.zeros();
        _leaf_mean.zeros();
        if(!skin) return;

        _skin_state = state_map[skin];

        std::vector<TreeNode*> nodes;
        skin->postorder(nodes);
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

      void buildMapFeatures(Vec& feat_out, bool include_bias){
        my_int need = computeMapFeatSize() - (include_bias ? 0 : 1);
        if(feat_out.size() < need) feat_out = Vec(need, 0);
        feat_out.zeros();
        my_int off = 0;
        const Vec* a = &_skin_state;
        if(_map_mode == TESN_MAP_MEAN) a = &_mean_state;
        else if(_map_mode == TESN_MAP_SUM) a = &_sum_state;
        else if(_map_mode == TESN_MAP_LEAVES) a = &_leaf_mean;

        if(_map_mode == TESN_MAP_ROOT_MEAN){
          for(my_int i = 0; i < _HIDDEN; i++)
            feat_out[off++] = _skin_state[i];
          for(my_int i = 0; i < _HIDDEN; i++)
            feat_out[off++] = _mean_state[i];
        } else {
          for(my_int i = 0; i < _HIDDEN; i++)
            feat_out[off++] = (*a)[i];
        }
        if(include_bias)
          feat_out[off++] = 1.0;
      }

      void buildCombinedFeatures(){
        // map without bias + full bseries features
        Vec mapf(computeMapFeatSize() - 1, 0);
        buildMapFeatures(mapf, false);
        _bseries.computeDifferentials(_skin_state);
        _bseries.buildFeatures();
        _feat.zeros();
        my_int off = 0;
        for(my_int i = 0; i < mapf.size(); i++)
          _feat[off++] = mapf[i];
        for(my_int i = 0; i < _bseries._FEAT; i++)
          _feat[off++] = _bseries._feat[i];
      }

      void forward(Tree& tree, my_int target = -1)
      {
        if(target < 0) target = tree.target;
        _last_target = target;
        _out.zeros();

        if(!tree.root){
          _skin_state.zeros();
          _mean_state.zeros();
          _sum_state.zeros();
          _leaf_mean.zeros();
        } else {
          std::map<TreeNode*, Vec> state_map;
          encodeMembranes(tree.root, state_map);
          aggregateStates(tree.root, state_map);
        }

        if(_readout_mode == PESN_READOUT_BSERIES){
          _bseries.forwardFromState(_skin_state, target);
          _out = _bseries._out;
          _feat = _bseries._feat;
          return;
        }

        if(_readout_mode == PESN_READOUT_MAP){
          buildMapFeatures(_feat, true);
        } else {
          buildCombinedFeatures();
        }
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

        if(_readout_mode == PESN_READOUT_BSERIES){
          // encode then collect via B-series
          if(!tree.root){
            _skin_state.zeros();
          } else {
            std::map<TreeNode*, Vec> state_map;
            encodeMembranes(tree.root, state_map);
            aggregateStates(tree.root, state_map);
          }
          _bseries.collectFromState(_skin_state, target);
          _out = _bseries._out;
          _feat = _bseries._feat;
          _ncollect = _bseries._ncollect;
          return;
        }

        forward(tree, target);
        // re-ensure features
        if(_readout_mode == PESN_READOUT_MAP)
          buildMapFeatures(_feat, true);
        else
          buildCombinedFeatures();

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
        if(_readout_mode == PESN_READOUT_BSERIES){
          bool ok = _bseries.fitReadout();
          _ncollect = _bseries._ncollect;
          return ok;
        }
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
        if(_readout_mode == PESN_READOUT_BSERIES){
          _bseries._out = _out;
          _bseries._last_target = _last_target;
          _bseries.update(lr);
          _out = _bseries._out;
          return;
        }
        for(my_int i = 0; i < _OUT; i++)
          _err_out[i] = -_out[i];
        if(_last_target >= 0 && _last_target < _OUT)
          _err_out[_last_target] += 1;
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

      void copy(const PESN& other)
      {
        assert(_IN == other._IN);
        assert(_HIDDEN == other._HIDDEN);
        assert(_OUT == other._OUT);

        _MAX_CHILDREN = other._MAX_CHILDREN;
        _spectral_radius = other._spectral_radius;
        _input_scaling = other._input_scaling;
        _sparsity = other._sparsity;
        _ridge = other._ridge;
        _map_mode = other._map_mode;
        _nonlin = other._nonlin;
        _child_scaling = other._child_scaling;
        _readout_mode = other._readout_mode;
        _bs_order = other._bs_order;
        _bs_feat = other._bs_feat;
        _bs_dim = other._bs_dim;
        _bs_step = other._bs_step;
        _FEAT = other._FEAT;
        _ncollect = other._ncollect;
        _last_target = other._last_target;

        _in2res._data = other._in2res._data;
        _mem2res._data = other._mem2res._data;
        _res2out._data = other._res2out._data;
        _out = other._out;
        _feat = other._feat;
        _skin_state = other._skin_state;
        _mean_state = other._mean_state;
        _sum_state = other._sum_state;
        _leaf_mean = other._leaf_mean;
        _gram = other._gram;
        _B = other._B;
        if(_readout_mode == PESN_READOUT_BSERIES ||
           _readout_mode == PESN_READOUT_BOTH){
          if(_bseries._HIDDEN == other._bseries._HIDDEN &&
             _bseries._FEAT == other._bseries._FEAT)
            _bseries.copy(other._bseries);
          else
            _bseries = other._bseries;
        }
      }

      void save(std::string filename)
      {
        FILE* f = fopen(filename.c_str(), "w");
        if(!f) return;
        fprintf(f, "%d %d %d %d %d %d %d %d %d %d %d\n",
            _IN, _HIDDEN, _OUT, _MAX_CHILDREN, _FEAT, _map_mode, _nonlin,
            _readout_mode, _bs_order, _bs_feat, _bs_dim);
        fprintf(f, "%.10f %.10f %.10f %.10f %.10f %.10f\n",
            _spectral_radius, _input_scaling, _sparsity, _ridge,
            _child_scaling, _bs_step);
        for(my_int i = 0; i < _in2res.size(); i++) fprintf(f, "%f,", _in2res._data[i]);
        for(my_int i = 0; i < _mem2res.size(); i++) fprintf(f, "%f,", _mem2res._data[i]);
        if(_readout_mode != PESN_READOUT_BSERIES){
          for(my_int i = 0; i < _res2out.size(); i++) fprintf(f, "%f,", _res2out._data[i]);
        } else {
          for(my_int i = 0; i < _bseries._res2out.size(); i++)
            fprintf(f, "%f,", _bseries._res2out._data[i]);
        }
        fclose(f);
      }

      void load(const std::string& filename)
      {
        FILE* f = fopen(filename.c_str(), "r");
        if(!f) return;
        fscanf(f, "%d %d %d %d %d %d %d %d %d %d %d\n",
            &_IN, &_HIDDEN, &_OUT, &_MAX_CHILDREN, &_FEAT, &_map_mode, &_nonlin,
            &_readout_mode, &_bs_order, &_bs_feat, &_bs_dim);
        fscanf(f, "%lf %lf %lf %lf %lf %lf\n",
            &_spectral_radius, &_input_scaling, &_sparsity, &_ridge,
            &_child_scaling, &_bs_step);
        _in2res = Linear(_HIDDEN, _IN);
        _mem2res = Linear(_HIDDEN, _HIDDEN);
        for(my_int i = 0; i < _in2res.size(); i++) fscanf(f, "%lf,", &_in2res._data[i]);
        for(my_int i = 0; i < _mem2res.size(); i++) fscanf(f, "%lf,", &_mem2res._data[i]);

        if(_readout_mode == PESN_READOUT_BSERIES ||
           _readout_mode == PESN_READOUT_BOTH){
          _bseries = BSeriesReadout(_HIDDEN, _OUT, _bs_order, _bs_feat,
                                    _bs_dim, _ridge, _nonlin, _bs_step);
        }
        if(_readout_mode == PESN_READOUT_BSERIES){
          _FEAT = _bseries._FEAT;
          for(my_int i = 0; i < _bseries._res2out.size(); i++)
            fscanf(f, "%lf,", &_bseries._res2out._data[i]);
          _res2out = Linear(1, _OUT);
        } else {
          _res2out = Linear(_FEAT, _OUT);
          for(my_int i = 0; i < _res2out.size(); i++)
            fscanf(f, "%lf,", &_res2out._data[i]);
        }
        fclose(f);
        _out = Vec(_OUT, 0);
        _err_out = Vec(_OUT, 0);
        _feat = Vec(_FEAT, 0);
        _skin_state = Vec(_HIDDEN, 0);
        _mean_state = Vec(_HIDDEN, 0);
        _sum_state = Vec(_HIDDEN, 0);
        _leaf_mean = Vec(_HIDDEN, 0);
        if(_readout_mode != PESN_READOUT_BSERIES){
          _gram = Vec2D(_FEAT, _FEAT, 0);
          _B = Vec2D(_FEAT, _OUT, 0);
        }
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
      my_int _readout_mode;
      my_int _bs_order;
      my_int _bs_feat;
      my_int _bs_dim;
      my_real _bs_step;
      my_int _ncollect;
      my_int _last_target;

      Linear _in2res;
      Linear _mem2res; // inner-membrane communication
      Linear _res2out;

      Vec _out;
      Vec _err_out;
      Vec _feat;
      Vec _skin_state;
      Vec _mean_state;
      Vec _sum_state;
      Vec _leaf_mean;

      Vec2D _gram;
      Vec2D _B;

      BSeriesReadout _bseries;
  };

} // namespace rnn
#endif
