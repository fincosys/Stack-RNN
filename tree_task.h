/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Tree-structured tasks for TESN / DTESN / PESN.
 *
 *  ntask:
 *    1 - parity of leaf labels (mod 2)  — binary classification
 *    2 - sum of leaf labels mod nclass
 *    3 - tree depth class (bucketed)
 *    4 - majority leaf label
 *    5 - sequential task encoded as chain tree (uses task.h sequences)
 *    6 - sequential task encoded as balanced tree
 *    7 - unordered rooted tree (A000081) leaf-parity (membrane-shaped)
 *    8 - unordered rooted tree sum-mod labels
 *    9 - A000081 shape-id class (membrane structure classification)
 */
#ifndef _TREE_TASK_
#define _TREE_TASK_
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>

#include "common.h"
#include "Tree.h"
#include "RootedTree.h"
#include "task.h"

namespace rnn
{

  enum TreeTaskId {
    TTASK_PARITY = 1,
    TTASK_SUM_MOD = 2,
    TTASK_DEPTH = 3,
    TTASK_MAJORITY = 4,
    TTASK_SEQ_CHAIN = 5,
    TTASK_SEQ_BALANCED = 6,
    TTASK_UTREE_PARITY = 7,
    TTASK_UTREE_SUM_MOD = 8,
    TTASK_A000081_SHAPE = 9
  };

  inline my_int tree_nclass(my_int ntask, my_int nlabels, my_int nchar_seq = 2,
                            my_int nshape_class = 4){
    switch(ntask){
      case TTASK_PARITY: return 2;
      case TTASK_SUM_MOD: return nlabels;
      case TTASK_DEPTH: return 4; // depth buckets
      case TTASK_MAJORITY: return nlabels;
      case TTASK_SEQ_CHAIN:
      case TTASK_SEQ_BALANCED: return nchar_seq;
      case TTASK_UTREE_PARITY: return 2;
      case TTASK_UTREE_SUM_MOD: return nlabels;
      case TTASK_A000081_SHAPE: return nshape_class < 2 ? 2 : nshape_class;
      default: return 2;
    }
  }

  // Build a random labeled binary tree and assign a task target.
  // For unordered / A000081 tasks, nmin_leaves/nmax_leaves are node counts.
  inline void generate_tree_task(Tree& tree,
                                 my_int ntask,
                                 my_int nlabels,
                                 my_int nmin_leaves,
                                 my_int nmax_leaves,
                                 my_int nchar_seq = 2,
                                 my_int nrep = 1,
                                 my_int seq_task = 1,
                                 my_int nshape_class = 4,
                                 my_int a000081_max_order = 6)
  {
    if(nmax_leaves <= nmin_leaves) nmax_leaves = nmin_leaves + 1;
    my_int nleaves = nmin_leaves + (rand() % (nmax_leaves - nmin_leaves));

    if(ntask == TTASK_SEQ_CHAIN || ntask == TTASK_SEQ_BALANCED){
      // Build from a sequential toy task; target = last symbol of sequence
      std::string p = generate_next_sequence(nmax_leaves, nmin_leaves,
                                             nchar_seq, nrep, seq_task);
      std::vector<my_int> seq;
      for(size_t i = 0; i < p.size(); i++)
        seq.push_back(p[i] - 'a');
      my_int target = seq.empty() ? 0 : seq.back();
      TreeNode* root = NULL;
      if(ntask == TTASK_SEQ_CHAIN)
        root = sequence_to_chain(seq, true); // left-branching: root = last symbol
      else
        root = sequence_to_balanced_tree(seq, nchar_seq, 0);
      tree.reset(root, target);
      return;
    }

    // Unordered rooted trees (membrane shapes / A000081)
    if(ntask == TTASK_UTREE_PARITY || ntask == TTASK_UTREE_SUM_MOD){
      my_int nnodes = nleaves;
      if(nnodes < 1) nnodes = 1;
      TreeNode* root = random_unordered_rooted_tree(nnodes, nlabels);
      std::vector<my_int> leaves;
      collect_leaves(root, leaves);
      my_int target = 0;
      my_int s = 0;
      for(size_t i = 0; i < leaves.size(); i++) s += leaves[i];
      if(ntask == TTASK_UTREE_PARITY) target = s % 2;
      else target = s % nlabels;
      tree.reset(root, target);
      return;
    }

    if(ntask == TTASK_A000081_SHAPE){
      // Classify membrane structure by A000081 shape bucket
      static UTreeCatalog cat;
      static my_int built_ord = -1;
      my_int mord = a000081_max_order;
      if(mord < 2) mord = 2;
      if(mord > 8) mord = 8;
      if(built_ord != mord){
        cat.build(mord);
        built_ord = mord;
      }
      my_int nclass = nshape_class < 2 ? 2 : nshape_class;
      if(cat.size() < 1){
        tree.reset(new TreeNode(0), 0);
        return;
      }
      my_int id = 1 + (rand() % cat.size());
      TreeNode* root = instantiate_shape(cat, id, nlabels, -1);
      my_int target = (id - 1) % nclass;
      tree.reset(root, target);
      return;
    }

    TreeNode* root = random_binary_tree(nleaves, nlabels, 0);
    std::vector<my_int> leaves;
    collect_leaves(root, leaves);

    my_int target = 0;
    if(ntask == TTASK_PARITY){
      my_int s = 0;
      for(size_t i = 0; i < leaves.size(); i++) s += leaves[i];
      target = s % 2;
    } else if(ntask == TTASK_SUM_MOD){
      my_int s = 0;
      for(size_t i = 0; i < leaves.size(); i++) s += leaves[i];
      target = s % nlabels;
    } else if(ntask == TTASK_DEPTH){
      my_int d = root->depth();
      if(d <= 1) target = 0;
      else if(d <= 3) target = 1;
      else if(d <= 5) target = 2;
      else target = 3;
    } else if(ntask == TTASK_MAJORITY){
      std::vector<my_int> counts(nlabels, 0);
      for(size_t i = 0; i < leaves.size(); i++)
        if(leaves[i] >= 0 && leaves[i] < nlabels) counts[leaves[i]]++;
      target = 0;
      for(my_int c = 1; c < nlabels; c++)
        if(counts[c] > counts[target]) target = c;
    } else {
      // default: parity
      my_int s = 0;
      for(size_t i = 0; i < leaves.size(); i++) s += leaves[i];
      target = s % 2;
    }

    tree.reset(root, target);
  }

} // namespace rnn
#endif
