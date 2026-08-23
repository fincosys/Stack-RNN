/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Unordered rooted trees (OEIS A000081).
 *
 *  A000081(n) = number of unlabeled rooted trees with n nodes:
 *    n: 1  2  3  4  5   6   7    8    9   10
 *       1  1  2  4  9  20  48  115  286  719
 *
 *  Used as:
 *    - membrane structure of cell-like P-systems
 *    - index set of Butcher B-series / elementary differentials
 *
 *  Children of a node form a multiset (order irrelevant). Canonical form
 *  sorts child subtrees by a structural key so isomorphic trees compare equal.
 */
#ifndef _ROOTED_TREE_
#define _ROOTED_TREE_
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <cstdlib>
#include <assert.h>
#include <stdio.h>

#include "common.h"
#include "Tree.h"

namespace rnn
{

  // ---- OEIS A000081 counts (n = 1..16; a(0) unused) ----
  inline my_int a000081(my_int n){
    static const my_int seq[] = {
      0,
      1, 1, 2, 4, 9, 20, 48, 115, 286, 719,
      1842, 4766, 12486, 32973, 87811, 235381
    };
    if(n < 0 || n > 16) return -1;
    return seq[n];
  }

  // Cumulative number of unordered rooted trees with 1..n nodes
  inline my_int a000081_cumulative(my_int n){
    my_int s = 0;
    for(my_int i = 1; i <= n; i++){
      my_int a = a000081(i);
      if(a < 0) return -1;
      s += a;
    }
    return s;
  }

  // Lightweight structural node for enumeration (not owning heap graph of TreeNode).
  // Represented as: root + sorted multiset of child UTree shapes (by shape-id).
  struct UTreeShape
  {
    my_int order;                 // |τ| = number of nodes
    my_int symmetry;              // σ(τ) Butcher symmetry factor
    my_int density;               // γ(τ) density (tree factorial related)
    std::string key;              // canonical string key
    std::vector<my_int> child_ids; // ids of child shapes (sorted, with repeats)

    UTreeShape() : order(0), symmetry(1), density(1), key(), child_ids() {}
  };

  // Catalog of all unordered rooted trees with order in [1, max_order].
  // Index 0 is unused; valid ids are 1..ntrees (enumeration order: by increasing
  // order, then by canonical key).
  struct UTreeCatalog
  {
    my_int max_order;
    std::vector<UTreeShape> shapes; // shapes[0] dummy; shapes[1..] real
    std::map<std::string, my_int> key2id;
    std::vector<std::vector<my_int> > by_order; // by_order[n] = ids with |τ|=n

    UTreeCatalog() : max_order(0), shapes(), key2id(), by_order() {}

    my_int size() const { return (my_int)shapes.size() - 1; }

    const UTreeShape& get(my_int id) const {
      assert(id >= 1 && id < (my_int)shapes.size());
      return shapes[id];
    }

    // Build all trees with |τ| <= max_ord (max_ord >= 1, capped for safety).
    void build(my_int max_ord){
      if(max_ord < 1) max_ord = 1;
      if(max_ord > 12) max_ord = 12; // combinatorial explosion
      max_order = max_ord;
      shapes.clear();
      key2id.clear();
      by_order.assign(max_order + 1, std::vector<my_int>());
      shapes.push_back(UTreeShape()); // dummy id 0

      // order 1: single node •
      {
        UTreeShape t;
        t.order = 1;
        t.symmetry = 1;
        t.density = 1;
        t.key = "()";
        t.child_ids.clear();
        shapes.push_back(t);
        key2id[t.key] = 1;
        by_order[1].push_back(1);
      }

      // Growing by order: a tree of order n is a root plus a multiset of
      // smaller trees whose orders sum to n-1.
      for(my_int n = 2; n <= max_order; n++){
        // Generate all multisets of existing trees with total order n-1.
        std::vector<my_int> pick;
        gen_multisets(n - 1, 1, pick);
      }

      // Verify A000081 for small n
      for(my_int n = 1; n <= max_order && n <= 16; n++){
        my_int expect = a000081(n);
        my_int got = (my_int)by_order[n].size();
        if(expect >= 0 && got != expect){
          // Soft check only — do not abort; counts should match.
          fprintf(stderr, "[RootedTree] A000081(%d) expected %d got %d\n",
                  n, expect, got);
        }
      }
    }

    // Recursively enumerate nondecreasing sequences of shape ids whose orders
    // sum to `remain`, each id >= min_id (to keep multisets canonical).
    void gen_multisets(my_int remain, my_int min_id, std::vector<my_int>& pick){
      if(remain == 0){
        add_tree_from_children(pick);
        return;
      }
      if(remain < 0) return;
      my_int nsh = size();
      for(my_int id = min_id; id <= nsh; id++){
        my_int o = shapes[id].order;
        if(o > remain) continue;
        pick.push_back(id);
        gen_multisets(remain - o, id, pick); // id >= previous => multiset order
        pick.pop_back();
      }
    }

    void add_tree_from_children(const std::vector<my_int>& kids){
      UTreeShape t;
      t.child_ids = kids;
      t.order = 1;
      for(size_t i = 0; i < kids.size(); i++)
        t.order += shapes[kids[i]].order;

      // Canonical key: sorted child keys already ensured by nondecreasing ids
      t.key = "(";
      for(size_t i = 0; i < kids.size(); i++){
        if(i) t.key += ",";
        // use id for compactness
        char buf[32];
        sprintf(buf, "%d", kids[i]);
        t.key += buf;
      }
      t.key += ")";

      if(key2id.count(t.key)) return; // already have it

      // Symmetry factor σ(τ) = product over distinct child-types:
      //   σ(child)^mult * mult!
      // times product of child symmetries (recursive definition).
      t.symmetry = 1;
      {
        size_t i = 0;
        while(i < kids.size()){
          size_t j = i;
          while(j < kids.size() && kids[j] == kids[i]) j++;
          my_int mult = (my_int)(j - i);
          my_int child_sig = shapes[kids[i]].symmetry;
          // mult! * σ(child)^mult
          my_int fact = 1;
          for(my_int k = 2; k <= mult; k++) fact *= k;
          my_int p = 1;
          for(my_int k = 0; k < mult; k++) p *= child_sig;
          t.symmetry *= fact * p;
          i = j;
        }
      }

      // Density γ(τ) = |τ| * product γ(children)
      t.density = t.order;
      for(size_t i = 0; i < kids.size(); i++)
        t.density *= shapes[kids[i]].density;

      shapes.push_back(t);
      my_int id = (my_int)shapes.size() - 1;
      key2id[t.key] = id;
      if(t.order <= max_order)
        by_order[t.order].push_back(id);
    }
  };

  // ---- Convert unordered structure helpers on TreeNode ----

  // Canonical structural key of a TreeNode ignoring labels (shape only).
  inline std::string utree_shape_key(const TreeNode* n){
    if(!n) return "";
    std::vector<std::string> ck;
    ck.reserve(n->children.size());
    for(size_t i = 0; i < n->children.size(); i++)
      ck.push_back(utree_shape_key(n->children[i]));
    std::sort(ck.begin(), ck.end());
    std::string s = "(";
    for(size_t i = 0; i < ck.size(); i++){
      if(i) s += ",";
      s += ck[i];
    }
    s += ")";
    return s;
  }

  inline bool utree_child_less(const TreeNode* a, const TreeNode* b){
    std::string ka = utree_shape_key(a);
    std::string kb = utree_shape_key(b);
    if(ka != kb) return ka < kb;
    return a->label < b->label;
  }

  // Sort children of every node into canonical unordered order (by shape key,
  // then by label). Makes subsequent position-wise ops order-invariant.
  inline void canonicalize_unordered(TreeNode* n){
    if(!n) return;
    for(size_t i = 0; i < n->children.size(); i++)
      canonicalize_unordered(n->children[i]);
    std::sort(n->children.begin(), n->children.end(), utree_child_less);
  }

  // Random unordered rooted tree with exactly n nodes (labels in [0,nlabels)).
  // Built by random root degree / recursive partition of n-1 among children.
  inline TreeNode* random_unordered_rooted_tree(my_int n, my_int nlabels){
    if(n <= 0) return NULL;
    TreeNode* root = new TreeNode(rand() % nlabels);
    if(n == 1) return root;

    my_int remain = n - 1;
    // number of child subtrees: at least 1, at most remain
    my_int k = 1 + (rand() % remain);
    // random positive composition of remain into k parts
    std::vector<my_int> parts(k, 1);
    my_int left = remain - k;
    for(my_int i = 0; i < left; i++)
      parts[rand() % k]++;

    for(my_int i = 0; i < k; i++)
      root->addChild(random_unordered_rooted_tree(parts[i], nlabels));

    canonicalize_unordered(root);
    return root;
  }

  // Build a TreeNode whose shape matches catalog id (labels random / fixed).
  inline TreeNode* instantiate_shape(const UTreeCatalog& cat, my_int id,
                                     my_int nlabels, my_int fixed_label = -1){
    if(id < 1 || id > cat.size()) return NULL;
    const UTreeShape& sh = cat.get(id);
    my_int lab = (fixed_label >= 0) ? fixed_label : (rand() % nlabels);
    TreeNode* root = new TreeNode(lab);
    for(size_t i = 0; i < sh.child_ids.size(); i++)
      root->addChild(instantiate_shape(cat, sh.child_ids[i], nlabels, fixed_label));
    return root;
  }

} // namespace rnn
#endif
