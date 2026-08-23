/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Labeled ordered trees for Tree-ESN / Deep Tree-ESN.
 */
#ifndef _TREE_
#define _TREE_
#include <vector>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <assert.h>

#include "common.h"

namespace rnn
{

  struct TreeNode
  {
    my_int label;                     // discrete symbol in [0, nlabels)
    std::vector<TreeNode*> children;

    TreeNode() : label(0), children() {}
    explicit TreeNode(my_int lab) : label(lab), children() {}

    ~TreeNode(){
      for(size_t i = 0; i < children.size(); i++)
        delete children[i];
      children.clear();
    }

    void addChild(TreeNode* c){
      children.push_back(c);
    }

    my_int nChildren() const { return (my_int)children.size(); }

    bool isLeaf() const { return children.empty(); }

    // Number of nodes in the subtree (including self)
    my_int size() const {
      my_int n = 1;
      for(size_t i = 0; i < children.size(); i++)
        n += children[i]->size();
      return n;
    }

    my_int depth() const {
      my_int d = 0;
      for(size_t i = 0; i < children.size(); i++){
        my_int cd = children[i]->depth();
        if(cd + 1 > d) d = cd + 1;
      }
      return d;
    }

    // Post-order traversal (children before parent) — natural for recursive RC
    void postorder(std::vector<TreeNode*>& out){
      for(size_t i = 0; i < children.size(); i++)
        children[i]->postorder(out);
      out.push_back(this);
    }

    void preorder(std::vector<TreeNode*>& out){
      out.push_back(this);
      for(size_t i = 0; i < children.size(); i++)
        children[i]->preorder(out);
    }

  private:
    // Non-copyable (owns children)
    TreeNode(const TreeNode&);
    TreeNode& operator=(const TreeNode&);
  };

  // Own a tree root and free it
  struct Tree
  {
    TreeNode* root;
    my_int target; // classification / next-symbol target for the whole tree

    Tree() : root(NULL), target(0) {}
    explicit Tree(TreeNode* r, my_int t = 0) : root(r), target(t) {}
    ~Tree(){ clear(); }

    void clear(){
      delete root;
      root = NULL;
      target = 0;
    }

    void reset(TreeNode* r, my_int t = 0){
      clear();
      root = r;
      target = t;
    }

    my_int size() const { return root ? root->size() : 0; }
    my_int depth() const { return root ? root->depth() : 0; }

  private:
    Tree(const Tree&);
    Tree& operator=(const Tree&);
  };

  // ---- builders ----

  // Encode a sequence as a right-branching unary chain (leaf = first symbol).
  // Useful to run TreeESN on sequential tasks.
  inline TreeNode* sequence_to_chain(const std::vector<my_int>& seq, bool left_branching = false){
    if(seq.empty()) return NULL;
    if(left_branching){
      TreeNode* cur = new TreeNode(seq[0]);
      for(size_t i = 1; i < seq.size(); i++){
        TreeNode* p = new TreeNode(seq[i]);
        p->addChild(cur);
        cur = p;
      }
      return cur;
    } else {
      TreeNode* leaf = new TreeNode(seq[0]);
      TreeNode* cur = leaf;
      for(size_t i = 1; i < seq.size(); i++){
        TreeNode* p = new TreeNode(seq[i]);
        cur->addChild(p);
        cur = p;
      }
      return leaf; // root is first symbol; walk children for the chain
    }
  }

  // Balanced binary tree over a sequence of leaf labels; internal nodes get
  // label = (left_label + right_label) % nlabels (or internal_label if >= 0).
  inline TreeNode* sequence_to_balanced_tree(const std::vector<my_int>& leaves,
                                            my_int nlabels,
                                            my_int internal_label = -1){
    if(leaves.empty()) return NULL;
    std::vector<TreeNode*> level;
    for(size_t i = 0; i < leaves.size(); i++)
      level.push_back(new TreeNode(leaves[i]));
    while(level.size() > 1){
      std::vector<TreeNode*> next;
      for(size_t i = 0; i < level.size(); i += 2){
        if(i + 1 >= level.size()){
          next.push_back(level[i]);
        } else {
          my_int lab = internal_label;
          if(lab < 0)
            lab = (level[i]->label + level[i+1]->label) % nlabels;
          TreeNode* p = new TreeNode(lab);
          p->addChild(level[i]);
          p->addChild(level[i+1]);
          next.push_back(p);
        }
      }
      level.swap(next);
    }
    return level[0];
  }

  // Random full binary tree of given leaf count; leaf labels in [0, nlabels).
  // Internal nodes: label 0 by default.
  inline TreeNode* random_binary_tree(my_int nleaves, my_int nlabels, my_int internal_label = 0){
    if(nleaves <= 0) return NULL;
    if(nleaves == 1)
      return new TreeNode(rand() % nlabels);
    // Random split
    my_int left = 1 + (rand() % (nleaves - 1));
    my_int right = nleaves - left;
    TreeNode* p = new TreeNode(internal_label);
    p->addChild(random_binary_tree(left, nlabels, internal_label));
    p->addChild(random_binary_tree(right, nlabels, internal_label));
    return p;
  }

  // Collect leaf labels in left-to-right order
  inline void collect_leaves(const TreeNode* n, std::vector<my_int>& out){
    if(!n) return;
    if(n->isLeaf()){
      out.push_back(n->label);
      return;
    }
    for(size_t i = 0; i < n->children.size(); i++)
      collect_leaves(n->children[i], out);
  }

} // namespace rnn
#endif
