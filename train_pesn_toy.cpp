/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 *  Train membrane P-system recursive reservoirs (PESN) with optional
 *  Butcher B-series ridge readouts. Membrane structures and B-series trees
 *  both follow OEIS A000081 (unordered rooted trees).
 */
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>

#include "common.h"
#include "tree_task.h"
#include "RootedTree.h"
#include "PESN.h"

using namespace std;
using namespace rnn;

void print_help(){
  printf("train_pesn_toy — Membrane P-system reservoir + Butcher B-series ridge\n");
  printf("  P-system membrane structures and B-series index set: OEIS A000081\n");
  printf("usage: train_pesn_toy [options]\n");
  printf("options:\n");
  printf("-nhid [integer]\t\t membrane reservoir size. Default: 50\n");
  printf("-nlabels [integer]\t object alphabet size. Default: 2\n");
  printf("-maxch [integer]\t max inner membranes in recursion. Default: 6\n");
  printf("-ntree [integer]\t trees/membranes per epoch. Default: 1000\n");
  printf("-lr [float]\t\t SGD learning rate (fit=sgd). Default: 0.1\n");
  printf("-rho [float]\t\t spectral radius (membrane ESP). Default: 0.9\n");
  printf("-is [float]\t\t input / object scaling. Default: 0.5\n");
  printf("-cscale [float]\t\t inner-membrane scale factor. Default: 1.0\n");
  printf("-sparsity [float]\t fraction of nonzero W_mem. Default: 0.1\n");
  printf("-ridge [float]\t\t ridge regression lambda. Default: 1e-4\n");
  printf("-map [integer]\t\t 0=skin 1=mean 2=sum 3=leaves 4=skin+mean. Default: 4\n");
  printf("-nl [integer]\t\t 0=tanh, 1=sigmoid. Default: 0\n");
  printf("-fit [string]\t\t ridge|sgd. Default: ridge\n");
  printf("-readout [integer]\t 0=map 1=bseries 2=both. Default: 1\n");
  printf("-bsorder [integer]\t B-series tree order (A000081, <=8). Default: 4\n");
  printf("-bsfeat [integer]\t 0=mean 1=full 2=proj 3=mean+skin. Default: 3\n");
  printf("-bsdim [integer]\t projection dim per tree (bsfeat=2). Default: 2\n");
  printf("-bsstep [float]\t\t B-series step weight h. Default: 1.0\n");
  printf("-nepoch [integer]\t number of epochs. Default: 15\n");
  printf("-ntask [integer]\t task id (1..9). Default: 7\n");
  printf("\t\t\t 7=unordered parity 8=unordered sum-mod 9=A000081 shape\n");
  printf("-nseqtask [integer]\t underlying seq task for ntask 5/6. Default: 1\n");
  printf("-nchar [integer]\t alphabet for seq-tree tasks. Default: 2\n");
  printf("-nrep [integer]\t\t task2 repetitions. Default: 1\n");
  printf("-nshape [integer]\t classes for task 9. Default: 4\n");
  printf("-seed [integer]\t\t RNG seed. Default: 1\n");
  printf("-nmin [integer]\t\t min nodes/leaves. Default: 2\n");
  printf("-nmax [integer]\t\t max nodes/leaves. Default: 10\n");
  printf("-save\t\t\t save model and logs\n");
  printf("-check-a000081\t\t verify tree catalog vs OEIS A000081 and exit\n");
}

int check_a000081(int max_ord){
  printf("OEIS A000081 unordered rooted trees (catalog check)\n");
  UTreeCatalog cat;
  cat.build(max_ord);
  int ok = 1;
  for(int n = 1; n <= max_ord; n++){
    int expect = a000081(n);
    int got = (int)cat.by_order[n].size();
    printf("  n=%2d  A000081=%5d  catalog=%5d  %s\n",
           n, expect, got, (expect == got ? "OK" : "FAIL"));
    if(expect != got) ok = 0;
  }
  printf("total trees order<=%d: %d (cumulative A000081=%d)\n",
         max_ord, cat.size(), a000081_cumulative(max_ord));
  // print a few B-series symmetry factors
  printf("sample trees (id, order, sigma, density, key):\n");
  int lim = cat.size() < 20 ? cat.size() : 20;
  for(int id = 1; id <= lim; id++){
    const UTreeShape& sh = cat.get(id);
    printf("  id=%d |t|=%d sigma=%d gamma=%d key=%s\n",
           id, sh.order, sh.symmetry, sh.density, sh.key.c_str());
  }
  return ok ? 0 : 1;
}

int main(int argc, char **argv){

  int nhid = 50;
  int nlabels = 2;
  int maxch = 6;
  float lr = 0.1;
  int nmaxmax = 10;
  int nmin = 2;
  int nchar = 2;
  int nrep = 1;
  int ntask = 7; // default: unordered membrane parity
  int nseqtask = 1;
  int nshape = 4;
  int ntree = 1000;
  int seed = 1;
  bool save = false;
  double rho = 0.9;
  double input_scaling = 0.5;
  double cscale = 1.0;
  double sparsity = 0.1;
  double ridge = 1e-4;
  int map_mode = TESN_MAP_ROOT_MEAN;
  int nl = NL_TANH;
  string fitmode = "ridge";
  int nepoch = 15;
  int readout = PESN_READOUT_BSERIES;
  int bsorder = 4;
  int bsfeat = BS_FEAT_MEAN_ROOT;
  int bsdim = 2;
  double bsstep = 1.0;
  bool do_check = false;

  printf("For help: train_pesn_toy --help\n");

  int ai = 1;
  while(ai < argc){
    if( strcmp( argv[ai], "--help") == 0){
      print_help();
      return 1;
    }
    if( strcmp( argv[ai], "-check-a000081") == 0){
      do_check = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-nhid") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nhid = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nlabels") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nlabels = atoi(argv[ai+1]);
      if(nlabels < 2) {printf("error nlabels should be >= 2\n");return -1;}
    }
    else if( strcmp( argv[ai], "-maxch") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      maxch = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-ntree") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      ntree = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-lr") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      lr = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-rho") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      rho = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-is") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      input_scaling = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-cscale") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      cscale = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-sparsity") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      sparsity = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-ridge") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      ridge = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-map") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      map_mode = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nl") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nl = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-fit") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      fitmode = argv[ai+1];
    }
    else if( strcmp( argv[ai], "-readout") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      readout = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-bsorder") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      bsorder = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-bsfeat") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      bsfeat = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-bsdim") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      bsdim = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-bsstep") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      bsstep = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nepoch") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nepoch = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nrep") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nrep = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-ntask") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      ntask = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nseqtask") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nseqtask = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nchar") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nchar = atoi(argv[ai+1]);
      if(nchar < 2) {printf("error nchar should be >= 2\n");return -1;}
    }
    else if( strcmp( argv[ai], "-nshape") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nshape = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nmin") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nmin = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-seed") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      seed = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nmax") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nmaxmax = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-save") == 0){
      save = true;
      ai--;
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }

  if(do_check)
    return check_a000081(bsorder > 6 ? bsorder : 6);

  int nlab_in = nlabels;
  if(ntask == TTASK_SEQ_CHAIN || ntask == TTASK_SEQ_BALANCED)
    nlab_in = nchar;
  int nout = tree_nclass(ntask, nlabels, nchar, nshape);

  cout<<"seed: "<<seed<<endl<<"nhid: "<<nhid<<endl<<"nlabels: "<<nlab_in
      <<endl<<"nout: "<<nout<<endl<<"maxch: "<<maxch
      <<endl<<"rho: "<<rho<<endl<<"is: "<<input_scaling
      <<endl<<"cscale: "<<cscale<<endl<<"sparsity: "<<sparsity
      <<endl<<"ridge: "<<ridge<<endl<<"map: "<<map_mode
      <<endl<<"nl: "<<nl<<endl<<"fit: "<<fitmode
      <<endl<<"readout: "<<readout<<endl<<"bsorder: "<<bsorder
      <<endl<<"bsfeat: "<<bsfeat<<endl<<"task: "<<ntask<<endl;

  srand(seed);

  char buff[1000];
  sprintf(buff,"pesn_ntask%d_nhid%d_rho%.2f_ro%d_bso%d_seed%d",
      ntask, nhid, rho, readout, bsorder, seed);
  string modelname = "data/model_";
  modelname.append(buff);
  string testfilename = "data/test_";
  testfilename.append(buff);

  cout<<"create pesn (membrane P-system + B-series)...";
  PESN net(nlab_in, nhid, nout, maxch, rho, input_scaling, sparsity, ridge,
      map_mode, nl, cscale, readout, bsorder, bsfeat, bsdim, bsstep);
  PESN best(nlab_in, nhid, nout, maxch, rho, input_scaling, sparsity, ridge,
      map_mode, nl, cscale, readout, bsorder, bsfeat, bsdim, bsstep);
  cout<<"done (feat="<<net._FEAT;
  if(readout == PESN_READOUT_BSERIES || readout == PESN_READOUT_BOTH)
    cout<<", bseries_trees="<<net._bseries.ntrees();
  cout<<")"<<endl;

  // Quick A000081 note
  if(readout != PESN_READOUT_MAP){
    printf("B-series trees: order<=%d count=%d (A000081 cum=%d)\n",
           bsorder, net._bseries.ntrees(), a000081_cumulative(bsorder > 8 ? 8 : bsorder));
  }

  float last_acc = -1;
  bool use_ridge = (fitmode != "sgd");
  int nmax = 3;
  if(nmin >= nmax) nmax = nmin + 1;

  for(int e = 0; e < nepoch; e++){
    nmax = max(min(e+3, nmaxmax), 3);
    double lo = 0;
    int ne = 1;
    int ncorr = 0, ntot = 0;

    if(use_ridge) net.clearCollectors();

    for(int it = 0; it < ntree; it++){
      Tree tree;
      generate_tree_task(tree, ntask, nlabels, nmin, nmax, nchar, nrep, nseqtask,
                         nshape, bsorder > 6 ? 6 : bsorder);

      if(use_ridge)
        net.collect(tree);
      else {
        net.forward(tree);
        net.update(lr);
      }

      double pr = net.eval(tree.target);
      if(pr < 1e-12) pr = 1e-12;
      lo -= log(pr) / log(10.0);
      ne++;
      if(net.pred() == tree.target) ncorr++;
      ntot++;

      if(it % 50 == 0){
        fprintf(stdout, "\r [train] it=%7d nmax:%d  entropy: %.3f  acc: %.3f ",
            it, nmax, lo / ne, ntot ? (ncorr / (float)ntot) : 0.f);
      }
    }

    if(use_ridge){
      bool ok = net.fitReadout();
      fprintf(stdout, "\n [ridge] collected=%d fit=%s\n", net._ncollect, ok ? "ok" : "fail");
      lo = 0; ne = 1; ncorr = 0; ntot = 0;
      for(int it = 0; it < min(ntree, 300); it++){
        Tree tree;
        generate_tree_task(tree, ntask, nlabels, nmin, nmax, nchar, nrep, nseqtask,
                           nshape, bsorder > 6 ? 6 : bsorder);
        net.forward(tree);
        double pr = net.eval(tree.target);
        if(pr < 1e-12) pr = 1e-12;
        lo -= log(pr) / log(10.0);
        ne++;
        if(net.pred() == tree.target) ncorr++;
        ntot++;
      }
    }

    fprintf(stdout, "\r [train] epoch=%d nmax:%d  entropy: %.3f  acc: %.3f \n",
        e, nmax, lo / ne, ntot ? (ncorr / (float)ntot) : 0.f);

    // valid
    nmax = max(nmaxmax, 12); nmin = 2;
    lo = 0; ne = 1; ncorr = 0; ntot = 0;
    for(int it = 0; it < 500; it++){
      Tree tree;
      generate_tree_task(tree, ntask, nlabels, nmin, nmax, nchar, nrep, nseqtask,
                         nshape, bsorder > 6 ? 6 : bsorder);
      net.forward(tree);
      double pr = net.eval(tree.target);
      if(pr < 1e-12) pr = 1e-12;
      lo -= log(pr) / log(10.0);
      ne++;
      if(net.pred() == tree.target) ncorr++;
      ntot++;
    }
    float vacc = ntot ? (ncorr / (float)ntot) : 0.f;
    fprintf(stdout, " [valid] entropy: %.3f  acc: %.3f \n", lo / ne, vacc);

    if(e == 0 || vacc > last_acc){
      last_acc = vacc;
      best.copy(net);
      if(save) best.save(modelname);
    } else if(!use_ridge && e > nmaxmax/2){
      lr /= 2;
      net.copy(best);
    }
    if(!use_ridge && lr < 1e-5) break;
    nmin = 2;
  }

  // size generalization test
  srand(10);
  net.copy(best);
  fprintf(stdout,"Test set (by nnodes / n):\n");
  FILE* fres = NULL;
  if(save){
    fres = fopen(testfilename.c_str(),"w");
    if(fres) fprintf(fres,"validation:\t %f\n", last_acc);
  }

  for(int nm = 2; nm < 30; nm++){
    nmin = nm; nmax = nm + 1;
    int ncorr = 0, ntot = 0;
    for(int it = 0; it < 200; it++){
      Tree tree;
      generate_tree_task(tree, ntask, nlabels, nmin, nmax, nchar, nrep, nseqtask,
                         nshape, bsorder > 6 ? 6 : bsorder);
      net.forward(tree);
      if(net.pred() == tree.target) ncorr++;
      ntot++;
    }
    float acc = ntot ? (ncorr / (float)ntot) : 0.f;
    if(fres) fprintf(fres,"%d \t %f\n", nm, acc);
    fprintf(stdout,"n: %d \t accuracy: %f \n", nm, acc);
  }
  fprintf(stdout, "\n");
  if(fres) fclose(fres);

  return 0;
}
