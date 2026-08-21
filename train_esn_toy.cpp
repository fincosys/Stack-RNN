/*
 *  Copyright (c) 2015-present, Facebook, Inc.
 *  All rights reserved.
 *
 *  This source code is licensed under the BSD-style license found in the
 *  LICENSE file in the root directory of this source tree. An additional grant
 *  of patent rights can be found in the PATENTS file in the same directory.
 *
 */
#include <ctime>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>

#include "common.h"
#include "task.h"
#include "ESN.h"

using namespace std;
using namespace rnn;

void print_help(){
  printf("train_esn_toy trains an Echo-State Network (reservoir computer) on toy tasks\n");
  printf("usage: train_esn_toy [options]\n");
  printf("options:\n");
  printf("-nhid [integer]\t\t reservoir size. Default: 100\n");
  printf("-nstack [integer]\t number of stacks (0 = plain ESN). Default: 0\n");
  printf("-depth [integer]\t stack depth used in features / recurrence. Default: 1\n");
  printf("-stack_size [integer]\t stack capacity. Default: 200\n");
  printf("-mem [integer]\t\t state history length. Default: 50\n");
  printf("-nseq [integer]\t\t sequences per epoch. Default: 2000\n");
  printf("-mod [integer]\t\t 0=no rec, 1=stack rec, 2=full W_res (+stacks). Default: 2\n");
  printf("-lr [float]\t\t SGD learning rate (fit=sgd). Default: 0.1\n");
  printf("-rho [float]\t\t spectral radius. Default: 0.9\n");
  printf("-is [float]\t\t input scaling. Default: 0.5\n");
  printf("-alpha [float]\t\t leaking rate. Default: 1.0\n");
  printf("-sparsity [float]\t fraction of nonzero W_res. Default: 0.1\n");
  printf("-ridge [float]\t\t ridge regression lambda. Default: 1e-4\n");
  printf("-washout [integer]\t washout steps after reset. Default: 0\n");
  printf("-feat [integer]\t\t 0=res, 1=res+in, 2=res+stack, 3=res+in+stack. Default: 0\n");
  printf("-nl [integer]\t\t 0=tanh, 1=sigmoid. Default: 0\n");
  printf("-fit [string]\t\t ridge|sgd. Default: ridge\n");
  printf("-nepoch [integer]\t number of epochs. Default: 20\n");
  printf("-nreset [integer]\t empty state every N sequences. Default: 1\n");
  printf("-ntask [integer]\t task id. Default: 1\n");
  printf("-nchar [integer]\t alphabet size. Default: 2\n");
  printf("-nrep [integer]\t\t task2 repetitions. Default: 1\n");
  printf("-seed [integer]\t\t RNG seed. Default: 1\n");
  printf("-nmax [integer]\t\t max n in tasks. Default: 10\n");
  printf("-noop\t\t\t enable stack no-op action\n");
  printf("-hard\t\t\t discrete stack actions at valid/test\n");
  printf("-save\t\t\t save model and logs\n");
}

int main(int argc, char **argv){

  int nhid = 100;
  int nstack = 0;
  int stack_size = 200;
  int mem = 50;
  float lr = 0.1;
  int mod = 2;
  int nmaxmax = 10;
  int nmin = 2;
  bool isnoop = false;
  bool ishard = false;
  int nchar = 2;
  int nrep = 1;
  int nreset = 1; // ESN: reset reservoir each sequence by default
  int ntask = 1;
  int depth = 1;
  int nseq = 2000;
  int seed = 1;
  bool save = false;
  double rho = 0.9;
  double input_scaling = 0.5;
  double alpha = 1.0;
  double sparsity = 0.1;
  double ridge = 1e-4;
  int washout = 0;
  int feat = FEAT_RES;
  int nl = NL_TANH;
  string fitmode = "ridge";
  int nepoch = 20;

  printf("For help: train_esn_toy --help\n");

  int ai = 1;
  while(ai < argc){
    if( strcmp( argv[ai], "--help") == 0){
      print_help();
      return 1;
    }
    if( strcmp( argv[ai], "-nhid") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nhid = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nseq") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nseq = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nstack") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nstack = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-stack_size") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      stack_size = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-mem") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      mem = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-mod") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      mod = atoi(argv[ai+1]);
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
    else if( strcmp( argv[ai], "-alpha") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      alpha = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-sparsity") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      sparsity = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-ridge") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      ridge = atof(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-washout") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      washout = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-feat") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      feat = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nl") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nl = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-fit") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      fitmode = argv[ai+1];
    }
    else if( strcmp( argv[ai], "-nepoch") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nepoch = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nreset") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nreset = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nrep") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nrep = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-ntask") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      ntask = atoi(argv[ai+1]);
    }
    else if( strcmp( argv[ai], "-nchar") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nchar = atoi(argv[ai+1]);
      if(nchar < 2) {printf("error nchar should be >= 2\n");return -1;}
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
    else if( strcmp( argv[ai], "-noop") == 0){
      isnoop = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-save") == 0){
      save = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-hard") == 0){
      ishard = true;
      ai--;
    }
    else if( strcmp( argv[ai], "-depth") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      depth = atoi(argv[ai+1]);
      if(depth < 1) {printf("error depth < 1\n"); return -1;}
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }

  // Auto-adjust feat mode if stacks requested with default feat
  if(nstack > 0 && feat == FEAT_RES)
    feat = FEAT_RES_STACK;

  cout<<"seed: "<<seed<<endl<<"nhid: "<<nhid<<endl<<"nstack: "<<nstack
      <<endl<<"mod: "<<mod<<endl<<"depth: "<<depth<<endl<<"rho: "<<rho
      <<endl<<"is: "<<input_scaling<<endl<<"alpha: "<<alpha
      <<endl<<"sparsity: "<<sparsity<<endl<<"ridge: "<<ridge
      <<endl<<"feat: "<<feat<<endl<<"nl: "<<nl<<endl<<"fit: "<<fitmode
      <<endl<<"task: "<<ntask<<" nchar:"<<nchar<<" nrep: "<<nrep<<endl;

  srand(seed);

  char buff[1000];
  sprintf(buff,"esn_ntask%d_nchar%d_nhid%d_nstack%d_mod%d_depth%d_rho%.2f_feat%d_seed%d",
      ntask, nchar, nhid, nstack, mod, depth, rho, feat, seed);
  string modelname = "data/model_";
  modelname.append(buff);
  string testfilename = "data/test_";
  testfilename.append(buff);

  cout<<"create esn...";
  ESN esn(nchar, nhid, nchar, nstack, stack_size, mem, mod, isnoop, depth,
      rho, input_scaling, alpha, sparsity, ridge, feat, nl);
  ESN best(nchar, nhid, nchar, nstack, stack_size, mem, mod, isnoop, depth,
      rho, input_scaling, alpha, sparsity, ridge, feat, nl);
  cout<<"done (feat="<<esn._FEAT<<", ncollect path="<<fitmode<<")"<<endl;

  int cur = nchar - 1, next = 0;
  int nmax = 3;
  if(nmin >= nmax) nmax = nmin + 1;

  string spred(50,'#');
  string sgoal(50,'#');

  float last_ent = 1e9;
  bool use_ridge = (fitmode != "sgd");

  for(int e = 0; e < nepoch; e++){
    nmax = max(min(e+3, nmaxmax), 3);
    int ne = 1;
    double lo = 0;
    int count = 0;

    esn.resetState();

    if(use_ridge)
      esn.clearCollectors();

    // ---- train / collect ----
    for(int iseq = 0; iseq < nseq; iseq++){
      string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      spred += '_'; sgoal += '_';
      if(nreset == 1 || (nreset > 0 && iseq % nreset == 0)) esn.resetState();
      cur = nchar - 1;

      for(size_t ip = 0; ip < p.size(); ip++){
        next = p[ip] - 'a';

        if(use_ridge){
          esn.collect(cur, next, washout);
        } else {
          esn.forward(cur, next);
          esn.update(lr);
        }

        if(esn._steps_since_reset > washout){
          double pr = esn.eval(next);
          if(pr < 1e-12) pr = 1e-12;
          lo -= log(pr) / log(10.0);
          ne++;
        }

        spred += 'a' + esn.pred();
        sgoal += 'a' + next;
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);

        if(ip == 0){
          fprintf(stdout, "\r [train] it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s ",
              count, nmax, lo / ne, sgoal.c_str(), spred.c_str());
        }
        cur = next;
        count++;
      }
    }

    if(use_ridge){
      bool ok = esn.fitReadout();
      fprintf(stdout, "\n [ridge] collected=%d fit=%s\n", esn._ncollect, ok ? "ok" : "fail");
      // re-eval train entropy after fit on a small pass
      esn.resetState();
      ne = 1; lo = 0;
      for(int iseq = 0; iseq < min(nseq, 200); iseq++){
        string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
        if(nreset == 1 || (nreset > 0 && iseq % nreset == 0)) esn.resetState();
        for(size_t ip = 0; ip < p.size(); ip++){
          next = p[ip] - 'a';
          esn.forward(cur, next);
          if(esn._steps_since_reset > washout){
            double pr = esn.eval(next);
            if(pr < 1e-12) pr = 1e-12;
            lo -= log(pr) / log(10.0);
            ne++;
          }
          cur = next;
        }
      }
    }

    fprintf(stdout, "\r [train] it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s \n",
        count, nmax, lo / ne, sgoal.c_str(), spred.c_str());

    // ---- valid ----
    nmax = max(nmaxmax, 20); nmin = 2;
    if(nstack==0 && mod != 2) nmax = nmaxmax;
    ne = 1; lo = 0;
    count = 0;

    for(int iseq = 0; iseq < 1000; iseq++){
      string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      spred += '_'; sgoal += '_';
      esn.resetState();
      cur = nchar - 1;

      for(size_t ip = 0; ip < p.size(); ip++){
        next = p[ip] - 'a';
        esn.forward(cur, next, ishard);

        double pr = esn.eval(next);
        if(pr < 1e-12) pr = 1e-12;
        lo -= log(pr) / log(10.0);
        ne++;

        spred += 'a' + esn.pred();
        sgoal += 'a' + next;
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);

        fprintf(stdout, "\r [valid] it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s ",
            count, nmax, lo / ne, sgoal.c_str(), spred.c_str());
        cur = next;
        count++;
      }
    }
    fprintf(stdout, "\r [valid] it=%7d nmax:%d  entropy: %.3f  goal: %s pred: %s \n",
        count, nmax, lo / ne, sgoal.c_str(), spred.c_str());

    if(e == 0 || lo / ne < last_ent){
      last_ent = lo / ne;
      best.copy(esn);
      if(save) best.save(modelname);
    } else if(!use_ridge && e > nmaxmax/2){
      lr /= 2;
      esn.copy(best);
    }
    if(!use_ridge && lr < 1e-5) break;

    // For ridge, one good collect+fit is often enough; still allow multi-epoch curriculum
    nmin = 2;
  }

  // ---- test length generalization ----
  srand(10);
  esn.copy(best);
  fprintf(stdout,"Test set:\n");
  int ntest = 200;
  bool iscountfirstelement = (ntask != 4);

  FILE* fres = NULL;
  if(save){
    fres = fopen(testfilename.c_str(),"w");
    if(fres) fprintf(fres,"validation:\t %f\n", last_ent);
  }

  for(int nm = 2; nm < 60; nm++){
    nmin = nm; nmax = nm + 1;
    float ecorr = 0;
    int neval = 0;

    for(int iseq = 0; iseq < ntest; iseq++){
      string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      esn.resetState();
      cur = nchar - 1;
      bool iseval = false;
      float corr = 0;
      int sseq = 0;

      for(size_t ip = 0; ip < p.size(); ip++){
        next = p[ip] - 'a';
        esn.forward(cur, next, ishard);

        if(iseval){
          sseq++;
          if(next == esn.pred()) corr++;
        }

        if( (ntask == 1 && cur == 0 && next != 0)
            || (ntask == 2 && cur == 0 && next!= 0)
            || (ntask == 3 && cur == nchar -2 && next == nchar - 1)
            || (ntask == 4 && next == 0)
            || (ntask == 6 && cur == 1 && next == 2)
            || (ntask == 5 && cur == nchar -2 && next == nchar - 1) ){
          iseval = true;
        }

        cur = next;
      }
      // end-of-sequence accuracy (optionally require first-symbol continuity — skipped with per-seq reset)
      neval++;
      if(sseq > 0 && corr == sseq) ecorr++;
      (void)iscountfirstelement;
    }
    if(neval == 0) neval = 1;
    if(fres) fprintf(fres,"%d \t %f\n", nm, ecorr / neval);
    fprintf(stdout,"n: %d \t accuracy: %f \n", nm, ecorr / neval);
  }
  fprintf(stdout, "\n");
  if(fres) fclose(fres);

  return 0;
}
