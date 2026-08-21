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
#include <unordered_map>
#include <string>
#include <vector>

#include "common.h"
#include "task.h"
#include "ESN.h"

using namespace std;
using namespace rnn;

int main(int argc, char **argv){

  int nhid = 200;
  int nstack = 0;
  int stack_size = 200;
  int mem = 50;
  float lr = 0.1;
  int mod = 2;
  int nmaxmax = 20;
  int nmin = 2;
  bool isnoop = true;
  bool ishard = false;
  int nreset = 10;
  int base = 2;
  int depth = 2;
  int nseq = 5000;
  int seed = 22;
  bool save = false;
  int nvalidmax = 20;
  double rho = 0.9;
  double input_scaling = 0.5;
  double alpha = 1.0;
  double sparsity = 0.1;
  double ridge = 1e-4;
  int washout = 5;
  int feat = FEAT_RES_IN;
  int nl = NL_TANH;
  string fitmode = "ridge";
  int nepoch = 15;

  int ai = 1;
  while(ai < argc){
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
    else if( strcmp( argv[ai], "-base") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      base = atoi(argv[ai+1]);
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
    else if( strcmp( argv[ai], "-nvalidmax") == 0){
      if(ai + 1 >= argc) { printf("error need argument for option %s\n",argv[ai]); return -1;}
      nvalidmax = atoi(argv[ai+1]);
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
      if(depth < 1) {printf("error in depth...\n"); return -1;}
    }
    else{
      printf("unknown option: %s\n",argv[ai]);
      return -1;
    }
    ai += 2;
  }
  srand(seed);

  if(nstack > 0 && feat == FEAT_RES) feat = FEAT_RES_IN_STACK;

  cout<<"seed: "<<seed<<endl<<"nhid: "<<nhid<<endl<<"nstack: "<<nstack
      <<endl<<"mod: "<<mod<<endl<<"depth: "<<depth<<endl<<"rho: "<<rho
      <<endl<<"fit: "<<fitmode<<endl<<"base: "<<base<<endl;

  int nchar = 3 + base;
  unordered_map<char, int> dic;
  vector<char> rdic(nchar,0);
  dic['+'] = 0; rdic[0] = '+';
  dic['='] = 1; rdic[1] = '=';
  dic['.'] = 2; rdic[2] = '.';
  for(int i = 0; i < nchar -3; i++)
  {  dic['0'+i] = 3 + i; rdic[3 + i] = '0' + i;}

  cout<<"create esn...";
  ESN esn(nchar, nhid, nchar, nstack, stack_size, mem, mod, isnoop, depth,
      rho, input_scaling, alpha, sparsity, ridge, feat, nl);
  ESN best(nchar, nhid, nchar, nstack, stack_size, mem, mod, isnoop, depth,
      rho, input_scaling, alpha, sparsity, ridge, feat, nl);
  cout<<"done (feat="<<esn._FEAT<<")"<<endl;

  char buff[1000];
  sprintf(buff,"esn_addition_base%d_nhid%d_nstack%d_mod%d_seed%d",
      base, nhid, nstack, mod, seed);
  string modelname = "data/model_";
  modelname.append(buff);
  string testfilename = "data/test_";
  testfilename.append(buff);

  int cur = nchar - 1, next = 0;
  int nmax = 3;
  if(nmin >= nmax) nmax = nmin + 1;
  int nseqv = 500;
  bool use_ridge = (fitmode != "sgd");
  float last_ent = 1e9;
  string spred, sgoal;

  for(int e = 0; e < nepoch; e++){
    nmax = max(min(e+3, nmaxmax), 3);
    nmin = 0;
    int ne = 1; double lo = 0;

    esn.resetState();
    if(use_ridge) esn.clearCollectors();

    for(int iseq = 0; iseq < nseq; iseq++) {
      string p = generate_addition(nmax, nmin, base);
      if(nreset > 0 && iseq % nreset == 0 ) esn.resetState();
      bool iseval = false;
      for(size_t ip = 0; ip < p.size(); ip++){
        next = dic[p[ip]];
        if(rdic[cur] == '=') iseval = true;

        if(use_ridge){
          // only collect supervised part (after '=') for readout; still drive reservoir always
          esn.forward(cur, next);
          if(iseval && esn._steps_since_reset > washout){
            esn.buildFeatures();
            for(my_int i = 0; i < esn._FEAT; i++){
              my_real fi = esn._feat[i];
              if(fi == 0) continue;
              for(my_int j = 0; j < esn._FEAT; j++)
                esn._gram(i, j) += fi * esn._feat[j];
            }
            if(next >= 0 && next < esn._OUT){
              for(my_int i = 0; i < esn._FEAT; i++)
                esn._B(i, next) += esn._feat[i];
            }
            esn._ncollect++;
          }
        } else {
          esn.forward(cur, next);
          if(iseval) esn.update(lr);
        }

        spred += (iseval)? rdic[esn.pred()] : '_'; sgoal += rdic[next];
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);

        if(iseval){
          double pr = esn.eval(next);
          if(pr < 1e-12) pr = 1e-12;
          lo -= log(pr) / log(10.0); ne++;
          fprintf(stdout, "\r[train] nmax: %02d entropy: %.3f goal: %s pred: %s prog=%.1f%%",
              nmax, lo / ne, sgoal.c_str(), spred.c_str(), 100.0 * iseq / nseq);
        }
        cur = next;
      }
    }

    if(use_ridge){
      bool ok = esn.fitReadout();
      fprintf(stdout, "\n[ridge] collected=%d fit=%s\n", esn._ncollect, ok ? "ok" : "fail");
    }

    fprintf(stdout, "\r[train] nmax: %02d entropy: %.3f goal: %s pred: %s\n",
        nmax, lo / ne, sgoal.c_str(), spred.c_str());

    // valid
    nmax = max(nmaxmax, nvalidmax);
    nmin = min(nmaxmax, nvalidmax);
    ne = 1; lo = 0;
    esn.resetState();

    for(int iseq = 0; iseq < nseqv; iseq++){
      string p = generate_addition(nmax, nmin, base);
      bool iseval = false;
      for(size_t ip = 0; ip < p.size(); ip++){
        next = dic[p[ip]];
        if(rdic[cur] == '=') iseval = true;
        esn.forward(cur, next, ishard);
        spred += (iseval)? rdic[esn.pred()] : '_'; sgoal += rdic[next];
        if (spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if (sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);
        if(iseval){
          double pr = esn.eval(next);
          if(pr < 1e-12) pr = 1e-12;
          lo -= log(pr) / log(10.0);
          ne++;
          fprintf(stdout, "\r[valid] nmax: %d entropy: %.3f goal: %s pred: %s prog=%.1f%%",
              nmax, lo / ne, sgoal.c_str(), spred.c_str(), 100.0 * iseq / nseqv);
        }
        cur = next;
      }
    }
    fprintf(stdout, "\r[valid] nmax: %02d entropy: %.3f goal: %s pred: %s\n",
        nmax, lo / ne, sgoal.c_str(), spred.c_str());

    if(e == 0 || lo / ne < last_ent){
      last_ent = lo / ne;
      best.copy(esn);
      if(save) best.save(modelname);
    } else if(!use_ridge && e > nmaxmax/2){
      lr /= 2;
      esn.copy(best);
    }
    if(!use_ridge && lr < 1e-5) break;
  }

  esn.copy(best);
  fprintf(stdout,"Test set:\n");
  FILE* fres = NULL;
  if(save){
    fres = fopen(testfilename.c_str(),"w");
    if(fres) fprintf(fres,"validation:\t %f\n", last_ent);
  }
  int ntest = 200;
  cur = nchar - 1;
  esn.resetState();

  for(int nm = 2; nm < 40; nm++){
    nmin = nm; nmax = nm + 1;
    float corr = 0, ecorr = 0;
    int sseq = 0;
    int neval = 0;

    for(int iseq = 0; iseq < ntest; iseq++){
      string p = generate_addition(nmax, nmin, base);
      bool iseval = false;
      if(nreset > 0 && iseq % nreset == 0 ) esn.resetState();

      for(size_t ip = 0; ip < p.size(); ip++){
        next = dic[p[ip]];
        esn.forward(cur, next, ishard);

        if (ip == 0) {
          if(iseq != 0){
            neval++;
            if( corr == sseq ) ecorr++;
          }
          sseq=0; corr = 0;
          iseval = false;
        }

        if(iseval && next == esn.pred()) corr++;
        if(iseval) sseq++;

        if(rdic[next] == '=') iseval = true;
        cur = next;
      }
    }
    if(neval == 0) neval = 1;
    if(fres) fprintf(fres,"%d \t %f\n", nm, ecorr / neval);
    fprintf(stdout,"n: %d \t accuracy: %f \n", nm, ecorr / neval);
  }
  fprintf(stdout, "\n");
  if(fres) fclose(fres);
  return 0;
}
