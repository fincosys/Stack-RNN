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
#include <vector>

#include "common.h"
#include "task.h"
#include "ESN.h"

using namespace std;
using namespace rnn;

/****************
  Train an Echo-State Network (optional Stack-ESN) on toy algorithmic tasks.
  Readout is trained with ridge regression (default) or SGD.
 **************/

void print_help(){
  printf("train_esn trains an Echo-State Network on simple toy tasks\n");
  printf("usage: train_esn [options]\n");
  printf("options:\n");
  printf("-nhid [integer]\t\t reservoir size. Default: 200\n");
  printf("-nstack [integer]\t number of stacks (0 = classic ESN). Default: 0\n");
  printf("-depth [integer]\t stack depth used for reservoir/readout. Default: 1\n");
  printf("-stack_size [integer]\t stack capacity. Default: 200\n");
  printf("-spectral_radius [float]\t reservoir spectral radius. Default: 0.9\n");
  printf("-sparsity [float]\t fraction of nonzero reservoir weights. Default: 0.1\n");
  printf("-input_scaling [float]\t scale of input weights. Default: 0.5\n");
  printf("-leak [float]\t\t leaky integrator rate in (0,1]. Default: 1.0\n");
  printf("-washout [integer]\t steps discarded per sequence for ridge. Default: 0\n");
  printf("-ridge [float]\t\t ridge regression lambda. Default: 1e-6\n");
  printf("-train_mode [ridge|sgd]\t readout training mode. Default: ridge\n");
  printf("-act [tanh|sigmoid]\t reservoir activation. Default: tanh\n");
  printf("-lr [float]\t\t learning rate (sgd mode). Default: 0.1\n");
  printf("-nseq [integer]\t\t sequences per training epoch. Default: 2000\n");
  printf("-nreset [integer]\t how often stacks/reservoir reset. Default: 1\n");
  printf("-ntask [integer]\t task id (see task.h / script_esn.sh). Default: 1\n");
  printf("-nchar [integer]\t alphabet size. Default: 2\n");
  printf("-nrep [integer]\t\t repetition factor (task 2). Default: 1\n");
  printf("-seed [integer]\t\t RNG seed. Default: 1\n");
  printf("-nmax [integer]\t\t max n for curriculum. Default: 10\n");
  printf("-nepoch [integer]\t number of epochs. Default: 20\n");
  printf("-mod [integer]\t\t 0=no stack->res, 1=stack tops into reservoir. Default: 1\n");
  printf("-noop \t\t\t enable no-op stack action. Default: false\n");
  printf("-hard \t\t\t discrete stack actions at valid/test. Default: false\n");
  printf("-save \t\t\t save model and logs under data/. Default: false\n");
  printf("Example:\n ./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 0 -ridge 1e-4 -nmax 10\n");
}

void print_step(ESN& esn, FILE* f, int cur, int next){
  fprintf(f, "cur: %c next: %c pred: %c ", 'a' + cur, 'a' + next, 'a' + esn.pred());
  fprintf(f, "prob[%c]: %f\n", 'a' + next, esn.eval(next));
}

int main(int argc, char **argv){
  int nhid = 200;
  int nstack = 0;
  int stack_size = 200;
  float lr = 0.1;
  string modelname = "model";
  int mod = 1;
  int nmaxmax = 10;
  int nmin = 2;
  bool isnoop = false;
  bool ishard = false;
  int nchar = 2;
  int nrep = 1;
  int nreset = 1;
  int ntask = 1;
  int depth = 1;
  int nseq = 2000;
  int seed = 1;
  bool save = false;
  int nepoch = 20;
  float spectral_radius = 0.9;
  float sparsity = 0.1;
  float input_scaling = 0.5;
  float leak = 1.0;
  int washout = 0;
  float ridge = 1e-6;
  string train_mode = "ridge";
  int act_type = ESN::ACT_TANH;

  printf("For help: train_esn --help\n");

  int ai = 1;
  while(ai < argc){
    if(strcmp(argv[ai], "--help") == 0){
      print_help();
      return 1;
    }
    if(strcmp(argv[ai], "-nhid") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nhid = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nseq") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nseq = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nstack") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nstack = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-stack_size") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      stack_size = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-mod") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      mod = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-lr") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      lr = atof(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nreset") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nreset = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nrep") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nrep = atoi(argv[ai+1]);
      if(nrep < 1){ printf("error nrep should be >= 1\n"); return -1; }
    }
    else if(strcmp(argv[ai], "-ntask") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      ntask = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nchar") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nchar = atoi(argv[ai+1]);
      if(nchar < 2){ printf("error nchar should be >= 2\n"); return -1; }
    }
    else if(strcmp(argv[ai], "-nmin") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nmin = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-seed") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      seed = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nmax") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nmaxmax = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-nepoch") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      nepoch = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-spectral_radius") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      spectral_radius = atof(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-sparsity") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      sparsity = atof(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-input_scaling") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      input_scaling = atof(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-leak") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      leak = atof(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-washout") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      washout = atoi(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-ridge") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      ridge = atof(argv[ai+1]);
    }
    else if(strcmp(argv[ai], "-train_mode") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      train_mode = argv[ai+1];
      if(train_mode != "ridge" && train_mode != "sgd"){
        printf("error -train_mode must be ridge or sgd\n"); return -1;
      }
    }
    else if(strcmp(argv[ai], "-act") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      if(strcmp(argv[ai+1], "tanh") == 0) act_type = ESN::ACT_TANH;
      else if(strcmp(argv[ai+1], "sigmoid") == 0) act_type = ESN::ACT_SIGMOID;
      else { printf("error -act must be tanh or sigmoid\n"); return -1; }
    }
    else if(strcmp(argv[ai], "-noop") == 0){
      isnoop = true;
      ai--;
    }
    else if(strcmp(argv[ai], "-save") == 0){
      save = true;
      ai--;
    }
    else if(strcmp(argv[ai], "-hard") == 0){
      ishard = true;
      ai--;
    }
    else if(strcmp(argv[ai], "-depth") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      depth = atoi(argv[ai+1]);
      if(depth < 1){ printf("error depth should be >= 1\n"); return -1; }
    }
    else if(strcmp(argv[ai], "-name") == 0){
      if(ai + 1 >= argc){ printf("error need argument for option %s\n", argv[ai]); return -1; }
      modelname = argv[ai+1];
    }
    else{
      printf("unknown option: %s\n", argv[ai]);
      return -1;
    }
    ai += 2;
  }

  cout << "seed: " << seed
       << " nhid: " << nhid
       << " nstack: " << nstack
       << " depth: " << depth
       << " spectral_radius: " << spectral_radius
       << " sparsity: " << sparsity
       << " leak: " << leak
       << " ridge: " << ridge
       << " train_mode: " << train_mode
       << " task: " << ntask
       << " nchar: " << nchar
       << endl;

  srand(seed);

  char buff[1000];
  sprintf(buff,
      "esn_ntask%d_nchar%d_nhid%d_nstack%d_mod%d_depth%d_sr%.2f_sp%.2f_leak%.2f_mode%s_seed%d",
      ntask, nchar, nhid, nstack, mod, depth,
      spectral_radius, sparsity, leak, train_mode.c_str(), seed);
  modelname = "data/model_";
  modelname.append(buff);
  string logfilename("data/log_");
  logfilename.append(buff);
  string testfilename = "data/test_";
  testfilename.append(buff);

  if(save){
    cout << "Model saved in: " << modelname << endl;
    cout << "Log file: " << logfilename << endl;
    cout << "Test results: " << testfilename << endl;
  }

  cout << "create esn...";
  ESN esn(nchar, nhid, nstack, stack_size, nchar,
      spectral_radius, input_scaling, sparsity, leak,
      washout, ridge, isnoop, depth, mod, act_type, true);
  ESN back_up(nchar, nhid, nstack, stack_size, nchar,
      spectral_radius, input_scaling, sparsity, leak,
      washout, ridge, isnoop, depth, mod, act_type, true);
  // Re-init backup from same seed state is wrong; copy after first good epoch.
  cout << "done (feat_dim=" << esn.featureDim() << ")" << endl;

  int cur = nchar - 1, next = 0;
  int nmax = 3;
  if(nmin >= nmax) nmax = nmin + 1;

  string spred(50, '#');
  string sgoal(50, '#');
  vector<string> sstacks(nstack);
  for(int s = 0; s < nstack; s++) sstacks[s] = string(50, '#');

  float last_ent = 1e9;
  bool have_backup = false;
  FILE* f = NULL;

  for(int e = 0; e < nepoch; e++){
    if(save) f = fopen(logfilename.c_str(), "w");
    nmax = max(min(e + 3, nmaxmax), 3);
    int count = 0;
    int ne = 0;
    double lo = 0;

    esn.reset();
    esn.clearCollected();

    // ---- training ----
    for(int iseq = 0; iseq < nseq; iseq++){
      string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      if(save) fprintf(f, "begin sequence\n");
      spred += '_'; sgoal += '_';
      for(int s = 0; s < nstack; s++) sstacks[s] += '_';

      if(nreset == 1 || (nreset > 0 && iseq % nreset == 0)) esn.reset();
      else {
        // keep reservoir continuous but still mark sequence boundary for washout
        esn._step_in_seq = 0;
      }

      cur = nchar - 1;
      for(int ip = 0; ip < (int)p.size(); ip++){
        next = p[ip] - 'a';
        esn.forward(cur, next, false);

        if(train_mode == "ridge"){
          esn.collectState();
        } else {
          // SGD on readout only
          esn.backwardReadoutOnly();
          esn.updateReadout(lr);
        }

        double prob = esn.eval(next);
        if(prob < 1e-15) prob = 1e-15;
        lo -= log(prob) / log(10.0);
        ne++;

        if(save) print_step(esn, f, cur, next);

        spred += 'a' + esn.pred();
        sgoal += 'a' + next;
        for(int s = 0; s < nstack; s++){
          if(esn._act[s][ESN_POP] > 0.7) sstacks[s] += '-';
          else if(esn._act[s][ESN_PUSH] > 0.7) sstacks[s] += '+';
          else if(isnoop && esn._act[s][ESN_NOOP] > 0.7) sstacks[s] += '|';
          else sstacks[s] += 'X';
        }
        if(spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if(sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);
        for(int s = 0; s < nstack; s++)
          if(sstacks[s].size() > 30)
            sstacks[s].erase(sstacks[s].begin(), sstacks[s].end() - 30);

        cur = next;
        count++;
      }
      if(iseq % 50 == 0){
        fprintf(stdout,
            "\r [train] epoch:%d nmax:%d it=%7d entropy: %.3f goal: %s pred: %s ",
            e, nmax, count, (ne > 0 ? lo / ne : 0), sgoal.c_str(), spred.c_str());
        fflush(stdout);
      }
    }

    if(train_mode == "ridge"){
      fprintf(stdout, "\n fitting ridge on %d samples...\n", esn.numCollected());
      bool ok = esn.fitReadoutRidge();
      if(!ok) fprintf(stdout, "warning: ridge solve failed\n");
      // Re-evaluate train entropy quickly on a small subset after fit.
      lo = 0; ne = 0;
      esn.reset();
      for(int iseq = 0; iseq < min(nseq, 200); iseq++){
        string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
        if(nreset == 1 || (nreset > 0 && iseq % nreset == 0)) esn.reset();
        cur = nchar - 1;
        for(int ip = 0; ip < (int)p.size(); ip++){
          next = p[ip] - 'a';
          esn.forward(cur, next, false);
          double prob = esn.eval(next);
          if(prob < 1e-15) prob = 1e-15;
          lo -= log(prob) / log(10.0);
          ne++;
          cur = next;
        }
      }
    }

    fprintf(stdout,
        "\r [train] epoch:%d nmax:%d it=%7d entropy: %.3f goal: %s pred: %s ",
        e, nmax, count, (ne > 0 ? lo / ne : 0), sgoal.c_str(), spred.c_str());
    for(int s = 0; s < min(nstack, 5); s++)
      fprintf(stdout, "| stack[%d]=%s", s, sstacks[s].c_str());
    fprintf(stdout, "\n");

    // ---- validation ----
    nmax = max(nmaxmax, 20);
    nmin = 2;
    lo = 0; ne = 0; count = 0;
    esn.reset();
    cur = nchar - 1;
    if(save) fprintf(f, "[VALID]\n");

    for(int iseq = 0; iseq < 500; iseq++){
      string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      spred += '_'; sgoal += '_';
      if(nreset == 1) esn.reset();
      else esn._step_in_seq = 0;

      if(save) fprintf(f, "begin sequence\n");
      cur = nchar - 1;
      for(int ip = 0; ip < (int)p.size(); ip++){
        next = p[ip] - 'a';
        esn.forward(cur, next, ishard);

        double prob = esn.eval(next);
        if(prob < 1e-15) prob = 1e-15;
        lo -= log(prob) / log(10.0);
        ne++;

        if(save) print_step(esn, f, cur, next);
        spred += 'a' + esn.pred();
        sgoal += 'a' + next;
        if(spred.size() > 30) spred.erase(spred.begin(), spred.end() - 30);
        if(sgoal.size() > 30) sgoal.erase(sgoal.begin(), sgoal.end() - 30);

        cur = next;
        count++;
      }
      if(iseq % 50 == 0){
        fprintf(stdout,
            "\r [valid] epoch:%d nmax:%d it=%7d entropy: %.3f goal: %s pred: %s ",
            e, nmax, count, (ne > 0 ? lo / ne : 0), sgoal.c_str(), spred.c_str());
        fflush(stdout);
      }
    }
    fprintf(stdout,
        "\r [valid] epoch:%d nmax:%d it=%7d entropy: %.3f goal: %s pred: %s \n",
        e, nmax, count, (ne > 0 ? lo / ne : 0), sgoal.c_str(), spred.c_str());

    float vent = (ne > 0 ? lo / ne : 1e9);
    if(!have_backup || vent < last_ent){
      last_ent = vent;
      back_up.copy(esn);
      have_backup = true;
      if(save) back_up.save(modelname);
    } else if(train_mode == "sgd" && e > nmaxmax / 2){
      lr /= 2;
      esn.copy(back_up);
    }

    if(save) fclose(f);
    if(train_mode == "sgd" && lr < 1e-5) break;
  }

  if(have_backup) esn.copy(back_up);

  // ---- short test ----
  fprintf(stdout, "Test set:\n");
  FILE* fres = NULL;
  if(save){
    fres = fopen(testfilename.c_str(), "w");
    fprintf(fres, "validation:\t %f\n", last_ent);
  }

  int ntest = 100;
  for(int nm = 2; nm < 30; nm++){
    nmin = nm; nmax = nm + 1;
    double lo = 0; int ne = 0;
    int correct_seq = 0;
    esn.reset();
    for(int iseq = 0; iseq < ntest; iseq++){
      string p = generate_next_sequence(nmax, nmin, nchar, nrep, ntask);
      if(nreset == 1) esn.reset();
      cur = nchar - 1;
      bool ok = true;
      for(int ip = 0; ip < (int)p.size(); ip++){
        next = p[ip] - 'a';
        esn.forward(cur, next, ishard);
        double prob = esn.eval(next);
        if(prob < 1e-15) prob = 1e-15;
        lo -= log(prob) / log(10.0);
        ne++;
        // For counting tasks, evaluate second half roughly
        if(ip >= (int)p.size() / 2 && esn.pred() != next) ok = false;
        cur = next;
      }
      if(ok) correct_seq++;
    }
    fprintf(stdout, "n=%d entropy: %.4f seq_acc(~2nd half): %.3f\n",
        nm, (ne > 0 ? lo / ne : 0), correct_seq / (double)ntest);
    if(save)
      fprintf(fres, "n=%d entropy: %.4f seq_acc: %.3f\n",
          nm, (ne > 0 ? lo / ne : 0), correct_seq / (double)ntest);
  }
  if(save) fclose(fres);

  return 0;
}
