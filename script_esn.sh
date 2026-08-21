#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

# Example experiments for Echo-State Networks (classic ESN and Stack-ESN).
# Compare with script_tasks.sh (Stack-RNN baselines).
# Uses train_esn (alias of train_esn_toy) and train_esn_add.

make esn
make esn_add

# ---- Classic ESN (no stack) ----

# a^nb^n
./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 0 -rho 0.9 -sparsity 0.1 -alpha 1.0 -ridge 1e-4 -nmax 10 -nseq 1000 -nepoch 10 -seed 1 -feat 1

# a^nb^nc^n
./train_esn -ntask 1 -nchar 3 -nhid 300 -nstack 0 -rho 0.9 -sparsity 0.1 -alpha 0.5 -ridge 1e-4 -nmax 10 -nseq 1000 -nepoch 10 -seed 1 -feat 1

# memorization
./train_esn -ntask 4 -nchar 3 -nhid 400 -nstack 0 -rho 0.95 -sparsity 0.1 -alpha 0.3 -ridge 1e-5 -nmax 10 -nseq 1000 -nepoch 15 -nreset 1 -seed 1 -feat 1

# ---- Stack-ESN (frozen stack controllers) ----

# a^nb^n with one stack
./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 1 -depth 2 -mod 1 -feat 2 -rho 0.9 -sparsity 0.1 -alpha 1.0 -ridge 1e-4 -nmax 10 -nseq 1000 -nepoch 10 -seed 1

# a^nb^n with several stacks
./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 5 -depth 2 -mod 1 -feat 2 -rho 0.9 -sparsity 0.1 -alpha 1.0 -ridge 1e-4 -nmax 15 -nseq 1000 -nepoch 10 -seed 1

# a^nb^mc^{n+m}
./train_esn -ntask 3 -nchar 3 -nhid 300 -nstack 2 -depth 2 -mod 1 -feat 2 -rho 0.9 -sparsity 0.1 -alpha 1.0 -ridge 1e-4 -nmax 10 -nseq 1000 -nepoch 10 -seed 1 -hard

# ---- SGD readout (online) ----

./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 0 -fit sgd -lr 0.05 -nmax 10 -nseq 2000 -nepoch 15 -seed 1 -feat 1

# ---- Activation ablation ----

./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 0 -nl 0 -ridge 1e-4 -nmax 10 -nseq 500 -nepoch 5 -seed 1 -feat 1
./train_esn -ntask 1 -nchar 2 -nhid 200 -nstack 0 -nl 1 -ridge 1e-4 -nmax 10 -nseq 500 -nepoch 5 -seed 1 -feat 1

# ---- Binary addition ----
# ./train_esn_add -nhid 200 -rho 0.9 -ridge 1e-4 -nseq 2000 -nmax 15 -nepoch 10
