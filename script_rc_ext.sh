#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
#

# Example experiments for DESN (Deep-ESN), TESN (Tree-ESN), DTESN (Deep-Tree-ESN).

make desn_toy
make tesn_toy
make dtesn_toy
make pesn_toy

# ---- Deep ESN on sequential toy tasks ----

# a^nb^n with 3 layers
./train_desn_toy -ntask 1 -nchar 2 -nhid 50 -nlayers 3 -rho 0.9 -sparsity 0.1 -alpha 1.0 -ridge 1e-4 -nmax 10 -nseq 800 -nepoch 8 -seed 1 -feat 0

# a^nb^nc^n
./train_desn_toy -ntask 1 -nchar 3 -nhid 40 -nlayers 3 -rho 0.9 -alpha 0.5 -ridge 1e-4 -nmax 10 -nseq 800 -nepoch 8 -seed 1 -feat 2

# memorization
./train_desn_toy -ntask 4 -nchar 3 -nhid 60 -nlayers 2 -rho 0.95 -alpha 0.3 -ridge 1e-5 -nmax 10 -nseq 800 -nepoch 10 -seed 1 -feat 0

# ---- Tree-ESN on tree tasks ----

# leaf-label parity
./train_tesn_toy -ntask 1 -nlabels 2 -nhid 50 -map 4 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1

# sum of leaves mod nlabels
./train_tesn_toy -ntask 2 -nlabels 3 -nhid 60 -map 4 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1

# sequential a^nb^n as left-branching chain tree
./train_tesn_toy -ntask 5 -nchar 2 -nseqtask 1 -nhid 50 -map 0 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 10 -nepoch 10 -seed 1

# ---- Deep Tree-ESN ----

./train_dtesn_toy -ntask 1 -nlabels 2 -nhid 40 -nlayers 2 -map 4 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1

./train_dtesn_toy -ntask 2 -nlabels 3 -nhid 40 -nlayers 3 -map 4 -rho 0.9 -iscale 0.5 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1

./train_dtesn_toy -ntask 6 -nchar 2 -nseqtask 1 -nhid 40 -nlayers 2 -map 4 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 10 -nepoch 10 -seed 1

# ---- Membrane P-system reservoir + Butcher B-series ridge (A000081) ----

# verify unordered rooted tree catalog matches OEIS A000081
./train_pesn_toy -check-a000081 -bsorder 6

# unordered membrane tree leaf-parity with B-series readout
./train_pesn_toy -ntask 7 -nlabels 2 -nhid 40 -readout 1 -bsorder 4 -bsfeat 3 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1

# A000081 membrane-shape classification
./train_pesn_toy -ntask 9 -nshape 4 -nlabels 2 -nhid 40 -readout 1 -bsorder 5 -ridge 1e-4 -ntree 800 -nmax 10 -nepoch 10 -seed 1

# binary tree parity with combined map + B-series features
./train_pesn_toy -ntask 1 -nlabels 2 -nhid 40 -readout 2 -map 4 -bsorder 4 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1

# classical skin/mean map readout (no B-series)
./train_pesn_toy -ntask 8 -nlabels 3 -nhid 50 -readout 0 -map 4 -rho 0.9 -ridge 1e-4 -ntree 800 -nmax 12 -nepoch 10 -seed 1
