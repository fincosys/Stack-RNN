# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.


CC = g++
CFLAGS = -std=c++0x  -lm -O3 -march=native -Wall -funroll-loops -ffast-math

all: toy add esn_toy esn_add esn desn_toy tesn_toy dtesn_toy pesn_toy

toy : train_toy.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_toy.cpp -o train_toy

add : train_add.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_add.cpp -o train_add

esn_toy : train_esn_toy.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_esn_toy.cpp -o train_esn_toy

esn_add : train_esn_add.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_esn_add.cpp -o train_esn_add

# Compatibility alias (master used `make esn` / ./train_esn for toy tasks).
esn : train_esn_toy.cpp
	$(CC) $(CFLAGS) $(OPT_DEF) train_esn_toy.cpp -o train_esn

desn_toy : train_desn_toy.cpp DESN.h
	$(CC) $(CFLAGS) $(OPT_DEF) train_desn_toy.cpp -o train_desn_toy

tesn_toy : train_tesn_toy.cpp TESN.h Tree.h tree_task.h
	$(CC) $(CFLAGS) $(OPT_DEF) train_tesn_toy.cpp -o train_tesn_toy

dtesn_toy : train_dtesn_toy.cpp DTESN.h TESN.h Tree.h tree_task.h
	$(CC) $(CFLAGS) $(OPT_DEF) train_dtesn_toy.cpp -o train_dtesn_toy

# Membrane P-system recursive reservoir + Butcher B-series ridge (A000081)
pesn_toy : train_pesn_toy.cpp PESN.h BSeries.h RootedTree.h Tree.h tree_task.h
	$(CC) $(CFLAGS) $(OPT_DEF) train_pesn_toy.cpp -o train_pesn_toy

clean:
	rm -f train_toy train_add train_esn_toy train_esn_add train_esn \
	      train_desn_toy train_tesn_toy train_dtesn_toy train_pesn_toy
