# Stack RNN
Stack RNN is a project gathering the code from the paper 
*Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets* by Armand Joulin and Tomas Mikolov ([pdf](http://arxiv.org/abs/1503.01007)).
In this research project, we focus on extending Recurrent Neural Networks (RNN) with a stack to allow them to learn sequences which require
some form of persistent memory. 

Examples are given in the script `script_tasks.sh`. The code is still under construction. 
We are working on releasing the code for the list RNN. If you have any suggestion, please let us know (contacts below).


## Examples
To run the code on a task:
```
> make toy
> ./train_toy  -ntask 1 -nchar 2 -nhid 10 -nstack 1 -lr .1 -nmax 10 -depth 2 -bptt 50 -mod 1
```
To run the code on binary addition:
```
> make add
> ./train_add 
```

## Reservoir Computing / ESN
This repository also includes an Echo-State Network (ESN) implementation for reservoir computing on the same algorithmic tasks.

Unlike Stack-RNN (which trains all weights with BPTT), the ESN keeps a **random frozen reservoir** (and optional frozen stack controllers) and trains **only the readout**:
- primary method: closed-form **ridge regression** on the Gram matrix \(\Phi^\top\Phi\)
- optional online **SGD** on the softmax readout (`-fit sgd`)

### Build and run
```
> make esn_toy
> ./train_esn_toy -ntask 1 -nchar 2 -nhid 100 -rho 0.9 -ridge 1e-4 -nseq 500 -nmax 10 -seed 1
```
Stack-augmented ESN (Stack-ESN):
```
> ./train_esn_toy -ntask 1 -nchar 2 -nhid 100 -nstack 2 -depth 2 -mod 1 -feat 2 -rho 0.9 -ridge 1e-4
```
Binary addition with ESN:
```
> make esn_add
> ./train_esn_add -nhid 200 -rho 0.9 -ridge 1e-4 -nseq 2000 -nmax 15
```

### ESN hyperparameters
| Flag | Meaning | Default |
|------|---------|---------|
| `-nhid` | reservoir size | 100 |
| `-rho` | spectral radius of \(W_{res}\) | 0.9 |
| `-is` | input scaling | 0.5 |
| `-alpha` | leaking rate | 1.0 |
| `-sparsity` | fraction of nonzero \(W_{res}\) | 0.1 |
| `-ridge` | ridge regression \(\lambda\) | 1e-4 |
| `-washout` | steps ignored after reset | 10 |
| `-feat` | 0=res, 1=res+in, 2=res+stack, 3=res+in+stack | 0 |
| `-nl` | 0=tanh, 1=sigmoid | 0 |
| `-fit` | `ridge` or `sgd` | ridge |
| `-nstack` / `-depth` / `-mod` | Stack-ESN options (`mod=1` stack-only rec.) | 0 / 1 / 2 |

See `./train_esn_toy --help` and the ESN section of `script_tasks.sh`.

## Requirements
Stack RNN works on:
* Mac OS X
* Linux

It was not tested on Windows. To compile the code a relatively recent version of g++ is required.

## Building Stack RNN
Run `make` to compile everything (Stack-RNN and ESN trainers). 


## Options
For more help about the options:
```
> make toy
> ./train_toy --help
> make esn_toy
> ./train_esn_toy --help
```
Note that `train_add` can take the same options as `train_toy`.


## Join the Stack RNN community
* Paper: http://arxiv.org/abs/1503.01007
* Facebook page: https://www.facebook.com/fair
* Contact: ajoulin@fb.com

See the CONTRIBUTING file for how to help out.

## License
Stack RNN is BSD-licensed. We also provide an additional patent grant





