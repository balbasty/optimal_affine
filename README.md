# optimal_affine
Build optimal "subject to mean space" affines from "subject to subject" pairwise affines

## Installation

```shell
pip install git+https://github.com/balbasty/optimal_affine
```
Note that we require `python >= 3.7`

## Usage

```shell
optimal_affine \
  -i mtw pdw mtw_to_pdw.lta \
  -i mtw t1w mtw_to_t1w.lta \
  -i pdw mtw pdw_to_mtw.lta \
  -i pdw t1w pdw_to_t1w.lta \
  -i t1w mtw t1w_to_mtw.lta \
  -i t1w pdw t1w_to_pdw.lta \
  -o out/{label}_to_mean.lta
```
Type `optimal_affine --help` for more options.
