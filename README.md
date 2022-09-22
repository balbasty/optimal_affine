# Optimal Affine

Build optimal "subject to mean space" affines from several "subject to subject" 
pairwise affines.

This package implements an algorithm for "symmetrizing coordinate spaces", 
which is largely based on observations made in the paper 
["Consistent multi-time-point brain atrophy estimation from the boundary shift integral"](https://adni.loni.usc.edu/adni-publications/Leung-Neuroimage-2011.pdf) by Leung et al. 

In Leung et al, all possible pairs of images are registered (in both
forward and backward directions), such that the "subject-to-template"
transforms can easily be computed as `T[i -> meanspace] = expm(mean_j(logm(T[i -> j])))`.

Our implementation differs in several aspects:
- We allow some transformation pairs to be missing, at the cost of
  introducing bias in the mean space estimation. This bias can be
  overcome in the statistical sense if the number of subjects is large
  and evaluated pairs are randomly sampled.
- Instead of first symmetrizing pairwise transforms, we fit the mean
  space from all possible forward and backward transformations.
- Instead of minimizing the L2 norm in the matrix Lie algebra
  (which is done implicitly by Leung et al's method), 
  we minimize the L2 norm in the embedding space (i.e.,
  the Frobenius norm of affine matrices). 
  While our method requires the use of iterative optimization,
  it is more accurate when pairwise transformations are large, 
  in which case affine composition is badly approximated by log-matrix addition.

## Installation

```shell
pip install git+https://github.com/balbasty/optimal_affine
```
Note that we require `python >= 3.7`

## Usage

```
usage:
    optimal_affine --input <fix> <mov> <path> [--input ...]

arguments:
    -i, --input         Affine transform for one pair of images
                          <fix>   Index (or label) of fixed image
                          <mov>   Index (or label) of moving image
                          <path>  Path to an LTA file that warps <mov> to <fix>
    -o, --output        Path to output transforms (default: {label}_optimal.lta)
    -l, --log           Minimize L2 in Lie algebra as in Leung et al. (default: L2 in matrix space)
    -a, --affine        Assume transforms are all affine (default)
    -s, --similitude    Assume transforms are all similitude
    -r, --rigid         Assume transforms are all rigid

example:
    optimal_affine \
      -i mtw pdw mtw_to_pdw.lta \
      -i mtw t1w mtw_to_t1w.lta \
      -i pdw mtw pdw_to_mtw.lta \
      -i pdw t1w pdw_to_t1w.lta \
      -i t1w mtw t1w_to_mtw.lta \
      -i t1w pdw t1w_to_pdw.lta \
      -o out/{label}_to_mean.lta

```
