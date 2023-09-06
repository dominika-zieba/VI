#! /usr/bin/bash

mkdir results/$1

python variational_sine_gaussian_jax.py $1
