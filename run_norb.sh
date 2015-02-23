#!/usr/bin/env bash
set -e

echo "*** Forming dataset"
python ./norb2cifar.py NORB_DATA
cwd=`pwd`
cd cuda-convnet
echo "*** Training net"
python convnet.py --data-path=$cwd/NORB_DATA --save-path=/tmp --test-range=28 --train-range=0-27 --layer-def=$cwd/layers_norb.cfg --layer-params=$cwd/params_norb.cfg --data-provider=norb --test-freq=60 --epochs=60 --conserve-mem=1 | tee /tmp/train.log
echo "*** Testing net"
file=`fgrep "ConvNet" /tmp/train.log | tail -n1 | awk '{ print $4 }'`
acc=`python convnet.py -f $file  --test-only=1 --logreg-name=logprob --test-range=6 | grep "^logprob" | awk '{ print $3 }'`
echo "*** Got accuracy $acc"
python shownet.py -f $file --show-cost=logprob
#python shownet.py -f $file --show-filters=C1 --no-rgb=1
#python shownet.py -f $file --show-preds=probs --only-errors=1

