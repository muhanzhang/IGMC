#! /bin/bash

# transfer models trained on ML-100K to the following datasets

data=${1}

num_relations=5
case ${data} in 
yahoo_music)
  multiply_by=20
;;
flixster)
  multiply_by=1
;;
douban)
  multiply_by=1
;;
*) echo 'Dataset does not exist.'
;;
esac

python Main.py --data-name ${data} --epochs 40 --testing --no-train --ensemble --transfer results/ml_100k_mnph200_testmode/ --dynamic-dataset --num-relations ${num_relations} --multiply-by ${multiply_by}

# Note: The --transfer specifies where the ml_100k model to transfer is located. The --num-relations specifies how many edge types are there in the transferred model so that the target datasets can group their edge types to the same number. The --multiply-by specifies how many times to multiply the final prediction of the transferred model to accomodate different rating scales between source and target.

