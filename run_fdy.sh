#! /bin/bash

# Run IGMC for five times on datasets Flixster, Douban or YahooMusic. Usage:
# ./run_FDY.sh DATANAME
# Replace DATANAME with flixster, douban or yahoo_music.
# After running, type python summarize_fdy.py to summarize the results.


data=${1}
for i in $(seq 1 5)  # to run with different seeds
do
  python Main.py --data-name ${data} --save-appendix _s${i} --hop 1 --epochs 40 --testing --seed ${i} --ensemble
done

