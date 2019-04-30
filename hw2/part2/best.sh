#!/bin/bash

PREDICT_CSV=./saved/ensemble_prediction/base_uncased_epoch_3-correct_epoch_1-correct_less_epoch_3-large_cased_no_weightdecay_epoch_3.csv

if [ ! -f $PREDICT_CSV ]
then
    echo "Start testing!"
    python3.7 -m src.predict\
        saved/base_uncased/ saved/correct/ saved/correct_less/ saved/large_cased_no_weightdecay/\
        -e 3 1 3 3\
        --csv $1\
        -g 0\
        --output $2
else
    echo "$PREDICT_CSV exist!"
    cp $PREDICT_CSV $2
fi

