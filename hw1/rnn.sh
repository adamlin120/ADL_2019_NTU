
if [ ! -f ./data/test.pkl ]; then
    echo "Test Pickle File not found!"
    # prepare dataset
    python3.7 ./src/make_dataset.py ./data/ --test $1 --embed ./data/embedding.pkl
fi

cd src
python3.7 predict.py ../models/rnn_baseline_saveEntireModel/ --epoch 5

cp ../models/rnn_baseline_saveEntireModel/predict-5.csv $2
