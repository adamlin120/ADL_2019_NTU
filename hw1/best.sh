
if [ ! -f ./data/test.pkl ]; then
    echo "Test Pickle File not found!"
    # prepare dataset
    python3.7 ./src/make_dataset.py ./data/ --test $1 --embed ./data/embedding.pkl
fi

cd src
python3.7 predict.py ../models/rnn_attention_fix_tri/ --epoch 6

cp ../models/rnn_attention_fix_tri/predict-6.csv $2
