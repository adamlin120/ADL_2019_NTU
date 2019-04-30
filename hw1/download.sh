# download spacy model
python3.7 -m spacy download en

# download embedding.pkl
wget -O ./data/embedding.pkl "https://www.dropbox.com/s/il25rz699simvss/embedding.pkl?dl=1"

# download models
wget -O ./models/rnn_attention_fix_tri/model.pkl.6 "https://www.dropbox.com/s/xldwsp6zqztyisp/rnn_attention_fix_tri_model.pkl.6?dl=1"

wget -O ./models/rnn_baseline_saveEntireModel/model.pkl.5 "https://www.dropbox.com/s/o0uks70s8k7txz7/rnn_baseline_saveEntireModel_model.pkl.5?dl=1"

wget -O ./models/rnn_attention_co/model.pkl.5 "https://www.dropbox.com/s/uftzeb0m724o8x5/rnn_attention_co_model.pkl.5?dl=1"
