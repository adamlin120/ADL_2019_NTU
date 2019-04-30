# How to run

1. Prepare the dataset and pre-trained embeddings (FastText is used here) in `./data`:

```
./data/train.json
./data/valid.json
./data/test.json
./data/crawl-300d-2M.vec
```

2. Preprocess the data
```
cd src/
python make_dataset.py ../data/
```

3. Then you can train a poooor example model as follow:
```
python train.py ../models/example/
```

4. To predict, run
```
python predict.py ../models/example/ --epoch 3
```
where `--epoch` specifies the save model of which epoch to use.


# Plot attention weight

1. Download `embedding.pkl` by running `download.sh`.

2. Uncomment line 68 on `src/rnnattention_predictor.py` and line 85 on `src/modules/rnnattention_net.py` to enable two breakpoints.

3. Change test file path in model config to `../data/valid.pkl`.

4. Run predict command as mentioned above.

5. On the first breakpoint: run the following commands in the interactive session.
```
import numpy as np
np.save('options.npy', np.array(batch['options'][0, 0]))
np.save('context.npy', np.array(batch['context'][0]))
```

6. On the second breakpoint: run the following commands in the interactive session.
```
import numpy as np
np.save('attn_mat.npy', np.array(attn_mat[0].cpu()))
```
7. Exit process.

8. run `src/plot.py` and the result will be saved at `src/attention_weight.png`

9. Comment breakpoints in step 2.
