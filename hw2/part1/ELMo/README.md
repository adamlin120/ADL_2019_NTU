# ELMo
## Prepare ELMo Dataset
1. `cd` into `part1/`
2. run command `python3.7 -m ELMo.elmo.prepare -c CORPUS_TXT -g GLOVE_EMBEDDING --save_folder SAVED_FOLDER --ratio PERCENTAGE_OF_WHOLE_CORPUS -t VOCAB_THRESHOLD --val_ratio PERCENTAGE_FOR_VALIDATION`
## Train ELMo
1. `cd` into `part1/`
2. run command `python3.7 -m ELMo.elmo.train MODEL_DIR -g CUDA_VISIBLE_DEVICES -c NUM_CPU`
## Plot train/dev perplexity 
1. run the jupyter notebook in `ELMo/report` and modify the first line to select which model
2. PNG file will be saved at the same dir.
