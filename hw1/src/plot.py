import pickle
import numpy as np
import matplotlib.pyplot as plt

def cut_padding(arr, pad=79585):
    return arr[arr!=pad]

def softmax(arr):
    arr = np.exp(arr)
    arr /= np.sum(arr)
    return arr


attn_mat = np.load('./attn_mat.npy')
context = np.load('./context.npy')
option = np.load('./options.npy')

context = cut_padding(context)
option = cut_padding(option)

len_con = len(context)
len_opt = len(option)

attn_mat = attn_mat[:len_con, :len_opt]
attn_mat = np.apply_along_axis(softmax, 1, attn_mat)
attn_mat = np.around(attn_mat, 2)

with open('../data/embedding.pkl', 'rb') as f:
    word_dict = pickle.load(f).word_dict
    word_dict = list(word_dict.keys())
    
context_words = [word_dict[c] for c in context]
option_words = [word_dict[c] for c in option]

fig, ax = plt.subplots(figsize=(100,100))
im = ax.imshow(attn_mat)

ax.set_xticks(np.arange(len(option_words)))
ax.set_yticks(np.arange(len(context_words)))

ax.set_xticklabels(option_words)
ax.set_yticklabels(context_words)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

for i in range(len(context_words)):
    for j in range(len(option_words)):
        text = ax.text(j, i, attn_mat[i, j],
                       ha="center", va="center", color="w")

ax.set_title("Attention Weight (context2option)")
plt.savefig('attention_weight.png')