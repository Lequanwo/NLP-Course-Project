# %%

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import csv

# %%




inFile1 = open("P_testDialogues1.csv", 'r', encoding="utf8")
inFile2 = open("P_testDialogues2.csv", 'r', encoding="utf8")
inFile3 = open("P_TFreqDialogues0.csv", 'r', encoding="utf8")
inFile4 = open("P_VFreqDialogues0.csv", 'r', encoding="utf8")
files = [inFile1, inFile2, inFile3, inFile4]


# %%

allInputs = []
allHumanResponses = []
allAlexaResponses = []
inputPlusHuman = []


# %%

def getLines(inFile):
    for line in inFile:
        if "Input,Response,TurkResponse" in line:
            continue
        sL = line.split('","')
        if len(sL) > 3:
            print("TOO LONG: ", line)
            continue
        # print(line)
        sL0 = sL[0].strip().lower()
        while sL0[0] == '"':
            sL0 = sL0[1:]
        allInputs.append(sL0.strip())

        sL1 = sL[1].strip().lower()
        allAlexaResponses.append(sL1)

        sL2 = sL[2].strip().lower()
        while sL2[-1] == '"':
            sL2 = sL2[:-1]
        allHumanResponses.append(sL2.strip())

        iph = sL0 + " " + sL2
        inputPlusHuman.append(iph.strip())


# %%

for f in files:
    getLines(f)

# %%

corpus = []
count_vect = CountVectorizer()
count_vect.fit(inputPlusHuman)
tokenizer = count_vect.build_tokenizer()
for s in inputPlusHuman:
    toks = tokenizer(s)
    # print(toks)
    if len(toks) > 0:
        # corpus.append(toks)
        corpus += toks


# %%

def build_vocabulary(corpus, limit=30000):
    """Builds a vocabulary.

    Args:
        corpus: A list of words.
    """
    counts = Counter(corpus)  # Count the word occurances.
    counts = counts.items()  # Transform Counter to (word, count) tuples.
    counts = sorted(counts, key=lambda x: x[1], reverse=True)  # Sort in terms of frequency.
    counts = counts[:limit]
    reverse_vocab = ['<UNK>'] + [x[0] for x in counts]  # Use a list to map indices to words.
    vocab = {x: i for i, x in enumerate(reverse_vocab)}  # Invert that mapping to get the vocabulary.
    data = [vocab[x] if x in vocab else 0 for x in corpus]  # Get ids for all words in the corpus.
    return data, vocab, reverse_vocab


data, vocab, reverse_vocab = build_vocabulary(corpus)

# %%

reverse_vocab[data[0]]

# %%

data = torch.tensor(data)


# Replace the python loop-based approach to getting context windows
# with tensor operations.
def window(x, window_size=2):
    chunks = x.unfold(0, 2 * window_size + 1, 1)
    w = chunks[:, window_size]
    c_left = chunks[:, :window_size]
    c_right = chunks[:, window_size + 1:]
    c = torch.cat((c_left, c_right), dim=1)
    return w, c


# %%

w, c = window(data)
print(reverse_vocab[w[0]])
print(' '.join(reverse_vocab[_c] for _c in c[0]))

# %%

train = data

if torch.cuda.is_available():
    train = train.cuda()

# %%

window_size = 2
batch_size = 128

train_w, train_c = window(train, window_size=window_size)
train_dataset = torch.utils.data.TensorDataset(train_w, train_c)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)

# val_w, val_c = window(val, window_size=window_size)
# val_dataset = torch.utils.data.TensorDataset(*window(val))
# val_loader = torch.utils.data.DataLoader(val_dataset, batch_size)

# %%

import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):

    def __init__(self, vocab_size, embedding_size):
        super(CBOW, self).__init__()
        # Parameters
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        # Layers
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size, bias=False)
        # Share projection weights between the input and output
        # embeddings.
        self.embeddings.weight.data = self.fc.weight.data

    def forward(self, c):
        embeddings = self.embeddings(c)
        sum_of_embeddings = embeddings.mean(dim=1)
        projection = self.fc(sum_of_embeddings)
        log_probabilities = F.log_softmax(projection, dim=1)
        return log_probabilities


# %%

from torch import optim

# Training settings / hyperparameters
vocab_size = len(vocab)
embedding_size = 128

# Create model
model = CBOW(vocab_size, embedding_size)
if torch.cuda.is_available():
    model = model.cuda()

# Initialize optimizer.
# Note: The learning rate is often one of the most important hyperparameters
# to tune.
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-6)

# We will train using negative log-likelihood loss.
# Note: Sometimes the loss is computed inside the model (e.g., in AllenNLP).
# Different people have different preferences.
loss_function = nn.NLLLoss()

# %%

epochs = 10
best_train_loss = float('inf')

for epoch in range(epochs):

    # Training loop
    model.train()
    for i, batch in enumerate(train_loader):
        w, c = batch
        optimizer.zero_grad()
        output = model(c)
        loss = loss_function(output, w)
        loss.backward()
        optimizer.step()
        if (i % 100) == 0:
            print(f'Epoch: {epoch}, Iteration: {i} - Train Loss: {loss.item()}', end='\r')
    torch.save(model.state_dict(), 'cbow_model.pt')

# %%

state_dict = torch.load('cbow_model.pt')
model.load_state_dict(state_dict)

# %%

embeddings = model.embeddings.weight.data

# %%

if torch.cuda.is_available():
    embedding_array = embeddings.cpu().numpy()
else:
    embedding_array = embeddings.numpy()

# Get the t-SNE embeddings.
tsne = TSNE(n_components=2).fit_transform(embedding_array[:2000])

# Create a scatter plot
fig = plt.figure(figsize=(24, 24), dpi=300)
ax = fig.add_subplot(111)
ax.scatter(tsne[:, 0], tsne[:, 1], color='#ff7f0e')

# Show labels for the most common words
for i in range(1, 301):
    ax.text(tsne[i, 0], tsne[i, 1], reverse_vocab[i], fontsize=12, color='#1f77b4')

# %%