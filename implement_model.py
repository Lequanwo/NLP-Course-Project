import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
inFile1 = open("P_testDialogues1.csv", 'r')
inFile2 = open("P_testDialogues2.csv", 'r')
inFile3 = open("P_TFreqDialogues0.csv", 'r')
inFile4 = open("P_VFreqDialogues0.csv", 'r')
files = [inFile1, inFile2, inFile3, inFile4]

allInputs = []
allHumanResponses = []
allAlexaResponses = []
inputPlusHuman = []


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

for f in files:
    getLines(f)

corpus = []
count_vect = CountVectorizer()
count_vect.fit(inputPlusHuman)
tokenizer = count_vect.build_tokenizer()

for s in inputPlusHuman:
    toks = tokenizer(s)
    #print(toks)
    if len(toks) > 0:
        #corpus.append(toks)
        corpus += toks

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

def cosine_distance(W, q):
    """
    Parameters:
    W : torch.FloatTensor
        shape (vocab_size, embedding_dim)
    q : torch.FloatTensor
        shape (embedding_dim)
    """
    # Normalize
    Wp = W / W.norm(dim=1, keepdim=True)
    q = q / q.norm()
    return 1 - torch.mv(Wp, q)


def similar_words(word):
    print('Most similar words to: %s' % word)

    # Get embedding for query word.
    word_idx = vocab[word]
    query = embeddings[word_idx]

    # Compute cosine distance between query word embedding and all other embeddings.
    # `torch.mv` is matrix vector multiplication.
    distance = cosine_distance(embeddings, query)

    # Find closest embeddings and print out corresponding words.
    closest_word_ids = distance.argsort()[:10]

    for i, close_word_idx in enumerate(closest_word_ids):
        print('%i - %s' % (i + 1, reverse_vocab[close_word_idx]))


vocab_size = len(vocab)
embedding_size = 128
model = CBOW(vocab_size, embedding_size)
state_dict = torch.load('cbow_model.pt')
model.load_state_dict(state_dict)

embeddings = model.embeddings.weight.data

similar_words('he')