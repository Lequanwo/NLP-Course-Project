import csv
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('punkt')


inFile1 = open("P_testDialogues1.csv", 'r', encoding="utf8")
inFile2 = open("P_testDialogues2.csv", 'r', encoding="utf8")
inFile3 = open("P_TFreqDialogues0.csv", 'r', encoding="utf8")
inFile4 = open("P_VFreqDialogues0.csv", 'r', encoding="utf8")
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

for file in files:
    getLines(file)

stop_words = set(stopwords.words('english'))

def removeStopWords(input_list):

    list_filtered = []
    for sentence in input_list:
        #print(sentence)
        #print(type(sentence))
        words = sentence.split()
        filtered_words = []
        #print(words)
        for r in words:
            if not r in stop_words:
                filtered_words.append(r)
        filtered_sentence = ' '.join(filtered_words)
        list_filtered.append(filtered_sentence)

    return list_filtered

# allInputs_filter = removeStopWords(allInputs)
# allHumanResponses_filter = removeStopWords(allHumanResponses)
# allAlexaResponses_filter = removeStopWords(allAlexaResponses)
# inputPlusHuman_filter = removeStopWords(inputPlusHuman)


def measureCosineSimilarity(sen1, sen2):
    X = sen1
    Y = sen2

    # tokenization
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    # sw contains the list of stopwords
    sw = stopwords.words('english')
    l1 = []
    l2 = []

    # remove stop words from string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    # form a set containing keywords of both strings
    rvector = X_set.union(Y_set)
    for w in rvector:
        if w in X_set:
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in Y_set:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]
    cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
    # print("similarity: ", cosine)
    return cosine

print(len(allInputs), len(allHumanResponses), len(allAlexaResponses))
num_samples = len(allInputs)

cossim_list1 = []
cossim_list2 = []
for i in range(num_samples):
    cossim1 = measureCosineSimilarity(allInputs[i], allAlexaResponses[i])
    cossim_list1.append(cossim1)
    cossim2 = measureCosineSimilarity(allInputs[i], allHumanResponses[i])
    cossim_list1.append(cossim2)
    if (abs(cossim1-cossim2) < 0.01) & (cossim1 != 0.0):
        print(cossim1, cossim2)
        print(allInputs[i])
        print(allAlexaResponses[i])
        print(allHumanResponses[i])





