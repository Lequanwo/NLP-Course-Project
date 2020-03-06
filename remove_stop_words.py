import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import csv
import nltk
from nltk.translate.bleu_score import sentence_bleu

# nltk.download('stopwords')
# nltk.download('punkt')




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



def removeStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_words = []
    for r in words:
        if not r in stop_words:
            filtered_words.append(r)
    filtered_sentence = ' '.join(filtered_words)

    return filtered_sentence

def changeNumbertoTokens(sentence):

    words = sentence.split()
    filtered_words = []
    for word in words:
        if word.isdigit():
            filtered_words.append('NUM')
        else:
            filtered_words.append(word)
    filtered_sentence = ' '.join(filtered_words)

    return filtered_sentence

def measureCosineSimilarity(sen1, sen2):
    X = sen1
    Y = sen2

    # remove stop words
    X = removeStopWords(X)
    Y = removeStopWords(Y)

    #Change numbers to token
    X = changeNumbertoTokens(X)
    Y = changeNumbertoTokens(Y)

    # tokenization
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    # sw contains the list of stopwords
    # sw = stopwords.words('english')
    l1 = []
    l2 = []

    # change format to use Union
    X_set = {w for w in X_list}
    Y_set = {w for w in Y_list}


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

def measureBleuScore(sen1, sen2):
    s1 = sen1
    s2 = sen2
    # remove stop words
    s1 = removeStopWords(s1)
    s2 = removeStopWords(s2)

    # Change numbers to token
    s1 = changeNumbertoTokens(s1)
    s2 = changeNumbertoTokens(s2)

    # change to list
    s1 = s1.split()
    s2 = s2.split()

    #calculate score
    score = sentence_bleu(s1, s2, weights=(1, 0, 0, 0))
    return score


if __name__ == "__main__":
    print('Reading files...')
    inFile1 = open("P_testDialogues1.csv", 'r', encoding="utf8")
    inFile2 = open("P_testDialogues2.csv", 'r', encoding="utf8")
    inFile3 = open("P_TFreqDialogues0.csv", 'r', encoding="utf8")
    inFile4 = open("P_VFreqDialogues0.csv", 'r', encoding="utf8")
    files = [inFile1, inFile2, inFile3, inFile4]

    allInputs = []
    allHumanResponses = []
    allAlexaResponses = []
    inputPlusHuman = []

    for file in files:
        getLines(file)



    print('Start Calculating...')
    # print(len(allInputs), len(allHumanResponses), len(allAlexaResponses))
    num_samples = len(allInputs)
    cossim_list1 = []
    cossim_list2 = []
    bleu_list1 = []
    bleu_list2 = []

    for i in range(num_samples):
        #calculate BLEU Score and append int a list
        bleu1 = measureBleuScore(allInputs[i], allAlexaResponses[i])
        bleu_list1.append(bleu1)

        bleu2 = measureBleuScore(allInputs[i], allHumanResponses[i])
        bleu_list2.append(bleu2)

      # cossim1 = measureCosineSimilarity(allInputs[i], allAlexaResponses[i])
      # cossim_list1.append(cossim1)
      #
      # cossim2 = measureCosineSimilarity(allInputs[i], allHumanResponses[i])
      # cossim_list2.append(cossim2)


      # if (abs(cossim1-cossim2) < 0.01) & (cossim1 != 0.0):
      #     print(cossim1, cossim2)
      #     print(allInputs[i])
      #     print(allAlexaResponses[i])
      #     print(allHumanResponses[i])

    # if (len(cossim_list1) !=len(cossim_list2)):
    #     print("Two results are not same!")
    #
    # with open('result.csv', 'w', newline='') as file:
    #   writer = csv.writer(file)
    #   writer.writerow(['HumanResponseCosineSimilarity', 'AlexaResponseCosineSimilarity'])
    #   for i in range(len(cossim_list1)):
    #       writer.writerow([cossim_list1[i], cossim_list2[i]])

    if (len(bleu_list1) !=len(bleu_list2)):
        print("Two results are not same!")

    with open('result2.csv', 'w', newline='') as file:
      writer = csv.writer(file)
      writer.writerow(['HumanResponseBleuScore', 'AlexaResponseBleuScore'])
      for i in range(len(bleu_list1)):
          writer.writerow([bleu_list2[i], bleu_list1[i]])

    print('Program is done successfully!')
