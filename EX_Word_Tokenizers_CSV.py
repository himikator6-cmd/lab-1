import csv
from L1_Modules import text_cleaner as tc
from L1_Modules import word_tokenizers as wt


FTR1 = 'train1.jsonl'
FTR2 = 'train2.jsonl'
FTR3 = 'train3.jsonl'
FTR4 = 'train4.jsonl'
FTS1 = 'test1.jsonl'
FTS2 = 'test2.jsonl'
FTS3 = 'test3.jsonl'

FTR = [FTR1, FTR2, FTR3, FTR4]
FTS = [FTS1, FTS2, FTS3]


TrainText = tc.text_extractor(FTR[0])
for f in range(1, len(FTR)):
    TrainText = TrainText + " " + tc.text_extractor(FTR[f])

TestText = tc.text_extractor(FTS[0])
for f in range(1, len(FTS)):
    TestText = TestText + " " + tc.text_extractor(FTS[f])


fieldnames = ["Метод токенизации", "Количество слов в словаре", "Доля OOV", "Время токенизации на 1000000 символов"]
data = [[]] * 10
data[0] = fieldnames
i = 1


print()
STV = wt.Space_Learning_Tokenizer(TrainText)
ST = wt.Space_Tokenizer(TestText, STV)
print("Токенизация по пробелам")
print("Количество слов в словаре - " + str(len(STV)))
print("Доля OOV - " + str(ST[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(ST[2]) + " секунд")
print()
print()
print()
data[i] = ["Токенизация по пробелам", str(len(STV)), str(ST[1]) + "%", str(ST[2]) + " секунд"]
i = i + 1


print()
STSV = wt.Space_Learning_Tokenizer_Stemming(TrainText)
STS = wt.Space_Tokenizer_Stemming(TestText, STSV)
print("Токенизация по пробелам со стеммингом")
print("Количество слов в словаре - " + str(len(STSV)))
print("Доля OOV - " + str(STS[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(STS[2]) + " секунд")
print()
print()
print()
data[i] = ["Токенизация по пробелам со стеммингом", str(len(STSV)), str(STS[1]) + "%", str(STS[2]) + " секунд"]
i = i + 1


print()
STLV = wt.Space_Learning_Tokenizer_Lemmatization(TrainText)
STL = wt.Space_Tokenizer_Lemmatization(TestText, STLV)
print("Токенизация по пробелам с лемматизацией")
print("Количество слов в словаре - " + str(len(STLV)))
print("Доля OOV - " + str(STL[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(STL[2]) + " секунд")
print()
print()
print()
data[i] = ["Токенизация по пробелам с лемматизацией", str(len(STLV)), str(STL[1]) + "%", str(STL[2]) + " секунд"]
i = i + 1


print()
SPTV = wt.Space_Preprocessed_Learning_Tokenizer(TrainText)
SPT = wt.Space_Preprocessed_Tokenizer(TestText, SPTV)
print("Токенизация по пробелам с предобработкой")
print("Количество слов в словаре - " + str(len(SPTV)))
print("Доля OOV - " + str(SPT[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(SPT[2]) + " секунд")
print()
print()
print()
data[i] = ["Токенизация по пробелам с предобработкой", str(len(SPTV)), str(SPT[1]) + "%", str(SPT[2]) + " секунд"]
i = i + 1


print()
SPTSV = wt.Space_Preprocessed_Learning_Tokenizer_Stemming(TrainText)
SPTS = wt.Space_Preprocessed_Tokenizer_Stemming(TestText, SPTSV)
print("Токенизация по пробелам с предобработкой и стеммингом")
print("Количество слов в словаре - " + str(len(SPTSV)))
print("Доля OOV - " + str(SPTS[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(SPTS[2]) + " секунд")
print()
print()
print()
data[i] = ["Токенизация по пробелам с предобработкой и стеммингом", str(len(SPTSV)), str(SPTS[1]) + "%", str(SPTS[2]) + " секунд"]
i = i + 1


print()
SPTLV = wt.Space_Preprocessed_Learning_Tokenizer_Lemmatization(TrainText)
SPTL = wt.Space_Preprocessed_Tokenizer_Lemmatization(TestText, SPTLV)
print("Токенизация по пробелам с предобработкой и лемматизацией")
print("Количество слов в словаре - " + str(len(SPTLV)))
print("Доля OOV - " + str(SPTL[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(SPTL[2]) + " секунд")
print()
print()
print()
data[i] = ["Токенизация по пробелам с предобработкойи лемматизацией", str(len(SPTLV)), str(SPTL[1]) + "%", str(SPTL[2]) + " секунд"]
i = i + 1


print()
NLTKV = wt.NLTK_Word_Learning_Tokenizer(TrainText)
NLTK = wt.NLTK_Word_Tokenizer(TestText, NLTKV)
print("NLTK токенизация по словам")
print("Количество слов в словаре - " + str(len(NLTKV)))
print("Доля OOV - " + str(NLTK[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(NLTK[2]) + " секунд")
print()
print()
print()
data[i] = ["NLTK токенизация по словам", str(len(NLTKV)), str(NLTK[1]) + "%", str(NLTK[2]) + " секунд"]
i = i + 1


print()
NLTKSV = wt.NLTK_Word_Learning_Tokenizer_Stemming(TrainText)
NLTKS = wt.NLTK_Word_Tokenizer_Stemming(TestText, NLTKSV)
print("NLTK токенизация по словам со стеммингом")
print("Количество слов в словаре - " + str(len(NLTKSV)))
print("Доля OOV - " + str(NLTKS[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(NLTKS[2]) + " секунд")
print()
print()
print()
data[i] = ["NLTK токенизация по словам со стеммингом", str(len(NLTKSV)), str(NLTKS[1]) + "%", str(NLTKS[2]) + " секунд"]
i = i + 1


print()
NLTKLV = wt.NLTK_Word_Learning_Tokenizer_Lemmatization(TrainText)
NLTKL = wt.NLTK_Word_Tokenizer_Lemmatization(TestText, NLTKLV)
print("NLTK токенизация по словам с лемматизацией")
print("Количество слов в словаре - " + str(len(NLTKLV)))
print("Доля OOV - " + str(NLTKL[1]) + "%")
print("Время токенизации на 1000000 символов - " + str(NLTKL[2]) + " секунд")
print()
print()
print()
data[i] = ["NLTK токенизация по словам с лемматизацией", str(len(NLTKLV)), str(NLTKL[1]) + "%", str(NLTKL[2]) + " секунд"]
i = i + 1



with open("tokenization_metrics.csv", "w", newline = '', encoding = "utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerows(data)

