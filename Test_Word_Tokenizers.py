import csv
from L1_Modules import text_cleaner as tc
from L1_Modules import word_tokenizers as wt


FTR1 = 'Train_Samplings/train1.jsonl'
FTR2 = 'Train_Samplings/train2.jsonl'
FTR3 = 'Train_Samplings/train3.jsonl'
FTR4 = 'Train_Samplings/train4.jsonl'

FTR = [FTR1, FTR2, FTR3, FTR4]


TrainText = tc.text_extractor(FTR[0])
for f in range(1, len(FTR)):
    TrainText = TrainText + " " + tc.text_extractor(FTR[f])

TestText1 = """События игр серии показываются с точки неназванного космического пехотинца,
работающего на Объединённую Аэрокосмическую Корпорацию (англ. Union Aerospace Corporation, UAC)
и сражающегося против полчищ демонов, для того чтобы выжить и спасти Землю от их нападения.
Действие почти всех частей серии происходит на Марсе, его спутниках, Фобосе и Деймосе, в лабораториях ОАК или в Аду."""

TestText2 = """Оригинальная игра 1993 года является одной из первых игр от первого лица для IBM-PC-совместимых компьютеров
с псевдотрёхмерной графикой,многопользовательским режимом, а также с поддержкой пользовательских модификаций.
К большей части игр серии были выпущены многочисленные модификации.
Сопутствующая продукция представлена комиксом, двумя одноимёнными фильмами и несколькими книгами по мотивам сюжета."""


print()
STV = wt.Space_Learning_Tokenizer(TrainText)[1]
ST1 = wt.Space_Tokenizer(TestText1, STV)
ST2 = wt.Space_Tokenizer(TestText2, STV)
print("Токенизация по пробелам")
print()
print()
print(ST1[0])
print()
print()
print(ST2[0])
print()
print()
print()
print()
STSV = wt.Space_Learning_Tokenizer_Stemming(TrainText)[1]
STS1 = wt.Space_Tokenizer_Stemming(TestText1, STSV)
STS2 = wt.Space_Tokenizer_Stemming(TestText2, STSV)
print("Токенизация по пробелам со стеммингом")
print()
print()
print(STS1[0])
print()
print()
print(STS2[0])
print()
print()
print()
print()
STLV = wt.Space_Learning_Tokenizer_Lemmatization(TrainText)[1]
STL1 = wt.Space_Tokenizer_Lemmatization(TestText1, STLV)
STL2 = wt.Space_Tokenizer_Lemmatization(TestText2, STLV)
print("Токенизация по пробелам с лемматизацией")
print()
print()
print(STL1[0])
print()
print()
print(STL2[0])
print()
print()
print()
print()
SPTV = wt.Space_Preprocessed_Learning_Tokenizer(TrainText)[1]
SPT1 = wt.Space_Preprocessed_Tokenizer(TestText1, SPTV)
SPT2 = wt.Space_Preprocessed_Tokenizer(TestText2, SPTV)
print("Токенизация по пробелам с предобработкой")
print()
print()
print(SPT1[0])
print()
print()
print(SPT2[0])
print()
print()
print()
print()
SPTSV = wt.Space_Preprocessed_Learning_Tokenizer_Stemming(TrainText)[1]
SPTS1 = wt.Space_Preprocessed_Tokenizer_Stemming(TestText1, SPTSV)
SPTS2 = wt.Space_Preprocessed_Tokenizer_Stemming(TestText2, SPTSV)
print("Токенизация по пробелам с предобработкой и стеммингом")
print()
print()
print(SPTS1[0])
print()
print()
print(SPTS2[0])
print()
print()
print()
print()
SPTLV = wt.Space_Preprocessed_Learning_Tokenizer_Lemmatization(TrainText)[1]
SPTL1 = wt.Space_Preprocessed_Tokenizer_Lemmatization(TestText1, SPTLV)
SPTL2 = wt.Space_Preprocessed_Tokenizer_Lemmatization(TestText2, SPTLV)
print("Токенизация по пробелам с предобработкой и лемматизацией")
print()
print()
print(SPTL1[0])
print()
print()
print(SPTL2[0])
print()
print()
print()
print()
NLTKV = wt.NLTK_Word_Learning_Tokenizer(TrainText)[1]
NLTK1 = wt.NLTK_Word_Tokenizer(TestText1, NLTKV)
NLTK2 = wt.NLTK_Word_Tokenizer(TestText2, NLTKV)
print("NLTK токенизация по словам")
print()
print()
print(NLTK1[0])
print()
print()
print(NLTK2[0])
print()
print()
print()
print()
NLTKSV = wt.NLTK_Word_Learning_Tokenizer_Stemming(TrainText)[1]
NLTKS1 = wt.NLTK_Word_Tokenizer_Stemming(TestText1, NLTKSV)
NLTKS2 = wt.NLTK_Word_Tokenizer_Stemming(TestText2, NLTKSV)
print("NLTK токенизация по словам со стеммингом")
print()
print()
print(NLTKS1[0])
print()
print()
print(NLTKS2[0])
print()
print()
print()
print()
NLTKLV = wt.NLTK_Word_Learning_Tokenizer_Lemmatization(TrainText)[1]
NLTKL1 = wt.NLTK_Word_Tokenizer_Lemmatization(TestText1, NLTKLV)
NLTKL2 = wt.NLTK_Word_Tokenizer_Lemmatization(TestText2, NLTKLV)
print("NLTK токенизация по словам с лемматизацией")
print()
print()
print(NLTKL1[0])
print()
print()
print(NLTKL2[0])
print()
print()
print()
print()



