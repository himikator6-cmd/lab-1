import pickle
from L1_Modules import text_cleaner as tc
from L1_Modules import word_tokenizers as wt


TestText1 = """События игр серии показываются с точки неназванного космического пехотинца,
работающего на Объединённую Аэрокосмическую Корпорацию (англ. Union Aerospace Corporation, UAC)
и сражающегося против полчищ демонов, для того чтобы выжить и спасти Землю от их нападения.
Действие почти всех частей серии происходит на Марсе, его спутниках, Фобосе и Деймосе, в лабораториях ОАК или в Аду."""

TestText2 = """Оригинальная игра 1993 года является одной из первых игр от первого лица для IBM-PC-совместимых компьютеров
с псевдотрёхмерной графикой,многопользовательским режимом, а также с поддержкой пользовательских модификаций.
К большей части игр серии были выпущены многочисленные модификации.
Сопутствующая продукция представлена комиксом, двумя одноимёнными фильмами и несколькими книгами по мотивам сюжета."""


print()
STV = None
with open("Vocabularies/Space_Tokenizer_Vocabulary.pkl", 'rb') as f:
    STV = pickle.load(f)
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
STSV = None
with open("Vocabularies/Space_Tokenizer_Stemming_Vocabulary.pkl", 'rb') as f:
    STSV = pickle.load(f)
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
STLV = None
with open("Vocabularies/Space_Tokenizer_Lemmatization_Vocabulary.pkl", 'rb') as f:
    STLV = pickle.load(f)
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
SPTV = None
with open("Vocabularies/Space_Preprocessed_Tokenizer_Vocabulary.pkl", 'rb') as f:
    SPTV = pickle.load(f)
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
SPTSV = None
with open("Vocabularies/Space_Preprocessed_Tokenizer_Stemming_Vocabulary.pkl", 'rb') as f:
    SPTSV = pickle.load(f)
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
SPTLV = None
with open("Vocabularies/Space_Preprocessed_Tokenizer_Lemmatization_Vocabulary.pkl", 'rb') as f:
    SPTLV = pickle.load(f)
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
NLTKV = None
with open("Vocabularies/NLTK_Word_Tokenizer_Vocabulary.pkl", 'rb') as f:
    NLTKV = pickle.load(f)
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
NLTKSV = None
with open("Vocabularies/NLTK_Word_Tokenizer_Stemming_Vocabulary.pkl", 'rb') as f:
    NLTKSV = pickle.load(f)
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
NLTKLV = None
with open("Vocabularies/NLTK_Word_Tokenizer_Lemmatization_Vocabulary.pkl", 'rb') as f:
    NLTKLV = pickle.load(f)
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



