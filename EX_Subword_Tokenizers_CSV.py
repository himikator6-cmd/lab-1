import csv
import pickle
from L1_Modules import text_cleaner as tc
from L1_Modules import subword_tokenizers as st



FTS1 = 'Test_Samplings/test1.jsonl'
FTS2 = 'Test_Samplings/test2.jsonl'
FTS3 = 'Test_Samplings/test3.jsonl'
FTS4 = 'Test_Samplings/test4.jsonl'
FTS5 = 'Test_Samplings/test5.jsonl'

FTS = [FTS1, FTS2, FTS3, FTS4, FTS5]


TestText = tc.text_extractor(FTS[0])
for f in range(1, len(FTS)):
    TestText = TestText + " " + tc.text_extractor(FTS[f])


fieldnames = ["Метод токенизации", "Количество слов в словаре", "Минимальная частота", "Реальное количество слов в словаре",
              "Процент фрагментации слов","Коэффициент сжатия", "Время токенизации на 1000000 символов"]
data = [[]] * 33
data[0] = fieldnames
i = 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_8000_2.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "8000", "2", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_8000_3.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "8000", "3", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_8000_4.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "8000", "4", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_8000_5.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "8000", "5", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_16000_2.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "16000", "2", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_16000_3.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "16000", "3", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_16000_4.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "16000", "4", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_16000_5.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "16000", "5", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_24000_2.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "24000", "2", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_24000_3.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "24000", "3", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_24000_4.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "24000", "4", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_24000_5.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "24000", "5", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_32000_2.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "32000", "2", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_32000_3.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "32000", "3", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_32000_4.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "32000", "4", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
BPET = None
with open("Encoders/BPE_Tokenizer_Encoder_32000_5.pkl", 'rb') as f:
    BPET = pickle.load(f)
BPE = st.Tokenizer(TestText, BPET)
print("BPE Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(BPE[1]))
print("Процент фрагментации слов - " + str(BPE[2]) + "%")
print("Коэффициент сжатия - " + str(BPE[3]))
print("Время токенизации на 1000000 символов - " + str(BPE[4]) + " секунд")
print()
print()
print()
data[i] = ["BPE Токенизация", "32000", "5", str(BPE[1]), str(BPE[2]) + "%", str(BPE[3]), str(BPE[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_2.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "8000", "2", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_3.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "8000", "3", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_4.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "8000", "4", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_5.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 8000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "8000", "5", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_2.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "16000", "2", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_3.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "16000", "3", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_4.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "16000", "4", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_5.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 16000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "16000", "5", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_2.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "24000", "2", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_3.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "24000", "3", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_4.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "24000", "4", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_5.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 24000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "24000", "5", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_2.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 2")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "32000", "2", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_3.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 3")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "32000", "3", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_4.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 4")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "32000", "4", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


print()
WPT = None
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_5.pkl", 'rb') as f:
    WPT = pickle.load(f)
WP = st.Tokenizer(TestText, WPT)
print("WordPiece Токенизация")
print("Количество слов в словаре - 32000")
print("Минимальная частота - 5")
print("Реальное количество слов в словаре - " + str(WP[1]))
print("Процент фрагментации слов - " + str(WP[2]) + "%")
print("Коэффициент сжатия - " + str(WP[3]))
print("Время токенизации на 1000000 символов - " + str(WP[4]) + " секунд")
print()
print()
print()
data[i] = ["WordPiece Токенизация", "32000", "5", str(WP[1]), str(WP[2]) + "%", str(WP[3]), str(WP[4]) + " секунд"]
i = i + 1


with open("subword_tokenization_metrics.csv", "w", newline = '', encoding = "utf-8") as csvfile:
    writer = csv.writer(csvfile, delimiter = ',')
    writer.writerows(data)
