import pickle
from L1_Modules import text_cleaner as tc
from L1_Modules import word_tokenizers as wt


FTR1 = 'Train_Samplings/train1.jsonl'
FTR2 = 'Train_Samplings/train2.jsonl'
FTR3 = 'Train_Samplings/train3.jsonl'
FTR4 = 'Train_Samplings/train4.jsonl'
FTR5 = 'Train_Samplings/train5.jsonl'
FTR6 = 'Train_Samplings/train6.jsonl'
FTR7 = 'Train_Samplings/train7.jsonl'
FTR8 = 'Train_Samplings/train8.jsonl'

FTR = [FTR1, FTR2, FTR3, FTR4, FTR5, FTR6, FTR7, FTR8]


TrainText = tc.text_extractor(FTR[0])
for f in range(1, len(FTR)):
    TrainText = TrainText + " " + tc.text_extractor(FTR[f])


STV = wt.Space_Learning_Tokenizer(TrainText)[1]
with open("Vocabularies/Space_Tokenizer_Vocabulary.pkl", 'wb') as f:
    pickle.dump(STV, f)


STSV = wt.Space_Learning_Tokenizer_Stemming(TrainText)[1]
with open("Vocabularies/Space_Tokenizer_Stemming_Vocabulary.pkl", 'wb') as f:
    pickle.dump(STSV, f)


STLV = wt.Space_Learning_Tokenizer_Lemmatization(TrainText)[1]
with open("Vocabularies/Space_Tokenizer_Lemmatization_Vocabulary.pkl", 'wb') as f:
    pickle.dump(STLV, f)


SPTV = wt.Space_Preprocessed_Learning_Tokenizer(TrainText)[1]
with open("Vocabularies/Space_Preprocessed_Tokenizer_Vocabulary.pkl", 'wb') as f:
    pickle.dump(SPTV, f)


SPTSV = wt.Space_Preprocessed_Learning_Tokenizer_Stemming(TrainText)[1]
with open("Vocabularies/Space_Preprocessed_Tokenizer_Stemming_Vocabulary.pkl", 'wb') as f:
    pickle.dump(SPTSV, f)


SPTLV = wt.Space_Preprocessed_Learning_Tokenizer_Lemmatization(TrainText)[1]
with open("Vocabularies/Space_Preprocessed_Tokenizer_Lemmatization_Vocabulary.pkl", 'wb') as f:
    pickle.dump(SPTLV, f)


NLTKV = wt.NLTK_Word_Learning_Tokenizer(TrainText)[1]
with open("Vocabularies/NLTK_Word_Tokenizer_Vocabulary.pkl", 'wb') as f:
    pickle.dump(NLTKV, f)


NLTKSV = wt.NLTK_Word_Learning_Tokenizer_Stemming(TrainText)[1]
with open("Vocabularies/NLTK_Word_Tokenizer_Stemming_Vocabulary.pkl", 'wb') as f:
    pickle.dump(NLTKSV, f)


NLTKLV = wt.NLTK_Word_Learning_Tokenizer_Lemmatization(TrainText)[1]
with open("Vocabularies/NLTK_Word_Tokenizer_Lemmatization_Vocabulary.pkl", 'wb') as f:
    pickle.dump(NLTKLV, f)
