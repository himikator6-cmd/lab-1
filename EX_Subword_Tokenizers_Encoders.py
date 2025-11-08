import pickle
from L1_Modules import text_cleaner as tc
from L1_Modules import subword_tokenizers as st


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


BPE = st.BPE_Learning_Tokenizer(TrainText, 8000, 2)
with open("Encoders/BPE_Tokenizer_Encoder_8000_2.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 8000, 3)
with open("Encoders/BPE_Tokenizer_Encoder_8000_3.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 8000, 4)
with open("Encoders/BPE_Tokenizer_Encoder_8000_4.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 8000, 5)
with open("Encoders/BPE_Tokenizer_Encoder_8000_5.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 16000, 2)
with open("Encoders/BPE_Tokenizer_Encoder_16000_2.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 16000, 3)
with open("Encoders/BPE_Tokenizer_Encoder_16000_3.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 16000, 4)
with open("Encoders/BPE_Tokenizer_Encoder_16000_4.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 16000, 5)
with open("Encoders/BPE_Tokenizer_Encoder_16000_5.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 24000, 2)
with open("Encoders/BPE_Tokenizer_Encoder_24000_2.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 24000, 3)
with open("Encoders/BPE_Tokenizer_Encoder_24000_3.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 24000, 4)
with open("Encoders/BPE_Tokenizer_Encoder_24000_4.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 24000, 5)
with open("Encoders/BPE_Tokenizer_Encoder_24000_5.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 32000, 2)
with open("Encoders/BPE_Tokenizer_Encoder_32000_2.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 32000, 3)
with open("Encoders/BPE_Tokenizer_Encoder_32000_3.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 32000, 4)
with open("Encoders/BPE_Tokenizer_Encoder_32000_4.pkl", 'wb') as f:
    pickle.dump(BPE, f)


BPE = st.BPE_Learning_Tokenizer(TrainText, 32000, 5)
with open("Encoders/BPE_Tokenizer_Encoder_32000_5.pkl", 'wb') as f:
    pickle.dump(BPE, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 8000, 2)
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_2.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 8000, 3)
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_3.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 8000, 4)
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_4.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 8000, 5)
with open("Encoders/WordPiece_Tokenizer_Encoder_8000_5.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 16000, 2)
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_2.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 16000, 3)
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_3.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 16000, 4)
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_4.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 16000, 5)
with open("Encoders/WordPiece_Tokenizer_Encoder_16000_5.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 24000, 2)
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_2.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 24000, 3)
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_3.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 24000, 4)
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_4.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 24000, 5)
with open("Encoders/WordPiece_Tokenizer_Encoder_24000_5.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 32000, 2)
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_2.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 32000, 3)
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_3.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 32000, 4)
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_4.pkl", 'wb') as f:
    pickle.dump(WP, f)


WP = st.WordPiece_Learning_Tokenizer(TrainText, 32000, 5)
with open("Encoders/WordPiece_Tokenizer_Encoder_32000_5.pkl", 'wb') as f:
    pickle.dump(WP, f)


