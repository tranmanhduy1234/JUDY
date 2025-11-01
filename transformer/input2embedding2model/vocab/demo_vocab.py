import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file=r'D:\chuyen_nganh\NT-HTKT\transformer\input2embedding2model\vocab\unigram_32000.model')

with open(r"D:\chuyen_nganh\NT-HTKT\transformer\input2embedding2model\vocab\text.txt", 'r', encoding='utf-8') as f:
    text = f.readlines()
tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print("Tokens:", tokens)
print("IDs:", ids)