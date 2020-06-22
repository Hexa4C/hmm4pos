import pickle as pk
import collections

corpus = []
token2idx = {}
idx2token = {}
frequency = collections.OrderedDict()
attr2idx = {}
idx2attr = {}
for line in open("corpus.txt", encoding='utf8'):
    line = line.strip()
    pieces = line.split(" ")
    for pair in pieces:
        if "/" not in pair:
            continue
        apairs = pair.split("/")
        token, attr = apairs[0], apairs[1]
        if token in frequency.keys():
            frequency[token] += 1
        else:
            frequency[token] = 1
        if attr in attr2idx.keys():
            attr_idx = attr2idx[attr]
        else:
            attr_idx = len(attr2idx)
            attr2idx[attr] = attr_idx
            idx2attr[attr_idx] = attr
print("[INFO] ", len(attr2idx), " attributes.")
print(idx2attr)

# truncate dictionary
dropped_token = []
for k, v in frequency.items():
    if v <= 1:
        dropped_token.append(k)
for token in dropped_token:
    frequency.pop(token)
items = sorted(frequency.items(), key=lambda obj:obj[1], reverse=True)
new_frequency = collections.OrderedDict()
for item in items:
    new_frequency[item[0]] = frequency[item[0]]
new_frequency["UNK"] = 1
token_list = list(new_frequency.keys())
for i in range(len(token_list)):
    token2idx[token_list[i]] = i
    idx2token[i] = token_list[i]

# revisit corpus
unk_cnt = 0
total_cnt = 0
for line in open("corpus.txt", encoding='utf8'):
    line = line.strip()
    pieces = line.split(" ")
    sent = []
    attrs = []
    for pair in pieces:
        #if "/" not in pair:
        #    continue
        total_cnt += 1
        try:
            apairs = pair.split("/")
            token, attr = apairs[0], apairs[1]
        except:
            print("*" + pair + "*" + line)
            quit()
        if token not in token2idx.keys():
            token = "UNK"
            unk_cnt += 1
        token_idx = token2idx[token]
        attr_idx = attr2idx[attr]
        sent.append(token_idx)
        attrs.append(attr_idx)
    corpus.append((sent, attrs))

print("[INFO] ", len(corpus), " sentences.")
print("[INFO] ", "After truncating...")
print("[INFO] ", len(token2idx), " tokens.")
print("[INFO] ", "UNK tokens: ", unk_cnt, "/", total_cnt, "\t%.5f" % (unk_cnt * 1.0 / total_cnt))
print("[EXAMPLE] ", )
example_sent = []
example_attr = []
for t in corpus[0][0]:
    example_sent.append(idx2token[t])
for t in corpus[0][1]:
    example_sent.append(idx2attr[t])
print(" ".join(example_sent) + "\n")
print(" ".join(example_attr))

pk.dump(
    {
        "corpus": corpus,
        "token2idx": token2idx,
        "idx2token": idx2token,
        "attr2idx": attr2idx,
        "idx2attr": idx2attr
    },
    open("corpus_all.pkl", 'wb')
)