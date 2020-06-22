import pickle as pk
import numpy as np
import hmm.model as model
import sklearn.metrics as met
from tqdm import tqdm


def read_data(path):
    corpus_all = pk.load(open(path, 'rb'))
    return corpus_all["corpus"], corpus_all["token2idx"], \
            corpus_all["idx2token"], corpus_all["attr2idx"], \
            corpus_all["idx2attr"]


def get_observation(corpus):
    observations = []
    for pair in corpus:
        obs = pair[0]
        observations.append(obs)
    return observations


def main():
    path = "./corpus_all.pkl"
    corpus, token2idx, idx2token, attr2idx, idx2attr = \
        read_data(path)
    observations = get_observation(corpus)
    total_size = len(corpus)
    train_size = total_size * 4 // 5
    train_corpus = corpus[:train_size]
    test_corpus = corpus[train_size:]
    train_obs = observations[:train_size]
    test_obs = observations[train_size:]
    my_model = model.hmm_model(len(attr2idx), len(token2idx))
    print("[INFO]: init model parameters...")
    my_model.init_probmat(train_corpus)
    ground_truth = []
    preds = []
    print("[INFO]: prediction of test corpus...")
    for pair in tqdm(test_corpus):
        obs = pair[0]
        ground_truth_states = pair[1]
        pred_states = my_model.viterbi(obs)
        ground_truth += ground_truth_states
        preds += pred_states
    ground_truth = np.array(ground_truth)
    preds = np.array(preds)
    p_score = met.precision_score(ground_truth, preds, average='micro')
    r_score = met.recall_score(ground_truth, preds, average='micro')
    print("[INFO]: precision score:", p_score)
    print("[INFO]: recall score:", r_score)
    pk.dump(
        {
            "model": my_model,
            "token2idx": token2idx,
            "idx2token": idx2token,
            "attr2idx": attr2idx,
            "idx2attr": idx2attr
        },
        open("model.pkl", 'wb')
    )


if __name__ == "__main__":
    main()