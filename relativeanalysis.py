"""
Relative performance analysis
"""
filenames = ["count_mnb.csv", "count_mnb.csv", "count_svm.csv"
             "tfidf_mnb.csv", "tfidf_mlp.csv", "count_svm.csv"]


def read_from_csv(filename):
    true = []
    pred = []
    with open(filename, 'r') as fin:
        for line in fin.readlines():
            t, p = line.strip().split(',')
            true.append(t)
            pred.append(p)
    return true, pred


def main():
    trues = []
    preds = []
    for filename in filenames:
        true, pred = read_from_csv(filename)
        trues.append(true)
        preds.append(pred)
    import pdb
    pdb.set_trace()

    # All correct
    total_num = len(trues[0])

    all_correct = 0
    all_wrong = 0
    for idx, ans in enumerate(trues[0]):
        if all(p[idx] == ans for p in preds):
            all_correct += 1
        if not any(p[idx] == ans for p in preds):
            all_wrong += 1

    print("Total:", total_num)
    print("All correct:", all_correct)
    print("All wrong:", all_wrong)
    pass


if __name__ == "__main__":
    main()
