import bisect
import math

from collections import defaultdict


def is_subseq(pattern,seq):
    if isinstance(seq, int):
        seq = [seq]
    if isinstance(pattern, int):
        pattern = [pattern]  
    iter_seq = iter(seq)
    return all(char in iter_seq for char in pattern)

def split_by_label(seqs, labels,obj_label):
    obj_seqs=[]
    other_seqs=[]
    for seq, label in zip(seqs, labels):
        if label == obj_label:
            obj_seqs.append(seq)
        else:
            other_seqs.append(seq)
    return obj_seqs,other_seqs

def datainput(filepath):
    max_sequence_length = 0
    db = []
    data_label = []
    itemset = []
    with open(filepath, 'r') as file:
        for line in file:
            temp = line.strip().split('\t')
            seq_db = temp[1].split(" ")
            max_sequence_length = max(max_sequence_length, len(seq_db))
            db.append(seq_db)
            data_label.append(str(temp[0]))
    itemset = set([item for sublist in db for item in sublist])
    itemset = list(itemset)
    int_itemset = [str(x) for x in itemset]
    int_itemset.sort()
    itemset = [str(x) for x in int_itemset]
    return db, data_label, itemset, max_sequence_length

def predict_seq(seq,rules,labels_set):
    pred_label = labels_set[0]
    max_score = 0
    type= 'EXACT'
    for rule in rules:
        pre_score=LCS(seq,rule.pattern,rule.precision,max_score)
        if pre_score > max_score :
            max_score = pre_score
            pred_label = rule.label
            if pre_score < rule.precision:
                type = 'PARTIAL'
            else :
                type = 'EXACT'
    return pred_label,type



def LCS(seq,pattern,precision,max_score):
    if precision < max_score:
        return 0
    m, n = len(seq), len(pattern)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq[i - 1] == pattern[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    pre_score=precision*dp[m][n]/len(pattern)
    return pre_score



def hunt_szymanski(seq,pattern,presicion,max_score):
    if presicion < max_score:
        return 0
    pos = defaultdict(list)
    for idx, char in enumerate(seq):
        pos[char].append(idx)

    match_positions = []
    for char in pattern:
        if char in pos:
            match_positions.extend(reversed(pos[char]))

    lis = []
    for p in match_positions:
        idx = bisect.bisect_left(lis, p)
        if idx == len(lis):
            lis.append(p)
        else:
            lis[idx] = p
    return presicion*len(lis) /len(pattern)


def test_rules(rules):
    crr_cover=set()
    in_cover=set()
    len_pos=0
    len_neg=0
    for rule in rules:
        len_pos=rule.pos_class_size
        len_neg=rule.neg_class_size
        crr_cover.update(rule.pos_cover_set)
        in_cover.update(rule.neg_cover_set)
    return len(crr_cover)/len_pos,len(in_cover)/len_neg




def compute_gain_ratio(p, n, P, N):
    def log2(x):
        return math.log2(x) if x > 0 else 0
    total = P + N
    covered = p + n
    rest = total - covered
    H_total = - (P / total) * log2(P / total) - (N / total) * log2(N / total)
    H1 = - (p / covered) * log2(p / covered) - (n / covered) * log2(n / covered) if covered else 0
    H2 = - ((P - p) / rest) * log2((P - p) / rest) - ((N - n) / rest) * log2((N - n) / rest) if rest else 0
    H_after = (covered / total) * H1 + (rest / total) * H2
    IG = H_total - H_after
    SI = - (covered / total) * log2(covered / total) - (rest / total) * log2(
        rest / total) if covered and rest else 0
    return IG / SI if SI != 0 else 0

def compute_gini_index(p, n, P, N):
    total = P + N
    covered = p + n
    rest = total - covered

    def gini(pos, neg):
        s = pos + neg
        if s == 0:
            return 0
        p_pos = pos / s
        p_neg = neg / s
        return 1 - (p_pos ** 2 + p_neg ** 2)

    gini_covered = gini(p, n)
    gini_rest = gini(P - p, N - n)

    weighted_gini = (covered / total) * gini_covered + (rest / total) * gini_rest

    return weighted_gini

def compute_odds(p, n, P, N, epsilon=1e-6):

    odds_hit = (p + epsilon) / (n + epsilon)
    odds_non_hit = (P - p + epsilon) / (N - n + epsilon)
    odds_ratio = odds_hit / odds_non_hit
    return odds_ratio




def dataFeature(filepath):
    db = []
    data_label = []
    itemset = []
    lengths = []
    with open(filepath, 'r') as file:
        for line in file:
            temp = line.strip().split('\t')
            seq_db = temp[1].split(" ")
            db.append(seq_db)
            data_label.append(str(temp[0]))
            lengths.append(len(seq_db))
    itemset = set([item for sublist in db for item in sublist])
    itemset = list(itemset)
    int_itemset = [str(x) for x in itemset]
    int_itemset.sort()
    itemset = [str(x) for x in int_itemset]
    labels_set = sorted(set(data_label))
    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    print(f'{filepath} size:{len(db)} e:{len(itemset)} max:{max_len} min:{min_len} avg:{avg_len} C:{len(labels_set)}')