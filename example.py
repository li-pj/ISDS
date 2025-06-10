import numpy as np
from sklearn.metrics import accuracy_score

from ISDS.GLOBAL_VAR import LAMBDA
from ISDS.train import fit
from ISDS.utils import datainput, predict_seq
from sklearn.model_selection import train_test_split

def main(filename):
    filepath=f"./dataset/{filename}.txt"
    db, labels, itemset, _=datainput(filepath)
    labels_set=sorted(set(labels))
    char_to_index = {char: idx+1 for idx, char in enumerate(itemset)}
    db = [[char_to_index.get(char, -1) for char in seq] for seq in db]
    X_train, X_test, y_train, y_test = train_test_split(db, labels, test_size=0.2, random_state=42)
    rules= fit(X_train,y_train,k=700,max_rule=20)
    y_pred=[]
    for i in range(len(X_test)):
        seq = X_test[i]
        pred_label,_=predict_seq(seq,rules,labels_set)
        y_pred.append(pred_label)
    acc = accuracy_score(y_test, y_pred)
    print(f'{filename} acc:{acc}')
    
if __name__ == "__main__":
    main("question")
