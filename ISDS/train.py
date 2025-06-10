from .utils import split_by_label,test_rules,datainput
from .dpminer import DPMinier
from .sbpso import SBPSO
from .rule import fitness_function
from concurrent.futures import  ProcessPoolExecutor

def task(args):
    label, seqs, labels, k, max_rule = args
    obj_seqs,other_seqs=split_by_label(seqs,labels,label)
    cpm =DPMinier(obj_seqs,other_seqs,label)
    rules=cpm.run(k,max_rule)
    cover,incover=test_rules(rules)
    print(f'{label}, {cover}, {incover}')
    solution_rules=[]
    print(f'{label}:{len(rules)}')
    if len(rules) < 2:
        solution_rules=rules
    else:
        sso=SBPSO(len(rules),rules,fitness_function,max_iter=100)
        best,_=sso.run()
        solution_rules=[rules[i] for i in best]
        cover,incover=test_rules(solution_rules)
        print(f'{label}, {cover}, {incover}')
    return solution_rules

def fit(seqs,labels,k=700,max_rule=20):
    label_set=sorted(set(labels))
    rules_list = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(task, (label, seqs, labels, k, max_rule)): label for label in label_set}
        for future in futures:
            try:
                result = future.result()
                rules_list.extend(result)  
            except Exception as e:
                print(f'Error processing {futures[future]}: {e}')
    return rules_list
       
if __name__ == '__main__':
    datasets = ['activity', 'aslbu', 'auslan2', 'context', 'epitope', 'gene',
               'pioneer', 'question', 'skating']
    # for dataset in datasets:
    filename=f"../dataset/activity.txt"
    db, labels, itemset, _=datainput(filename)
    rules=fit(db,labels,k=700,max_rule=20)
    for rule in rules:
        print(f'if {rule.pattern} in seq. then {rule.label}')