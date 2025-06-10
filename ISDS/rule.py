
import numpy as np
from .GLOBAL_VAR import *


class Rule:

    def __init__(self,pattern:list,pos_cover_set:set,neg_cover_set:set,pos_class_size:int,neg_class_size:int,label) :
        self.pattern=pattern
        self.pos_cover_set=pos_cover_set
        self.neg_cover_set=neg_cover_set
        self.pos_class_size=pos_class_size
        self.neg_class_size=neg_class_size
        tp=len(pos_cover_set)
        tn=neg_class_size-len(neg_cover_set)
        fp=len(neg_cover_set)
        fn=pos_class_size-len(pos_cover_set)
        self.precision = tp/(tp+fp)
        self.label=label
    def print_rule(self):
        print(f'{self.pattern},label:{self.label},precision:{self.precision}')
        print(f'pos_cover:{len(self.pos_cover_set)},neg_cover:{len(self.neg_cover_set)}')


def get_obj_funtion(sub_rules,rules_size,max_rule_len):

    sub_rules_len=len(sub_rules)
    if sub_rules_len <= 0 :
        return 0
    size_term = 1 - sub_rules_len/rules_size
    sum_len = 0
    for rule in sub_rules:
        sum_len+=len(rule.pattern)
    len_term = 1 - sum_len/(sub_rules_len*max_rule_len)
    pos_class,neg_class=0,0
    pos_set_sum,neg_set_sum=set(),set()
    for idx,rule in enumerate(sub_rules):
        pos_class=rule.pos_class_size
        neg_class=rule.neg_class_size
        pos_set_sum.update(rule.pos_cover_set)
        neg_set_sum.update(rule.neg_cover_set)
    tp=len(pos_set_sum)
    tn=neg_class-len(neg_set_sum)
    fp=len(neg_set_sum)
    fn=pos_class-len(pos_set_sum)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    obj_f = mcc + LAMBDA * (size_term + len_term)
    return obj_f

def fitness_function(particles,rules,max_rule_len):
    fitness_list=[]
    for particle in particles:
        sub_rules=[rules[idx] for idx in particle]
        fitness_list.append(get_obj_funtion(sub_rules,len(rules),max_rule_len))
    return fitness_list