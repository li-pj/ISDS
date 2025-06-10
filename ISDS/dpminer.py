import math
from typing import *
from heapq import heappush, heappushpop
from extratools.dicttools import nextentries
from concurrent.futures import ThreadPoolExecutor
from .utils import is_subseq, compute_gain_ratio, compute_gini_index, compute_odds
from .rule import Rule
DB = List[List[int]]
Matches = List[Tuple[int, int]]
Occurs = Dict[int, Matches]
Pattern = List[int]
Results = Optional[List[Tuple[int, Pattern]]]
Key = Callable[[Pattern, Matches], int]
Filter = Callable[[Pattern, Matches], bool]
Callback = Callable[[Pattern, Matches], None]
import copy
T = TypeVar("T")
Entries = List[Tuple[int, int]]



class DPMinier(object):

    def __init__(self, p_db,n_db,label):
        self.p_db = p_db
        self.n_db = n_db
        self.len_p = len(p_db)
        self.len_n = len(n_db)
        self.label=label
        self.minlen, self.maxlen = 1, 1000
        self._results = [] # type: Any
        
    # def _mine(self, func):
    #     # type: (Callable[[Pattern, Matches], None]) -> Any
    #     self._results.clear()
    #     func([], [(i, -1) for i in range(len(self._db))])
    #     return self._results
    
    defaultkey = lambda patt, matches: len(matches)

    def topk(
            self, k,db,
            key=None, bound=None,
            filter=None, callback=None
        ):
        # type: (DPMinier, int,List[List[int]],Optional[Key], Optional[Key], Optional[Filter], Optional[Callback]) -> Results

        def canpass(sup):
            # type: (int) -> bool
            return len(results) == k and sup <= results[0][0]


        def verify(patt, matches):
            sup = key(patt, matches)
            if canpass(sup):
                return

            if (filter is None or filter(patt, matches)) :
                (heappush if len(results) < k else heappushpop)(results, (sup, patt, matches))


        def topk_rec(patt, matches):
            if len(patt) >= self.minlen:
                verify(patt, matches)

                if len(patt) == self.maxlen:
                    return

            occurs = nextentries(db, matches)

            for newitem, newmatches in sorted(
                    occurs.items(),
                    key=lambda x: key(patt + [x[0]], x[1]),
                    reverse=True
                ):
                newpatt = patt + [newitem]

                if canpass(bound(newpatt, newmatches)):
                    break

                topk_rec(newpatt, newmatches)



        if key is None:
            key = bound = DPMinier.defaultkey

        results =[] 
        topk_rec([], [(i, -1) for i in range(len(db))])

        if callback:
            for _, patt, matches in results:
                callback(patt, matches)
                
            return None
        len_=len(db)
        return [(sup/len_ , patt)for sup, patt, _ in results]

    def run(self, k,max_rule, key=None, bound=None,
            filter=None, callback=None):
        
    # type: (DPMinier, int,int,  Optional[Key], Optional[Key], Optional[Filter], Optional[Callback]) -> Results

        def get_sup(patterns,db):
            sup_class=[]
            for pattern in patterns:
                match=0
                for seq in db:
                    if is_subseq(pattern[1],seq):
                        match += 1
                sup_class.append(match/self.len_n)
            return sup_class
        
        def get_covers(pattern,db):
            covers_index=set()
            # for _,pattern in patterns:
            for idx,seq in enumerate(db):
                if is_subseq(pattern,seq):
                    covers_index.add(idx)
            return sorted(list(covers_index),reverse=True)



        
        fw_p_db=copy.deepcopy(self.p_db)


        while( len(fw_p_db)!=0 and len(self._results)<=max_rule):
            sup_patterns = []
            prefix_patterns = self.topk(k,fw_p_db, key, bound,filter, callback)
            sup_patterns.extend(prefix_patterns)
            sup_patterns.sort(key=lambda x: (-x[0], x[1]))
            sup_neg_class=get_sup(sup_patterns,self.n_db)
            cmp_patterns=[]

            for pattern,neg_sup in zip(sup_patterns,sup_neg_class):

                # measure = compute_gain_ratio(pattern[0],neg_sup,len(self.p_db),len(self.n_db))
                # measure = compute_odds(pattern[0], neg_sup, len(self.p_db), len(self.n_db))
                # measure = compute_gini_index(pattern[0],neg_sup,len(self.p_db),len(self.n_db))
                # cmp_patterns.append((measure, pattern[1], pattern[0], neg_sup))

                #relative risk
                if neg_sup==0:
                    cmp_patterns.append((float('inf'),pattern[1],pattern[0],neg_sup))
                else:
                    rr_ratio=pattern[0]/neg_sup
                    cmp_patterns.append((rr_ratio,pattern[1],pattern[0],neg_sup))

            # #gain odd rr
            cmp_patterns.sort( key=lambda x: (-x[0], -x[2],x[3]))

            # gini
            # cmp_patterns.sort(key=lambda x: (x[0], -x[2], x[3]))

            # rr
            if cmp_patterns[0][0] <= 1:
                break

            covers_index=get_covers(cmp_patterns[0][1],fw_p_db)
            for idx in covers_index:
                del fw_p_db[idx]

            pos_covers =set(get_covers(cmp_patterns[0][1],self.p_db))
            neg_covers =set(get_covers(cmp_patterns[0][1],self.n_db))
            self._results.append(Rule(cmp_patterns[0][1],pos_covers,neg_covers,self.len_p,self.len_n,self.label))
            if not isinstance(self._results,list):
                self._results=[self._results]
        return self._results

            