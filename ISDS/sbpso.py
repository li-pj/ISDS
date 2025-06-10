import random
from itertools import chain, combinations
from tqdm import tqdm

import numpy as np

random.seed(42)
np.random.seed(42)


class SBPSO:
    def __init__(self, particles_num, rules, fitness_function ,velocities=None, max_iter=100,
                 c_1=1, c_2=1, c_3=1,c_4=1):
        def random_subsets_from_set(s, num_samples, min_size=1, max_size=None):
            s = list(s)
            max_size = max_size if max_size else len(s)
            subsets = []
            for _ in range(num_samples):
                k = random.randint(min_size, max_size)
                subset = set(random.sample(s, k))
                subsets.append(subset)
            return subsets
        self.U = set([idx for idx,_ in enumerate(rules)])
        self.particles = random_subsets_from_set(self.U,particles_num)
        self.velocities = velocities
        self.fitness_function = fitness_function
        self.N = len(self.particles)
        self.max_iter = max_iter
        self.rules=rules
        self.max_rule_len=max([len(rule.pattern) for rule in rules])
        self.c_1 = c_1  # parameters to control personal best
        self.c_2 = c_2  # parameters to control global best respectively
        self.c_3 = c_3*self.N
        self.c_4 = c_4*self.N
        self.gbest_value_hist = []  # gbest_y of every iteration
        self.p_bests = self.particles
        self.p_bests_values = self.fitness_function(self.particles,self.rules,self.max_rule_len)
        self.fits=self.p_bests_values
        self.g_best = set(self.p_bests[0])
        self.g_best_value = self.p_bests_values[0]
        self.update_bests()
        self.iter = 0

        
    def move_particles(self):
        def k_Tournament_Selection(A, particle, N, k=5):
            k = min(len(A), k)
            v_temp = set()
            for i in range(N):
                scores = []
                for j in range(k):
                    e_j = random.choice(list(A))
                    particle_ = set(particle) 
                    particle_.add(e_j) 
                    score = self.fitness_function([particle_], self.rules,self.max_rule_len)
                    scores.append((score, e_j))
                score_max = max(scores, key=lambda x: x[0])
                v_temp.add(('+',score_max[1]))
            return v_temp
        
            
        r_1 = np.random.rand(self.N)
        r_2 = np.random.rand(self.N)
        r_3 = np.random.rand(self.N)
        r_4 = np.random.rand(self.N)

        new_velocities= [set() for _ in range(self.N)]
        for idx,pbest_particle in enumerate(zip(self.p_bests,self.particles)):
            #velocitie part1
            p_best,particle=pbest_particle
            
            to_add_1 = p_best - particle
            to_remove_1 = particle - p_best
            temp_set_1 = set([('+', item) for item in to_add_1]) | set([('-', item) for item in to_remove_1])
            select_num_1 = int(len(temp_set_1)*self.c_1*r_1[idx])
            if len(temp_set_1) > 0:
                select_set_1=random.sample(temp_set_1, select_num_1)
                new_velocities[idx] = new_velocities[idx] |set(select_set_1)

            #velocitie part2
            to_add_2 = self.g_best - particle
            to_remove_2 = particle - self.g_best
            temp_set_2 = set([('+', item) for item in to_add_2]) | set([('-', item) for item in to_remove_2])
            select_num_2= int(len(temp_set_2)*self.c_2*r_2[idx])
            if len(temp_set_2) > 0:
                select_set_2=random.sample(temp_set_2, select_num_2)
                new_velocities[idx] = new_velocities[idx] | set(select_set_2)

            #velocitie part3
            A_i =   self.U - (particle | p_best | self.g_best)
            if len(A_i) > 0:
                N_beta_A=int(self.c_3*r_3[idx])
                select_set_3=k_Tournament_Selection(A_i,particle,N_beta_A)
                if len(select_set_3)>0:
                    new_velocities[idx] = new_velocities[idx] | select_set_3

            #velocitie part4
            S_i = particle & p_best & self.g_best
            if len(S_i) > 0:
                beta=int(self.c_4*r_4[idx])
                beta_least=0
                r=random.uniform(0,1)
                N_beta_B = min(len(S_i),beta_least+int(r*(beta-beta_least)))
                temp_set_4 = set([('-', item) for item in S_i])
                if N_beta_B > 0:
                    select_set_4 = random.sample(temp_set_4, N_beta_B)
                    new_velocities[idx] = new_velocities[idx] | set(select_set_4)

        # self.is_running = np.sum(self.velocities - new_velocities) != 0
        self.velocities = new_velocities
        for idx,velocity in enumerate(self.velocities):
            if len(velocity)==0:
                continue
            for item in velocity:
                if item[0] == '+':
                    self.particles[idx].add(item[1])
                elif item[0] == '-':
                    self.particles[idx].remove(item[1])

    def update_bests(self):
        self.fits = self.fitness_function(self.particles,self.rules,self.max_rule_len)
        g_best_value=self.g_best_value
        idx = -1
        for i in range(len(self.particles)):
            # update best personnal value (cognitive)
            if self.fits[i] > self.p_bests_values[i]:
                self.p_bests_values[i] = self.fits[i]
                self.p_bests[i] = self.particles[i]
                # update best global value (social)
                if self.fits[i] > g_best_value or self.fits[i] ==g_best_value and len(self.particles[i]) < len(self.g_best):
                    g_best_value = self.fits[i]
                    idx = i
        if idx > 0:
            self.g_best_value = g_best_value
            self.g_best = set(self.particles[idx])

    def run(self):
        with tqdm(total=self.max_iter, desc="Epoch", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as t:
            for iter in range(self.max_iter):
                self.move_particles()
                self.update_bests()
                self.gbest_value_hist.append(self.g_best_value)

                t.update(1)
                t.set_postfix({
                    'solution': self.g_best,
                    'score': f'{self.g_best_value:.4f}',
                })

            return self.g_best,self.g_best_value
    fit=run