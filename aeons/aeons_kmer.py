# STANDARD LIBRARY
from itertools import product
from math import sqrt
# NON STANDARD LIBRARY
import numpy as np
# from scipy.stats import pearsonr
# CUSTOM
from .aeons_utils import reverse_complement



class KmerCounter:

    """
    kmer counter specifically for di-, tri-, and tetramers
    general purpose class that can be initialised once
    and then used for many sequences

    """

    def __init__(self):
        # initialise kmer_tables for n = {2, 3, 4}
        self.t2, self.k2 = self.kmer_table(2)
        self.t3, self.k3 = self.kmer_table(3)
        self.t4, self.k4 = self.kmer_table(4)


    @staticmethod
    def kmer_table(k):
        # initialise nucleotides and their ordinants
        NUC = ['A', 'C', 'G', 'T']
        ordinants = [ord(c) for c in NUC]
        # generate all possible kmers for k
        kmers_str = [''.join(nu) for nu in product(NUC, repeat=k)]
        # the respective integer kmers
        # numbers modified to reduce table size
        kmers = np.array(list(product(ordinants, repeat=k))) - 64
        kmers = np.clip(kmers, 1, 8)
        # create multi-dim array for indexing
        ishape = [np.max(kmers) + 1] * k
        table = np.zeros(shape=ishape, dtype=np.uint8)
        # populate table
        kmer_ind = np.arange(len(kmers))
        for j in range(len(kmers)):
            # using tuple to NOT trigger fancy indexing
            table[tuple(kmers[j])] = kmer_ind[j]
        return table, kmers_str


    @staticmethod
    def integer_seq(seq):
        # prepare a sequence with its reverse complement in byte array
        pseq = np.array([seq + reverse_complement(seq)], dtype=bytes)
        # use view to cast to integer array
        # apply same transformation to reduce table size
        iseq = pseq.view('|S1').view(np.uint8) - 64
        iseq = np.clip(iseq, 1, 8)
        return iseq


    @staticmethod
    def translate_into_kmer_indices(iseq, k, table):
        # use indexing to get unique numbers for imers
        ituple = tuple(iseq[i: -(k-i+1)] for i in range(k))
        imer_t = table[ituple]
        return imer_t


    def count(self, seq, k):
        # input seq is normal concatenated string
        # turn into integer array first
        iseq = self.integer_seq(seq)
        # grab the correct pre-computed table
        table = getattr(self, f't{k}')
        # translate to kmer indices
        tseq = self.translate_into_kmer_indices(iseq, k, table)
        # fast counting with bincount
        counts = np.bincount(tseq)
        # make a dict for results
        count_dict = dict(zip(getattr(self, f'k{k}'), counts))
        return count_dict


    @staticmethod
    def expected_tetramer_frequencies(km):
        tetra_exp = {}
        tetra = km[2]
        for tet in tetra:
            tetra_exp[tet] = (1.0 * km[1][tet[:3]] * km[1][tet[1:]] / km[0][tet[1:3]])
        return tetra_exp


    @staticmethod
    def estimate_zscores(km, tetra_exp):
        tetra_sd = {}
        for tet, exp in list(tetra_exp.items()):
            den = km[0][tet[1:3]]
            tetra_sd[tet] = sqrt(exp * (den - km[1][tet[:3]]) * (den - km[1][tet[1:]]) / (den * den))

        tetra_z_num = np.array([km[2][tet] - exp for tet, exp in tetra_exp.items()])
        tetra_z_denom = np.array([tetra_sd[tet] for tet in tetra_exp.keys()])
        tetra_z_arr = np.divide(tetra_z_num, tetra_z_denom, where=tetra_z_denom > 0)
        tetra_z = dict(zip(tetra_exp.keys(), tetra_z_arr))
        return tetra_z


    def tetra_zscores(self, seq):
        # wrapper to get tetramer zscores from a sequence
        # first count 2,3,4-mers
        kmers = [dict()] * 3
        kmers[0] = self.count(seq, 2)
        kmers[1] = self.count(seq, 3)
        kmers[2] = self.count(seq, 4)
        # calculate expected frequencies
        tetramer_exp = self.expected_tetramer_frequencies(kmers)
        tetramer_zscores = self.estimate_zscores(km=kmers, tetra_exp=tetramer_exp)
        return tetramer_zscores



class TetramerDist:
    """
    this object takes sequence objects, not raw sequences
    this is in order to not recompute kmers
    """

    def __init__(self):
        self.kmc = KmerCounter()


    # def pearson_cor(self, seqo1, seqo2):
    #     # check if sequences already have tetramer zscores
    #     tz1 = getattr(seqo1, 'tetramer_zscores', None)
    #     tz2 = getattr(seqo2, 'tetramer_zscores', None)
    #     # calculate them if not
    #     if not tz1:
    #         seqo1.tetramer_zscores = self.kmc.tetra_zscores(seq=seqo1.seq)
    #     if not tz2:
    #         seqo2.tetramer_zscores = self.kmc.tetra_zscores(seq=seqo2.seq)
    #     # grab the zscores
    #     t1 = seqo1.tetramer_zscores
    #     t2 = seqo2.tetramer_zscores
    #     # intersection to get all 4mers present in both sequences
    #     tetramers = set(sorted(t1.keys())) & set(sorted(t2.keys()))
    #     z1 = [t1[t] for t in tetramers]
    #     z2 = [t2[t] for t in tetramers]
    #     # calculate pearson correlation
    #     cor = pearsonr(z1, z2)
    #     return cor


    def euclidean_dist(self, seqo1, seqo2):
        # check if kmers have already been counted
        t1 = getattr(seqo1, 'tmers', None)
        t2 = getattr(seqo2, 'tmers', None)
        # count them if not
        if not t1:
            seqo1.tmers = self.kmc.count(seqo1.seq, 4)
        if not t2:
            seqo2.tmers = self.kmc.count(seqo2.seq, 4)
        # grab the tetramers
        t1 = seqo1.tmers
        t2 = seqo2.tmers
        # intersection to get all 4mers present in both sequences
        tetramers = set(sorted(t1.keys())) & set(sorted(t2.keys()))
        c1 = np.array([t1[t] for t in tetramers])
        c2 = np.array([t2[t] for t in tetramers])
        # normalize
        n1 = c1 / np.sum(c1)
        n2 = c2 / np.sum(c2)
        # difference to calc Euclidean distance
        diff = n1 - n2
        euc = np.sqrt(np.sum(diff * diff))
        return euc




class IntraProb:

    def __init__(self):
        from scipy.stats import norm
        mean = 0
        std = 0.01037897 / 2
        mean2 = 0.0676654
        std2 = 0.03419337
        self.norm_intra = norm(mean, std)
        self.norm_inter = norm(mean2, std2)


    def intra_prob(self, e):
        prob = self.norm_intra.pdf(e) / (self.norm_inter.pdf(e) + self.norm_intra.pdf(e))
        return prob

    def calc_threshold(self):
        e = np.arange(0, 0.1, 0.001)
        ip = self.intra_prob(e)
        t = e[np.where(ip > 1e-10)][-1]
        return t


# instantiate to make available when imported
kmc = KmerCounter()
count_kmers = kmc.count                                 # basic counting function
tetramer_zscores = kmc.tetra_zscores                    # wrapper for tetramer z-scores

tdist = TetramerDist()
euclidean_dist = tdist.euclidean_dist                   # distance function for sequence OBJECTS
# pearson_dist = tdist.pearson_cor                        # distance function for sequence OBJECTS
euclidean_threshold = IntraProb().calc_threshold()      # constant from empirical parameters











