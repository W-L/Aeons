# STANDARD LIBRARY
from copy import deepcopy  # , copy
from random import randint
import os
import mmap
import gzip
import time
import subprocess
from shutil import which
from sys import executable
from itertools import combinations
# import argparse
# from pathlib import Path
# import shlex
# from difflib import SequenceMatcher

# CUSTOM IMPORTS
from rownorm import row_norm

# NON STANDARD LIBRARY
import numpy as np
from scipy.special import gammaln, digamma
from scipy.sparse import csr_matrix, coo_matrix  # , diags, tril
# from scipy import sparse
# from scipy.stats import dirichlet
import graph_tool as gt
from graph_tool.topology import label_components
import khmer
import matplotlib.pyplot as plt

# backend for interactive plots
# plt.switch_backend("GTK3cairo")
# plt.switch_backend("Qt5cairo")

import line_profiler
# import memory_profiler



# TODO for later, proper implementation
# class MyArgumentParser(argparse.ArgumentParser):
#     def convert_arg_line_to_args(self, arg_line):
#         return arg_line.split()



class Constants:
    def __init__(self, mu, lam, sd, N, k, err, maxbatch, batchsize, size):
        self.mu = mu
        self.lam = lam
        self.sd = sd
        self.N = N
        self.k = k

        self.rho = 10
        self.alpha = 10

        self.maxbatch = maxbatch
        self.batchsize = batchsize

        self.err = err
        self.errD = 0.0

        self.prior = 1e-10
        self.perc = 0.0

        self.size = size

        # self.eta = 11
        # self.windowSize = 2000
        # self.alphaPrior = 0.1
        # self.p0 = 0.1
        # self.mu = 5
        # self.lam = 50
        # self.rho = 10  # time cost for rejecting
        # self.alpha = 10  # time cost to acquire new fragment
        # self.substitution_error = 0.06
        # self.theta = 0.01  # expected genetic diversity from reference
        # self.deletion_substitution_ratio = 0.4
        # self.deletion_error = 0.05
        # self.err_missed_deletion = 0.1

    def __repr__(self):
        return str(self.__dict__)







class SparseGraph:
    def __init__(self, bloom, const, size=1e5):
        # size determines the shape of multiple objects and also the collision probability
        size = int(size)
        self.size = size
        self.prior = const.prior

        # initialise time cost   - not super accurate probably atm
        self.const = const
        self.t0 = (self.const.alpha + self.const.mu + self.const.rho)
        self.tc = (self.const.lam - self.const.mu - self.const.rho)

        # attach all objects that we need
        UniHash = UniversalHashing(N=size)
        self.uhash = UniHash.draw()
        self.bloom = bloom
        self.rld = LengthDist(lam=const.lam, sd=const.sd, mu=const.mu)


        # initialise the sparse adjacency matrix of the graph
        self.adjacency = csr_matrix((size, size), dtype=np.uint8)
        # initialise also the k+1 mer graph for path counting and verification
        self.adjacency_p = csr_matrix((size, size), dtype=np.uint8)
        # sparse array to save the scores
        # TODO this could be an array with fixed size later, since we know how large it has to be
        self.scores = csr_matrix((size, 1), dtype=np.float64)
        self.benefit = csr_matrix((size, size), dtype=np.float64)
        self.strat = csr_matrix((size, size), dtype=np.bool_)

        self.benefit_raw = csr_matrix((size, size), dtype=np.float64)

        # initialise the score array for fast indexing
        self.score_array = init_s_array(prior=self.prior, paths=16, maxcount=10)

        # TODO tmp sanity check
        self.pathsum = csr_matrix((size, 1), dtype=np.float64)
        self.n_paths = csr_matrix((size, 1), dtype=np.float64)







    def kmer2index(self, kmer, m=False, p=False):
        # transform a kmer to an index in the adjacency matrix
        # first hash using khmer to treat reverse complement correctly
        if m:
            bf = self.bloom.bloomf_m
        elif p:
            bf = self.bloom.bloomf_p
        else:
            bf = self.bloom.bloomf

        kmer_hash = bf.hash(kmer)
        # second step: compress using universal hash
        kmer_index = self.uhash(kmer_hash)
        return kmer_index


    def update_incremented_edges(self, increments, increments_p):
        # update the counts of edges that were found from mapping reads
        # saved in increments. updated symmetrically in lower and upper tri
        if len(increments) > 0 and len(increments_p) > 0:
            # swap axes so that unique works on each index tuple
            increments = increments.swapaxes(0, 1)
            # sort axis 1, so that duplicate reverse complements are also removed
            increments = np.sort(increments, axis=1)
            unique_incr, cnt = np.unique(increments, axis=0, return_counts=True)

            increments_p = increments_p.swapaxes(0, 1)
            increments_p = np.sort(increments_p, axis=1)
            unique_incr_p, cnt_p = np.unique(increments_p, axis=0, return_counts=True)

            self.adjacency[unique_incr[:, 0], unique_incr[:, 1]] += cnt.astype("uint8")
            self.adjacency[unique_incr[:, 1], unique_incr[:, 0]] += cnt.astype("uint8")
            self.adjacency_p[unique_incr_p[:, 0], unique_incr_p[:, 1]] += cnt_p.astype("uint8")
            self.adjacency_p[unique_incr_p[:, 1], unique_incr_p[:, 0]] += cnt_p.astype("uint8")

            # we only update the scores at nodes that have path updates
            # so we need to also add the updates from the increment array
            incremented_nodes = np.unique(increments)

            return incremented_nodes
        else:
            return np.array([], dtype="int")


    def add_novel_kmers_p(self, updated_kmers_p, threshold):
        # add new kmers to the graph, i.e. from unmapped reads
        n_updates_p = len(updated_kmers_p)
        updated_edges_p = np.zeros(shape=(n_updates_p * 2, 2), dtype='int64')

        km1 = set()
        indices = set()

        for i in range(n_updates_p):
            km = updated_kmers_p[i]

            bloom_count = self.bloom.bloomf_p.get(km)

            # solid threshold
            if bloom_count < threshold:
                continue

            # here we cut the lmer and rmer by subtracting 2. i.e. skipping a node in between to make the path
            # but use the same indices as in the other matrix
            lmer = km[:-2]
            rmer = km[2:]
            # get their index
            lmer_ind = self.kmer2index(kmer=lmer, m=True)
            rmer_ind = self.kmer2index(kmer=rmer, m=True)

            # collect the indices of source & target
            updated_edges_p[i, :] = (lmer_ind, rmer_ind)
            updated_edges_p[i + n_updates_p, :] = (rmer_ind, lmer_ind)

            # just for testing, collect all k-1mers
            if not reverse_complement(lmer) in km1:
                km1.add(lmer)
            if not reverse_complement(rmer) in km1:
                km1.add(rmer)

            # and all indices
            indices.add(lmer_ind)
            indices.add(rmer_ind)

        # check for collisions
        coll = len(km1) - len(indices)
        if coll > 0:
            print(f"hash collisions: {coll}")

        # apply the updated edges to the adjacency matrix
        # np.unique to circumvent weird behaviour of += 1 with duplicated indices
        unique_edges, cnt = np.unique(updated_edges_p, axis=0, return_counts=True)
        self.adjacency_p[unique_edges[:, 0], unique_edges[:, 1]] += cnt.astype("uint8")
        return unique_edges


    def add_novel_kmers(self, updated_kmers, threshold):
        # add the novel kmers to the standard graph
        n_updates = len(updated_kmers)
        updated_edges = np.zeros(shape=(n_updates * 2, 2), dtype='int64')

        # tmp testing for GFA writing
        novel_edges = np.zeros(shape=(n_updates, 2), dtype='int64')

        # collect indices of this batch for writing to file
        kmer_dict = dict()

        km1 = set()
        indices = set()

        for i in range(n_updates):
            km = updated_kmers[i]
            # km = list(updated_kmers)[4]

            # check if the bool threshold has been overcome,
            # otherwise we won't add a new kmer
            bloom_count = self.bloom.bloomf.get(km)

            # solid threshold
            if bloom_count < threshold:
                continue

            # slice the vertices
            lmer = km[:-1]
            rmer = km[1:]
            # and get their index
            lmer_ind = self.kmer2index(kmer=lmer, m=True)
            rmer_ind = self.kmer2index(kmer=rmer, m=True)

            # add to dict for dumping
            kmer_dict[lmer] = lmer_ind
            kmer_dict[rmer] = rmer_ind

            # collect the indices of source & target (index tuples can be duplicated)
            updated_edges[i, :] = (lmer_ind, rmer_ind)
            updated_edges[i + n_updates, :] = (rmer_ind, lmer_ind)

            novel_edges[i, :] = (lmer_ind, rmer_ind)

            # just for testing, collect all k-1mers
            if not reverse_complement(lmer) in km1:
                km1.add(lmer)
            if not reverse_complement(rmer) in km1:
                km1.add(rmer)

            # and all indices
            indices.add(lmer_ind)
            indices.add(rmer_ind)

        # check for collisions
        coll = len(km1) - len(indices)
        if coll > 0:
            print(f"hash collisions: {coll}")

        # reduce the updated arrays to exclude duplicated reverse comps
        # updated_edges = updated_edges[~np.all(updated_edges == 0, axis=1)]
        # updated_paths = updated_paths[~np.all(updated_paths == 0, axis=1)]

        # apply the updated edges to the adjacency matrix
        unique_edges, cnt = np.unique(updated_edges, axis=0, return_counts=True)
        nov_edges, cnt_nov = np.unique(novel_edges, axis=0, return_counts=True)

        self.adjacency[unique_edges[:, 0], unique_edges[:, 1]] += cnt.astype("uint8")
        return unique_edges, kmer_dict, nov_edges



    def path_counting(self, updated_edges, incr_nodes, threshold):
        # for all edges, which were either incremented or novel we need to check the path counts
        # function to get path counts for all updated kmers in a batch

        # count the paths spanning a node
        # - get the index of the node
        # - check the adjacency for all existing neighbors
        # - make all pairwise combinations of the existing neighbors
        # - check the existance of paths in the adjacency_p

        # gather all node indices for which we check the path counts
        updated_nodes = np.concatenate((np.unique(updated_edges), incr_nodes), dtype="int")

        # 17 slots, one for the index, rest for the path counts
        updated_paths = np.zeros(shape=(updated_nodes.shape[0], 17), dtype='int64')

        neighbors = self.adjacency[updated_nodes, :].tolil().rows

        for i in range(len(neighbors)):

            neighbor_combos = list(combinations(neighbors[i], 2))
            counts = [0] * 16

            for j in range(len(neighbor_combos)):
                n1, n2 = neighbor_combos[j]
                c = self.adjacency_p[n1, n2]
                if c >= threshold:
                    counts[j] = c

            # collect the indices and counts for the paths
            updated_paths[i, 0] = updated_nodes[i]
            updated_paths[i, 1:] = counts

        return updated_paths



    def update_graph(self, increments, increments_p, updated_kmers, updated_kmers_p):
        # here we perform different kinds of updates.
        # first we update the counts of edges that were found from mapped reads
        incr_nodes = self.update_incremented_edges(increments, increments_p)

        # find the threshold for solid kmers
        threshold = self.bloom.solid_kmers()
        # print(f"using threshold {threshold}")

        # then we add new kmers from reads that were not mapped to the graph
        # starting with the k+1 graph
        _ = self.add_novel_kmers_p(updated_kmers_p, threshold=threshold)
        # then the kmer graph
        unique_edges, kmer_dict, nov_edges = self.add_novel_kmers(updated_kmers, threshold=threshold)

        # then we collect the updated paths
        updated_paths = self.path_counting(updated_edges=unique_edges, incr_nodes=incr_nodes, threshold=threshold)

        # save the updated paths - used in a separate function to update the scores
        self.updated_paths = updated_paths
        # also save the edges    - used for writing the GFA file
        # and new kmer           - also used in writing the GFA
        self.updated_edges = unique_edges  # TODO tmp
        self.nov_edges = nov_edges  # TODO tmp

        self.kmer_dict = kmer_dict





    def update_scores(self):
        # wrapper to update the scores, either using the score array or calculating them from scratch
        # uses the updated_paths array, which holds the indices at the first column and the counts in the rest
        updated_paths = self.updated_paths[:, 1:]
        updated_indices = self.updated_paths[:, 0]

        # create container that will hold the scores
        scores_tmp = np.zeros(updated_paths.shape[0])

        # sort the path counts
        updated_paths_sorted = np.flip(np.sort(updated_paths, axis=1), axis=1)
        # if there are more than 3 options, we definitely have to calculate it
        highly_complex_nodes = np.where(updated_paths_sorted[:, 3] > 0)[0]
        # if any are higher than the shape of the score array
        highly_cov_nodes = np.unique(np.where(updated_paths_sorted >= self.score_array.shape[0])[0])

        # grab the patterns to calculate
        calc = np.zeros(updated_paths.shape[0], dtype="bool")
        calc[np.unique(np.append(highly_complex_nodes, highly_cov_nodes))] = True
        calc_paths = updated_paths_sorted[calc]
        n_calc = calc_paths.shape[0]
        print(f'calc: {n_calc}')

        # calc and assign the ones that need to be
        # we can not put these back into the score array though since that one is filled as much as it can be
        if calc_paths.shape[0] > 0:
            calc_s = node_score(counts=calc_paths, prior=self.prior)
            scores_tmp[calc] = calc_s

        # for the rest, we should be able to just index into the score array
        # TODO maybe wrap this into a try - except
        p = updated_paths_sorted[~calc]
        scores_tmp[~calc] = self.score_array[p[:, 0], p[:, 1], p[:, 2]]

        # check positions where scores were not found in the scoreArray (remain 0.0)
        missing = np.argwhere(scores_tmp == 0.0).flatten()
        nmiss = missing.shape[0]
        print(f'nmiss: {nmiss}') if nmiss > 0 else None

        # finally transfer the scores back to the permanent container using the initial indices
        self.scores[updated_indices] = scores_tmp


        # TODO tmp path sum
        pathsum = np.sum(updated_paths, axis=1)
        self.pathsum[updated_indices] = pathsum
        n_paths = np.count_nonzero(updated_paths, axis=1)
        self.n_paths[updated_indices] = n_paths



    #
    # def reduce_matrix(self):
    #     # TODO this might be useful for hashimoto calc or for checking symmetry
    #     # because of the hashing strategy, the matrix is bigger than necessary
    #     # so we reduce it to all existing nodes and remember their indices in the big matrix
    #     # (for rehashing purposes)
    #     adjr = csr_matrix(self.adjacency)
    #     # save the indices of the big matrix that are actually filled
    #
    #     filled_rows = adjr.getnnz(1) > 0
    #     filled_cols = adjr.getnnz(0) > 0
    #
    #     adjr = adjr[filled_rows][:, filled_cols]
    #     self.adjr = adjr




    def gt_format(self, mat=None):
        if mat is None:
            mat = self.adjacency

        # transfrom the matrix to an adjacency list so it can be visualised with gt
        # init graph
        gtg = gt.Graph(directed=False)

        # init edge weights
        bprop = gtg.new_edge_property("float")
        bprop.a.fill(0)
        sprop = gtg.new_edge_property("float")
        sprop.a.fill(0)
        # internalise the property
        gtg.ep["weights"] = bprop
        gtg.ep["strat"] = sprop


        cx = mat.tocoo()
        strat_coo = self.strat.tocoo()
        edges = []
        for i, j, v, s in zip(cx.row, cx.col, cx.data, strat_coo.data):
            edges.append((str(i), str(j), v, s))

        kmer_indices = gtg.add_edge_list(edges, hashed=True, hash_type="string", eprops=[gtg.ep.weights, gtg.ep.strat])

        ki_list = []
        for ki in kmer_indices:
            ki_list.append(int(ki))
        ki = np.array(ki_list)

        gtg.vp["ind"] = kmer_indices

        # argsort the kmer indices to then associate the properties
        ki_sorting = np.argsort(ki)

        # add scores as property for plotting
        scprop = gtg.new_vertex_property("float")
        scprop.a[ki_sorting] = self.scores.data
        gtg.vp["scores"] = scprop

        # add pathsum as property for plotting
        pprop = gtg.new_vertex_property("float")
        pprop.a[ki_sorting] = self.pathsum.data
        gtg.vp["pathsum"] = pprop

        # add npaths as property for plotting
        nprop = gtg.new_vertex_property("float")
        nprop.a[ki_sorting] = self.n_paths.data
        gtg.vp["npaths"] = nprop


        comp_label, comp_hist = label_components(gtg, directed=False)
        ncomp = len(set(comp_label))
        # print(ncomp)
        print(f'component histogram: {comp_hist}')

        # mean weight
        # mean_weight = np.mean(gtg.ep.weights.a)
        # print(mean_weight)

        self.gtg = gtg
        return gtg



    def gt_format_p(self):
        # for testing, transfrom the matrix to an adjacency list so it can be visualised with gt
        # init graph
        gtg = gt.Graph(directed=False)


        cx = self.adjacency_p.tocoo()
        edges = []
        for i, j, v in zip(cx.row, cx.col, cx.data):
            edges.append((str(i), str(j), v))

        kmer_indices = gtg.add_edge_list(edges, hashed=True, hash_type="string")

        # ki_list = []
        # for ki in kmer_indices:
        #     ki_list.append(int(ki))
        # ki = np.array(ki_list)

        gtg.vp["ind"] = kmer_indices

        # # argsort the kmer indices to then associate the properties
        # ki_sorting = np.argsort(ki)
        #
        # # add scores as property for plotting
        # sprop = gtg.new_vertex_property("float")
        # sprop.a[ki_sorting] = self.scores.data
        # gtg.vp["scores"] = sprop
        #
        # # add pathsum as property for plotting
        # pprop = gtg.new_vertex_property("float")
        # pprop.a[ki_sorting] = self.pathsum.data
        # gtg.vp["pathsum"] = pprop
        #
        # # add npaths as property for plotting
        # nprop = gtg.new_vertex_property("float")
        # nprop.a[ki_sorting] = self.n_paths.data
        # gtg.vp["npaths"] = nprop


        comp_label, comp_hist = label_components(gtg, directed=False)
        # ncomp = len(set(comp_label))
        # print(ncomp)
        # print(comp_hist)

        self.gtg_p = gtg



    def add_absorbers(self):
        # Because ccl can not extend further at dead ends they will have a biased, lowered benefit.
        # Therefore we add some absorbing triangles that will propagate ccl at dead-ends

        # find all dead-ends by checking where there is only one nonzero field in a row
        # TODO is np where efficiet for sparse matrices?
        culdesac = np.where(self.adjacency.getnnz(axis=1) == 1)[0]
        n_culdesac = len(culdesac)

        # if there are no dead ends. This will probably never happen
        # except if the function is called on a graph that already has absorbers
        if n_culdesac == 0:
            return

        # add absorber for each end
        # this means 2 new vertices and 3 new edges (i.e. 6 in symmetric graph)
        abs_vertices = set()
        abs_edges = np.zeros(shape=(n_culdesac * 3, 2), dtype="int")

        # find free indices where we can put the absorbers
        # uses the fact that indptr stays the same if a row has no values in it
        # only look into the first few indptr at first
        free_indices = np.where(np.diff(self.adjacency.indptr[: n_culdesac * 4]) == 0)[0]
        # if there were not enough free_indices at this point look into the whole thing
        if free_indices.shape[0] < (n_culdesac + 1) * 2:
            free_indices = np.where(np.diff(self.adjacency.indptr) == 0)[0][(n_culdesac + 1) * 2]


        for i in range(n_culdesac):
            c = culdesac[i]
            abs_a = free_indices[i]
            abs_b = free_indices[i + n_culdesac]

            # add the new indices to a set
            abs_vertices.update([abs_a, abs_b])

            # construct new edges
            abs_edges[i] = (c, abs_a)
            abs_edges[i + n_culdesac] = (c, abs_b)
            abs_edges[i + n_culdesac * 2] = (abs_a, abs_b)

        # duplicate and flip for symmetric (reverse comp) edges
        abs_edges = np.vstack((abs_edges, np.fliplr(abs_edges)))
        # create a copy of the graph and apply new edges
        adjacency_absorbers = self.adjacency.copy()
        adjacency_absorbers[abs_edges[:, 0], abs_edges[:, 1]] = 99

        # transfer scores to absorbers
        # culdesac_scores = np.repeat(self.scores[culdesac].toarray().squeeze(), 2)
        # make indices for the new vertices from the set
        abs_ind = list(abs_vertices)
        scores_absorbers = self.scores.copy()

        # TODO tmp deactivate boost for the component ends
        fill_val = np.median(self.scores.data)
        culdesac_scores = np.full(shape=(culdesac.shape[0] * 2), fill_value=fill_val)
        scores_absorbers[culdesac] = fill_val  # TODO tmp
        scores_absorbers[abs_ind] = culdesac_scores

        # combine the absorber vertices and culdesac for returning
        culdesac_vertices = list(culdesac) + abs_ind

        # TODO tmp
        # self.n_paths[abs_ind] = 1
        # self.pathsum[abs_ind] = 1

        return adjacency_absorbers, scores_absorbers, culdesac_vertices


    # @profile
    def update_benefit(self):
        # calculate both the probability of transitioning to a node and arriving at that node

        # first create the absorbing structures at the tips
        # returns copies of both the graph and the scores
        adj, scores, culdesac = self.add_absorbers()

        # transform adjacency to hashimoto matrix
        # and return the mapping of edges to their source and target vertices
        hashimoto, edge_mapping = fast_hashimoto(adj)

        # using the k+1mer graph check which transitions actually don't exist
        hashimoto = self.eliminate_paths(hashimoto=hashimoto, edge_mapping=edge_mapping, culdesac=culdesac)

        # turn hashimoto into a probability matrix (rows sum to 1)
        # also normalizes per row and filters low probabilities
        hp = probability_hashimoto(hashimoto=hashimoto, edge_mapping=edge_mapping, adj=adj)

        # get the scores for each edge (rather the score of their target node)
        # simply index into the score vector from the absorber addition with the edge mapping
        self.edge_scores = np.squeeze(scores[edge_mapping[:, 2]].A)

        # modify the partition lengths according to the approximation and the reduction factor
        self.part_lengths, self.probs = prepare_partitions(ccl=self.rld.ccl, mu=self.rld.mu)

        # perform the graph hopping by matrix multiplications to fill the caches
        score_cache, hp_cache = self.graph_hopping(hp=hp)

        # using the caches collected before, figure out the scores of each node
        nnodes = self.edge_scores.shape[0]
        arr_scores = better_hops(score_cache=score_cache, hp_cache=hp_cache, nnodes=nnodes)

        # row sums are utility for each edge
        utility = np.squeeze(arr_scores)
        s_mu_vec = np.squeeze(self.s_mu)
        # add back the original score of the starting node
        utility += self.edge_scores
        s_mu_vec += self.edge_scores

        # not sure what the proper data structure is for this
        # but maybe return the edge mapping & benefit & smu, all with culdesac edges removed
        # or a dictionary that maps (source, target) -> benefit
        # or just another sparse matrix for each of them
        # remove culdesac edges
        culdesac_edges = np.all(np.isin(edge_mapping[:, 1:], culdesac), axis=1)
        em_notri = edge_mapping[~culdesac_edges, :]
        # do not save the benefit and s_mu separately, but already subtract smu
        self.benefit[em_notri[:, 1], em_notri[:, 2]] = utility[~culdesac_edges] - s_mu_vec[~culdesac_edges]
        self.benefit.eliminate_zeros()

        # TODO tmp
        # also save the raw benefit for now - plotting etc
        # self.benefit_raw[em_notri[:, 1], em_notri[:, 2]] = utility[~culdesac_edges]
        # self.benefit_raw.eliminate_zeros()

        # but also keep track of the average benefit when all fragments are rejected (Ubar0)
        self.ubar0 = np.mean(s_mu_vec)


    # @profile
    def graph_hopping(self, hp):
        # the number of matrix multiplications we need to perform to cover all hops
        maxsteps = int(self.part_lengths[0] + 1)

        # init caches that hold the scores and the hashimoto for all layers
        cache_size = self.probs.shape[0]
        score_cache = [0] * cache_size
        hp_cache = [0] * cache_size

        # how much steps do we take per partition?
        partition_order = np.argsort(self.part_lengths)
        partition_steps = self.part_lengths[partition_order]

        # keep a copy of the basic matrix for multiplication
        hp_base = deepcopy(hp)

        # first transition with H^1
        # multiply the scores of traversing edges with the probability state distribution
        arrival_scores = hp.multiply(self.edge_scores).tocsr()
        traversal_scores = np.array(arrival_scores.sum(axis=1))

        self.s_mu = 0

        # fill the caches everytime the length of a partition is hit
        for i in range(1, maxsteps + 1):

            # calculate s_mu on the fly, i.e. save it once i reaches mu
            if i == self.rld.mu:
                self.s_mu = traversal_scores.copy()

            # check if any of the partition lengths has been reached
            reached = (i) == partition_steps  #  - 1
            if np.any(reached):
                # which partition has been reached?
                partition_num = partition_order[np.where(reached)[0]]

                # to get the targets and their probabilities at each layer we save the hashimoto
                hpc = deepcopy(hp)

                for pn in partition_num:
                    # save the score after X steps and multiply with probability of traversing that partition
                    score_cache[pn] = traversal_scores.copy() * self.probs[pn]
                    hp_cache[pn] = hpc

            # transition another step of the hashimoto (matrix multiplication)
            hp = hp @ hp_base

            # reduce the density of the probability matrix and normalise
            hp = filter_low_prob(hp)

            # multiply by scores (element-wise per row)
            transition_score = hp.multiply(self.edge_scores).tocsr()

            # element-wise multiplication of csr matrix and float
            traversal_scores += np.array(transition_score.sum(axis=1))

        # return the scores and the state of the hashimoto at each layer
        return score_cache, hp_cache




    def eliminate_paths(self, hashimoto, edge_mapping, culdesac):
        # we want to verify which paths actually exist in the data
        # and eliminate the rest of the transitions in the hashimoto
        # first find the edges involved in the absorbing structures
        culdesac_edges = np.nonzero(np.isin(edge_mapping[:, 2], culdesac))[0]
        # find edges which have more than one next possible step
        multistep = np.where(np.sum(hashimoto, axis=1) > 1)[0]
        # exception for absorer tri-edges
        multistep = multistep[~np.isin(multistep, culdesac_edges)]
        # probably better to save the coordinates than having another massive array?
        # mask = np.zeros(shape=hashimoto.shape, dtype=bool)
        mask_list = []

        for edge in multistep:
            # edge = multistep[0] # tmp
            # get the source node of the multistep edge
            path_source = edge_mapping[:, 1][edge]

            # next possible edges and nodes
            next_steps = hashimoto[edge, :].nonzero()[1]
            step_targets = edge_mapping[:, 2][next_steps]
            for t in range(len(step_targets)):
                # t = 1 # tmp
                # now check whether the edge between the multistep source
                # and one of the targets exists in the k+1mer graph
                # the adjacency is symmetric, so no need to index twice
                path_target = step_targets[t]
                exists = self.adjacency_p[path_source, path_target]

                if not exists:
                    mask_list.append([edge, next_steps[t]])
                    # I don't think we need to add reciprocal,
                    # since they get checked in the outer loop anyway

        # transform mask_list to an array of coords
        mask_coords = list(zip(*mask_list))

        # set 0 for the non-observed paths
        if len(mask_coords) > 0:
            hashimoto[mask_coords[0], mask_coords[1]] = 0

        return hashimoto



    def find_strat(self):
        # keep track of the indices of the benefit
        benefit_indices = np.array(self.benefit.nonzero()).T
        # get the ratios of (benefit - S_mu) / time cost
        uot = self.benefit.data / self.tc
        # argsort the u/t array
        # i.e. list of edges from most to least valuable
        forwarded_i = np.argsort(uot)[::-1]

        # average benefit of strategy, initialized to the case that all fragments are rejected
        # gets calculated during the benefit update (no need to save smu)
        ubar0 = self.ubar0
        tbar0 = self.t0

        # we assume that every fragment has the same time cost, so we just fill an array with the same value
        time_cost = np.zeros(uot.shape[0])
        time_cost.fill(self.tc)

        # find the number of positions to accept
        # start with a strategy that rejects all fragments and increase its size one by one.
        ordered_benefit = self.benefit.data[forwarded_i]
        cumsum_u = np.cumsum(ordered_benefit) + ubar0
        cumsum_t = np.cumsum(time_cost) + tbar0
        # total number of accepted positions of the current strategy
        strat_size = np.argmax(cumsum_u / cumsum_t) + 1
        # threshold = ordered_benefit[strat_size - 1]

        # strat indices that should be set to 1
        accept_indices = benefit_indices[forwarded_i[:strat_size], :]
        self.strat[accept_indices[:, 0], accept_indices[:, 1]] = 1

        # TODO tmp for viz we can set the rejected ones to explicit 0s
        reject_indices = benefit_indices[forwarded_i[strat_size:], :]
        self.strat[reject_indices[:, 0], reject_indices[:, 1]] = 0

        # print number of accepted and fraction of accepted
        print(f'Accepting nodes: {strat_size}, {round(strat_size / uot.shape[0], 5)}')



class Bloom:
    """
    class that handles all things kmer
    """
    def __init__(self, k, genome_estimate):
        # constants and vars for the size of the bloom filter
        self.target_table_size = 5e7  # TODO test these params (and check paper)
        self.num_tables = 6
        self.k = k
        self.genome_estimate = genome_estimate

        # init 3 bloom filters - edges, nodes and paths
        self.bloomf = khmer.Counttable(self.k, self.target_table_size, self.num_tables)
        self.bloomf_p = khmer.Counttable(self.k + 1, self.target_table_size, self.num_tables)
        self.bloomf_m = khmer.Counttable(self.k - 1, self.target_table_size, self.num_tables)
        # init two sets: observed kmers for both filters - might be too large in the future?
        # once this is too big, write them to file and pass trough hash function again
        self.obs_kmers = dict()
        # self.obs_kmers_p1 = set()



    def __repr__(self):
        return f'{self.__dict__.keys()}'


    def fill(self, reads):
        # fill bloom filters with a batch of sequencing reads
        # reads is a dict of {id: seq}

        # fill the standard filter first
        # return all kmers that were updated in this batch
        self.updated_kmers, self.bloomf = self.consume_and_return(filt=self.bloomf, reads=reads)

        # bloom filter with (k+1)-mers to find the paths through each node for score calc
        self.updated_kmers_p, self.bloomf_p = self.consume_and_return(filt=self.bloomf_p, reads=reads)


    def consume_and_return(self, filt, reads):
        # take a bunch of reads, add them to a bloom filter and return all consumed kmers
        # reverse complement twins are counted in the same bin
        updated_kmers = set()

        # loop through all reads
        for _, seq in reads.items():
            # split the read into its kmers
            try:
                kmers = filt.get_kmers(seq)
            except ValueError:
                continue

            # add to the filter
            try:
                filt.consume(seq)
            except ValueError:
                # print("read shorter than k")
                continue

            updated_kmers.update(kmers)

        # return instead of attribute to make it more general
        return updated_kmers, filt



    def solid_kmers(self):
        # TODO fix this finally
        # from the current state of the bloom filter get threshold t that designates a kmer as solid
        # after Lin et al. 2016
        # for different vals of t - filter and count
        # if number of k-mers < genome_estimate: return t-1
        threshold = 0

        for t in range(0, 40):
            counter = 0
            # filter read kmers by abundance
            for km in self.obs_kmers:
                if self.bloomf.get(km) >= t:
                    counter += 1

            if counter > self.genome_estimate:
                # counter = 0
                continue

            else:
                # threshold = t - 1
                threshold = 1  # temp TODO this does not do anything at the moment
                break
        self.t = threshold
        return threshold




class LengthDist:
    """
    keeping track of the length distribution
    """
    def __init__(self, lam, sd, mu):
        # keep track of read lengths for the length distribution
        self.read_lengths = np.zeros(shape=int(1e6), dtype='uint16')

        # initialise as truncated normal distribution
        self.mu = mu
        self.lam = lam
        self.sd = sd
        self.prior_ld()

        # calc the complementary cumulative dist of L
        self.comp_cml_dist()

        # TODO come up with some approximation
        # approx_CCL = CCL_ApproxConstant(L=L, eta=eta)


    def prior_ld(self):
        # init a trunc normal as prior
        # get the maximum read length
        longest_read = int(self.lam + 10 * self.sd)
        # prob density of normal distribution
        x = np.arange(longest_read, dtype='int')
        L = np.exp(-((x - self.lam + 1) ** 2) / (2 * (self.sd ** 2))) / (self.sd * np.sqrt(2 * np.pi))
        # exclude reads shorter than mu
        L[: self.mu] = 0.0
        # normalise
        L /= sum(L)
        # transform lambda to read length from distribution mode
        mean_length = np.average(x, weights=L) + 1

        self.L = L
        self.mean_length = mean_length


    def record(self, reads):
        # keep track of read lengths
        for _, seq in reads.items():
            seq_len = len(seq)
            if seq_len > self.mu:
                self.read_lengths[seq_len] += 1


    def update(self):
        # update L and its approximation
        # get the lengths that have actually been observed
        observed_read_lengths = np.nonzero(self.read_lengths)
        length_sum = np.sum(observed_read_lengths * self.read_lengths[observed_read_lengths])
        # average length of all reads observed so far
        lam = length_sum / np.sum(self.read_lengths[observed_read_lengths])
        # longest read observed so far
        longest_read = np.max(np.where(self.read_lengths))
        L = np.copy(self.read_lengths[: longest_read]).astype('float64')
        # set everything shorter mu to 0 and normalise
        L[: self.mu] = 0.0
        L /= sum(L)

        self.L = L
        self.mean_length = lam

        # calc the complementary cumulative dist of L
        self.comp_cml_dist()


    def comp_cml_dist(self):
        # complement of cumulative distribtuion of read lengths
        ccl = np.zeros(len(self.L) + 1)
        ccl[0] = 1
        # subtract cml sum to get ccl
        ccl[1:] = 1 - self.L.cumsum()
        # correct numerical errors, that should be 0.0
        ccl[ccl < 1e-10] = 0
        # cut distribution off at some point to reduce complexity of calculating U
        ccl[ccl < 1e-6] = 0
        # or just trim zeros
        ccl = np.trim_zeros(ccl, trim='b')
        self.ccl = ccl


    def plot(self):
        plt.plot(self.L)
        plt.show()




class GFA:
    def __init__(self, file, k):
        self.segments = f'{file}.seg'
        self.links = f'{file}.lin'
        self.gfa = f'{file}.gfa'

        self.link_count = 0
        self.overlap = k - 2

        # delete the files if they exist from a previous run
        self.del_files()
        self.write_gfa()


    def del_files(self):
        # delete the files from a previous run
        comm = f'rm {self.segments} {self.links} {self.gfa}'
        execute(comm)


    def write_segments(self, kmer_dict):
        # append the new kmers and their indices to the segments file
        # TODO not sure this deals with reverse complements the correct way
        with open(self.segments, 'a') as f:
            for kmer, ind in kmer_dict.items():
                f.write(f'S\t{ind}\t{kmer}\n')


    def write_links(self, updated_edges):
        # append new edges to the links file
        # currently updated_edges contain both watson & crick
        # we only want to add one of them to the file
        # sorted_edges = np.sort(updated_edges, axis=1)
        # uniq_edges, cnt = np.unique(sorted_edges, axis=0, return_counts=True)
        # n_edges = int(updated_edges.shape[0] // 2)
        # n_edges = uniq_edges.shape[0]
        n_edges = updated_edges.shape[0]

        with open(self.links, 'a') as f:
            for i in range(n_edges):
                f.write(f'L\t'
                        f'{updated_edges[i, 0]}\t'
                        f'+\t'
                        f'{updated_edges[i, 1]}\t'
                        f'+\t'
                        f'{self.overlap}M\n')

                # # create also the rev comp link
                # f.write(f'L\t'
                #         f'{updated_edges[i, 1]}\t'
                #         f'-\t'
                #         f'{updated_edges[i, 0]}\t'
                #         f'-\t'
                #         f'{self.overlap}M\n')
                self.link_count += 1


    def write_gfa(self):
        # combine the segments and links into one file
        with open(self.gfa, "w") as gfa:
            gfa.write(f'H\tVN:Z:1.0\n')

        comm = f'cat {self.segments} {self.links} >> {self.gfa}'
        execute(comm)


    def size(self):
        # return size of the graph file
        return os.path.getsize(self.gfa)




class GraphMapper:
    def __init__(self, ref, mu, gaf=None, fa_tmp=None):
        # find the executable
        self.exe = find_exe(name="GraphAligner")
        self.ref = ref
        self.gfa = ref.gfa
        self.mu = mu

        # a filename for the temporary fastq file that the mapper uses
        if not fa_tmp:
            self.fa_tmp = "tmp"
        else:
            self.fa_tmp = fa_tmp

        if not gaf:
            self.gaf = "mapped.gaf"
        else:
            self.gaf = f'{gaf}.gaf'



    def write_fasta(self, reads, truncate=False):
        # the mapper needs to read sequences from a file
        # so we truncate and write them to a temporary file
        if truncate:
            filename = f'{self.fa_tmp}.trunc.fa'
        else:
            filename = f'{self.fa_tmp}.full.fa'

        f = open(filename, "w")

        for rid, seq in reads.items():
            if truncate:
                fa_line = f'>{rid}\n{seq[: self.mu]}\n'
            else:
                fa_line = f'>{rid}\n{seq}\n'

            f.write(fa_line)
        f.close()
        return filename


    def map(self, reads, truncate=False):
        # if the graph file was just built, skip mapping
        if self.ref.size() < 50:
            return None

        # params might need tuning, but don't seem to have too much of an effect
        length = 11
        wsize = 17

        # create a tmp fasta file for truncated and full length
        fasta = self.write_fasta(reads=reads, truncate=truncate)

        # build the command
        ga = f"{self.exe} -g {self.gfa} -a {self.gaf} -f {fasta}" \
             f" -x dbg --seeds-minimizer-length {length} --seeds-minimizer-windowsize {wsize}"

        # execute the mapper
        out, err = execute(ga)
        self.out = out
        self.err = err
        return self.gaf


    def size(self):
        # return size of the mapping file
        return os.path.getsize(self.gaf)










class AeonsRun:

    def __init__(self, const, fq_source):
        self.const = const
        self.switch = False
        # self.args = args

        # for writing batches to files
        self.batch = 0
        self.fa_out = "seq_data"
        self.cache_naive = dict()
        self.cache_aeons = dict()

        # after how much time should the sequenced be written to file
        # dump time is incremented every time a batch is written, which happens once that is overcome
        self.dump_every = int(1e5)
        self.dump_number_naive = 1
        self.dump_number_aeons = 1
        self.dump_time = self.dump_every

        # for keeping track of the sequencing time
        self.time_naive = 0
        self.time_aeons = 0

        # set up the fq stream
        self.fq = FastqStream(source=fq_source)
        self.fq.scan_offsets()
        self.fq.load_offsets(seed=0, shuffle=False, batchsize=const.batchsize, maxbatch=const.maxbatch)

        # set up the bloom filter, mapper ...
        self.bloom = Bloom(k=const.k, genome_estimate=const.N)
        self.gfa = GFA(file="test", k=const.k)
        self.mapper = GraphMapper(ref=self.gfa, mu=const.mu)
        self.dbg = SparseGraph(size=self.const.size, const=const, bloom=self.bloom)



    def process_batch(self, reads=None, no_incr=False):
        print(f'BATCH {self.batch}  ' + '#' * 30)

        # first get a new batch of reads from the mmap
        if not reads:
            read_sequences = self.fq.get_batch()
            self.read_sequences = read_sequences
        else:
            read_sequences = reads

        # map them to the graph as truncated
        gaf = self.mapper.map(reads=read_sequences, truncate=True)

        # this then modifies the reads as if they had gone through the acceptance/rejection
        reads_decision = self.make_decision(gaf=gaf, read_sequences=read_sequences)

        # then we map again with whatever we have after decision making
        gaf = self.mapper.map(reads=reads_decision, truncate=False)

        # now check which reads mapped and which ones are getting decomposed
        self.check_mappings(gaf=gaf, reads=reads_decision, no_incr=no_incr)

        ########################

        # fill the bloom filter with new seqs and update the read length distribution
        self.bloom.fill(reads=reads_decision)
        self.dbg.rld.record(reads=reads_decision)

        # new updating function that does not rely on the bloom for all counts
        self.dbg.update_graph(increments=self.increments,
                              increments_p=self.increments_p,
                              updated_kmers=self.updated_kmers,
                              updated_kmers_p=self.updated_kmers_p)

        # update the scores of nodes with new path counts
        self.dbg.update_scores()
        # TODO tmp to keep scores at tips down
        self.dbg.scores.data[self.dbg.scores.data > 2] = np.median(self.dbg.scores.data)

        # TODO tmp
        # self.dbg.scores[361504] = 3
        # self.dbg.scores[708901] = 3

        # update the benefit at each node
        self.dbg.update_benefit()

        # find the next strategy
        self.dbg.find_strat()

        ###########################

        # update the files for mapping
        self.gfa.write_segments(kmer_dict=self.dbg.kmer_dict)
        self.gfa.write_links(updated_edges=self.dbg.updated_edges)
        self.gfa.write_links(updated_edges=self.dbg.nov_edges)
        self.gfa.write_gfa()

        # create a graph for viz
        # self.dbg.gt_format()
        # self.dbg.gt_format(mat=self.dbg.benefit_raw)

        ############################

        # periodically update rld
        if (self.batch + 1) % 10 == 0:
            self.dbg.rld.update()

        self.update_times(read_sequences=read_sequences, reads_decision=reads_decision)
        self.write_batch(read_sequences=read_sequences, reads_decision=reads_decision)
        self.batch += 1

        #############################
        ind = nonzero_indices(self.dbg.adjacency)
        print(f'occupancy: {ind.shape[0] / self.dbg.size} \n')
        print()



    def make_decision(self, gaf, read_sequences):
        # decide accept/reject for each read

        # TODO looks like GA now produces an empty, non-binary file if nothing maps
        # so these two are not needed anymore
        # if gaf is None:
        #     return read_sequences

        # if is_binary(gaf):
        #     return read_sequences

        if self.mapper.size() < 10:
            return read_sequences

        if self.batch < 3:
            return read_sequences

        reads_decision = dict()
        strat = self.dbg.strat

        reject_count = 0

        # loop over gaf entries with generator function
        gaf_file = open(gaf, 'r')

        for record in parse_gaf(gaf_file):
            # record = list(parse_gaf(gaf_file))[4]
            # print(record)

            # only consider alignments with identity > 0.9
            # i.e. only reject if we are sure where the read comes from
            if record['idt'] < 0.9:
                continue

            # extract the first transition and check what the strategy is for that edge
            if record['path'].startswith('>'):
                transition = [_conv_type(x, int) for x in record['path'].split('>') if x != ''][0:2]
                # decision = strat[transition[0], transition[1]]
            elif record['path'].startswith('<'):
                transition = [_conv_type(x, int) for x in record['path'].split('<') if x != ''][-2:]
                transition.reverse()
            else:
                transition = []

            if len(transition) > 1:
                decision = strat[transition[0], transition[1]]
            else:
                continue

            # ACCEPT
            if decision:
                record_seq = read_sequences[record["qname"]]
            # REJECT
            else:
                record_seq = read_sequences[record["qname"]][: self.const.mu]
                reject_count += 1

            # append the read's sequence to a new dictionary of the batch after decision making
            reads_decision[record['qname']] = record_seq

        gaf_file.close()

        # all unmapped reads also need to be accepted, i.e. added back into the dict
        mapped_ids = set(reads_decision.keys())

        for read_id, seq in read_sequences.items():
            if read_id in mapped_ids:
                continue
            else:
                reads_decision[read_id] = seq

        # simple check for read lengths
        # osl = [len(i) for i in read_sequences.values()]
        # nsl = [len(i) for i in reads_decision.values()]
        # print(f'reads lengths before {osl}'
        #       f'reads lengths after  {nsl}')

        print(f'rejecting: {reject_count}')
        return reads_decision


    def collect_mappings(self, gaf):
        # check which reads map to the graph and which ones we will decompose into kmers
        # loop over gaf entries with generator function
        gaf_file = open(gaf, 'r')

        # container to save the increments
        sources = []
        targets = []
        sources_p = []
        targets_p = []
        processed_reads = set()

        # track the chunks (start -> end) of partial mapping reads
        partial_mappers = dict()

        # keep track of which edge counts to increment in the graph
        for record in parse_gaf(gaf_file):
            # record = list(parse_gaf(gaf_file))[0]

            # skip very short alignments only if the read is actually not that short
            if record['n_matches'] < self.const.k and record['qlen'] > self.const.k * 2:
                continue

            # only consider alignments with identity > 0.9
            if record['idt'] < 0.9:
                continue

            # check if the read aligns only partially
            # we record which bits did not align so that we can add them as kmers instead
            # but the found path still just gets incremented
            if record['alignment_block_length'] < record['qlen'] * 0.75:
                prt_map = (record['qstart'], record['qend'])
                partial_mappers[record['qname']] = prt_map

            # extract the transitions
            if record['path'].startswith('>'):
                transitions = [_conv_type(x, int) for x in record['path'].split('>') if x != '']
            elif record['path'].startswith('<'):
                # include a reverse
                transitions = [_conv_type(x, int) for x in record['path'].split('<') if x != '']
                transitions.reverse()
            else:
                continue

            # get the increments
            sources.extend(transitions[: -1])
            targets.extend(transitions[1: ])

            # get the +1 transitions by skipping another node
            sources_p.extend(transitions[: -2])
            targets_p.extend(transitions[2:])

            # keep track of the ids of the mapped reads
            processed_reads.add(record['qname'])

        gaf_file.close()

        # transform the transitions into an array for indexing
        increments = np.array((sources, targets), dtype=int)
        increments_p = np.array((sources_p, targets_p), dtype=int)

        return increments, increments_p, processed_reads, partial_mappers




    def check_mappings(self, gaf, reads, no_incr=False):
        # wrapper that checks which reads mapped to the graph, and thus their edges are simply incremented
        # TODO this is gross
        # if gaf is not None and not is_binary(gaf) and not no_incr and not self.mapper.size() < 10 and self.batch > 3:
        if not no_incr and not self.mapper.size() < 10 and self.batch > 3:
            increments, increments_p, processed_reads, partial_mappers = self.collect_mappings(gaf)
        else:
            increments, increments_p, processed_reads, partial_mappers = [], [], set(), dict()

        # filter out mapped reads
        reads_filt = filter_unmapped_reads(reads=reads,
                                           processed_reads=processed_reads,
                                           partial_mappers=partial_mappers)

        # filtered reads are decomposed and added to the graph as kmers
        updated_kmers, updated_kmers_p = decompose_into_kmers(reads=reads_filt, k=self.const.k)

        self.reads_filt = reads_filt
        self.increments = increments
        self.increments_p = increments_p
        self.updated_kmers = updated_kmers
        self.updated_kmers_p = updated_kmers_p


    def update_times(self, read_sequences, reads_decision):
        # increment the timer counts for naive and aeons

        # for naive: take all reads as they come out of the sequencer (memorymap)
        # total bases + (#reads * alpha)
        bases_total = np.sum([len(seq) for seq in read_sequences.values()])
        acquisition = self.const.batchsize * self.const.alpha
        self.time_naive += (bases_total + acquisition)

        # for aeons: bases of the fully sequenced reads (accepted & unmapped) and of the truncated reads
        read_lengths_decision = np.array([len(seq) for seq in reads_decision.values()])
        n_reject = np.sum(np.where(read_lengths_decision == self.const.mu, 1, 0))
        bases_aeons = np.sum(read_lengths_decision)
        acquisition = self.const.batchsize * self.const.alpha
        rejection_cost = n_reject * self.const.rho
        self.time_aeons += (bases_aeons + acquisition + rejection_cost)


    def write_batch(self, read_sequences, reads_decision):
        # add the current sequences to the cache
        for rid, seq in read_sequences.items():
            self.cache_naive[rid] = seq

        for rid, seq in reads_decision.items():
            self.cache_aeons[rid] = seq


        def write_seqdict(filename, seqdict):
            with open(filename, "w") as f:
                for rid, seq in seqdict.items():
                    fa_line = f'>{rid}\n{seq}\n'
                    f.write(fa_line)

        # check if any of the times exceeds the dump time
        if self.time_naive > (self.dump_time * self.dump_number_naive):
            print(f'dump naive #{self.dump_number_naive}. # of reads {len(list(self.cache_naive.keys()))}')
            # generate filename and write to file
            filename = f'{self.fa_out}_{self.dump_number_naive}_naive.fa'
            write_seqdict(filename, self.cache_naive)
            # increment the dump counter
            self.dump_number_naive += 1
            # reset the cache
            self.cache_naive = dict()

        if self.time_aeons > (self.dump_time * self.dump_number_aeons):
            print(f'dump aeons #{self.dump_number_aeons}. # of reads {len(list(self.cache_aeons.keys()))}')
            # generate filename and write to file
            filename = f'{self.fa_out}_{self.dump_number_aeons}_aeons.fa'
            write_seqdict(filename, self.cache_aeons)
            # increment the dump counter
            self.dump_number_aeons += 1
            # reset the cache
            self.cache_aeons = dict()



    def __repr__(self):
        return str(self.__dict__)


class FastqStream:
    """
    Stream reads from a fastq file (4 lines per read).
    Class only used for in silico experiments when we have a large fastq file that we randomly sample reads from
    """
    def __init__(self, source, log_each=int(1e5)):
        self.source = source  # path to file-like object

        # check if file is gzipped. Not very good
        suffix = source.split('.')[-1]
        if suffix == 'gz':
            self.gzipped = True
        else:
            self.gzipped = False

        self.log_each = int(log_each)  # defining logging frequency of scanning for offsets
        self.filesize = int(os.stat(source).st_size)
        print(f"Representing {self.filesize / 1e6} Mbytes of data from source: {self.source}")


    def scan_offsets(self, k=4):
        """
        Scan file to find byte offsets. Offsets are created for chunks of k lines each (4 for fastq)
        """
        tic = time.time()
        tmp_offsets = []
        read_num = 0

        # f = open(self.source, 'rb')
        with open(self.source, 'rb') as f:
            k_tmp = 1
            # memory-map the file; lazy eval-on-demand via POSIX filesystem
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # handle gzipped seemlessly
            if self.gzipped:
                mm = gzip.GzipFile(mode="rb", fileobj=mm)

            for _ in iter(mm.readline, b''):
                if k_tmp % k == 0:
                    pos = mm.tell()
                    tmp_offsets.append(pos)
                    k_tmp = 1
                    read_num += 1
                    # status update in case there are many reads
                    if read_num % self.log_each == 0:
                        print(f"{read_num} reads scanned")
                else:
                    k_tmp += 1

                # if read_num >= limit:
                #     break

        toc = time.time()
        # convert to numpy array
        # uint32 for small and medium corpora?
        offsets = np.asarray(tmp_offsets, dtype='uint64')
        del tmp_offsets
        # write the offsets to a file
        np.save(f'{self.source}.offsets', offsets)
        print(f"DONE scanning {read_num} reads")
        print(f'wrote {len(offsets)} offsets to {self.source}.offsets.npy')
        print(f"{round(toc - tic, 4)} seconds elapsed scanning file for offsets")


    def load_offsets(self, seed=0, shuffle=False, batchsize=1, maxbatch=1):
        if seed == 0:
            seed = np.random.randint(low=0, high=int(1e6))
        np.random.seed(seed)
        offsets = np.load(f'{self.source}.offsets.npy')
        # add one batch for initialising length dist
        maxbatch = maxbatch + 1

        if shuffle:
            np.random.shuffle(offsets)
            print(f"offsets shuffled using random seed: {seed}")

        # shorten the offsets to number of reads we need
        len_offsets = len(offsets)
        n_reads = batchsize * maxbatch
        if n_reads < len_offsets:
            offsets = offsets[: n_reads]
        else:
            print("requested more reads than there are available in the fastq")
            # sys.exit()

        # restructure the offsets into 2D array to represent batches (rows)
        offsets = offsets.reshape((maxbatch, batchsize))
        self.offsets = offsets



    def get_batch(self):
        """
        return a batch of reads from the fastq file
        """
        batch = ''
        # check if offsets are empty
        if self.offsets.shape[0] == 0:
            print("no more batches left")
            # sys.exit()

        # f = open(self.source, 'rb')
        with open(self.source, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # handle gzipped files
            if self.gzipped:
                mm = gzip.GzipFile(mode="rb", fileobj=mm)
            # the first row of the offsets are the next batch
            batch_offsets = self.offsets[0, :]

            # this is probably LINUX specific and not POSIX
            # here we tell the kernel what to do with the mapped memory
            # we tell it to preload specific "pages" of the file into memory
            # which magically makes it even faster to access
            # pagesize is a LINUX (system)-specific constant of 4096 bytes per "page"
            pagesize = 4096  #
            # the start of WILLNEED needs to be a multiple of pagesize
            # so we take the modulo and move the start of the offset a little bit earlier if needed
            new_offsets = batch_offsets - (batch_offsets % pagesize)

            for new_offset in new_offsets:
                # we preload 20 pages of data following each read start
                # 20 pages = 80 kbytes (read of up to ~40 kbases, I think..)
                mm.madvise(mmap.MADV_RANDOM)
                mm.madvise(mmap.MADV_WILLNEED, int(new_offset), 20)

            # offset = batch_offsets[0]
            batch_offsets = np.sort(batch_offsets)
            for offset in batch_offsets:
                try:
                    # here's the magic. Use the offset to jump to the position in the file
                    # then return the next 4 lines, i.e. one read
                    chunk = self.get_single_read(mm=mm, offset=offset)
                    # append the fastq entry to the batch
                    batch += chunk
                except:
                    print(f"Error at location: {offset}")
                    continue
                if len(chunk) == 0:
                    continue

            # add call to close memory map, only file itself is under with()
            mm.close()

        if not batch.startswith('@'):
            print("The batch is broken")


        # remove the row from the offsets so it does not get sampled again
        new_offsets = np.delete(self.offsets, 0, 0)
        self.offsets = new_offsets
        # parse the batch, which is just a long string into dicts
        read_lengths, read_sequences, basesTOTAL = self.parse_batch(batch_string=batch)
        return read_sequences  # read_lengths, basesTOTAL



    def get_single_read(self, mm, offset):
        # return 4 lines from a memory-mapped fastq file given a byte-wise position
        mm.seek(offset)
        chunk_size = 4
        chunk = b''
        # read the 4 lines of the fastq entry
        for _ in range(chunk_size):
            chunk += mm.readline()

        chunk = chunk.decode("utf-8")
        return chunk


    def parse_batch(self, batch_string):
        # take a batch in string format and parse it into some containers. Imitates reading from an actual fq
        read_lengths = {}
        read_sequences = {}

        batch_lines = batch_string.split('\n')
        n_lines = len(batch_lines)

        i = 0
        # since we increment i by 4 (lines for read in fq), loop until n_lines - 4
        while i < (n_lines - 4):
            # grab the name of the read. split on space, take first element, trim the @
            name = batch_lines[i].split(' ')[0][1:]
            seq = batch_lines[i + 1]
            read_len = len(seq)
            # update the containers
            read_lengths[name] = read_len
            read_sequences[name] = seq
            i += 4
        # get the total length of reads in this batch
        basesTOTAL = np.sum(np.array(list(read_lengths.values())))
        return read_lengths, read_sequences, basesTOTAL




class UniversalHashing:
    """
    Classic universal hashing with [(ax + b) % p] % N

    N = # bins in the hash table
    p = prime number (p >= N)
    """
    def __init__(self, N, p=None):
        self.N = N
        if p is None:
            p = find_prime(N)
        assert p >= N, 'Prime p should be > N'
        self.p = p

    def draw(self):
        # a = randint(1, self.p - 1)  # TODO tmp
        # b = randint(0, self.p - 1)
        a = 10
        b = 20
        return lambda x: ((a * x + b) % self.p) % self.N



############################


def replace(a1, ind, x):
    a2 = np.copy(a1)
    a2.put(ind,x)
    return(a2)


def unq(a):
    unique_vals, counts = np.unique(a.round(decimals=5), return_counts=True)
    return unique_vals, counts



def approx_ccl(ccl, eta=11):
    approx_ccl = np.zeros(eta, dtype='int32')

    i = 0
    for part in range(eta - 1):
        prob = 1 - (part + 0.5) / (eta - 1)
        while ccl[i] > prob:
            i += 1
        approx_ccl[part] = i
    # approx_ccl[0] gives the length that reads reach with probability 1
    # approx_ccl[1] length that reads reach with prob 0.9
    # etc..

    partition_lengths = np.diff(np.concatenate((np.zeros(1), approx_ccl)))[:-1]

    return partition_lengths



def prepare_partitions(ccl, mu, eta=11):
    # TODO could do with some clean-up
    # ccl = self.dbg.rld.ccl
    # get the lengths of the piecewise constant approximation
    partition_lengths = approx_ccl(ccl, eta)
    # the probabilities of the approximate intervals
    probs = np.linspace(0.9, 0.1, eta - 2)

    # make sure the first partition is the longest
    partition_lengths_sort = np.argsort(partition_lengths)[::-1]
    longest_part_index = partition_lengths_sort[0]
    longest_part = partition_lengths[longest_part_index]
    assert longest_part_index == 0

    # try to reduce the number of matrix multiplications even more
    # by splitting up the first partition until
    # it is not longer than the second biggest anymore
    # or longer than mu if that's shorter than the second partition
    second_longest = partition_lengths[partition_lengths_sort[1]]
    if longest_part < 100:
        reduction = 1
        maxsteps = np.ceil(longest_part / reduction)
        # maxsteps = np.floor(longest_part / reduction) # TODO tmp
    else:
        reduction = int(np.floor(longest_part / second_longest) - 1)
        # the new maximum amount of matrix mult we need
        maxsteps = np.ceil(longest_part / reduction)
        # if the reduction is too much, revert it so that the maxsteps is at least 100
        while maxsteps < 100 and reduction > 2:
            reduction -= 1
            maxsteps = np.ceil(longest_part / reduction)

    assert longest_part / reduction > second_longest


    # check that the longest part is still larger than mu
    if maxsteps < mu:
        reduction = np.floor(longest_part / mu) + 1
        maxsteps = np.ceil(longest_part / reduction)
        assert longest_part / reduction > second_longest
        assert longest_part / reduction > mu

    # adapt the partition lengths with the reduction factor
    if reduction == 1:
        part_len = partition_lengths
    else:
        # prepend the partition size of the reduced first partition (maxsteps)
        part_len = np.concatenate((np.full(shape=reduction, fill_value=maxsteps), partition_lengths[1:]))

    # prepend 1s to the probabilities for the chopped up first partition
    probs = np.concatenate((np.ones(reduction), probs))

    print(f"using reduction {reduction}")
    print(f"part lengths {part_len}")

    return part_len, probs



def better_hops(score_cache, hp_cache, nnodes):
    # init container for the benefit of each node
    acc = np.zeros(nnodes)
    # dummy targets of the first layer: just a range
    targets = np.arange(nnodes)
    scores0 = np.bincount(targets, np.squeeze(score_cache[0][targets]))
    acc += scores0

    # for the second layer, grab the indices of the nonzero elements
    target_tuples = np.swapaxes(np.array(hp_cache[0].nonzero()), 0, 1)
    targets = target_tuples[:, 1]
    targets_origin = target_tuples[:, 0]
    # the probabilities of the targets are just the data in the matrix
    targetprobs = hp_cache[0].data
    scores1 = np.bincount(targets_origin, np.squeeze(score_cache[1][targets] * targetprobs[:, None]))
    acc += scores1
    # plt.plot(acc, '.')
    # plt.show()

    # initialise all the necessary caches to save the targets,
    # their probabilities and their origin mappings of each layer
    mapping_cache = [0] * len(score_cache)
    mapping_cache[0] = targets_origin
    expand_cache = [0] * len(score_cache)
    expand_cache[0] = targets
    tp_cache = [0] * len(score_cache)
    tp_cache[0] = targetprobs

    # i = 1
    for i in range(1, len(score_cache) - 1):
        # use the previous targets to get the new targets
        new_targets = hp_cache[i][targets].nonzero()
        nt_flat = new_targets[1]
        expand_cache[i] = nt_flat

        # create a mapping between the previous targets and the new ones
        _, target_multiplicity = np.unique(new_targets[0], return_counts=True)
        mapping = np.repeat(np.arange(len(targets)), target_multiplicity)
        mapping_cache[i] = mapping

        # grab the probabilities of the splits between previous and next targets
        target_probs = hp_cache[i][targets].data
        tp_cache[i] = target_probs

        # new targets become the previous targets for next iteration (layer)
        # print(targets.shape)
        targets = nt_flat

        # use the new targets to grab the scores, multiply with probs and map to origins
        scores = np.squeeze(score_cache[i + 1][nt_flat])
        scores = scores * target_probs
        scores_red = np.bincount(mapping, scores)
        # print(np.unique(scores_red, return_counts=True))

        # now map this layer all the way back to the initial origin nodes
        for j in list(reversed(range(i))):
            lower_mapping = mapping_cache[j]
            lower_tp = tp_cache[j]
            scores_red = scores_red * lower_tp
            scores_red = np.bincount(lower_mapping, scores_red)

        acc += scores_red

        # debugging code for visualisations
        # print(unq(acc))
        # plt.plot(acc, '.')
        # plt.show()
        # i += 1
        # utility = np.squeeze(acc)
        # # utility += self.dbg.edge_scores
        # culdesac_edges = np.all(np.isin(edge_mapping[:, 1:], culdesac), axis=1)
        # em_notri = edge_mapping[~culdesac_edges, :]
        # self.benefit_raw[em_notri[:, 1], em_notri[:, 2]] = utility[~culdesac_edges]
        # self.benefit_raw.eliminate_zeros()
        # self.gt_format(mat=self.benefit_raw)
        # plot_benefit(graph=self.gtg, name="benefit_raw", esize=50)

    return acc




def parse_stdout(out):
    # parse stdout from GraphAligner run
    # stdout contains some useful info: total bases, mapped reads
    n_reads = 0
    n_mapped = 0
    n_bases = 0

    for line in out.split("\n"):
        if line.startswith("Input reads:"):
            # total amount of bases
            n_bases = line.split(" ")[-1]
            n_bases = int(''.join(c for c in n_bases if c.isdigit()))
            # number of reads
            n_reads = line.split(" ")[-2]
            n_reads = int(''.join(c for c in n_reads if c.isdigit()))

        if line.startswith("Reads with an"):
            # number of mapped reads
            n_mapped = line.split(" ")[-1]
            n_mapped = int(''.join(c for c in n_mapped if c.isdigit()))

    # unmapped reads
    n_unmapped = n_reads - n_mapped

    print(f'# bases: {n_bases}\n'
          f'# reads: {n_reads}\n'
          f'# mapped: {n_mapped}\n'
          f'# unmapped: {n_unmapped}\n')

    return n_bases, n_reads, n_mapped, n_unmapped


def diagnose_mappings(gaf):
    # loop over gaf entries with generator function
    gaf_file = open(gaf, 'r')

    for record in parse_gaf(gaf_file):
        # record = list(parse_gaf(gaf_file))[4]

        # extract the transitions
        if record['path'].startswith('>'):
            transitions = [_conv_type(x, int) for x in record['path'].split('>') if x != '']
        elif record['path'].startswith('<'):
            # include a reverse
            transitions = [_conv_type(x, int) for x in record['path'].split('<') if x != '']
            transitions.reverse()
        else:
            continue


        aln = record['alignment_block_length']
        rid = record['qname']
        qlen = record['qlen']
        idt = record["idt"]
        trans_len = len(transitions)
        unq, cnt = np.unique(np.array(transitions), return_counts=True)

        print(f'{rid}  {qlen}  {aln}  {idt}  {trans_len}  {cnt}')

    gaf_file.close()
    return




def size(graph):
    print(f'v: {graph.num_vertices()}, e: {graph.num_edges()}')
    return graph.num_vertices(), graph.num_edges()


def memsize(m):
    # return the approximate memory size of a sparse matrix
    m = csr_matrix(m)
    mem = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    print(f"{mem / 1e6} Mb")


def is_symm(a, rtol=1e-05, atol=1e-05):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def reverse_complement(dna):
    """
    Return the reverse complement of a dna string. Used when parsing the cigar
    of a read that mapped on the reverse strand.

    Parameters
    ----------
    dna: str
        string of characters of the usual alphabet

    Returns
    -------
    rev_comp: str
        the reverse complemented input dna

    """
    trans = str.maketrans('ATGC', 'TACG')
    rev_comp = dna.translate(trans)[::-1]
    return rev_comp



def densify(csr):
    arr = np.array(csr.todense())
    # print(arr)
    return arr


def kl_dirichlet(a, b):
    """
    matlab code from bariskurt.com
    D = gammaln(sum(alpha)) – gammaln(sum(beta)) – sum(gammaln(alpha)) + …
    sum(gammaln(beta)) + (alpha – beta) * (psi(alpha) – psi(sum(alpha)))’;

    psi == digamma function to calculate the geometric mean
    psi(alpha) – psi(sum(alpha)
    """
    # Kullback-Leibler divergence of two dirichlet distributions
    # a, b are arrays of parameters of the dirichlets
    # vectorised version
    a0 = np.sum(a, axis=1)
    b0 = np.sum(b, axis=1)

    D = np.sum((a - b) * np.subtract(digamma(a), digamma(a0)[:, None]), axis=1) +\
        np.sum(gammaln(b) - gammaln(a), axis=1) +\
        gammaln(a0) - gammaln(b0)
    return D




def isPrime(n):
    # Corner cases
    if n <= 1:
        return False
    if n <= 3:
        return True

    # This is checked so that we can skip
    # middle five numbers in below loop
    if (n % 2 == 0 or n % 3 == 0):
        return False

    i = 5
    while (i * i <= n):
        if (n % i == 0 or n % (i + 2) == 0):
            return False
        i = i + 6
    return True


def find_prime(N):
    prime = False
    n = N
    while not prime:
        if isPrime(n):
            prime = True
        else:
            n += 1
    return n



def filter_unmapped_reads(reads, processed_reads, partial_mappers):
    # takes a dict of reads and a set of read ids and filters the dictionary
    # used to figure out which reads mapped to the graph
    reads_filt = dict()
    for read_id, seq in reads.items():
        # if the read was found to map, skip it
        if read_id in processed_reads:
            continue
        else:
            reads_filt[read_id] = seq

    print(f'decomposing: {len(reads_filt)}')
    print(f'partial mappers: {len(partial_mappers)}')

    # then loop over the partial mappers and add the sequence chunks that was not mapped
    for read_id, (start, stop) in partial_mappers.items():
        # grab the sequence from the complete dict
        read_seq = reads[read_id]
        # trim it to the non-mapped bit
        read_seq_trimmed = read_seq[start: stop + 1]
        # also add these to the filtered reads
        reads_filt[read_id] = read_seq_trimmed

    return reads_filt


def decompose_into_kmers(reads, k):
    # container for the kmers that get added
    updated_kmers = []
    updated_kmers_p = []

    # then loop over the unmapped reads
    for read_id, seq in reads.items():
        # decompose kmers & filter
        kmers = [seq[i: i + k] for i in range(len(seq))]
        kmers = [km for km in kmers if len(km) == k]
        updated_kmers.extend(kmers)

        # decompose k+1mers & filter
        kmers = [seq[i: i + k + 1] for i in range(len(seq))]
        kmers = [km for km in kmers if len(km) == k + 1]
        updated_kmers_p.extend(kmers)

    return updated_kmers, updated_kmers_p







#
# def kmers(read, k=59):
#     for i in range(len(read) - k + 1):
#         yield read[i: i + k]


# def forward_neighbors(km):
#     for x in 'ACGT':
#         yield km[1: ] + x
#
#
# def backwards_neighbors(km):
#     for x in 'ACGT':
#         yield x + km[ :-1]



def init_s_array(prior, paths=16, maxcount=10):
    # up to which count the array should be filled
    crange = np.arange(maxcount)

    # generate patterns for up to (m x m x m x 0 ...) counts
    # where at most 3 paths are present
    xx, yy, zz = np.meshgrid(crange, crange, crange)
    counts = np.array((zz.ravel(), xx.ravel(), yy.ravel())).T
    counts = np.flip(np.sort(counts, axis=1), axis=1)
    unique_counts = np.unique([tuple(row) for row in counts], axis=0)

    # transfer to an array with the rest of the counts
    a = np.zeros(shape=(unique_counts.shape[0], paths))
    a[:, 0:3] = unique_counts

    # calculate the scores
    scores = node_score(counts=a, prior=prior)

    # only make it 3d, as soon as there are more than 3 paths, we calculate
    # could be extended to 4 later or something
    score_array = np.zeros(shape=(maxcount, maxcount, maxcount))

    # load the score array
    score_array[unique_counts[:, 0], unique_counts[:, 1], unique_counts[:, 2]] = scores

    return score_array



def node_score(counts, prior):
    # function to calculate the potential increase of KL
    # transform to array
    count_arr = np.array(counts, dtype='float')
    count_arr = np.flip(np.sort(count_arr, axis=1), axis=1)

    # add the prior values
    count_arr = np.add(count_arr, prior)
    # calculate potential information increase by increasing every path by 1
    # potential info increase is weighted by already existing links, cf Polya Urn model
    # for vectorisation: broadcast into third (actually second) dimension
    potential_counts = np.add(count_arr[:, None, :], np.identity(n=count_arr.shape[1]))

    # observation probabilities as weights
    p_obs = np.divide(count_arr, np.sum(count_arr, axis=1)[:, None])
    # KL divergence with every potential change after observing new data
    n_pattern = count_arr.shape[0]
    score = np.zeros(n_pattern)

    for p in range(n_pattern):
        score[p] = np.sum(p_obs[p, :] * kl_dirichlet(a=potential_counts[p, :, :], b=count_arr[p, None]))

    return score


def nonzero_indices(sparse):
    return np.where(np.diff(sparse.indptr) != 0)[0]



def iterate_sparse(x):
    cx = x.tocoo()
    contents = []
    for i, j, v in zip(cx.row, cx.col, cx.data):
        contents.append((i, j, v))
    return np.array(contents)


def allequal(a, b):
    # compare two sparse matrices
    content_a = iterate_sparse(a)
    content_b = iterate_sparse(b)

    r = np.allclose(content_a, content_b)
    print(r)














def half_incidence(adj):
    """
    Return the 'half-incidence' matrices from an adjacency matrix.
    If the graph has n nodes and m *undirected* edges, then the half-incidence matrices are two matrices,
    P and Q, with n rows and 2m columns.  One row for each node, and one column for each *directed* edge.
    For P, the entry at (n, e) is equal to 1 if node n is the source (or tail) of edge e, and 0 otherwise.
    For Q, the entry at (n, e) is equal to 1 if node n is the target (or head) of edge e, and 0 otherwise.

    Params
    ------
    mat: sparse adjacency matrix

    ordering (str):
        'blocks' (default), the two columns corresponding to the i'th edge
        are placed at i and i+m. The first m columns correspond to a random orientation,
        while the latter m columns correspond to the reversed orientation.
        Columns are sorted following the indices of the sparse graph matrix.

        'consecutive', first two columns correspond both orientations of first edge,
        the third and fourth row are the two orientations of the second edge..
        In general, the two columns for the i'th edge are placed at 2i and 2i+1.

    return_ordering (bool): return a function that maps an edge id to the column placement.
        'blocks'       lambda x: (x, m+x)
        'consecutive'  lambda x: (2*x, 2*x + 1)

    Returns
    -------
    P (sparse matrix), Q (sparse matrix), ordering (function or None).
    """
    # transform to coo format
    adj = adj.tocoo()

    # number of edges and nodes
    numedges = int(adj.getnnz() / 2)
    numnodes = adj.shape[0]

    # containers that hold the mapping of
    # edgeindex, n1, n2
    # source node, edge index
    # target node, edge index
    edge_mapping = np.zeros(shape=(numedges * 2, 3), dtype="int64")
    src_pairs = np.zeros(shape=(numedges * 2, 2))
    tgt_pairs = np.zeros(shape=(numedges * 2, 2))

    # loop over the edges and get the node indices of them
    for idx, (node1, node2) in enumerate(zip(adj.row, adj.col)):
        edge_mapping[idx, :] = (idx, node1, node2)
        src_pairs[idx] = (node1, idx)
        tgt_pairs[idx] = (node2, idx)

    # construct the sparse incidence matrices
    data = np.ones(2 * numedges)
    src_coo = coo_matrix((data, list((src_pairs[:, 0], src_pairs[:, 1]))), shape=(numnodes, 2 * numedges))
    tgt_coo = coo_matrix((data, list((tgt_pairs[:, 0], tgt_pairs[:, 1]))), shape=(numnodes, 2 * numedges))
    return src_coo, tgt_coo, edge_mapping



def fast_hashimoto(adj):
    """
    Construct Hashimoto (non-backtracking) matrix.
    modified from Torres 2018

    Params
    ------
    adjacency matrix

    Returns
    -------
    hashimoto: sparse csr matrix
    edge_mapping
    """
    # get the half incidence matrices
    sources, targets, em = half_incidence(adj)

    # calculate the coordinates of the hashimoto matrix
    temp = np.dot(targets.T, sources).asformat('coo')
    temp_coords = set(zip(temp.row, temp.col))
    coords = [(r, c) for r, c in temp_coords if (c, r) not in temp_coords]

    # fill a coo matrix with 1s
    data = np.ones(len(coords))
    shape = adj.getnnz()
    hashimoto = coo_matrix((data, list(zip(*coords))), shape=(shape, shape))
    return hashimoto.asformat('csr'), em






def probability_hashimoto(hashimoto, edge_mapping, adj):
    # transform a matrix to probabilities by multiplying with edge weights & normalising
    # grab the edge weights from the adjacency matrix of the graph
    weights = adj[edge_mapping[:, 1], edge_mapping[:, 2]].astype(int)
    weights[weights == 99] = 1e9

    hashimoto_weighted = csr_matrix(hashimoto.multiply(weights))
    # normalise to turn into probabilities
    row_norm.row_norm(hashimoto_weighted)
    # filter very low probabilities, e.g. edges escaping absorbers
    hashimoto_prob = filter_low_prob(prob_mat=hashimoto_weighted)
    return hashimoto_prob



def filter_low_prob(prob_mat, threshold=0.001):
    # filter very low probabilities by setting them to 0
    # this prevents the probability matrix from getting denser and denser because
    # of circular structures and the absorbers
    # simply index into the data, where it is smaller than some threshold
    prob_mat.data[np.where(prob_mat.data < threshold)] = 0
    # unset the 0 elements
    prob_mat.eliminate_zeros()
    # normalise the matrix again by the rowsums
    row_norm.row_norm(prob_mat)
    # prob_mat.eliminate_zeros()
    return prob_mat



def parse_gaf(gaf):
    # parse the raw gaf from GraphAligner
    # generator that yields named tuple
    fields = [
        "qname",
        "qlen",
        "qstart",
        "qend",
        "strand",
        "path",
        "plen",
        "pstart",
        "pend",
        "n_matches",
        "alignment_block_length",
        "mapping_quality",
        "tags",
    ]

    for record in gaf:
        record = record.strip().split("\t")
        record_dict = {fields[x]: _conv_type(record[x], int) for x in range(12)}

        tag_dict = dict()
        tags = {record[x] for x in range(13, len(record))}
        for tag in tags:
            tag = tag.strip().split(":")
            tag_dict[tag[0]] = tag[-1]

        record_dict['idt'] = float(tag_dict['id'])

        yield record_dict



def execute(command):
    # create the unix process
    running = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding='utf-8', shell=True)
    # run on a shell and wait until it finishes
    stdout, stderr = running.communicate()
    return stdout, stderr



def find_exe(name):
    # shutil.which seems to work mostly but is still not completely portable
    exe = which(name, path='/'.join(executable.split('/')[0:-1]))
    # exe = subprocess.run('which minimap2', shell=True, capture_output=True, universal_newlines=True).stdout
    return exe.strip()


def _conv_type(s, func):
    # Generic converter, to change strings to other types
    try:
        return func(s)
    except ValueError:
        return s


def is_binary(file, length=1024):
    # open the file in binary mode to read a little bit at the beginning
    chunk = []
    try:
        with open(file, 'rb') as f:
            chunk = f.read(length)
    except IOError as e:
        print(e)

    if not chunk:
        print("file not found")
        return

    # then try to decode it to utf-8
    # if that fails, the file is not a text file
    try:
        chunk.decode(encoding="utf-8")
        return False
    except UnicodeDecodeError:
        return True




# software eng side of things
# def setup_parser_realdata():
#     parser = MyArgumentParser(fromfile_prefix_chars='@')
#     parser.add_argument('--ref', dest='ref', type=str, default=None, required=True, nargs='*',
#                         help='path to (generated) reference; can be multiple')
#     parser.add_argument('--ref_name', dest='ref_name', type=str, default=None, required=True, nargs='*',
#                         help='name for reference(s); can be multiple')
#     parser.add_argument('--roi_mask', dest='roi_mask', type=str, default=None, nargs='*',
#                         help='path to the roi mask that corresponds to the generated reference; can be multiple')
#     parser.add_argument('--fq', dest='fq', type=str, default=None, help='path to fastq for streaming')
#     parser.add_argument('--run_name', dest='run_name', default="test", type=str, help='name for sequencing run')
#     parser.add_argument('--linearity', dest='linearity', default=None, required=True, nargs='*',
#                         help='specify if genome is linear with 1 or 0; can be multiple')
#     parser.add_argument('--ploidy', dest='ploidy', default=None, required=True, nargs='*',
#                         help='1 == haploid, 2==diploid; can be multiple')
#     parser.add_argument('--regions', dest='regions', default=None, nargs='*',
#                         help='whether sequencing only within regions of interest. 1/0')
#     parser.add_argument('--batch_size', dest='batch_size', default=100, type=int,
#                         help='Number of reads in a batch')
#     parser.add_argument('--cov_until', dest='cov_until', default=5, type=int,
#                         help='avrg cov. before first strategy update (0 = after first batch)')
#     # parser.add_argument('--stoptime', dest='stoptime', default=int(200e9), type=int,
#     #                     help='Stop after this time (~ final coverage * genome size)')
#     # parser.add_argument('--time_until', dest='time_until', default=int(100e9), type=int,
#     #                     help='Time before first strategy update (0 = after first batch)')
#     parser.add_argument('--maxbatch', dest='maxbatch', default=100, type=int,
#                         help='maximum number of batches of reads to process')
#     # parser.add_argument('--checkpoint_every', dest='checkpoint_every', default=20e6, type=float,
#     #                     help='sample entropy every X seq. bases')
#     parser.add_argument('--maxCov', dest='maxCov', default=30, type=int,
#                         help='maximum coverage that the scoreArray is initialised to; trade-off memory & speed')
#     parser.add_argument('--mu', dest='mu', type=int, default=500,
#                         help='len of truncated mapping')
#     parser.add_argument('--seed', dest='seed', default=0, type=int,
#                         help='input seed for fastq stream. If 0, seed will be randomized')
#     parser.add_argument('--deletion_error', dest='deletion_error', default=0.05, type=float,
#                         help='probability that a deletion is actually not a deletion')
#     parser.add_argument('--model', default=0, type=int, help="which model for posterior to use")
#     return parser



#
# def read_args_fromfile(parser, file):
#     with open(file, 'r') as f:
#         arguments = [ll for line in f for ll in line.rstrip().split()]
#     args = parser.parse_args(args=arguments)
#     return args
#



# random funcs that might be useful at some point

#
# def edges2adjmat(e): # convert edge list to adjacency matrix
#   ea = array(e)
#   numverts = ea.max() + 1
#   a = sparse.lil_matrix((numverts,numverts))
#
#   for edge in e:
#     a[edge[0].__int__(),edge[1].__int__()] = 1
#     a[edge[1].__int__(),edge[0].__int__()] = 1
#
#   return a
#
# def adjmat2lap(a): # convert adjacency matrix to a laplacian
#   numverts = a.shape[0]
#   degrees = a*ones(numverts)
#   degmat = sparse.lil_matrix((numverts,numverts))
#   degmat.setdiag(degrees)
#   lap = degmat - a
#
#   return lap


#
# def detect_cycles(hashimoto):
#     edges = int(hashimoto.shape[0] / 2)
#     h_base = deepcopy(hashimoto)
#     # count the cycles
#     cycles = []
#     trace = hashimoto.diagonal().sum()
#     cycles.append(trace)
#     for i in range(edges * 3):
#         hashimoto = hashimoto @ h_base
#         trace = hashimoto.diagonal().sum()
#         cycles.append(trace)
#     return cycles
#
#
# def has_cycles(hashimoto):
#     return hashimoto.diagonal().sum()
#
#




#
#
# def merge(source_node, target_node):
#     m = SequenceMatcher(None, source_node, target_node)
#     for o, i1, i2, j1, j2 in m.get_opcodes():
#         if o == 'equal':
#             yield source_node[i1:i2]
#         elif o == 'delete':
#             yield source_node[i1:i2]
#         elif o == 'insert':
#             yield target_node[j1:j2]
#         elif o == 'replace':
#             yield source_node[i1:i2]
#             yield target_node[j1:j2]
#
