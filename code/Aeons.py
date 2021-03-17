#! /usr/bin/env python3
# standard library imports
from copy import deepcopy  # , copy
from random import randint
import os
import mmap
import gzip
import time
import subprocess
from shutil import which
from sys import executable

# import shlex
# from difflib import SequenceMatcher
# import argparse
# from pathlib import Path
# from itertools import combinations

# non standard library
import numpy as np
from scipy.special import gammaln, digamma
from scipy.sparse import csr_matrix, coo_matrix, diags  # tril
# from scipy import sparse
# from scipy.stats import dirichlet


import graph_tool as gt
from graph_tool.all import graph_draw
from graph_tool.topology import label_components
import khmer
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend("GTK3cairo")
# plt.switch_backend("Qt5cairo")






# TODO for later, proper implementation
# class MyArgumentParser(argparse.ArgumentParser):
#     def convert_arg_line_to_args(self, arg_line):
#         return arg_line.split()



class Constants:
    def __init__(self, mu, lam, sd, N, k):
        self.mu = mu
        self.lam = lam
        self.sd = sd
        self.N = N
        self.k = k

        self.rho = 10
        self.alpha = 10

        self.maxbatch = 10
        self.batchsize = 10

        self.err = 0.01
        self.errD = 0.0

        self.prior = 0.01
        self.perc = 0.1

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




    def update_graph(self, updated_kmers):
        # take updated kmers of a batch and perform updates to the graph structure
        # derive indices of the adjacency matrix

        # find the threshold for solid kmers
        threshold = self.bloom.solid_kmers()
        # print(f"using threshold {threshold}")

        # the updates need to be coordinates in the graph and a weight (count)
        updated_kmers = list(updated_kmers)
        n_updates = len(updated_kmers)

        updated_edges = np.zeros(shape=(n_updates * 2, 3), dtype='int64')
        # 17 slots, one for the index, the others for the path counts
        updated_paths = np.zeros(shape=(n_updates * 2, 17), dtype='int64')

        # kmer indices dict - collect the indices of this batch and write to file
        kmer_dict = dict()
        # to check if reverse comp is already updated, keep another set
        processed_kmers = set()

        km1 = set()
        indices = set()


        for i in range(n_updates):
            km = updated_kmers[i]

            # check if reverse comp is already updated
            if km in processed_kmers:
                continue

            # km = list(updated_kmers)[4]
            count = self.bloom.bloomf.get(km)

            # solid threshold
            if count < threshold:
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

            # collect the indices of source & target & the weight in an array
            updated_edges[i, :] = (lmer_ind, rmer_ind, count)
            updated_edges[i + n_updates, :] = (rmer_ind, lmer_ind, count)

            # now check the possible paths for both the lmer & rmer
            lpaths = count_paths(node=lmer, bloom_paths=self.bloom.bloomf_p, t_solid=threshold)
            rpaths = count_paths(node=rmer, bloom_paths=self.bloom.bloomf_p, t_solid=threshold)

            # collect the indices and counts for the paths
            updated_paths[i, 0] = lmer_ind
            updated_paths[i, 1:] = lpaths
            updated_paths[i + n_updates, 0] = rmer_ind
            updated_paths[i + n_updates, 1:] = rpaths

            # add the processed kmer to the set
            processed_kmers.add(km)
            processed_kmers.add(reverse_complement(km))

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
            print(f"{coll} hash collisions")

        # reduce the updated arrays to exclude duplicated reverse comps
        updated_edges = updated_edges[~np.all(updated_edges == 0, axis=1)]
        updated_paths = updated_paths[~np.all(updated_paths == 0, axis=1)]

        # apply the updated edges to the adjacency matrix
        self.adjacency[updated_edges[:, 0], updated_edges[:, 1]] = updated_edges[:, 2]

        # save the updated paths - used in a separate function to update the scores
        # also save the edges    - used for writing the GFA file
        self.updated_paths = updated_paths
        self.updated_edges = updated_edges
        self.kmer_dict = kmer_dict


    def update_graph_p(self, updated_kmers):
        """
        Updating the k+1 mer graph
        """
        # the updates need to be just coordinates in the graph
        updated_kmers = list(updated_kmers)
        n_updates = len(updated_kmers)

        updated_edges = np.zeros(shape=(n_updates * 2, 2), dtype='int64')

        # to check if reverse comp is already updated, keep another set
        processed_kmers = set()

        km1 = set()
        indices = set()

        for i in range(n_updates):
            km = updated_kmers[i]

            # check if reverse comp is already updated
            if km in processed_kmers:
                continue

            # slice the vertices
            # here we cut the lmer and rmer by subtracting 2. i.e. skipping a node in between to mark the path
            # but use the same indices as in the other matrix
            lmer = km[:-2]
            rmer = km[2:]
            # and get their index
            lmer_ind = self.kmer2index(kmer=lmer, m=True)
            rmer_ind = self.kmer2index(kmer=rmer, m=True)


            # collect the indices of source & target & the weight in an array
            updated_edges[i, :] = (lmer_ind, rmer_ind)
            updated_edges[i + n_updates, :] = (rmer_ind, lmer_ind)

            # add the processed kmer to the set
            processed_kmers.add(km)
            processed_kmers.add(reverse_complement(km))

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
            print(f"{coll} hash collisions")

        # reduce the updated arrays to exclude duplicated reverse comps
        updated_edges = updated_edges[~np.all(updated_edges == 0, axis=1)]

        # apply the updated edges to the adjacency matrix
        self.adjacency_p[updated_edges[:, 0], updated_edges[:, 1]] = 1



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
        highly_cov_nodes = np.unique(np.where(updated_paths_sorted > self.score_array.shape[0])[0])

        # grab the patterns to calculate
        calc = np.zeros(updated_paths.shape[0], dtype="bool")
        calc[np.unique(np.append(highly_complex_nodes, highly_cov_nodes))] = True
        calc_paths = updated_paths_sorted[calc]
        n_calc = calc_paths.shape[0]
        print(n_calc)

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
        print(nmiss)

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
        # for testing, transfrom the matrix to an adjacency list so it can be visualised with gt
        # init graph
        gtg = gt.Graph(directed=False)

        # init edge weights
        eprop = gtg.new_edge_property("int")
        eprop.a.fill(0)
        # internalise the property
        gtg.ep["weights"] = eprop

        cx = mat.tocoo()
        edges = []
        for i, j, v in zip(cx.row, cx.col, cx.data):
            edges.append((str(i), str(j), v))

        kmer_indices = gtg.add_edge_list(edges, hashed=True, hash_type="string", eprops=[gtg.ep.weights])

        ki_list = []
        for ki in kmer_indices:
            ki_list.append(int(ki))
        ki = np.array(ki_list)

        gtg.vp["ind"] = kmer_indices

        # argsort the kmer indices to then associate the properties
        ki_sorting = np.argsort(ki)

        # add scores as property for plotting
        sprop = gtg.new_vertex_property("float")
        sprop.a[ki_sorting] = self.scores.data
        gtg.vp["scores"] = sprop

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
        print(ncomp)
        print(comp_hist)

        # mean weight
        mean_weight = np.mean(gtg.ep.weights.a)
        print(mean_weight)

        self.gtg = gtg


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
        ncomp = len(set(comp_label))
        print(ncomp)
        print(comp_hist)

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

        for i in range(n_culdesac):
            # find available indices for the absorbers
            # brute force search
            c = culdesac[i]

            abs_a = c + 1
            while self.adjacency[abs_a, :].sum() != 0:
                abs_a += 1

            abs_b = abs_a + 1
            while self.adjacency[abs_b, :].sum() != 0:
                abs_b += 1

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
        culdesac_scores = np.repeat(self.scores[culdesac].toarray().squeeze(), 2)
        # make indices for the new vertices from the set
        abs_ind = list(abs_vertices)
        scores_absorbers = self.scores.copy()
        scores_absorbers[abs_ind] = culdesac_scores

        # combine the absorber vertices and culdesac for returning
        culdesac_vertices = list(culdesac) + abs_ind

        # TODO tmp
        # self.n_paths[abs_ind] = 1
        # self.pathsum[abs_ind] = 1

        return adjacency_absorbers, scores_absorbers, culdesac_vertices



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

        # keep a copy of the basic matrix for multiplication
        hp_base = deepcopy(hp)

        # get the scores for each edge (rather the score of their target node)
        # simply index into the score vector from the absorber addition with the edge mapping
        edge_scores = np.squeeze(scores[edge_mapping[:, 2]].A)

        # first transition with H^1
        # I think arrival scores can be a vector instead of a matrix?
        arrival_scores = hp.multiply(edge_scores).tocsr()  # * ccl[0]
        arr_scores = np.array(arrival_scores.sum(axis=1))

        # # In this function we calculate both utility and S_mu at the same time
        # s_mu = deepcopy(arrival_scores).tocsr()
        s_mu = 0  # dummy init

        # then transition each step of the ccl
        ccl = self.rld.ccl
        n_steps = ccl.shape[0]

        for i in range(1, n_steps):
            # increment transition step
            # (multiplication instead of power for efficiency)
            hp = hp @ hp_base

            # reduce the density of the probability matrix
            # if i % 5 == 0:
            hp = filter_low_prob(hp)

            # multiply by scores and add - this is element-wise per row
            transition_score = hp.multiply(edge_scores).tocsr()

            # # calculate smu on the fly
            # if i <= self.rld.mu:
            #     s_mu += transition_score

            # element-wise multiplication of csr matrix and float
            tsp = transition_score.multiply(ccl[i]).tocsr()
            arr_scores += np.array(tsp.sum(axis=1))
            # since we calculate s_mu on the fly, we save it once i reaches mu
            if i == self.rld.mu:
                s_mu = arr_scores.copy()


        # row sums are utility for each edge
        # utility = np.squeeze(np.array(arrival_scores.sum(axis=1)))
        utility = np.squeeze(arr_scores)
        s_mu_vec = np.squeeze(s_mu)
        # add back the original score of the starting node
        utility += edge_scores
        s_mu_vec += edge_scores

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
        # but also keep track of the average benefit when all fragments are rejected (Ubar0)
        self.ubar0 = np.mean(s_mu_vec)




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
        print(f'Accepting nodes: {strat_size}, {strat_size / uot.shape[0]}')



class Bloom:
    """
    class that handles all things kmer
    """
    def __init__(self, k, genome_estimate):
        # constants and vars for the size of the bloom filter
        self.target_table_size = 5e6  # TODO test these params (and check paper)
        self.num_tables = 5
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

        # legacy
        self.kcount = 0




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
            kmers = filt.get_kmers(seq)
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
            self.read_lengths[len(seq)] += 1


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
        n_edges = int(updated_edges.shape[0] // 2)
        with open(self.links, 'a') as f:
            for i in range(n_edges):
                f.write(f'L\t'
                        f'{updated_edges[i, 0]}\t'
                        f'+\t'
                        f'{updated_edges[i, 1]}\t'
                        f'+\t'
                        f'{self.overlap}M\n')

                # create also the rev comp link
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




class GraphMapper:
    def __init__(self, ref, mu, gaf=None, fa_tmp=None):
        # find the executable
        self.exe = find_exe(name="GraphAligner")
        self.ref = ref
        self.mu = mu

        # a filename for the temporary fastq file that the mapper uses
        if not fa_tmp:
            self.fa_tmp = "tmp"
        else:
            self.fa_tmp = fa_tmp

        if not gaf:
            self.gaf = "mapped"
        else:
            self.gaf = gaf





    def truncate_fasta(self, reads):
        # the mapper needs to read sequences from a file
        # so we truncate and write them to a temporary file
        filename = f'{self.fa_tmp}.fa'
        f = open(filename, "w")

        for rid, seq in reads.items():
            fa_line = f'>{rid}\n{seq[: self.mu]}\n'
            f.write(fa_line)
        f.close()
        return filename


    def map(self, reads):
        # create and map a temporary sequence file
        # TODO tune params

        # create a tmp fasta file
        fasta = self.truncate_fasta(reads=reads)
        self.gaf_out = f'{self.gaf}.gaf'

        # build the command
        ga = f"{self.exe} -g {self.ref} -a {self.gaf_out} -f {fasta}" \
             f" -x dbg --seeds-minimizer-length 7 --seeds-minimizer-windowsize 11"
        # execute the mapper
        out, err = execute(ga)

        self.out = out
        self.err = err
        return out, err


    def parse_stdout(self):
        # parse stdout from GraphAligner run
        # stdout contains some useful info: total bases, mapped reads
        n_reads = 0
        n_mapped = 0
        n_bases = 0

        for line in self.out.split("\n"):
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
        return n_bases, n_reads, n_mapped, n_unmapped










class AeonsRun:

    def __init__(self, const, fq_source):
        self.const = const
        self.switch = False
        # self.args = args

        # set up the fq stream
        self.fq = FastqStream(source=fq_source)
        self.fq.scan_offsets()
        self.fq.load_offsets(seed=0, shuffle=False, batchsize=const.batchsize, maxbatch=const.maxbatch)

        # set up the bloom filter, mapper ...
        self.bloom = Bloom(k=const.k, genome_estimate=const.N)
        self.gfa = GFA(file="test", k=const.k)
        self.mapper = GraphMapper(ref=self.gfa.gfa, mu=const.mu)
        self.dbg = SparseGraph(size=int(1e7), const=const, bloom=self.bloom)





    def process_batch(self):
        # first get a new batch of reads from the mmap
        read_lengths, read_sequences, n_bases = self.fq.get_batch()

        # map them to the graph (only if the strategy switch is active)
        # pretending that we only have the first mu bases
        if self.switch:
            out, err = self.mapper.map(reads=read_sequences)
            # this then modifies the reads as if they had gone through the acceptance/rejection
            add_sequences = self.make_decision(read_sequences=read_sequences)

            # TODO continue here
            osl = [len(i) for i in read_sequences.values()]
            nsl = [len(i) for i in add_sequences.values()]


        # fill the bloom filter with new seqs and update the read length distribution
        self.bloom.fill(reads=read_sequences)
        self.dbg.rld.record(reads=read_sequences)

        # add the edges to both the kmer and k+1mer graph
        self.dbg.update_graph(updated_kmers=self.bloom.updated_kmers)
        self.dbg.update_graph_p(updated_kmers=self.bloom.updated_kmers_p)

        # update the scores of nodes with new path counts
        self.dbg.update_scores()

        # update the benefit at each node
        self.dbg.update_benefit()

        # find the next strategy
        self.dbg.find_strat()

        # update the files for mapping
        self.gfa.write_segments(kmer_dict=self.dbg.kmer_dict)
        self.gfa.write_links(updated_edges=self.dbg.updated_edges)
        self.gfa.write_gfa()

        # create a graph for viz
        # self.gt_format()
        # self.gt_format(mat=self.benefit)
        # self.gt_format(mat=self.strat)

        # TODO some decision that flips the strategy switch


    def make_decision(self, read_sequences):
        # decide accept/reject for each read
        add_seqs = dict()
        strat = self.dbg.strat
        decision = False

        # loop over gaf entries with generator function
        gaf_file = open(self.mapper.gaf_out, 'r')

        for record in parse_gaf(gaf_file):
            # record = list(parse_gaf(gaf_file))[4]
            # print(record)

            # filter by mapping quality
            # if record["mapping_quality"] < 55:
            #     continue

            # decision process
            # TODO check if this handles strands correctly
            # strand = 0 if record['strand'] == '+' else 1

            # skip very short alignments
            if record['n_matches'] < self.const.k:
                continue

            # extract the first transition and check what the strategy is for that edge
            if record['path'].startswith('>'):
                transition = [_conv_type(x, int) for x in record['path'].split('>') if x != ''][0:2]
                decision = strat[transition[0], transition[1]]
            elif record['path'].startswith('<'):
                transition = [_conv_type(x, int) for x in record['path'].split('<') if x != ''][-2:]
                decision = strat[transition[1], transition[0]]

            # ACCEPT
            if decision:
                record_seq = read_sequences[record["qname"]]
            # REJECT
            else:
                record_seq = read_sequences[record["qname"]][: self.const.mu]

            add_seqs[record["qname"]] = record_seq

        gaf_file.close()
        return add_seqs



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
        return read_lengths, read_sequences, basesTOTAL



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
        a = randint(1, self.p - 1)
        b = randint(0, self.p - 1)
        return lambda x: ((a * x + b) % self.p) % self.N








def size(graph):
    print(f'v: {graph.num_vertices()}, e: {graph.num_edges()}')
    return graph.num_vertices(), graph.num_edges()


def memsize(m):
    # return the approximate memory size of a sparse matrix
    m = csr_matrix(m)
    mem = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    print(f"{mem / 1e6} Mb")


def is_symm(a, rtol=1e-05, atol=1e-08):
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


def is_palindrome(seq):
    if seq == reverse_complement(seq):
        return True
    else:
        return False


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



def count_paths(node, bloom_paths, t_solid):
    # count the paths spanning a node
    # - instead of estimating them from combinations of incident edges
    # - actual k+1 mers are stored in a separate bloom filter and can be checked
    # special treatment needed for palindromic kmers, otherwise they return each count twice
    counts = [0] * 16
    i = 0
    if is_palindrome(node):
        for km in kmer_neighbors_palindromic(node):
            c = bloom_paths.get(km)
            if c >= t_solid:
                counts[i] = c
            i += 1
    else:
        for km in kmer_neighbors(node):
            c = bloom_paths.get(km)
            if c >= t_solid:
                counts[i] = c
            i += 1
    return counts


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

def kmer_neighbors(km):
    for x in 'ACGT':
        for y in 'ACGT':
            yield x + km + y


def kmer_neighbors_palindromic(km):
    pal_tuples = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "GA", "GC", "TA"]
    for t in pal_tuples:
        yield t[0] + km + t[1]



def init_s_array(prior=0.001, paths=16, maxcount=10):
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







def plot_gt(graph, vcolor=None, ecolor=None, hcolor=None, comp=None):
    # initialise figure to plot on to
    _, ax = plt.subplots()

    # transform edge weights to float for plotting
    if ecolor is not None:
        ecolor = ecolor.copy(value_type="float")

    vcol = vcolor if vcolor is not None else "grey"
    hcol = hcolor if hcolor is not None else [0, 0, 0, 0, 0]
    ecol = ecolor if ecolor is not None else ""

    # overwrite vcol with components if set
    if comp is not None:
        comp_label, _ = label_components(graph, directed=False)  # can be used for vertex_fill_color
        # print(set(comp_label))
        vcol = comp_label
        print(len(set(comp_label)))

    # if color is not None:
    #     obj_col.a = color
    # else:
    #     obj_col = "grey"

    a = graph_draw(graph, mplfig=ax,
                   #vertex_fill_color="grey",# vcmap=cm.coolwarm,        # vertex fill is used to show score/util
                   vertex_halo=True, vertex_halo_color=hcol,
                   vertex_text=vcol,                         # just indices
                   edge_text=ecol, edge_text_distance=0,                   # show edge weights as text
                   # edge_text=graph.edge_index, edge_text_distance=0,     # edge indices
                   #edge_color=ecol,#, vertex_size=2)#, ecmap=cm.coolwarm,                  # color edge weights
                   output_size=(3000, 3000), vertex_size=1)
    # return a
    plt.show(block=True)





def plot_i(graph):
    ind = graph.vp.ind.copy(value_type="float")
    graph_draw(graph, vertex_fill_color=ind, vcmap=cm.coolwarm, vertex_text=ind)


def plot_s(graph):
    scores = graph.vp.scores.copy(value_type="float")
    graph_draw(graph, vertex_fill_color=scores, vcmap=cm.coolwarm, vertex_text=graph.vp.npaths)


def plot_w(graph):
    w = graph.ep.weights.copy(value_type="float")
    graph_draw(graph, edge_color=w, ecmap=cm.coolwarm, edge_pen_width=5, edge_end_marker="arrow",
               vertex_fill_color="grey", vertex_size=20, vertex_color="white", vertex_shape="circle")




# def plot_u(graph, direc):
#     # separate function to plot the edge-centric benefit
#     ecol = graph.ep.util_zig if direc == 0 else graph.ep.util_zag
#     e_weights = graph.ep.edge_weights.copy(value_type="float")
#
#     _, ax = plt.subplots()
#     a = graph_draw(graph, mplfig=ax,
#                    edge_color=ecol, ecmap=cm.coolwarm, edge_pen_width=1.5,
#                    vertex_fill_color="grey", vertex_size=0.1,
#                    edge_text=e_weights, edge_text_distance=1, edge_font_size=2)  #,
#                    # output_size=(3000, 3000),
#                    # vertex_text=graph.vertex_index)  # just indices
#                    # edge_text=graph.edge_index, edge_text_distance=0,     # edge indices
#     plt.show(block=a)













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
    hashimoto_prob = normalize_matrix_rowwise(mat=hashimoto_weighted)
    # filter very low probabilities, e.g. edges escaping absorbers
    hashimoto_prob = filter_low_prob(prob_mat=hashimoto_prob)
    return hashimoto_prob


def normalize_matrix_rowwise(mat):
    # rowsums for normalisation
    rowsums = csr_matrix(mat.sum(axis=1))
    rowsums.data = 1 / rowsums.data
    # find the diagonal matrix to scale the rows
    rowsums = rowsums.transpose()
    scaling_matrix = diags(rowsums.toarray()[0])
    norm = scaling_matrix.dot(mat)
    return norm


def filter_low_prob(prob_mat, threshold=0.001):
    # filter very low probabilities by setting them to 0
    # this prevents the probability matrix from getting denser and denser because
    # of circular structures and the absorbers
    # simply index into the data, where it is smaller than some threshold
    prob_mat.data[np.where(prob_mat.data < threshold)] = 0
    # unset the 0 elements
    prob_mat.eliminate_zeros()
    # normalise the matrix again by the rowsums
    prob_mat = normalize_matrix_rowwise(mat=prob_mat)
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
