#! /usr/bin/env python3
# standard library imports
# from copy import copy, deepcopy
# from pathlib import Path
import numpy as np
from random import randint
# import gzip
# import subprocess
# import shlex
# from difflib import SequenceMatcher
# import argparse


# scipy
from scipy.special import gammaln, digamma
# from scipy.stats import dirichlet
from scipy.sparse import csr_matrix, coo_matrix#, lil_matrix, csc_matrix  # diags
# from scipy import sparse
# from scipy.sparse.sparsetools import csr_scale_rows
# from itertools import combinations


import graph_tool as gt
from graph_tool.all import graph_draw
from graph_tool.topology import label_components
import khmer
import matplotlib.pyplot as plt
from matplotlib import cm
plt.switch_backend("GTK3cairo")
# plt.switch_backend("Qt5cairo")


#%%

# TODO for later, proper implementation
# class MyArgumentParser(argparse.ArgumentParser):
#     def convert_arg_line_to_args(self, arg_line):
#         return arg_line.split()


# TODO for later, proper implementation
# class Constants:
#     def __init__(self):
#         self.eta = 11
#         self.windowSize = 2000
#         self.alphaPrior = 0.1
#         self.p0 = 0.1
#         self.rho = 500  # time cost for rejecting
#         self.alpha = 300  # time cost to acquire new fragment
#         self.substitution_error = 0.06
#         self.theta = 0.01  # expected genetic diversity from reference
#         self.deletion_substitution_ratio = 0.4
#         self.deletion_error = 0.05
#         self.err_missed_deletion = 0.1
#
#     def __repr__(self):
#         return str(self.__dict__)
#


# TODO legacy implementation using gt directly
class Assembly:

    def __init__(self, bloom):
        # initialise the de bruijn graph for the assembly from a bloom filter
        self.bloom = bloom
        self.batch = 0
        # init a dict that keeps the index of each kmer - TODO not efficient!
        self.node_dict = dict()

    def initial_construction(self, reads):
        # init an empty graph
        self.dbg = gt.Graph(directed=False)
        # init edge weights
        self.init_ep(name="weights", val=0)
        # init temporary batch counter
        self.init_ep(name="batch", val=0)

        # use pre-loaded bloom filter to add edges to the graph
        self.bloom.fill(reads=reads)
        self.bloom.get_edge_list()
        edge_list = self.bloom.new_edges
        edge_weights = self.bloom.new_weights

        # add all edges in one swoop
        vertex_names = self.dbg.add_edge_list(edge_list, hashed=True, hash_type='string')
        # set the edge weights
        self.dbg.ep.weights.a = edge_weights

        # internalize the vertex names - TODO is this necessary for anything?
        # self.dbg.vp["nodes"] = vertex_names
        node_dict_tmp = {vertex_names[i]: i for i in range(self.dbg.num_vertices())}
        self.node_dict.update(node_dict_tmp)


    def update(self, new_reads):
        # update the graph after a new batch

        # the dbg's bloom is filled with new reads
        self.bloom.fill(reads=new_reads)
        # then we generate the new edges and their weights
        self.bloom.get_edge_list(from_scratch=False)
        # grab the new edges
        edge_list = self.bloom.new_edges
        edge_weights = self.bloom.new_weights

        # add the weight and batch value to the edges
        self.batch += 1

        # edge_list2 = [(self.node_dict[source], self.node_dict[target], weight, self.batch)
        #                 for ((source, target), weight) in zip(edge_list, edge_weights)]

        # add all edges in one swoop
        vertex_names = self.dbg.add_edge_list(edge_list,
                                              hashed=True,
                                              hash_type='string',
                                              eprops=[self.dbg.ep.weights, self.dbg.ep.batch])
        # internalize the vertex names
        self.dbg.vp["nodes"] = vertex_names

        comp_label, comp_hist = label_components(self.dbg, directed=False)
        ncomp = len(set(comp_label))
        print(ncomp)
        # print(comp_hist)


    def reconstruct(self):
        # reconstruct the graph from the bloom filter

        # the dbg's bloom is filled with new reads
        # self.bloom.fill(reads=new_reads)
        # then we generate all the edges and their weights
        _ = self.bloom.get_edge_list(from_scratch=True)
        # grab the new edges
        edge_list = self.bloom.new_edges
        edge_weights = self.bloom.new_weights

        # init an empty graph
        self.dbg = gt.Graph(directed=False)
        # init edge weights
        self.init_ep(name="weights", val=0)


        # edge_list = [(source, target, weight)
        #               for ((source, target), weight) in zip(edge_list, edge_weights)]

        # add all edges in one swoop
        vertex_names = self.dbg.add_edge_list(edge_list,
                                              hashed=True,
                                              hash_type='string')
        # set the edge weights
        self.dbg.ep.weights.a = edge_weights

        # internalize the vertex names
        self.dbg.vp["nodes"] = vertex_names

        # how many components are in the graph?
        comp_label, comp_hist = label_components(self.dbg, directed=False)
        ncomp = len(set(comp_label))
        print(ncomp)
        print(comp_hist)

        # mean weight
        mean_weight = np.mean(self.dbg.ep.weights.a)
        print(mean_weight)

        # tmp return for speed measures
        # return nkmers, ncomp


    def init_ep(self, name, val, t="int"):
        # init an edge property, e.g. edge weights
        eprop = self.dbg.new_edge_property(t)
        eprop.a.fill(val)  # temporary, need to implement edge merging if there are already parallel ones
        # internalise the property
        self.dbg.ep[name] = eprop


    def update_scores(self, prior):
        # loop over nodes - TODO more efficient vectorised calculation here
        for i in range(self.dbg.num_vertices()):
            # get corresponding k-1mer and loop through the k+1-mers to get the paths through the node
            node = self.dbg.vp.nodes[i]
            counts = count_paths(node=node, bloom_paths=self.bloom.bloomf_p1, t_solid=self.bloom.t)
            # actual calculation
            node_score = node_score(counts=counts, prior=prior)
            self.dbg.vp.scores[i] = node_score

        # all nodes with out-degree of 1 have adjusted scores to make sense in combo
        # with the absorbers
        culdesac = np.nonzero(self.dbg.get_out_degrees(self.dbg.get_vertices()) == 1)[0]
        self.dbg.vp.scores.a[culdesac] = max(self.dbg.vp.scores.a) / 4

        # return array of scores (pointer to vertex property)
        scores = self.dbg.vp.scores.a
        return scores


    # # TODO not very useful in the case of reconstructing it from the bloom
    # def init_prior(self, prior):
    #     # save the prior as graph property, for access
    #     gprior = self.dbg.new_graph_property("float")
    #     # internalise property
    #     self.dbg.gp["prior"] = gprior
    #     # set value for internal property
    #     self.dbg.gp.prior = prior

    # def init_scores(self):
    #     # initialise a score vector for all nodes
    #     scores = self.dbg.new_vertex_property("float")
    #     self.dbg.vp["scores"] = scores
    #     # fill the scores with pseudocounts
    #     scores.a = self.dbg.gp.prior
    #     # a new read can only do one thing here, which is increase one of the priors by one
    #     potential_counts = copy(scores.a)
    #     potential_counts[0] += 1
    #     # calculate KL divergence once and fill the array with that
    #     s = kl_diri(a=potential_counts, b=scores.a)
    #     scores.a.fill(s)


    def init_utility(self):
        # initialise 2 utility vectors for all edges and their reciprocal
        util_zig = self.dbg.new_edge_property("float")
        util_zag = self.dbg.new_edge_property("float")
        util_zig.a.fill(0)  # gets overwritten when updating anyways
        util_zag.a.fill(0)  # gets overwritten when updating anyways
        # internalise
        self.dbg.ep["util_zig"] = util_zig
        self.dbg.ep["util_zag"] = util_zag


    def init_timeCost(self, lam, mu, rho, alpha):
        # init time cost per EDGE
        # for now this is uniform
        # no prior about chromosome ends
        # add Fhat at some point? maybe we don't need that
        timeCost_zig = self.dbg.new_edge_property("float")
        timeCost_zag = self.dbg.new_edge_property("float")
        tc = (lam - mu - rho)
        timeCost_zig.a.fill(tc)
        timeCost_zag.a.fill(tc)
        # internalise
        self.dbg.ep["timeCost_zig"] = timeCost_zig
        self.dbg.ep["timeCost_zag"] = timeCost_zag

        # all edges leading to nodes with out-degree of 1 should have 0 timeCost
        # we always allow expansion of the graph
        # TODO needs to be done according to read length dist
        # TODO also only assign a time cost when a component is bigger than X?
        # culdesac are vertex indices
        culdesac = np.nonzero(self.dbg.get_out_degrees(self.dbg.get_vertices()) == 1)[0]
        # TODO get edge mapping only for the culdesac-edges
        _, edge_targets = self.get_edge_mapping(subset=culdesac)
        edge_targets_zig = edge_targets[:len(edge_targets) // 2]
        edge_targets_zag = edge_targets[len(edge_targets) // 2:]
        self.dbg.ep.timeCost_zig.a[edge_targets_zig[np.isin(edge_targets_zig, culdesac)]] = 1e-300
        self.dbg.ep.timeCost_zag.a[edge_targets_zag[np.isin(edge_targets_zag, culdesac)]] = 1e-300

        # add t0 - cost rejecting all
        t0 = self.dbg.new_graph_property("float")
        self.dbg.gp["t0"] = t0
        self.dbg.gp.t0 = (alpha + mu + rho)

    def init_smu(self):
        # initialise an edge property for S_mu, gets updated together with U
        s_mu_zig = self.dbg.new_edge_property("float")
        s_mu_zag = self.dbg.new_edge_property("float")
        self.dbg.ep["s_mu_zig"] = s_mu_zig
        self.dbg.ep["s_mu_zag"] = s_mu_zag
        # fill the scores with pseudocounts
        s_mu_zig.a.fill(0)  # gets updated together with U
        s_mu_zag.a.fill(0)  # gets updated together with U



    def get_edge_mapping(self, subset=None):
        # returns a list of source and target nodes for each edge
        # to use with the "blocks" ordering of the improved hashimoto implementation
        # TODO what does edges() return and why do I have to index into vertex_index again?
        if subset is not None:
            subset = set(subset)
            edge_mapping = np.array([(self.dbg.vertex_index[node1], self.dbg.vertex_index[node2])
                                     for (node1, node2) in self.dbg.edges()
                                     if self.dbg.vertex_index[node1] in subset
                                     or self.dbg.vertex_index[node2] in subset])
            # number of edges from the subset
            nedges = edge_mapping.shape[0]


        else:
            edge_mapping = np.array([(self.dbg.vertex_index[node1], self.dbg.vertex_index[node2])
                                     for (node1, node2) in self.dbg.edges()])
            # number of edges in whole graph
            nedges = self.dbg.num_edges()



        edge_mapping_rev = np.fliplr(edge_mapping)
        # fill a bigger array that lets us index into the source and targets
        # TODO this might get quite big if there are many edges
        edge_indices = np.empty(shape=(nedges * 2, 2), dtype="int")
        edge_indices[0: nedges] = edge_mapping
        edge_indices[nedges:] = edge_mapping_rev
        edge_sources = edge_indices[:, 0]
        edge_targets = edge_indices[:, 1]
        return edge_sources, edge_targets





class SparseGraph:
    def __init__(self, bloom, rld, size=1e5):
        size = int(size)
        # initialise the sparse adjacency matrix of the graph
        self.adjacency = csr_matrix((size, size), dtype=np.uint8)
        # sparse array to save the scores
        # TODO this could be an array with fixed size later, since we know how large it has to be
        self.scores = csr_matrix((size, 1), dtype=np.float64)

        # TODO tmp sanity check
        self.pathsum = csr_matrix((size, 1), dtype=np.float64)
        self.n_paths = csr_matrix((size, 1), dtype=np.float64)



        # initialise the universal hashing
        UniHash = UniversalHashing(N=size)
        self.uhash = UniHash.draw()

        self.bloom = bloom
        self.rld = rld

        # init vars and arrays
        self.prior = 0.01




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

            # km = list(updated_kmers)[4]  # TODO tmp
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

        # calculate and apply scores
        scores = node_score(counts=updated_paths[:, 1:], prior=self.prior)
        self.scores[updated_paths[:, 0]] = scores

        # TODO tmp path sum
        pathsum = np.sum(updated_paths[:, 1:], axis=1)
        self.pathsum[updated_paths[:, 0]] = pathsum
        # TODO tmp
        n_paths = np.count_nonzero(updated_paths[:, 1:], axis=1)
        self.n_paths[updated_paths[:, 0]] = n_paths




    def reduce_matrix(self):
        # TODO this might be useful for hashimoto calc or for checking symmetry
        # because of the hashing strategy, the matrix is bigger than necessary
        # so we reduce it to all existing nodes and remember their indices in the big matrix
        # (for rehashing purposes)
        adjr = csr_matrix(self.adjacency)
        # save the indices of the big matrix that are actually filled

        filled_rows = adjr.getnnz(1) > 0
        filled_cols = adjr.getnnz(0) > 0

        adjr = adjr[filled_rows][:, filled_cols]
        self.adjr = adjr




    def gt_format(self):
        # for testing, transfrom the matrix to an adjacency list so it can be visualised with gt
        # init graph
        gtg = gt.Graph(directed=False)

        # init edge weights
        eprop = gtg.new_edge_property("int")
        eprop.a.fill(0)
        # internalise the property
        gtg.ep["weights"] = eprop

        cx = self.adjacency.tocoo()
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



    def add_absorbers(self):
        # Because ccl can not extend further at dead ends they will have a biased, lowered benefit.
        # Therefore we add some absorbing triangles that will propagate ccl at dead-ends

        # find all dead-ends by checking where rowsums == 1
        culdesac = np.where(self.adjacency.sum(axis=1) == 1)[0]
        culdesac_scores = self.scores[culdesac].toarray().squeeze()
        n_culdesac = len(culdesac)

        # if there are no dead ends. This will probably never happen
        # except if the function is called on a graph that already has absorbers
        if n_culdesac == 0:
            return

        # add absorber for each end
        # this means 2 new vertices and 3 new edges (or 6 for symmetry)
        abs_vertices = set()
        abs_edges = np.zeros(shape=(n_culdesac * 3, 2), dtype="int")

        for i in range(n_culdesac):
            # find available indices for the absorbers
            # brute force search
            c = culdesac[i]

            abs_a = c + 1
            while self.adjacency[abs_a,: ].sum() != 0:
                abs_a += 1

            abs_b= abs_a + 1
            while self.adjacency[abs_b,: ].sum() != 0:
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
        culdesac_scores = np.repeat(culdesac_scores, 2)
        # make indices for the new vertices
        abs_ind = list(abs_vertices)
        scores_absorbers = self.scores.copy()
        scores_absorbers[abs_ind] = culdesac_scores

        # TODO tmp
        # self.n_paths[abs_ind] = 1
        # self.pathsum[abs_ind] = 1

        return adjacency_absorbers, scores_absorbers



    def update_benefit(self, graph, bloom_paths, ccl, mu):
        # calculate both the probability of transitioning to a node and arriving at that node
        adj, scores = self.add_absorbers()

        # TODO calculate hashimoto from the adjacency with absorbers


        # # graph_absorbers, culdesac, absorb_edges = add_absorbers(graph)
        # # original_v = graph.num_vertices()
        # # calc the hashimoto
        # hashimoto = fast_hashimoto(graph=graph_absorbers)
        #
        # # source and targets for edges in the hashimoto ("blocks" ordering)
        # edge_sources, edge_targets = get_edge_mapping(graph_absorbers)
        # edge_scores = edge_target_scores(graph=graph_absorbers, edge_targets=edge_targets)
        #
        # # the edges involved in the absorbers should not be deleted, despite not actually being observed
        # absorber_tri_edges = np.nonzero(np.isin(edge_targets, culdesac))
        #
        # # verify paths across hubs by the (k+1)-mer data observed
        # zero_mask = verify_paths(graph=graph_absorbers, bloom_paths=bloom_paths,
        #                          edge_sources=edge_sources, edge_targets=edge_targets,
        #                          absorber_tri_edges=absorber_tri_edges)
        #
        # # set 0 for the non-observed paths
        # hashimoto[zero_mask] = 0
        #
        # # turn into a probability matrix (rows sum to 1)
        # # this normalizes per row and filters low probabilities
        # hp = probability_mat(mat=hashimoto, edge_weights=graph_absorbers.ep.edge_weights.a)
        # hp_base = deepcopy(hp)  # save a copy for mat mult
        # # hp_dense = densify(hp)
        #
        # # first transition with H^1
        # arrival_scores = hp.multiply(edge_scores)  # * ccl[0]
        # # as_dense = densify(arrival_scores)
        # # In this function we calculate both utility and S_mu at the same time
        # s_mu = csr_matrix(deepcopy(arrival_scores))
        # # for all consecutive steps
        # for i in range(1, len(ccl)):
        #     # increment step of hashimoto_probs (multiplication instead of power for efficiency)
        #     hp = hp @ hp_base
        #
        #     # reduce the density of the probability matrix
        #     # if i % 5 == 0:
        #     hp = filter_low_prob(hp)
        #     # hp_dense = densify(hp)
        #
        #     # multiply by scores and add - this is element-wise per row
        #     transition_score = hp.multiply(edge_scores)
        #     # trans_dense = densify(transition_score)
        #
        #     if i <= mu:
        #         s_mu += transition_score
        #
        #     # element-wise multiplication of coo matrix and float
        #     # trans_intermed_dense = densify(transition_score * ccl[i])
        #     arrival_scores += transition_score * ccl[i]
        #     # as_dense2 = densify(arrival_scores)
        #
        # # row sums are utility for each edge
        # utility = np.squeeze(np.array(arrival_scores.sum(axis=1)))
        # s_mu_vec = np.squeeze(np.array(s_mu.sum(axis=1)))
        # # add back the original score of the starting node
        # utility += edge_scores
        # s_mu_vec += edge_scores
        #
        # # get rid of the utility of edges within the absorbers
        # # using masked array
        # util_mask = np.ma.array(utility, mask=False)
        # s_mu_mask = np.ma.array(s_mu_vec, mask=False)
        # util_mask[absorb_edges] = np.ma.masked
        # s_mu_mask[absorb_edges] = np.ma.masked
        #
        # # assign utility to edge properties
        # graph.ep.util_zig.a = util_mask[~util_mask.mask][:len(util_mask[~util_mask.mask]) // 2]
        # graph.ep.util_zag.a = util_mask[~util_mask.mask][len(util_mask[~util_mask.mask]) // 2:]
        # graph.ep.s_mu_zig.a = s_mu_mask[~s_mu_mask.mask][:len(s_mu_mask[~s_mu_mask.mask]) // 2]
        # graph.ep.s_mu_zag.a = s_mu_mask[~s_mu_mask.mask][len(s_mu_mask[~s_mu_mask.mask]) // 2:]
        # return None


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
        self.updated_kmers, self.bloomf = self.consume_and_return(filter=self.bloomf, reads=reads)

        # bloom filter with (k+1)-mers to find the paths through each node for score calc
        _, self.bloomf_p = self.consume_and_return(filter=self.bloomf_p, reads=reads)






    def legacy_fill(self, reads):
        # TODO this can go soon
        new_kmers, self.bloomf = self.consume_and_return(filter=self.bloomf, reads=reads)
        self.obs_kmers = self.add_novel_kmers(observed=self.obs_kmers, new_kmers=new_kmers)
        self.new_kmers = new_kmers
        print(f"{self.kcount} kmers")

        # bloom filter with (k+1)-mers to find the paths through each node for score calc
        # then add the new kmers to the total observed kmers
        # new_kmers_p1, self.bloomf_p1 = self.consume_and_return(filter=self.bloomf_p1, reads=reads)
        # self.obs_kmers_p1 = self.add_novel_kmers(observed=self.obs_kmers_p1, new_kmers=new_kmers_p1)



    def consume_and_return(self, filter, reads):
        # take a bunch of reads, add them to a bloom filter and return all consumed kmers
        # reverse complement twins are counted in the same bin
        updated_kmers = set()

        # loop through all reads
        for id, seq in reads.items():
            # split the read into its kmers
            kmers = filter.get_kmers(seq)
            # add to the filter
            try:
                filter.consume(seq)

            except ValueError:
                # print("read shorter than k")
                continue
            updated_kmers.update(kmers)

        # return instead of attribute to make it more general
        return updated_kmers, filter






    def add_novel_kmers(self, observed, new_kmers):
        # TODO legacy
        # update the set of observed kmers with new ones only if their reverse comp is not in the set yet
        # set from kmers
        observed_kmers = set(observed.keys())

        for kmer in new_kmers:
            if kmer in observed_kmers:
                continue
            if reverse_complement(kmer) in observed_kmers:
                continue
            inc_counter = self.kcount + 1
            observed[kmer] = inc_counter
            self.kcount = inc_counter
        # return set for generalisation
        return observed


    def get_edge_list(self, from_scratch=True):
        # TODO legacy
        # find the threshold for solid kmers
        threshold = self.solid_kmers()
        print(f"using threshold {threshold}")

        # construct the edge_list from (newly) observed kmers
        present_edges = khmer.SmallCounttable(self.k, self.target_table_size, self.num_tables)
        edge_list = []
        edge_weights = []

        edge_counter = 0
        edge_skipper = 0

        if from_scratch:
            kmers = self.obs_kmers
            print(f"observed {len(self.obs_kmers)} so far")
        else:
            kmers = self.new_kmers
            print(f'adding {len(kmers)} new kmers to graph')

        for km in kmers:
            # km = list(kmers)[1]  # TODO tmp
            count = self.bloomf.get(km)

            # make the reverse comp
            # mk = reverse_complement(km)
            # present = present_edges.get(mk)

            # ignore kmer if edge is already in the new edge list
            # if present:
            #     edge_skipper += 1
            #     continue
            # # also ignore if count is smaller than t
            if count < threshold:
                continue

            # only if the edge is not present yet and counter is over t, generate a new edge

            # construct the vertices
            lmer = km[:-1]
            rmer = km[1:]
            edge_list.append((lmer, rmer))

            # also the reverse edge
            # mk = reverse_complement(km)
            # lmer = mk[:-1]
            # rmer = mk[1:]
            # hash the reverse kmers
            # lmer_hash = str(self.bloomf_m1.hash(lmer))
            # rmer_hash = str(self.bloomf_m1.hash(rmer))

            # edge_list.append((lmer_hash, rmer_hash))

            edge_weights.append(count)
            # edge_weights.append(count)
            # set edge as present
            present_edges.add(km)

            edge_counter += 1

        # how many edes were made
        print(f'made {edge_counter} edges')
        print(f'skipped {edge_skipper} edges')

        self.new_edges = edge_list
        self.new_weights = edge_weights
        # tmp return for speed measures
        nkmer = len(self.obs_kmers)
        return nkmer






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
    def __init__(self, lam, sd, mu, eta=None):
        # keep track of read lengths for the length distribution
        self.read_lengths = np.zeros(shape=int(1e6), dtype='uint16')

        # initialise as truncated normal distribution
        self.mu = mu
        self.lam = lam
        self.sd = sd
        self.prior_ld()

        # calc the complementary cumulative dist of L
        self.comp_cml_dist()

        # TODO get the stepwise approx
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

        # TODO update approx CCL
        # otu.approx_ccl = CCL_ApproxConstant(L=otu.L, eta=eta)


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




# TODO for future expansion. class that ties everything together at the end
# class AeonsRun:
#
#     def __init__(self, args, const):
#         self.args = args
#         self.const = const
#




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





# def plot_benefit(graph, direc):
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
#
#





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


def kl_diri(a, b):
    # TODO check if this checks out - source of the equation etc.
    # Kullback-Leibler divergence of two dirichlet distributions
    # a, b are arrays of parameters of the dirichlets
    a0 = sum(a)
    b0 = sum(b)

    D = sum((a - b) * (digamma(a) - digamma(a0))) +\
        sum(gammaln(b) - gammaln(a)) +\
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

#
#
# def name2vertex(graph, km):
#     # takes a kmer and returns the vertex index
#     # NOT PROPERLY IMPLEMENTED YET - WILL BE NEEDED FOR MAPPING NEW READS
#     node_dict = {graph.vp.nodes[i]: i for i in range(graph.num_vertices())}
#     return node_dict[km]
#
#
# def kmer2edge(kmer, dbg):
#     # takes kmer and returns edge number
#
#     # get the vertex indices of the left and right node
#     lmer = kmer[:-1]
#     rmer = kmer[1:]
#     lnode = name2vertex(dbg, lmer)
#     rnode = name2vertex(dbg, rmer)
#
#     # use the node to get the edge
#     edge = dbg.edge(s=lnode, t=rnode)
#


# function to calculate the potential increase of KL
def node_score(counts, prior):
    # TODO this probably needs some tuning
    # transform to array
    count_arr = np.array(counts, dtype='float')
    # get rid of all 0 counts to avoid redundant calculation
    # TODO probably not a good idea. more low prob categories means larger difference is the scores
    # count_arr = count_arr[count_arr.nonzero()]
    # append a 0 for a potential new link
    # count_arr = np.append(count_arr, 0)
    # add the prior values
    count_arr = np.add(count_arr, prior)
    # calculate potential information increase by increasing every path by 1
    # or establishing a new link and average over all D_KL
    # potential info increase is weighted by already existing links, cf Polya Urn model
    # for vectorisation: broadcast into third dimension
    potential_counts = count_arr[:, :, None] + np.identity(n=count_arr.shape[1])

    # observation probabilities as weights
    p_obs = np.divide(count_arr, np.sum(count_arr, axis=1)[:, None])
    # KL divergence with every potential change after observing new data
    n_pattern = count_arr.shape[0]
    score = np.zeros(n_pattern)

    # TODO some more vectorisation possible here, but not at 10pm
    for p in range(n_pattern):
        for q in range(potential_counts.shape[1]):
            score[p] += p_obs[p, q] * kl_diri(a=potential_counts[p, :, q], b=count_arr[p])
    return score





# pr = 1e-6
# a = np.array([[5, 1, 0, 0], [0, 0, 2, 2], [0,0,0,10]])
# nodeScore(counts=a, prior=pr)

# b = csr_matrix(np.array([2, 2, 0, 0, 0, 0.0, 0, 0, 0, 0, 0]))
# nodeScore(b, prior=pr)
#
# #
# dirichlet(a).entropy()
# dirichlet(b).entropy()
# kl_diri(a, b)



#%%

def iterate_sparse(x):
    cx = x.tocoo()
    for i, j, v in zip(cx.row, cx.col, cx.data):
        print(i, j, v)
        if i > 1000000:
            break





#
#
# def fq2readpool(fq):
#     # init container for lengths and seqs
#     read_lengths = {}
#     read_sequences = {}
#
#     # fill the dicts read_lengths and read_sequences
#     _fq = Path(fq).expanduser()
#     print(f"Processing file: {_fq}")
#     # check whether fastq is gzipped - there might be a better way here
#     if _fq.name.endswith(('.gz', '.gzip')):
#         fh = gzip.open(_fq, 'rt')
#     else:
#         fh = open(_fq, 'rt')
#
#     # loop over all reads in the fastq file
#     for desc, name, seq, qual in readfq(fh):
#         bases_in_read = len(seq)
#         read_lengths[str(name)] = bases_in_read
#         read_sequences[str(name)] = seq
#     fh.close()
#
#     return read_lengths, read_sequences
#
#
#
#
#


#
#
#
#
# def probability_mat(mat, edge_weights):
#     # transform a matrix to probabilities with edge weights
#     # multiply by edge weights. They are repeated because hashimoto expands the edge set to 2E
#     # ew = np.repeat(edge_weights, 2) # this version is for "consecutive" ordering of the hashimoto
#     ew = np.concatenate((edge_weights, edge_weights))  # this version is for "blocks" ordering of hashimoto
#     mat_weighted = csr_matrix(mat.multiply(np.array(ew)))
#     # normalise to turn into probabilities
#     mat_prob = normalize_matrix_rowwise(mat=mat_weighted)
#     # filter very low probabilities, e.g. edges escaping absorbers
#     mat_prob = filter_low_prob(prob_mat=mat_prob)
#     return mat_prob
#
#
# def filter_low_prob(prob_mat):
#     # filter very low probabilities by setting them to 0
#     # this prevents the probability matrix from getting denser and denser because
#     # of circular structures and the absorbers
#     # indices of (nonzero < 0.01) relative to all nonzero
#     threshold_mask = np.array(prob_mat[prob_mat.nonzero()] < 0.01)[0]
#     # row and col indices of filtered items
#     threshold_rows = prob_mat.nonzero()[0][threshold_mask]
#     threshold_cols = prob_mat.nonzero()[1][threshold_mask]
#     prob_mat[threshold_rows, threshold_cols] = 0
#     prob_mat.eliminate_zeros()
#     # then normalise again
#     prob_mat = normalize_matrix_rowwise(mat=prob_mat)
#     return prob_mat
#
#
# def normalize_matrix_rowwise(mat):
#     factor = mat.sum(axis=1)
#     nnzeros = np.where(factor > 0)
#     factor[nnzeros] = 1 / factor[nnzeros]
#     factor = np.array(factor)[0]
#
#     if not mat.format == "csr":
#         raise ValueError("csr only")
#     # using csr_scale_rows from scipy.sparse.sparsetools
#     csr_scale_rows(mat.shape[0], mat.shape[1], mat.indptr,
#                    mat.indices, mat.data, factor)
#     return mat
#
#
# def source_and_targets(graph, eso):
#     # get the source and target nodes of the edges in the hashimoto 2|E| edge set
#     # sort the edge index first
#     edges = graph.get_edges()[eso]
#     # generate the reverse edge, just like the hashimoto does
#     edges_rev = np.fliplr(edges)
#     # fill a bigger array that lets us index into the source and targets
#     edge_indices = np.empty(shape=(edges.shape[0] * 2, 2), dtype="int")
#     edge_indices[0::2] = edges
#     edge_indices[1::2] = edges_rev
#     edge_sources = edge_indices[:, 0]
#     edge_targets = edge_indices[:, 1]
#     # index as in hashimoto - just for debugging
#     edge_indices = np.concatenate((edge_indices, np.array([np.arange(0,edges.shape[0]*2)]).T), axis=1)
#     return edge_sources, edge_targets, edge_indices
#
#
#
#
#
# def edge_target_scores(graph, edge_targets):
#     # use the edge targets to generate a score vector
#     edge_target_scores = graph.vp.scores.a[edge_targets]
#     return edge_target_scores
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
#
# def verify_paths(graph, bloom_paths, edge_sources, edge_targets, absorber_tri_edges):
#     # we want to verify which paths actually exist in the data
#     hashimoto = fast_hashimoto(graph=graph)
#     num_edges = int(hashimoto.shape[0] / 2)
#     # hash_py, ordering = fast_hashimoto(graph=graph, ordering="blocks", return_ordering=True)
#     # h_dense = densify(hashimoto)
#     # plot_gt(graph, ecolor=graph.ep.edge_weights)
#
#     # we might be able to use the k+1 mer graph to do this instead
#     # so that we do not have to rebuild the seqs
#     # dbg_pone_adj = gt.spectral.adjacency(dbg_pone).T
#     # dbg_pone_adj2 = dbg_pone_adj @ dbg_pone_adj
#
#     rowsums = np.squeeze(np.array(hashimoto.sum(axis=1)))
#     # find edges which have more than one next possible step
#     multistep = np.where(rowsums > 1)[0]
#     # exception for absorer tri-edges
#     multistep = multistep[~np.isin(multistep, absorber_tri_edges)]
#
#     # if multistep empty: return empty or something, probably not necessary
#     mask = np.zeros(shape=hashimoto.shape, dtype=bool)
#     # masklist = []
#     # edge = multistep[0] # tmp
#     for edge in multistep:
#         # rebuild the edge kmer
#         path_source = edge_sources[edge]
#         path_source_seq = graph.vp.nodes[path_source]
#         path_mid = edge_targets[edge]
#         path_mid_seq = graph.vp.nodes[path_mid]
#         path_source_mid_seq = ''.join(merge(path_source_seq, path_mid_seq))
#         # next possible edges and nodes
#         nextsteps = hashimoto[edge, :].nonzero()[1]
#         path_targets = edge_targets[nextsteps]
#         for t in range(len(path_targets)):
#             # t = 1 # tmp
#             path_target = path_targets[t]
#             edge_target = nextsteps[t]
#             # this is for using the k+1 mer adjacency trick
#             # print(f'{path_source} -> {path_target}: {dbg_pone_adj2[path_source, path_target]}')
#
#             path_target_seq = graph.vp.nodes[path_target]
#             path_seq = ''.join(merge(path_source_mid_seq, path_target_seq))
#             # check if the constructed sequence has been observed
#             observation_count = bloom_paths.get(path_seq)
#
#             # print(f'{path_source} -> {path_target}: {observation_count}')
#             # if not, set the mask at that point to 1
#             if observation_count == 0:
#                 mask[edge, edge_target] = 1
#                 # also add the reciprocal
#                 edge_recip = reciprocal_edge(edge, num_edges)
#                 target_recip = reciprocal_edge(edge_target, num_edges)
#                 mask[target_recip, edge_recip] = 1
#                 # masklist.append((path_source, path_target))
#
#     # print(masklist)
#     # print(np.nonzero(mask))
#     # plot_gt(graph, ecolor=graph.ep.edge_weights)
#     return mask
#
#
# def reciprocal_edge(edge, num_edges):
#     # returns the reciprocal edge under the "blocks" ordering of the hashimoto
#     # i.e. two reciprocal edges have the indices i & i + num_edges
#     if edge < num_edges:
#         return edge + num_edges
#     else:
#         return edge - num_edges
#
#





#%%






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


def plot_s(graph):
    # initialise figure to plot on to
    # _, ax = plt.subplots()



    graph_draw(graph, # mplfig=ax,
               vertex_fill_color=graph.vp.scores, vcmap=cm.coolwarm,  # vertex fill is used to show score/util
               # vertex_halo=True, vertex_halo_color=hcol,
               vertex_text=graph.vp.npaths)#,                         # just indices
               # edge_text=ecol, edge_text_distance=0,                   # show edge weights as text
               # edge_text=graph.edge_index, edge_text_distance=0,     # edge indices
               # edge_color=ecol,#, vertex_size=2)#, ecmap=cm.coolwarm,                  # color edge weights
               # output_size=(3000, 3000), vertex_size=1)
    # return a
    # plt.show(block=True)


#%%




# from Torres 2018
def half_incidence(graph, ordering='blocks', return_ordering=False):
    """Return the 'half-incidence' matrices of the graph.
    If the graph has n nodes and m *undirected* edges, then the
    half-incidence matrices are two matrices, P and Q, with n rows and 2m
    columns.  That is, there is one row for each node, and one column for
    each *directed* edge.  For P, the entry at (n, e) is equal to 1 if node
    n is the source (or tail) of edge e, and 0 otherwise.  For Q, the entry
    at (n, e) is equal to 1 if node n is the target (or head) of edge e,
    and 0 otherwise.
    Params
    ------
    graph (gt graph)
    ordering (str): If 'blocks' (default), the two columns corresponding to
    the i'th edge are placed at i and i+m.  That is, choose an arbitarry
    direction for each edge in the graph.  The first m columns correspond
    to this orientation, while the latter m columns correspond to the
    reversed orientation.  Columns are sorted following graph.edges().  If
    'consecutive', the first two columns correspond to the two orientations
    of the first edge, the third and fourth row are the two orientations of
    the second edge, and so on.  In general, the two columns for the i'th
    edge are placed at 2i and 2i+1.
    return_ordering (bool): if True, return a function that maps an edge id
    to the column placement.  That is, if ordering=='blocks', return the
    function lambda x: (x, m+x), if ordering=='consecutive', return the
    function lambda x: (2*x, 2*x + 1).  If False, return None.
    Returns
    -------
    P (sparse matrix), Q (sparse matrix), ordering (function or None).
    Notes
    -----
    The nodes in graph must be labeled by consecutive integers starting at
    0.  This function always returns three values, regardless of the value
    of return_ordering.
    """
    numnodes = graph.num_vertices()
    numedges = graph.num_edges()

    if ordering == 'blocks':
        src_pairs = lambda i, u, v: [(u, i), (v, numedges + i)]
        tgt_pairs = lambda i, u, v: [(v, i), (u, numedges + i)]
    if ordering == 'consecutive':
        src_pairs = lambda i, u, v: [(u, 2*i), (v, 2*i + 1)]
        tgt_pairs = lambda i, u, v: [(v, 2*i), (u, 2*i + 1)]

    def make_coo(make_pairs):
        """Make a sparse 0-1 matrix.
        The returned matrix has a positive entry at each coordinate pair
        returned by make_pairs, for all (idx, node1, node2) edge triples.
        """
        coords = list(zip(*(pair
                            for idx, (node1, node2) in enumerate(graph.edges())
                            for pair in make_pairs(idx, node1, node2))))
        data = np.ones(2*graph.num_edges())
        return coo_matrix((data, coords), shape=(numnodes, 2*numedges))

    src = make_coo(src_pairs).asformat('csr')
    tgt = make_coo(tgt_pairs).asformat('csr')

    if return_ordering:
        if ordering == 'blocks':
            func = lambda x: (x, numedges + x)
        else:
            func = lambda x: (2*x, 2*x + 1)
        return src, tgt, func
    else:
        return src, tgt

#
# # from Torres 2018
# def fast_hashimoto(graph, ordering='blocks', return_ordering=False):
#     """Make the Hashimoto (aka Non-Backtracking) matrix.
#     Params
#     ------
#     graph (gt graph)
#     ordering (str): Ordering used for edges (see `half_incidence`).
#     return_ordering (bool): If True, return the edge ordering used (see
#     `half_incidence`).  If False, only return the matrix.
#     Returns
#     -------
#     A sparse (csr) matrix.
#     """
#     if return_ordering:
#         sources, targets, ord_func = half_incidence(graph, ordering, return_ordering)
#     else:
#         sources, targets = half_incidence(graph, ordering, return_ordering)
#     temp = np.dot(targets.T, sources).asformat('coo')
#     temp_coords = set(zip(temp.row, temp.col))
#
#     coords = [(r, c) for r, c in temp_coords if (c, r) not in temp_coords]
#     data = np.ones(len(coords))
#     shape = 2*graph.num_edges()
#     hashimoto = coo_matrix((data, list(zip(*coords))), shape=(shape, shape))
#
#     if return_ordering:
#         return hashimoto.asformat('csr'), ord_func
#     else:
#         return hashimoto.asformat('csr')




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
# def find_strat_aeons(graph):
#
#     # utility_cont is the expected benefit from keeping reading a fragment
#     utility_cont = np.array(graph.vp.util.a - graph.vp.s_mu.a)
#     # timeCost is the expected cost increase by keeping reading a fragment
#     timeCost = np.array(graph.vp.timeCost.a)
#
#
#     # vector of ratios
#     uot = utility_cont / timeCost
#     # if time cost is negative (or almost zero), always accept.
#     # Does not do anything now, maybe adjust graph ends with this
#     uot = np.where(timeCost <= 0.000000000001, np.inf, uot)
#     # argsort the u/t array
#     # "I" contains the list of genome positions from the most valuable for the strategy to the least valuable.
#     forwarded_i = np.argsort(uot)[::-1]
#
#     # average benefit of strategy, initialized to the case that all fragments are rejected
#     # (like U^0 at the end of Supplementary Section 2).
#     # ubar0 = np.sum(Fhat[:, 0] * S_mu[:, 0]) + np.sum(Fhat[:, 1] * S_mu[:, 1])
#     ubar0 = np.average(graph.vp.s_mu.a)
#
#     # NICOLA: this is more operations than the old version,
#     # but by using numpy functions it should be almost always faster.
#     # Find the best strategy (or equivalently, the number of positions stratSize to accept).
#     # here we start with a strategy that rejects all fragments and increase its size one by one.
#     # in some scenario it might be faster to start from a strategy that
#     # accepts everything and then remove fragments one by one.
#     ordered_benefit = utility_cont[forwarded_i]
#     cumsum_u = np.cumsum(ordered_benefit) + ubar0
#     cumsum_t = np.cumsum(timeCost[forwarded_i]) + graph.gp.t0
#     # stratSize is the total number of accepted positions (the number of 1's) of the current strategy
#     strat_size = np.argmax(cumsum_u / cumsum_t) + 1
#     # print(f'stratsize: {strat_size}')
#     # threshold = ordered_benefit[strat_size-1]
#
#     # put strat in an array
#     strategy = np.zeros(graph.num_vertices(), dtype=bool)
#     np.put(strategy, forwarded_i[:strat_size], True)
#     # make a property map
#     strat = graph.new_vertex_property("float")
#     graph.vp["strat"] = strat
#     graph.vp.strat.a = strategy
#
#     print(f'Accepting nodes: {np.sum(strategy)}, {np.sum(strategy) / graph.num_vertices()}')
#     return strategy
#
#
#
#
# # FUNCTIONS FOR READING AND MAPPING
# # we need functions to write the current graph to GFA format
# # and to add the updates to the graph to the GFA (if de novo writing takes too long)
# def toGFA(graph, k, file):
#     # write the graph to GFA format
#     overlap = k - 2
#
#     with open(file, "w") as gfa:
#         gfa.write(f'H\tVN:Z:1.0\n')
#
#         # write nodes
#         for name, seq in enumerate(graph.vp.nodes):
#             gfa.write(f'S\t{name}\t{seq}\n')
#
#         # write edges
#         for source, target in graph.get_edges():
#             gfa.write(f'L\t{source}\t+\t{target}\t+\t{overlap}M\n')
#
#     return None
#
#
# # function to discover all sequence files in a directory
# def discover_seq_files(directory):
#     # return list of files
#     file_extensions = [".fq.gz", ".fastq", ".fq", ".fastq.gz"]
#     direc = Path(directory).expanduser()
#
#     if direc.is_dir():
#         file_list = [x for x in direc.iterdir() if "".join(x.suffixes).lower() in
#                       file_extensions and 'trunc' not in x.name]
#         return file_list
#     else:
#         return []
#
#
# def execute(command, cmd_out=subprocess.PIPE, cmd_err=subprocess.PIPE, cmd_in=subprocess.PIPE):
#     args = shlex.split(command)
#     running = subprocess.Popen(args, stdout=cmd_out, stderr=cmd_err, stdin=cmd_in,
#                                encoding='utf-8')
#     stdout, stderr = running.communicate()
#     return stdout, stderr
#
#
# def truncate_fq(fq, mu):
#     # truncates the seqs in a fastq file to some length
#     # using awk and subprocess
#     fq_base = Path(fq.stem).with_suffix('')
#     fq_trunc = str(fq_base) + "_trunc" + ''.join(fq.suffixes)
#     fq_trunc_name = fq.with_name(fq_trunc)
#
#     # case of gzipped file - gunzip | awk | gzip
#     if '.gz' in fq.suffixes:
#         fq_trunc_file = gzip.open(fq_trunc_name, 'w')
#         uncompress = subprocess.Popen(['gunzip', '-c', fq], stdout=subprocess.PIPE)
#         awk_cmd = f"awk '{{if ( NR%2 == 0 ) print substr($0,1,{mu}); else print}}'"
#         truncate = subprocess.Popen(shlex.split(awk_cmd), stdin=uncompress.stdout, stdout=subprocess.PIPE)
#         stdout, stderr = execute("gzip -", cmd_in=truncate.stdout, cmd_out=fq_trunc_file)
#
#     # uncompressed case - awk
#     else:
#         fq_trunc_file = open(fq_trunc_name, 'w')
#         cmd = f"awk '{{if ( NR%2 == 0 ) print substr($0,1,{mu}); else print}}' {fq}"
#         stdout, stderr = execute(command=cmd, cmd_out=fq_trunc_file)
#
#     fq_trunc_file.close()
#     return fq, fq_trunc_name
#
#
#
#
#
# # read the next batch of reads from a fastq file
# def map2graph(fastq_dir, processed_files, mu, exe_path, ref_path):
#     # get list of all sequence files
#     file_list = discover_seq_files(directory=fastq_dir)
#
#     # loop over fastqs; select next one
#     for _fq in file_list:
#         if _fq not in processed_files:
#             print(f"Processing file: {_fq}")
#
#             # first we truncate the file, deals with compressed and uncompressed
#             full_file, trunc_file = truncate_fq(fq=_fq, mu=mu)
#             full_gaf = full_file.with_suffix('.gaf')
#             trunc_gaf = trunc_file.with_suffix('.gaf')
#
#             # map the full file TODO tune parameters for mapping
#             graphaligner_full = f"{exe_path} -g {ref_path} -a {full_gaf} -f {full_file}" \
#                                 f" -x dbg --seeds-minimizer-length 7 --seeds-minimizer-windowsize 11"
#
#             # map the truncated file
#             graphaligner_trunc = f"{exe_path} -g {ref_path} -a {trunc_gaf} -f {trunc_file}" \
#                                  f" -x dbg --seeds-minimizer-length 7 --seeds-minimizer-windowsize 11"
#
#             stdout_full, stderr_full = execute(command=graphaligner_full)
#             stdout_trunc, stderr_trunc = execute(command=graphaligner_trunc)
#
#             # add the fastq file to the processed ones
#             processed_files.append(_fq)
#
#             # return the file descriptors for gaf files and the stdout of the two runs
#             return full_gaf, trunc_gaf, stdout_full, stdout_trunc, processed_files
#     # return (None, None, None) # if file list is empty, return Nones
#
#
#
# def parse_GA_stdout(GAout):
#     # this function parses the stdout from GraphAligner runs
#     # the stdout contains some useful info: total bases, mapped reads
#
#     # total amount of bases
#     basesTOTAL_str = [x for x in GAout.split('\n') if x.startswith("Input reads:")][0].split(" ")[-1]
#     basesTOTAL = int(''.join(c for c in basesTOTAL_str if c.isdigit()))
#
#     # number of reads
#     n_reads_str = [x for x in GAout.split('\n') if x.startswith("Input reads:")][0].split(" ")[-2]
#     n_reads = int(''.join(c for c in n_reads_str if c.isdigit()))
#
#     # number of mapped reads
#     n_mapped = [x for x in GAout.split('\n') if x.startswith("Reads with an")][0].split(" ")[-1]
#     n_mapped = int(''.join(c for c in n_mapped if c.isdigit()))
#
#     # unmapped reads
#     n_unmapped = n_reads - n_mapped
#
#     return basesTOTAL, n_reads, n_mapped, n_unmapped
#
#
# def _conv_type(s, func):
#     # Generic converter, to change strings to other types
#     try:
#         return func(s)
#     except ValueError:
#         return s
#
#
# def parseGAF(gaf_file):
#     # parse the raw gaf from GraphAligner
#     # generator that yields named tuple
#     fields = [
#         "qname",
#         "qlen",
#         "qstart",
#         "qend",
#         "strand",
#         "path",
#         "plen",
#         "pstart",
#         "pend",
#         "n_matches",
#         "alignment_block_length",
#         "mapping_quality",
#         "tags",
#     ]
#
#
#     for record in gaf_file:
#         record = record.strip().split("\t")
#         record_dict = {fields[x]:_conv_type(record[x], int) for x in range(12)}
#         yield record_dict
#
#
# def make_decision(gaf, strat, read_seqs, mu):
#     # decide accept/reject for each read
#     add_seqs = set()
#
#     # loop over gaf entries with generator function
#     gaf_file = open(gaf, 'r')
#     # with open(gaf, "r") as gaf_file:
#     for record in parseGAF(gaf_file):
#         # records = list(parseGAF(gaf_file))
#         # print(record)
#
#         # filter by mapping quality
#         if record["mapping_quality"] < 55:
#             continue
#
#         # decision process
#         strand = 0 if record['strand'] == '+' else 1
#         node = _conv_type([x for x in record['path'].split('>') if x != ''][0], int)
#         # ACCEPT
#         if strat[strand][node]:
#             record_seq = read_seqs[record["qname"]]
#         # REJECT
#         else:
#             record_seq = read_seqs[record["qname"]][:mu]  # TODO truncating again here after awk, efficiency loss
#
#         add_seqs.add(record_seq)
#
#         # this snippet extracts the alignment path without considering the exact sequence
#         # could be used if the graph should not be extended, but simply edge weights are added
#         # mapping = [_conv_type(x, int) for x in record['path'].split('>') if x != '']
#         # for edge in zip(mapping, mapping[1:]):
#         #     print(edge)
#
#     gaf_file.close()
#     return add_seqs
#
#


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


