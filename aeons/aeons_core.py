# CUSTOM
from .aeons_utils import execute, find_blocks_generic, random_id, empty_file,\
    init_logger, read_fa, find_exe, write_logs, readfq, MyArgumentParser
from .aeons_sampler import FastqStream, FastqStream_mmap
from .aeons_readlengthdist import ReadlengthDist
from .aeons_polisher import Polisher
from .aeons_merge import ArrayMerger
from .aeons_paf import Paf, PafLine
from .aeons_mapper import LinearMapper
from .aeons_kmer import euclidean_dist, euclidean_threshold

# STANDARD LIBRARY
import os
import gzip
import glob
import time
import logging
import re
from sys import exit
from collections import defaultdict
from pathlib import Path
from io import StringIO
from copy import deepcopy

# NON STANDARD LIBRARY
import pandas as pd
import numpy as np
from scipy.special import gammaln, digamma
from scipy.sparse import csr_matrix, coo_matrix, triu, lil_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import nbinom
import toml
import graph_tool as gt
import gfapy
import networkit as nk
from minknow_api.manager import Manager


# TODO tmp imports
import matplotlib.pyplot as plt
# backend for interactive plots
# plt.switch_backend("GTK3cairo")
# plt.switch_backend("Qt5cairo")
import line_profiler
# import memory_profiler




class Constants:
    def __init__(self):
        self.workers = 4                # so far used in fastqstream read retrieval
        self.mu = 450                   # initial bit of the read for decision making
        self.node_size = 512            # length of a single node in the graph in bases
        self.rho = 300
        self.alpha = 300
        self.window = 1                 # whether strategies are written per base or downsized
        self.wait = 60                  # waiting time in live version
        self.min_len = 1000
        self.contig_lim = 30000         # what to consider as contigs, when writing for mapping against


        # TODO tmp
        # self.redotable = "/hps/software/users/goldman/lukasw/redotable/redotable_v1.1/redotable"
        self.redotable = "/home/lukas/software/redotable/redotable_v1.1/redotable"




    def __repr__(self):
        return str(self.__dict__)


class SparseGraph():

    def __init__(self, args, approx_ccl, node_sources, node_positions):
        self.args = args
        self.approx_ccl = approx_ccl
        self.node_sources = node_sources
        self.node_positions = node_positions




    def load_adjacency(self, gfa):
        # check how large the matrix needs to be
        # by getting the largest node number
        out, _ = execute(f"grep '^S' {gfa} | tail -1")
        n_nodes = int(out.split("\t")[1])
        logging.info(f'loading graph with {n_nodes + 1} nodes')  # +1 because we have node 0
        # calculate a size for the adjacency matrix
        # extra_bit = np.max([n_nodes * 0.2, 100])
        # size = int(n_nodes + extra_bit)  # TODO extra bits for absorbers
        size = int(n_nodes + 1)

        # fill an array with coverage values
        cov = np.zeros(size, dtype='uint64')

        # collect all edges
        edges = []

        with open(gfa, 'r') as gfa_f:
            for line in gfa_f:
                if line.startswith('L'):
                    ll = line.split('\t')
                    start, stop = int(ll[1]), int(ll[3])
                    edges.append((start, stop))

                elif line.startswith('S'):
                    ll = line.split('\t')
                    nid, seq = ll[1], ll[2]
                    try:
                        dp = ll[3]
                        dp = dp.split(':')[-1]
                        cov[int(nid)] = dp
                    # if the gfa does not contain a depth value, assign 1
                    except IndexError:
                        cov[int(nid)] = 1


        # transform to array
        edge_array = np.array(edges)

        # create the sparse matrix
        adj = lil_matrix((size, size), dtype=np.uint16)
        adj[edge_array[:, 0], edge_array[:, 1]] = 1  # edge_array[:, 2]
        adj = adj.tocsr()

        self.n_nodes = n_nodes
        self.adjacency = adj
        self.size = adj.shape[0]
        self.cov = cov
        return n_nodes, adj, cov



    def update_scores(self, max_cov=41):
        # update scores of all nodes
        # calc average coverage at this moment
        cmean = np.mean(self.cov)
        # scores from neg binom
        max_cov_node = max_cov * self.args.node_size
        p = nbinom.pmf(np.arange(max_cov_node), cmean, 0.5)
        scores = np.max(p) - p
        lim = np.ceil(cmean) if cmean else 1
        scores[int(lim):] = 0

        # index into scores to create score vectors
        self.scores = np.zeros(shape=self.cov.shape[0])
        segment_scores = scores[self.cov]
        self.scores = segment_scores




    def get_components(self):
        # check if there are any nodes first
        if not self.n_nodes:
            return

        adjr = SparseGraph._remove_empty_rows(adj=self.adjacency.copy(), max_nodes=self.n_nodes)

        # find and label the connected components using scipy
        # we get the number of components
        # & a label for each of the nodes to which component it belongs
        _, labels = connected_components(adjr, directed=False, connection="weak")
        self.labels = labels
        # filter the components with some minimum size -> label_set
        vals, cnt = np.unique(labels, return_counts=True)

        # get label of biggest comp and the nodes that are part of it
        # label_largest = vals[np.argmax(cnt)]
        # nodes_largest = np.where(labels == label_largest)[0]

        # sort the components sizes - the largest size is used to flip the strategy switch
        label_hist = np.sort(cnt)
        # largest_comp = label_hist[-1]

        # ignore components with one node (should not happen)
        # ncomp = np.where(cnt > 1)[0].shape[0]
        ncomp = vals.shape[0]
        self.ncomp = ncomp
        total_component_size = np.sum(label_hist) * self.args.node_size
        self.total_length = total_component_size
        hist_summary = label_hist[::-1][:10] * self.args.node_size

        # some interesting info
        logging.info(f'num components: {ncomp}')
        logging.info(f'total comp length: {total_component_size}')
        logging.info(f'component histogram: {hist_summary}')
        return labels


    def component_ends_simple(self, lim):
        adjr = SparseGraph._remove_empty_rows(adj=self.adjacency.copy(), max_nodes=self.n_nodes)
        # simple way of finding all component ends: where the sum of the rows is 1
        culdesac = np.where(adjr.getnnz(axis=1) == 1)[0]
        logging.info(f'component ends: {culdesac.shape[0]}')
        # filter the component ends that already have lots of coverage
        culdesac_cov = self.cov[culdesac] < (lim * self.args.node_size)
        culdesac_filt = culdesac[culdesac_cov]
        return set(culdesac_filt)


    def find_lowcov(self, lim):
        # find the low coverage nodes we want to sample more from
        # use the components and labels found earlier
        label_set = list(set(self.labels))

        lowcov = set()
        # loop over components to find low cov and ignore if average high enough
        for lab in range(len(label_set)):
            # component indices
            cind = np.where(self.labels == label_set[lab])[0]
            clc = self.contig_lowcov(cind=cind, lim=lim)
            lowcov.update(clc)

        # transform to array
        lowcov = np.sort(np.array(list(lowcov)))
        # DEPR version not considering contigs
        # lowcov_old = np.where(self.cov < lim * self.args.node_size)[0]
        # print(np.allclose(lowcov, lowcov_old))

        self.lowcov = lowcov
        # here we filter for blocks of low cov sites, to chuck out single sites
        # this uses the difference between adjacent indices of lowcov sites
        # i.e. the difference between the indices of two neighboring sites
        # NOT selecting blocks of coverage 1
        lowcov_diff = np.diff(lowcov)
        lc_blocks = find_blocks_generic(lowcov_diff, 1, 3)
        lc = set()
        for start, end in lc_blocks:
            lc.update(lowcov[start: end + 1])

        logging.info(f'low coverage nodes: {lowcov.shape[0]}, filt: {len(lc)}')
        return lc



    def contig_lowcov(self, cind, lim):
        # check if some thresholds are overcome before declaring lowcov
        ccov = np.copy(self.cov[cind])
        # prevent extrem coverage areas to bias the mean
        ccov[np.where(ccov > 100 * self.args.node_size)] = 100 * self.args.node_size
        # mean after bias check
        contig_mean = np.mean(ccov)
        lc = np.where(ccov < lim * self.args.node_size)[0]
        l0 = 5
        l1 = 20

        # MEAN COV < l0
        if (contig_mean < (l0 * self.args.node_size)):
            lc_ind = lc
        # MEAN l0 < COV < l1
        elif (contig_mean > (l0 * self.args.node_size)) and (contig_mean < (l1 * self.args.node_size)):
            # dropout_idx = self.find_dropout(ccov, mod=8 * self.args.node_size)
            # logging.info(f'dropouts: {dropout_idx.shape[0]}')
            lc = set(lc) #- set(dropout_idx)  # TODO no dropout ignoring for now
            # index into original node indices
            lc_ind = np.array(list(lc), dtype='uint64')
        # MEAN COV > l1
        elif contig_mean > (l1 * self.args.node_size):
            lc_ind = np.array(list(set()), dtype='uint64')
        else:
            lc_ind = lc

        clc = cind[lc_ind]
        return clc


    def find_dropout(self, ccov, mod):
        cov_mean = np.mean(ccov)
        # ignore threshold is dependent on mean coverage
        threshold = int(cov_mean / mod) * self.args.node_size
        dropout = np.where(ccov <= threshold)[0]
        return dropout


    # def component_ends_centrality(self, perc_sampling=1, perc_ends=0.1):
    #     # remove empty rows from the adjancecy matrix
    #     adjr = SparseGraph._remove_empty_rows(adj=self.adjacency.copy(), max_nodes=self.n_nodes)
    #     # calculate centrality and return nodes within some quartile of lowest values
    #     lower_q = self.calc_centrality(perc_sampling, perc_ends)
    #     # check which of these is actually a component end
    #     culdesac = lower_q[np.where(adjr[lower_q].getnnz(axis=1) == 1)[0]]
    #
    #     logging.info(f'found {culdesac.shape[0]} component ends')
    #     return culdesac


    # def calc_centrality(self, perc_sampling=1, perc_ends=0.1):
    #     # remove empty rows from the adjancecy matrix
    #     adjr = SparseGraph._remove_empty_rows(adj=self.adjacency.copy(), max_nodes=self.n_nodes)
    #     # use the components and labels found earlier
    #     label_set = list(set(self.labels))
    #     # collect all indices that are within X% closeness of the component end
    #     lower_q = set()
    #
    #     # loop over all components and calculate their centrality
    #     for l_ind in range(len(label_set)):
    #         # first subset the adjacency to the component
    #         comp, nodes = SparseGraph._subset_adj(adjr.copy(), self.labels, label_set, l_ind)
    #         # generate the nk graph
    #         nkg = SparseGraph._sparse2nk(adj=comp)
    #         # approximate closeness - returns the farthest nodes
    #         nodes_q = SparseGraph._approx_closeness(nkg, perc_sampling=perc_sampling, perc_ends=perc_ends)
    #         # index into original node indices
    #         end_nodes = nodes[nodes_q]
    #         lower_q.update(end_nodes)
    #
    #     lower_q = np.array(list(lower_q))
    #     return lower_q


    @staticmethod
    def load_nodes(gfa):
        # load all nodes into a dictionary
        nodes = dict()
        node_sources = dict()
        links = defaultdict(list)
        with open(gfa, 'r') as gfa_file:
            for line in gfa_file:
                if line.startswith('S'):
                    ll = line.split('\t')
                    nid = ll[1]
                    seq = ll[2].strip()
                    contig = ll[3].strip()
                    nodes[nid] = seq
                    node_sources[nid] = contig

                if line.startswith('L'):
                    ll = line.split('\t')
                    # only save the POS links, and check that all are in the same orientation
                    if ll[2] != ll[4]:
                        print("unmatched orientations")
                    if ll[2] != '+' or ll[4] != '+':
                        continue
                    source = ll[1]
                    target = ll[3]
                    links[source].append(target)
        return nodes, node_sources, links






    @staticmethod
    def _remove_empty_rows(adj, max_nodes):
        # remove all empty rows of the graph
        # nnz = np.where(np.diff(adj.indptr) != 0)[0]
        # adj_rm = adj[nnz][:, nnz]
        return adj[: max_nodes + 1, : max_nodes + 1]


    @staticmethod
    def _subset_adj(adj, labels, label_set, lab):
        # select all nodes from one component
        nodes = np.where(labels == label_set[lab])[0]
        # subset the adjacency to the nodes of one component
        comp = adj[nodes][:, nodes]
        return comp, nodes


    @staticmethod
    def _sparse2nk(adj):
        # transform an adjacency matrix to a networkit graph
        # number of rows == number of nodes in graph
        adju = triu(adj)
        num_nodes = adju.shape[0]
        # initialise a graph, directed because adjacency is symmetric
        nkG = nk.graph.Graph(num_nodes, weighted=False, directed=False)
        # there does not seem to be a better way than iterating over all edges
        # hopefully this is not a bottleneck
        for x, y in zip(adju.nonzero()[0], adju.nonzero()[1]):
            nkG.addEdge(x, y)
        return nkG


    @staticmethod
    def _approx_closeness(nkg, perc_sampling=0.02, perc_ends=0.2):
        # approximate the closeness of a networkit graph
        # takes a graph of a connected component
        nnodes = nkg.numberOfNodes()
        n_samples = int(nnodes * perc_sampling)

        closeness = nk.centrality.ApproxCloseness(nkg, nSamples=n_samples)  # , epsilon=1)
        closeness.run()

        # get the nodes with the lowest scores
        quantile = int(nnodes * perc_ends)
        nodes_q = np.array([i for i, j in closeness.ranking()[-quantile:]])
        return nodes_q


    @staticmethod
    def filter_low_prob(prob_mat, threshold=0.01):
        # filter very low probabilities by setting them to 0
        # this prevents the probability matrix from getting denser and denser because
        # of circular structures and the absorbers
        # simply index into the data, where it is smaller than some threshold
        prob_mat.data[np.where(prob_mat.data < threshold)] = 0
        # unset the 0 elements
        prob_mat.eliminate_zeros()
        # normalise the matrix again by the rowsums
        # row_norm.row_norm(prob_mat)
        # prob_mat.eliminate_zeros()
        return prob_mat




    # @profile
    def approach_nodes(self, nodes, ccl):
        # check if it is possible to reach a desired node from any starting point
        # if it is, then the node goes into the accepted set
        acc = set()
        # transform adjacency to hashimoto matrix
        # and return the mapping of edges to their source and target vertices
        h, edge_mapping = fast_hashimoto(self.adjacency)
        # since hashimoto works in edge-space, translate the targets to edge indices
        target_edges = np.where(np.isin(edge_mapping[:, 2], list(nodes)))[0]
        acc.update(list(target_edges))
        # keep a copy of the basic matrix for multiplication
        h0 = deepcopy(h)
        # first transition from base hashimoto
        approachers = SparseGraph.find_approaching_transitions(h, target_edges)
        acc.update(list(approachers))

        # cover X% of ccl
        # this is steps in addition to the first transition
        n_steps = int(ccl[-1] / self.args.node_size)

        for i in range(n_steps):
            # increment transition step
            # (multiplication instead of power for efficiency)
            h = h @ h0
            # find which edges have reached a desired node
            approachers = SparseGraph.find_approaching_transitions(h, target_edges)
            acc.update(list(approachers))

        # transform edge indices to transitions
        target_transitions = edge_mapping[np.array(list(acc), dtype='uint32'), 1:3]
        return target_transitions




    def find_strat(self, desired_nodes):
        # reads from nodes close to desired ones will be accepted
        logging.info("finding new strategy")
        acc = self.approach_nodes(desired_nodes, ccl=self.approx_ccl)

        # dictionary for contig lengths
        contig_lengths = dict()
        for node, cname in self.node_sources.items():
            contig_lengths[cname] = self.node_positions[node][1]

        # create strategy arrays
        cnames = set(list(self.node_sources.values()))
        # create a boolean array for each contig
        carrays = {cname: np.zeros((int(contig_lengths[cname] / self.args.window), 2),
                                   dtype="bool") for cname in cnames}

        # mark the accepted transitions as accepted in the boolean array
        for source, target in acc:
            source_contig = self.node_sources[source]
            target_contig = self.node_sources[target]
            assert source_contig == target_contig

            # if the position within the contig of the target is larger than the source
            # it has to go into the reverse array
            source_pos = self.node_positions[source]
            target_pos = self.node_positions[target]
            orient = 0 if target_pos[0] > source_pos[0] else 1
            all_pos = [item for sublist in [source_pos, target_pos] for item in sublist]
            left_bound = min(all_pos)
            right_bound = max(all_pos)
            # finally set the ranges to accept
            carrays[source_contig][left_bound: right_bound, orient] = 1

        # plt.plot(carrays[list(cnames)[0]])
        # plt.show()

        # write the accepted transitions to a file TODO
        # TODO write out strategies to file for readfish
        # TODO check what we do in BR
        # np.save(f'{self.const.name}.masks', carrays, allow_pickle=True)
        # execute(f"touch {self.const.name}.masks.updated")
        # to load the set of transitions
        # t = np.load(f'{self.const.name}.strat.npy', allow_pickle=True)[()]
        return carrays



    def process_graph(self, gfa):
        # WRAPPER
        # load new graph and get component ends
        self.load_adjacency(gfa=gfa)
        self.get_components()
        # self.graph.update_scores()

        # find component ends
        culdesac = self.component_ends_simple(lim=self.args.lowcov * 5)
        lowcov = self.find_lowcov(lim=self.args.lowcov)
        desired_nodes = culdesac | lowcov
        # self.graph.update_benefit(culdesac=self.culdesac, ccl=self.rl_dist.approx_ccl)

        # get new strategy
        strat = self.find_strat(desired_nodes=desired_nodes)
        return strat


    # def update_benefit(self, culdesac, ccl):
    #     # check if it is possible to reach a culdesac from each node
    #     # if it is, then the node goes into the accepted set
    #     acc_list = []
    #
    #     # transform adjacency to hashimoto matrix
    #     # and return the mapping of edges to their source and target vertices
    #     h, edge_mapping = fast_hashimoto(self.adjacency)
    #
    #     # since hashimoto works in edge-space, translate the culdesac to edge indices
    #     csac_edges = np.where(np.isin(edge_mapping[:, 2], culdesac))[0]
    #     csac_edges_transitions = edge_mapping[csac_edges, 1:3]
    #     acc_list.extend(csac_edges_transitions)
    #
    #     # get the scores for each edge (rather the score of their target node)
    #     # simply index into the score vector from the absorber addition with the edge mapping
    #     edge_scores = self.scores[edge_mapping[:, 2]]
    #
    #     # keep a copy of the basic matrix for multiplication
    #     h0 = deepcopy(h)
    #
    #     # first transition with H^1
    #     # I think arrival scores can be a vector instead of a matrix?
    #     arrival_scores = h0.multiply(edge_scores).tocsr()  # * ccl[0]
    #     arr_scores = np.array(arrival_scores.sum(axis=1))
    #
    #     # # In this function we calculate both utility and S_mu at the same time
    #     # s_mu = deepcopy(arrival_scores).tocsr()
    #     s_mu = 0  # dummy init
    #
    #     # first transition from base hashimoto
    #     approachers = SparseGraph.find_approaching_transitions(h, edge_mapping, csac_edges)
    #     acc_list.extend(approachers)
    #
    #     # cover X% of ccl
    #     # this is steps in addition to the first transition
    #     n_steps = int(ccl[5] / self.args.node_size)
    #     ccl_modifier_steps = [int(i) for i in ccl / self.args.node_size]
    #     ccl_modifiers = np.arange(0.1, 1.1, 0.1)[::-1]
    #     ccl_mod = np.repeat(ccl_modifiers, ccl_modifier_steps)
    #
    #
    #     for i in range(n_steps):
    #         # increment transition step
    #         # (multiplication instead of power for efficiency)
    #         h = h @ h0
    #
    #         # reduce the density of the probability matrix
    #         # h = self.filter_low_prob(h)
    #
    #         # find which edges have reached a culdesac
    #         approachers = SparseGraph.find_approaching_transitions(h, edge_mapping, csac_edges)
    #         acc_list.extend(approachers)
    #
    #         transition_score = h.multiply(edge_scores).tocsr()
    #
    #         # element-wise multiplication of csr matrix and float
    #         tsp = transition_score.multiply(ccl_mod[i]).tocsr()
    #         arr_scores += np.array(tsp.sum(axis=1))
    #         # since we calculate s_mu on the fly, we save it once i reaches mu
    #         if i == int(self.args.mu / self.args.node_size):
    #             s_mu = arr_scores.copy()
    #
    #     # row sums are utility for each edge
    #     # utility = np.squeeze(np.array(arrival_scores.sum(axis=1)))
    #     utility = np.squeeze(arr_scores)
    #     s_mu_vec = np.squeeze(s_mu)
    #     # add back the original score of the starting node
    #     utility += edge_scores
    #     s_mu_vec += edge_scores
    #
    #     # not sure what the proper data structure is for this
    #     # but maybe return the edge mapping & benefit & smu, all with culdesac edges removed
    #     # or a dictionary that maps (source, target) -> benefit
    #     # or just another sparse matrix for each of them
    #     # remove culdesac edges
    #     # culdesac_edges = np.all(np.isin(edge_mapping[:, 1:], culdesac), axis=1)
    #     # em_notri = edge_mapping[~culdesac_edges, :]
    #     # do not save the benefit and s_mu separately, but already subtract smu
    #     # self.benefit[em_notri[:, 1], em_notri[:, 2]] = utility[~culdesac_edges] - s_mu_vec[~culdesac_edges]
    #     # self.benefit.eliminate_zeros()
    #     self.benefit = utility - s_mu_vec
    #
    #     # also save the raw benefit for now - plotting etc
    #     # self.benefit_raw[em_notri[:, 1], em_notri[:, 2]] = utility[~culdesac_edges]
    #     # self.benefit_raw.eliminate_zeros()
    #
    #     # also keep track of the average benefit when all fragments are rejected (Ubar0)
    #     self.ubar0 = np.mean(s_mu_vec)
    #
    #     # stack the collected list of transitions
    #     try:
    #         acc = np.stack(acc_list)
    #     except ValueError:
    #         acc = []
    #     self.acc = acc
    #     return acc



    @staticmethod
    def find_approaching_transitions(h, target_edges):
        # find the rows from which to reach targets
        # this version translates the matrix to coo for easy indexing
        hcoo = h.tocoo()
        indices_reached = np.where(np.isin(hcoo.col, target_edges))[0]
        # find out the original row
        # these are the edges from which another move will end up in a target
        rows_reached = hcoo.row[indices_reached]
        return rows_reached



    def approach_centrality(self):
        # find the nodes in a lower quantile of closeness values
        pass
        # lower_q = self.calc_centrality(perc_sampling=1, perc_ends=0.1)
        # TODO threshold taking into account read length distribution somehow?







class GFAio:


    def __init__(self, gfa):
        self.gfa = gfa


    @staticmethod
    def add_seqs_to_gfa(gfa, seqs):
        # fill a gfa that has * with sequences
        gfa_seq = open(f'{gfa}.tmp', 'w')

        with open(gfa, 'r') as gfaf:
            for line in gfaf:
                if line.startswith('S'):
                    # replace the * with the sequence
                    ll = line.split('\t')
                    rid = ll[1]
                    # grab sequence from input dict
                    try:
                        seq = seqs[rid].seq
                    except KeyError:
                        print("fatal problem with parsing seqs into gfa")
                    # replace * with seq
                    ll[2] = seq
                    # manage LN tag
                    ll[3] = GFAio.assert_ln_tag(ln=ll[3], seq=seq)
                    # assemble output line
                    line_out = '\t'.join(ll)
                    gfa_seq.write(line_out)
                else:
                    gfa_seq.write(line)
        gfa_seq.close()
        execute(f"mv {gfa}.tmp {gfa}")


    @staticmethod
    def assert_ln_tag(ln, seq):
        # make sure LN tag has correct length if sequence changed
        seq_len = len(seq)
        length = int(ln.strip().split(':')[-1])
        if seq_len != length:
            length = seq_len
            ln = f'LN:i:{length}\n'
        return ln


    @staticmethod
    def get_new_paths(gfa):
        # get the new paths and check which reads are involved
        used_rids = set()
        new_paths = dict()
        merged_headers = dict()
        header_dict = dict()  # associate new ids with merged headers for coverage dict
        for header, seq, tags in SequencePool.parse_gfa(gfa):
            if 'or' in tags.keys():  # this is specific to gfapy
                path_parts = tags['or'].split(',')
                path_parts_clean = [p.replace('^', '') for p in path_parts]  # marker for reverse
                used_rids.update(path_parts_clean)

                # new_paths[header] = seq
                # replace header with home-made instead of creating mega-headers
                # keep track of how many reads are in the segment
                num_merged = 0
                header_parts = header.split('_')
                for h in header_parts:
                    if '*' in h:  # already merged seq
                        num_merged += h.count('*')
                    else:  # new reads
                        num_merged += 1

                new_id = random_id() + '*' * num_merged
                new_paths[new_id] = seq
                merged_headers[new_id] = path_parts_clean
                header_dict[header] = new_id
        return new_paths, merged_headers, used_rids, header_dict


    @staticmethod
    def get_linear_sequences(gfa, seq_pool):
        # this uses gfapy to perform a search for paths in the overlaps
        # and to reconstruct the sequences for the paths
        # load the graph with gfapy
        batch_gfa = gfapy.Gfa.from_file(gfa)  # , vlevel=0)
        paths_gfa = batch_gfa.merge_linear_paths(enable_tracking=True)
        gfa_out = f'{gfa}.paths'
        paths_gfa.to_file(gfa_out)

        # merge the arrays of new sequences
        merge_gfa = gfapy.Gfa.from_file(gfa)  # reload to get the paths again
        am = ArrayMerger(gfa=merge_gfa)
        merged_covs = am.merge_linear_arrs(seq_pool=seq_pool)
        # merged_borders = am.merge_linear_arrs(arr_dict=borders)

        # extract the new sequences from gfa
        new_paths, merged_headers, used_rids, header_dict = GFAio.get_new_paths(gfa_out)
        # change the headers of the merged arrays
        new_covs = dict()
        # new_borders = dict()
        headers = list(merged_covs.keys())
        for i in range(len(merged_covs)):
            merged_header = headers[i]
            merged_cov = merged_covs[merged_header]
            # merged_border = merged_borders[merged_header]
            new_id = header_dict[merged_header]
            new_covs[new_id] = merged_cov
            # new_borders[new_id] = merged_border

        # check that the new sequences and coverage arrays are the same length
        assert len(new_paths) == len(new_covs)
        # assert len(new_paths) == len(new_borders)
        for header, seq in new_paths.items():
            assert len(seq) == len(new_covs[header])
            # assert len(seq) == len(new_borders[header])

        logging.info(f"new_paths: {len(new_paths)} used_rids: {len(used_rids)}")
        return new_paths, merged_headers, new_covs, used_rids





class AeonsRun:

    def __init__(self, args):
        # initialise constants and put into arguments
        const = Constants()
        for c, cval in const.__dict__.items():
            setattr(args, c, cval)

        self.args = args
        self.name = args.name
        self.processed_files = {}
        self.read_sources = dict()
        # for keeping track of the sequencing time
        self.time_naive = 0
        self.time_aeons = 0
        # initial strategy is accept
        self.strat = 1
        # for writing reads to file
        self.cache_naive = dict()
        self.cache_aeons = dict()
        # after how much time should the sequenced be written to file
        # dump time is incremented every time a batch is written, which happens once that is overcome
        self.dump_every = args.dumptime
        self.dump_number_naive = 1
        self.dump_number_aeons = 1
        self.dump_time = self.dump_every

        # for plotting afterwards, we keep a list of rows
        self.metrics = defaultdict(list)

        # make sure the run name does not have any spaces
        assert ' ' not in args.name

        args.out_dir = f'./out_{args.name}'
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
            os.mkdir(f'{args.out_dir}/masks')
            os.mkdir(f'{args.out_dir}/fq')
            os.mkdir(f'{args.out_dir}/logs')
            # os.mkdir(f'{args.out_dir}/fq/NV')
            # os.mkdir(f'{args.out_dir}/fq/AE')
            # os.mkdir(f'{args.out_dir}/metrics')

        # for storing the batches of reads for snakesembly
        if not os.path.exists('./00_reads'):
            os.mkdir('./00_reads')
        empty_file(f'00_reads/{self.args.name}_0_naive.fa')
        empty_file(f'00_reads/{self.args.name}_0_aeons.fa')

        # initialise a log file in the output folder
        init_logger(logfile=f'{args.out_dir}/{args.name}.aeons.log')

        logging.info("AEONS")
        for a, aval in self.args.__dict__.items():
            logging.info(f'{a} {aval}')
        logging.info('\n')

        # init fastq stream
        # continous blocks for local (i.e. for reading from usb)
        # seems like the mmap is also faster on the cluster? might depend on exact filesystem?
        # if not args.remote:
        self.stream = FastqStream_mmap(source=self.args.fq, batchsize=self.args.bsize,
                                       maxbatch=self.args.maxb, seed=self.args.seed)
        # else:
        #     self.stream = FastqStream(source=self.args.fq, bsize=self.args.bsize,
        #                               seed=self.args.seed, workers=self.args.workers)

        self.pool = SequencePool(name=args.name, min_len=args.min_len, out_dir=args.out_dir)
        self.ava = SequenceAVA(paf=f'{args.name}.ava', tetra=args.tetra)
        self.rl_dist = ReadlengthDist(mu=args.mu)


        # load some initial batches
        if not args.preload:
            self.load_init_batches(binit=self.args.binit)
        else:
            self.load_init_contigs(preload=self.args.preload, trusted=self.args.hybrid)

        self.batch = self.stream.batch

        # create first asm
        # makes contigs, saves graph, loads adj, finds comp ends
        self.create_init_asm()

        # if this is a live run, initialise the sequencing device output
        if args.live:
            fq, channels = LiveRun.init_live(device=args.device, host=args.host, port=args.port,
                                                    quadrants=args.quadrant, run_name=args.name)
            self.args.fq = fq
            self.channels = channels



    def scan_dir(self):
        # instead of making new masks after each file, make them periodically
        # this method gets triggered after some time has passed
        # it then scans Minknow's output dir for all deposited files
        # and we create a new batch from all NEW files
        patterns = ["*.fq.gz", "*.fastq.gz", "*.fastq.gzip", "*.fq.gzip", "*.fastq", "*.fq"]
        all_fq = set()
        for p in patterns:
            all_fq.update(glob.glob(f'{self.args.fq}/{p}'))  # TODO change to argument - live version

        # which files have we not seen before?
        new_fq = all_fq.difference(self.processed_files)
        logging.info(f"found {len(new_fq)} new fq files: \n {new_fq}")
        # add the new files to the set of processed files
        self.processed_files.update(new_fq)
        return list(new_fq)


    def load_init_batches(self, binit):
        # this is to load several batches from which to make an initial assembly
        for i in range(binit):
            self.stream.read_batch()
            self.pool.ingest(seqs=self.stream.read_sequences)
            # save the source file name for the reads
            for header in self.stream.read_sequences.keys():
                self.read_sources[header] = self.stream.source  # TODO live version


    def load_init_contigs(self, preload, trusted):
        # this is to load already built contigs
        # either finished or in construction
        prebuilt = dict()
        with open(preload, 'r') as contigs:
            for header, seq in read_fa(contigs):
                header = str(header)
                header = header.replace('>', '')
                header = header.replace('_', '-')
                prebuilt[header] = seq

        # save the source file name for the contigs (for polishing)
        for header in prebuilt.keys():
            self.read_sources[header] = preload

        contig_pool = SequencePool(sequences=prebuilt, min_len=self.args.min_len)
        # preload also some coverage, if we trust these contigs already
        # i.e. if we want to focus on component ends, not on covering everything all over again
        if trusted:
            oz = 10000  # allow overlap zone of this size
            for header, seqo in contig_pool.sequences.items():
                seqo.cov[oz: -oz] = self.args.lowcov
            # add to general pool
            self.pool.ingest(seqs=contig_pool)
        else:
            self.pool.ingest(seqs=contig_pool)




    def create_init_asm(self):
        # write out the current readpool & run complete all versus all
        logging.info("running first AVA")
        paf = self.pool.run_ava(sequences=self.pool.seqdict(), fa=self.pool.fa, paf=self.pool.ava)

        # load paf into ava object - includes filtering
        containments, overlaps, internals = self.ava.load_ava(paf=paf)
        self.pool.increment(containment=containments, overlaps=overlaps, internals=internals)
        self.remove_seqs(sequences=containments.keys())

        # after using coverage info, filter sequences before merging
        # here we only remove from AVA, otherwise we can't map against the contigs
        short_seqs = {header for header, seqo in self.pool.sequences.items() if len(seqo.seq) < self.args.min_len_ovl}
        if short_seqs:
            self.ava.remove_from_ava(sequences=short_seqs)

        # single links
        self.ava.single_links(seqpool=self.pool)

        # merge new sequences
        new_pool = self.merge_overlaps(paf=f'{self.ava.paf}.deg', gfa=self.ava.gfa)

        # add new sequences to the dict and to the ava
        new_ava, new_onto_pool = self.pool.add2ava(new_sequences=new_pool)
        self.pool.ingest(new_pool)
        # load new ava files
        cont_new, _, _ = self.ava.load_ava(new_ava)
        cont_onto, _, _ = self.ava.load_ava(new_onto_pool)
        cont = cont_new.keys() | cont_onto.keys()
        self.remove_seqs(sequences=cont)

        # write the current pool to file for mapping against
        contigs = self.pool.declare_contigs(min_len=self.args.contig_lim)
        self.pool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)




    def make_decision_paf(self, paf_out, read_sequences, strat):
        # decide accept/reject for each read
        # first, transform paf output into dictionary
        # filtering here is for alignment_block_length, not sequence length
        # i.e. at least half of the initial mu-sized fragment has to map
        paf_dict = Paf.parse_PAF(StringIO(paf_out), min_len=int(self.args.mu / 2))

        # if nothing mapped, just return. Unmapped = accept
        if len(paf_dict.items()) == 0:
            logging.info("nothing mapped")
            return read_sequences

        reads_decision = dict()

        reject_count = 0
        accept_count = 0
        unmapped_count = 0

        # loop over paf dictionary
        for record_id, record_list in paf_dict.items():
            # record_id, record_list = list(gaf_dict.items())[0]
            rec = record_list[0]   # TODO make better decision here / use best mapping instead of random

            # find the start and end position relative to the whole linearised genome
            if rec.strand == '+':
                rec.c_start = rec.tstart
                rec.c_end = rec.tend - 1
            elif rec.strand == '-':
                rec.c_start = rec.tend - 1
                rec.c_end = rec.tstart
            else:
                continue

            # index into strategy to find the decision
            try:
                decision = strat[str(rec.tname)][rec.c_start // self.args.window][rec.rev]
            except TypeError:
                # if we don't have a strategy yet, it's an integer so except this and accept all
                decision = 1

            # TODO random decision
            # decision = np.random.choice((0, 1))

            # ACCEPT
            if decision:
                record_seq = read_sequences[rec.qname]
                accept_count += 1

            # REJECT
            else:
                record_seq = read_sequences[rec.qname][: self.args.mu]
                reject_count += 1

            # append the read's sequence to a new dictionary of the batch after decision making
            reads_decision[rec.qname] = record_seq

        # all unmapped reads also need to be accepted, i.e. added back into the dict
        mapped_ids = set(reads_decision.keys())

        for read_id, seq in read_sequences.items():
            if read_id in mapped_ids:
                continue
            else:
                reads_decision[read_id] = seq
                unmapped_count += 1

        logging.info(f'decisions - rejecting: {reject_count} accepting: {accept_count} unmapped: {unmapped_count}')
        self.reject_count = reject_count
        self.accept_count = accept_count
        self.unmapped_count = unmapped_count
        return reads_decision





    def update_times(self, read_sequences, reads_decision):
        # increment the timer counts for naive and aeons

        # for naive: take all reads as they come out of the sequencer (memorymap)
        # total bases + (#reads * alpha)
        bases_total = np.sum([len(seq) for seq in read_sequences.values()])
        acquisition = self.args.bsize * self.args.alpha
        self.time_naive += (bases_total + acquisition)
        logging.info(f"time naive: {self.time_naive}")

        # for aeons: bases of the fully sequenced reads (accepted & unmapped) and of the truncated reads
        read_lengths_decision = np.array([len(seq) for seq in reads_decision.values()])
        n_reject = np.sum(np.where(read_lengths_decision == self.args.mu, 1, 0))
        bases_aeons = np.sum(read_lengths_decision)
        acquisition = self.args.bsize * self.args.alpha
        rejection_cost = n_reject * self.args.rho
        self.time_aeons += (bases_aeons + acquisition + rejection_cost)
        logging.info(f"time aeons: {self.time_aeons}")



    def _execute_dump(self, cond, dump_number, cache):
        # write out the next cumulative batch file
        logging.info(f'dump {cond} #{dump_number}. # of reads {len(list(cache.keys()))}')
        filename = f'00_reads/{self.args.name}_{dump_number}_{cond}.fa'
        # copy previous file to make cumulative
        previous_filename = f'00_reads/{self.args.name}_{dump_number - 1}_{cond}.fa'
        try:
            execute(f"cp {previous_filename} {filename}")
        except FileNotFoundError:
            # at the first batch, create empty 0th and copy to 1st
            # to make sure we don't append to the same file multiple times
            # otherwise we have duplicate reads and that causes flye to crash
            empty_file(previous_filename)
            execute(f"cp {previous_filename} {filename}")
        # writing operation
        with open(filename, "a") as f:
            for rid, seq in cache.items():
                r = random_id()
                fa_line = f'>{rid}.{r}\n{seq}\n'
                f.write(fa_line)

        # increment dump counter
        setattr(self, f'dump_number_{cond}', dump_number + 1)
        # reset cache
        setattr(self, f'cache_{cond}', dict())
        # launch snake  - NOT DONE ANYMORE
        # if self.args.snake:
        #     aeons_eval.launch_solo_snake(input=filename, ref=self.args.ref, gsize=self.args.gsize,
        #                                  remote=self.args.remote, meta=self.args.meta)


    def _prep_dump(self, cond):
        # grab the attributes of the condition
        dump_time = self.dump_time
        curr_time = getattr(self, f'time_{cond}')
        dump_number = getattr(self, f'dump_number_{cond}')
        cache = getattr(self, f'cache_{cond}')
        # check if it's time to write out the next file
        if curr_time > (dump_time * dump_number):
            self._execute_dump(cond=cond, dump_number=dump_number, cache=cache)


    def write_batch(self, read_sequences, reads_decision):
        # helper function for both conditions
        def add_to_cache(seqs, cache):
            for rid, seq in seqs.items():
                cache[rid] = seq

        # add the current sequences to the cache
        add_to_cache(seqs=read_sequences, cache=self.cache_naive)
        add_to_cache(seqs=reads_decision, cache=self.cache_aeons)

        # check if time to dump and execute
        self._prep_dump(cond='naive')
        self._prep_dump(cond='aeons')



    def collect_metrics(self):
        # TODO what would actually be useful here?
        # save a few things after each batch
        # add it as row into a list that can be transformed to a pandas frame and saved for plotting

        logging.info("total strat size")
        strat_perc = []
        strat_size = 0
        fwd_acc = 0
        rev_acc = 0
        for cname, carray in self.strat.items():
            csize = carray.shape[0]
            strat_size += csize
            fwd_acc += carray[:, 0].sum()
            rev_acc += carray[:, 1].sum()
            # for logging display
            fwd = round(carray[:, 0].sum() / csize, 2)
            rev = round(carray[:, 1].sum() / csize, 2)
            avg = (fwd + rev) / 2
            strat_perc.append((csize, avg))
        strat_perc = sorted(strat_perc, key=lambda tup: tup[0])
        # logging.info(f'{strat_perc}')

        # add new row to data frame of metrics
        bsize = len(self.stream.read_ids)

        row = {'name': self.args.name,
               'batch': self.batch,
               'ncomp': self.graph.ncomp,
               'total_length': self.graph.total_length,
               'acc_fwd_n': fwd_acc,
               'acc_rev_n': rev_acc,
               'acc_fwd_ratio': fwd_acc / strat_size,
               'acc_rev_ratio': rev_acc / strat_size,
               'n_mapped': self.mapped_ids / bsize,
               'n_unmapped': self.unmapped_ids / bsize,
               'n_reject': self.reject_count / bsize,
               'n_accept': self.accept_count / bsize,
               'n_unmapped_dec': self.unmapped_count / bsize,
               'n_reads': len(self.stream.read_ids),
               'time_aeons': self.time_aeons,
               'time_naive': self.time_naive,
               'pool_size': len(self.pool.sequences.keys()),
               'low_cov_nodes': self.graph.lowcov.shape[0]}



        self.metrics = append_row(self.metrics, row)
        # write to file
        df = pd.DataFrame(self.metrics)
        df_csv = f"{self.args.name}_metrics.csv"
        with open(df_csv, 'w'):
            pass
        df.to_csv(df_csv)



    # def find_strat_thread(self, benefit, ubar0, lam):
    #     '''
    #     Finding approximate decision strategy from the current read benefits
    #
    #     Parameters
    #     ----------
    #     benefit: np.array
    #         read benefit of each position
    #     s_mu: np.array
    #         benefit score of the first mu positions
    #
    #
    #     Returns
    #     -------
    #     strat: np.array
    #         boolean array of decisions for each position
    #     threshold: float
    #         benefit value at which reads are accepted from a position
    #
    #     '''
    #     # take downsampling into consideration
    #     window = self.args.node_size # self.const.window
    #     alpha = self.args.alpha // window
    #     rho = self.args.rho // window
    #     mu = self.args.mu // window
    #     tc = (lam / window) - mu - rho # // window
    #     # group benefit into bins of similar values
    #     # using binary exponent
    #     # benefit_flat = benefit.flatten('F')
    #     # benefit_nz_ind = np.nonzero(benefit_flat)
    #     # benefit_flat_nz = benefit_flat[benefit_nz_ind]
    #     # to make binary exponents work, normalise benefit values
    #     normaliser = np.max(benefit)
    #     benefit_norm = benefit / normaliser
    #     mantissa, benefit_exponents = np.frexp(benefit_norm)
    #     # count how often each exponent is present
    #     # absolute value because counting positive integers is quicker
    #     benefit_exponents_pos = np.abs(benefit_exponents)
    #     # multi-thread counting of exponents
    #     exponent_arrays = np.array_split(benefit_exponents_pos, 12)
    #     with TPE(max_workers=12) as executor:
    #         exponent_counts = executor.map(np.bincount, exponent_arrays)
    #     exponent_counts = list(exponent_counts)
    #     # aggregate results from threads
    #     # target array needs to have largest shape of the thread results
    #     max_exp = np.max([e.shape[0] for e in exponent_counts])
    #     bincounts = np.zeros(shape=max_exp, dtype='int')
    #     # sum up results from individual threads
    #     for exp in exponent_counts:
    #         bincounts[0:exp.shape[0]] += exp
    #
    #     exponents_unique = np.nonzero(bincounts)[0]  # filter empty bins
    #     counts = bincounts[exponents_unique]  # counts of the existing benefit exponents
    #     # perform weighted bincount in multiple threads
    #     # exponent_arrays = np.array_split(benefit_exponents_pos, 12)
    #     # fhat_arrays = np.array_split(fhat.flatten('F')[benefit_nz_ind], 12)
    #     # arguments = zip(exponent_arrays, fhat_arrays)
    #
    #     # with TPexe(max_workers=12) as executor:
    #     #     fgs = executor.map(binc, arguments)
    #     # fgs = list(fgs)
    #     # aggregates results of threads
    #     # max_fg = np.max([f.shape[0] for f in fgs])
    #     # f_grid = np.zeros(shape=max_fg, dtype='float')
    #     # aggregate results
    #     # for fg in fgs:
    #     #     f_grid[0: fg.shape[0]] += fg
    #
    #     # f_grid = f_grid[exponents_unique]  # filter empty bins
    #     # f_grid_mean = f_grid / counts  # mean fhat for exponent bin
    #     # use exponents to rebuild benefit values
    #     benefit_bin = np.power(2.0, -exponents_unique) * normaliser
    #     # average benefit of strategy in the case that all fragments are rejected
    #     # ubar0 = np.sum(s_mu)
    #     tbar0 = alpha + rho + mu
    #     # cumsum of the benefit (bins multiplied by how many sites are in the bin)
    #     # weighted by probability of a read coming from that bin
    #     cs_u = np.cumsum(benefit_bin * counts) + ubar0
    #     cs_t = np.cumsum(tc * counts) + tbar0
    #     peak = cs_u / cs_t
    #     strat_size = np.argmax(peak) + 1
    #     plt.plot(cs_u)
    #     plt.plot(cs_t)
    #     plt.plot(peak)
    #     plt.show()
    #
    #     # calculate threshold exponent and where values are geq
    #     try:
    #         threshold = benefit_bin[strat_size]
    #     except IndexError:
    #         threshold = benefit_bin[-1]
    #
    #     strat = np.where(benefit >= threshold, True, False)
    #     return strat, threshold








    def merge_overlaps(self, paf, gfa):
        self.ava.ava_dict2ava_file(paf_out=paf)
        # transform ava to gfa inorder to load as graph  # TODO could circumvent by transforming myself
        new_ovls = self.ava.aln2gfa(paf_in=paf, gfa_out=gfa)

        if not new_ovls:
            return SequencePool()

        # parse sequences into the gfa
        GFAio.add_seqs_to_gfa(seqs=self.pool.sequences, gfa=gfa)

        # parse the AVA to form new sequences
        new_paths, merged_headers, new_covs, used_rids = GFAio.get_linear_sequences(gfa=gfa, seq_pool=self.pool.sequences)

        # create sequence objects from new sequences
        new_sequences = {}
        for header, seq in new_paths.items():
            cov = new_covs[header]
            merged_atoms = self.pool.get_atoms(headers=merged_headers[header])
            merged_components = self.pool.get_components(headers=merged_headers[header])
            new_seqo = Sequence(header=header, seq=seq, cov=cov,
                                merged_components=merged_components, merged_atoms=merged_atoms)
            new_sequences[header] = new_seqo

        self.remove_seqs(sequences=used_rids)
        new_pool = SequencePool(sequences=new_sequences, min_len=self.args.min_len)
        return new_pool






    def strat_csv(self, strat, node2pos):
        # write the strategy to csv file so it can be displayed in bandage
        csv_file = f'{self.args.name}.strat.csv'
        header = ','.join(["node", "Color"])
        with open(csv_file, 'w') as csv:
            # write the header
            csv.write(f'{header}\n')
            # write the rows
            for cname, carray in strat.items():
                nodes = cname.split("-")
                for n in nodes:
                    n = int(n)
                    df = carray[node2pos[n][0], 0]
                    dr = carray[node2pos[n][0], 1]
                    cf = "darkgreen" if df else "darkred"
                    cr = "darkgreen" if dr else "darkred"
                    csv.write(f'{n}+,{cf}\n')
                    csv.write(f'{n}-,{cr}\n')






    def remove_seqs(self, sequences):
        # wrapper to remove sequences from pool, ava, coverage etc.
        if not sequences:
            return

        self.ava.remove_from_ava(sequences=sequences)
        self.pool.remove_sequences(sequences=sequences)




    def add_new_sequences(self, sequences, increment=True):
        # WRAPPER
        logging.info('')
        logging.info("adding new seqs")
        ava_new, ava_onto_pool = self.pool.add2ava(sequences)
        # ingest the new sequences
        self.pool.ingest(seqs=sequences)
        # load new alignments
        cont_new, ovl_new, int_new = self.ava.load_ava(ava_new)
        if increment:
            self.pool.increment(containment=cont_new, overlaps=ovl_new, internals=int_new)
        cont_onto, ovl_onto, int_onto = self.ava.load_ava(ava_onto_pool)
        if increment:
            self.pool.increment(containment=cont_onto, overlaps=ovl_onto, internals=int_onto)
        cont = cont_new.keys() | cont_onto.keys()
        self.remove_seqs(sequences=cont)
        # TODO here
        # after taking the coverage information, we can remove sequences that are too short
        # to contribute to contigs from the pool. They won't be useful for anything else
        # could also implement this earlier, i.e. in load_ava to prevent putting stuff into ava_dict
        # which we then remove again here anyway. But then load_ava is already complex enough
        short_seqs = {header for header, seqo in sequences.sequences.items() if len(seqo.seq) < self.args.min_len_ovl}
        self.remove_seqs(sequences=short_seqs)




    def overlap_pool(self):
        # WRAPPER
        # run AVA for the pool to find overlaps and remove contained sequences
        logging.info('')
        logging.info("ava pool")
        contigs = self.pool.declare_contigs(min_len=self.args.contig_lim)
        pool_paf = self.pool.run_ava(sequences=contigs.seqdict(), fa=self.pool.fa, paf=self.pool.ava)
        pool_contained, _, _ = self.ava.load_ava(paf=pool_paf)
        self.remove_seqs(sequences=pool_contained.keys())


    def trim_sequences(self):
        # WRAPPER
        # find which reads need trimming for potential big overlaps
        # needs to be after a load_ava where ava get marked with c=6
        # i.e. we only care about trimming stuff already in the pool
        # add trimmed sequences as new entities so that they go through
        # the same simplification as other sequences
        logging.info('')
        trim_dict = self.ava.to_be_trimmed()
        # trim_dict = dict()
        logging.info(f"trimming {len(trim_dict.keys())} seqs")
        trimmed_seqs = self.pool.trim_sequences(trim_dict=trim_dict)
        trim_paf = self.pool.run_ava(sequences=trimmed_seqs,
                                     fa=f'{self.pool.fa}.trim',
                                     paf=f'{self.pool.ava}.trim')
        trim_contained, _, _ = self.ava.load_ava(paf=trim_paf)
        to_remove = self.ava.trim_success(trim_dict=trim_dict, overlaps=self.ava.overlaps)
        # remove original sequences & failed mergers
        self.remove_seqs(sequences=to_remove)



    def process_batch_live(self):
        # this will ultimately be used for the live version
        # instead of having a separate subclass and all that
        logging.info(f"next {self.batch}")
        tic = time.time()

        # find new fastq files
        new_fastq = self.scan_dir()
        if not new_fastq:
            logging.info("no new files, deferring update ")
            return self.args.wait

        # TODO cont
        # update graph
        # get new strat
        # update rld

        toc = time.time()
        passed = toc - tic
        next_update = int(self.args.wait - passed)
        logging.info(f"batch took: {passed}")
        logging.info(f"finished updating masks, waiting for {next_update} ... \n")
        self.batch += 1
        return next_update



    def sim_batch(self):
        # start of batch processing when simulating
        # get new reads - real fastqs in live version
        # logging.info("getting new batch")
        self.stream.read_batch()

        # save the source file name for the reads
        for header in self.stream.read_sequences.keys():
            self.read_sources[header] = self.stream.source   # TODO live version

        # initialise a new LinearMapper for the current contigs
        # logging.info("mapping new batch")
        lm = LinearMapper(ref=self.pool.contig_fa, mu=self.args.mu, default=False)
        paf_trunc = lm.mappy_batch(sequences=self.stream.read_sequences, truncate=True, out=f'{self.args.name}.lm_out.paf')
        # for metrics collection
        self.mapped_ids = lm.mapped_ids
        self.unmapped_ids = lm.unmapped_ids

        # make decisions
        # logging.info("making decisions")
        reads_decision = self.make_decision_paf(paf_out=paf_trunc,
                                                read_sequences=self.stream.read_sequences,
                                                strat=self.strat)
        return reads_decision


    # @profile
    def process_batch(self):
        logging.info(f'\n NEW BATCH #############################  {self.batch}')
        tic = time.time()

        reads_decision = self.sim_batch()
        # update read length dist, time recording
        self.rl_dist.update(read_lengths=self.stream.read_lengths, recalc=True)
        self.update_times(read_sequences=self.stream.read_sequences, reads_decision=reads_decision)
        self.write_batch(read_sequences=self.stream.read_sequences, reads_decision=reads_decision)

        #  -------------------------------- POST DECISIONS
        logging.info("")

        if self.batch % 30 == 0:
            print("breakpoint")


        # load new sequences, incl length filter
        sequences = SequencePool(sequences=reads_decision, min_len=self.args.min_len)
        # add new sequences to AVA
        self.add_new_sequences(sequences=sequences)
        # check for overlaps and containment in pool
        self.overlap_pool()
        # trim sequences that might lead to overlaps
        self.trim_sequences()

        # process current alignments
        self.ava.single_links(seqpool=self.pool)
        new_sequences = self.merge_overlaps(paf=f'{self.ava.paf}.deg', gfa=self.ava.gfa)
        self.add_new_sequences(sequences=new_sequences, increment=False)

        contigs = self.pool.declare_contigs(min_len=self.args.contig_lim)
        if self.args.polish:
            cpolished = self.pool.polish_sequences(contigs=contigs, read_sources=self.read_sources)
            contigs = self.pool.declare_contigs(min_len=self.args.contig_lim)
            self.ava.remove_from_ava(sequences=cpolished)

        # write the current pool to file for mapping against
        self.pool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)

        # transform to generalised gfa file
        node_sources, node_positions = self.pool.contigs2gfa(gfa=self.pool.gfa, contigs=contigs, node_size=self.args.node_size)
        # wrapper for graph processing to get new strategy
        self.graph = SparseGraph(args=self.args, approx_ccl=self.rl_dist.approx_ccl, node_sources=node_sources, node_positions=node_positions)
        self.strat = self.graph.process_graph(gfa=self.pool.gfa)

        # self.strat_csv(self.strat, node2pos)  # this is for bandage viz

        # collect metrics
        self.collect_metrics()

        # if self.batch % 5 == 0:
        #     redotable(fa=self.pool.contig_fa,
        #                           out=f'{self.pool.contig_fa}.{self.batch}.redotable.png',
        #                           ref=self.args.ref,
        #                           prg=self.args.redotable,
        #                           logdir=f'{self.args.out_dir}/logs/redotable')

        self.batch += 1
        logging.info(f"batch took: {time.time() - tic}")


    def __repr__(self):
        return str(self.__dict__)




class LiveRun:

    def __init__(self):
        pass


    @staticmethod
    def init_live(device, host, port, quadrants, run_name):
        # initialise seq device dependent things
        # - find the output path where the fastq files are placed
        # - and where the channels toml is if we need that
        try:
            out_path = LiveRun._grab_output_dir(device=device, host=host, port=port)
            logging.info(f"grabbing Minknow's output path: \n{out_path}\n")
            fastq_dir = f'{out_path}/fastq_pass'
        except:
            logging.info("Minknow's output dir could not be inferred from device name. Exiting.")
            logging.info(f'{device}\n{host}\n{port}')
            # out_path = "/home/lukas/Desktop/BossRuns/playback_target/data/pb01/no_sample/20211021_2209_MS00000_f1_f320fce2"
            # args.fq = out_path
            exit()

        # grab channels of the condition - might be irrelevant if we don't have quadrants
        if quadrants:
            # out_path = "/nfs/research/goldman/lukasw/BR/data/zymo_all_live/20211124_boss_runs_log_live_001/20211124_1236_X2_FAQ09307_bfb985c5"
            channel_path = f'{out_path}/channels.toml'
            logging.info(f'looking for channels specification at : {channel_path}')
            channels_found = False
            while channels_found == False:
                if not os.path.isfile(channel_path):
                    logging.info("channels file does not exist (yet), waiting for 30s")
                    time.sleep(30)
                else:
                    channels = LiveRun._grab_channels(channels_toml=channel_path, run_name=run_name)
                    channels_found = True
            # channels successfully found
            logging.info(f"found channels specification: BOSS uses {len(channels)} channels.")
        else:
            # if we use a whole flowcell, use all channels
            channels = set(np.arange(1, 512 + 1))

        return fastq_dir, channels


    @staticmethod
    def _grab_output_dir(device, host='localhost', port=None):
        # minknow_api.manager supplies Manager (wrapper around MinKNOW's Manager gRPC)
        # Construct a manager using the host + port provided.
        manager = Manager(host=host, port=port, use_tls=False)
        # Find a list of currently available sequencing positions.
        positions = list(manager.flow_cell_positions())
        pos_dict = {pos.name: pos for pos in positions}

        # index into the dict of available devices
        try:
            target_device = pos_dict[device]
        except KeyError:
            logging.info(f"target device {device} not available")
            return None
        # connect to the device and navigate api to get output path
        device_connection = target_device.connect()
        out_path = device_connection.protocol.get_current_protocol_run().output_path
        return out_path


    @staticmethod
    def _grab_channels(channels_toml, run_name):
        # parse the channels TOML file
        toml_dict = toml.load(channels_toml)
        # find the condition that corresponds to BR
        correct_key = ''
        for key in toml_dict["conditions"].keys():
            name = toml_dict["conditions"][key]["name"]
            if name == run_name:
                correct_key = key
                break
        try:
            selected_channels = set(toml_dict["conditions"][correct_key]["channels"])
            logging.info("grabbing channel numbers for BOSS condition")
            return selected_channels
        except UnboundLocalError:
            logging.info("--name in .params not found in channel-specification toml. Exiting")
            exit()










class SequenceAVA:

    def __init__(self, paf, tetra=False):
        self.paf = paf
        self.gfa = f'{paf}.gfa'
        self.ava_dict = defaultdict(lambda: defaultdict(dict))  # 2 levels of defaultdict
        self.records = []
        self.tetra = tetra    # whether to use the tetramer distance to filter overlaps


    def load_ava(self, paf):
        # load all entries from a paf file as PafLines
        # filter the entries while loading
        skip = 0
        self.records = []  # used for trimming
        self.overlaps = {}  # used for trimming, and increments
        internals = {}  # used for trimming, and increments
        containments = defaultdict(list)  # collect, used for coverage incrementing
        ovl = 0

        with open(paf, 'r') as fh:
            for record in fh:
                rec = PafLine(record)

                # if one of the nodes is already contained, skip
                # if rec.qname in is_contained or rec.tname in is_contained:
                #     skip += 1
                #     continue

                # if '6Pi' in rec.qname and 'HAZK' in rec.tname:
                #     print("break")

                # classify the alignment
                rec.classify()

                if rec.c == 0:
                    # short or self-alignment
                    skip += 1
                    continue
                if rec.c == 1:
                    # internal match, use to increment
                    internals[(rec.qname, rec.tname)] = rec
                    continue
                elif rec.c == 2:
                    # first contained
                    containments[rec.qname].append(rec)
                elif rec.c == 3:
                    # second contained
                    containments[rec.tname].append(rec)
                elif rec.c in {4, 5}:
                    # append the alignment to both the query and the target
                    self.ava_dict[rec.qname][rec.qside][(rec.tname, rec.tside)] = rec
                    self.ava_dict[rec.tname][rec.tside][(rec.qname, rec.qside)] = rec
                    ovl += 1
                    self.overlaps[(rec.qname, rec.tname)] = rec
                else:
                    pass

                # keep the current records in a list
                self.records.append(rec)


        logging.info(f"ava load: skip {skip}, cont {len(containments.keys())} ovl: {ovl}")
        return containments, self.overlaps, internals




    def remove_from_ava(self, sequences):
        # remove alignments of reads from ava alignments
        for sid in sequences:
            # collect the reciprocal targets for removal
            for side in ['L', 'R']:
                targets = self.ava_dict[sid][side].keys()
                for tname, tside in targets:
                    self.ava_dict[tname][tside].pop((sid, side), None)

            # unlink the outgoing edges
            self.ava_dict.pop(sid, None)



    def to_be_trimmed(self):
        # after classification, identify which records need to be trimmed
        # and find the coordinates for trimming
        trim = [rec for rec in self.records if rec.c == 6]
        # save the name and coordinates for trimming
        to_trim = {}
        for rec in trim:
            sid, trim_start, trim_stop, other = rec.find_trim_coords()
            if sid == 0:  # don't trim if product would be short
                continue
            to_trim[sid] = (trim_start, trim_stop, other)
        return to_trim


    def trim_success(self, trim_dict, overlaps):
        # after the trimming, check which ones were successful
        # in order to remove the originals and the not successful ones
        success, unsuccess = set(), set()
        trim = set(trim_dict.keys())
        if not trim:
            to_remove = success | unsuccess
            return to_remove
        if not overlaps:
            unsuccess = {f'{t}%' for t in trim}
            to_remove = success | unsuccess
            return to_remove

        ovl_q, ovl_t = zip(*overlaps.keys())
        ovl = set(ovl_q) | set(ovl_t)
        trim_mod = {f'{t}%' for t in trim}
        success_raw = trim_mod & ovl
        unsuccess = trim_mod - success_raw
        # remove the percentage marker for removal
        success = {s[:-1] for s in success_raw}
        # merge the headers for removal
        to_remove = success | unsuccess
        return to_remove




    def single_links0(self):
        # ensure that we only keep one edge if there are multiple
        logging.info("single links")
        occupied = set()
        for node, edge_dict in list(self.ava_dict.items()):
            # loop over both sides of node
            # each side has its own ava_dict containing possible edges
            for side, avas in edge_dict.items():
                # if there are no edges on a side, skip
                if len(avas) == 0:
                    continue
                # if there is exactly 1 link
                elif len(avas) == 1:
                    (tname, tside), rec = next(iter(avas.items()))
                    # if the target is already occupied
                    if (tname, tside) in occupied:
                        self.ava_dict[node][side].pop((tname, tside), None) # eliminate edge
                    # otherwise we keep it
                    else:
                        self.ava_dict[tname][tside] = {}  # eliminiate reciprocal edge
                        occupied.add((tname, tside))
                        occupied.add((node, side))
                # this is if there are more than 1 possible edge from the node
                else:
                    not_occ = {(tname, tside): rec for (tname, tside), rec in avas.items()
                               if (tname, tside) not in occupied}
                    if not not_occ:  # if all targets are occupied
                        self.ava_dict[node][side] = {}
                        continue
                    # characteristic to choose target
                    # metric = [rec.alignment_block_length for rec in not_occ.values()]
                    metric = [rec.qlen + rec.tlen for rec in not_occ.values()]
                    targets = list(not_occ.keys())
                    max_idx = np.argmax(metric)
                    chosen_t, chosen_t_side = targets[max_idx]
                    rec = avas[(chosen_t, chosen_t_side)]
                    # for the chosen one, put into place
                    self.ava_dict[node][side] = {(chosen_t, chosen_t_side): rec}
                    # also eliminate reciprocal edge
                    self.ava_dict[chosen_t][chosen_t_side] = {}
                    # mark both as occupied
                    occupied.add((node, side))
                    occupied.add((chosen_t, chosen_t_side))


    def single_links(self, seqpool):
        # ensure that we only keep one edge if there are multiple
        logging.info("single links")
        occupied = set()
        for node, edge_dict in list(self.ava_dict.items()):
            # loop over both sides of node
            # each side has its own ava_dict containing possible edges
            for side, avas in edge_dict.items():
                # if there are no edges on a side, skip
                if len(avas) == 0:
                    continue
                # otherwise, we have link and need to check which one to keep
                else:
                    # reduce to only non-occupied targets
                    avas = self.check_occupancy(avas=avas, occupied=occupied)
                    # if specified, check other filters
                    # e.g. tetramer freq. dist.
                    if self.tetra:
                        avas = self.check_tetramer_dist(avas=avas, seqpool=seqpool, node=node)
                    # if there are no targets left
                    if not avas:
                        self.ava_dict[node][side] = {}
                        continue
                    # after filtering choose the link we retain
                    # using some characteristic, i.e. largest resulting sequence
                    # other options: alignment_block_length ..
                    metric = [rec.qlen + rec.tlen for rec in avas.values()]
                    targets = list(avas.keys())
                    max_idx = np.argmax(metric)
                    chosen_t, chosen_t_side = targets[max_idx]
                    rec = avas[(chosen_t, chosen_t_side)]
                    # for the chosen one, put into place
                    self.ava_dict[node][side] = {(chosen_t, chosen_t_side): rec}
                    # also eliminate reciprocal edge
                    self.ava_dict[chosen_t][chosen_t_side] = {}
                    # mark both as occupied
                    occupied.add((node, side))
                    occupied.add((chosen_t, chosen_t_side))


    def check_occupancy(self, avas, occupied):
        # check whether targets of some node are already occupied
        not_occ = {(tname, tside): rec for (tname, tside), rec in avas.items()
                   if (tname, tside) not in occupied}
        return not_occ


    def check_tetramer_dist(self, avas, seqpool, node):
        # check whether an edge fulfills some tetramer dist metric
        eligible = {(tname, tside): rec for (tname, tside), rec in avas.items()
                    if seqpool.is_intra(node, tname)}
        return eligible


    def ava_dict2ava_file(self, paf_out):
        # after making single links, write the alignments to file
        written_lines = set()
        with open(paf_out, 'w') as fh:
            # indexing ava_dict returns node, dict for each end
            for node, edge_dict in self.ava_dict.items():
                for side, avas in edge_dict.items():
                    # no overlaps on the end of this node
                    if not avas:
                        continue
                    # grab next target
                    _, rec = next(iter(avas.items()))
                    # exact overlap was already written (reciprocal)
                    if rec.line in written_lines:
                        continue
                    else:
                        fh.write(rec.line)
                        written_lines.add(rec.line)


    def aln2gfa(self, paf_in, gfa_out):
        # transform to GFA file for further processing
        if not os.path.getsize(paf_in):
            logging.info("no overlaps for merging")
            return False

        comm = f"fpa -i {paf_in} -o /dev/null gfa -o {gfa_out}"
        stdout, stderr = execute(comm)
        if stderr:
            logging.info(f"stderr: \n {stderr}")
        return True



class Sequence:

    def __init__(self, header, seq, cov=None, merged_components=None, merged_atoms=None):
        self.header = header
        self.seq = seq

        if cov is None:
            self.cov = np.zeros(shape=len(seq), dtype='uint16')
        else:
            self.cov = cov

        # merged_headers provides the info which reads were used to create a new sequence
        if merged_components is None:
            self.atoms = set()  # includes all sequence reads that are contained in this one
            self.components = set()  # all sequences used to build this one
        else:
            # if merged info are provided
            self.components = set(merged_components)
            self.atoms = set(merged_atoms)
        # threshold for first polish
        self.last_polish = 0
        self.next_polish = 15
        self.polish_step = 5
        # inits
        self.tetramer_zscores = 0
        self.kmers = 0



    def polish_sequence(self, read_sources):
        success = 0
        # dont polish raw reads, only derived sequences
        if '*' not in self.header:
            return success
        # threshold for polishing, do it only at some timepoint
        if np.mean(self.cov) < self.next_polish:
            return success
        # these are all constituent reads of the contig
        if not self.atoms:
            return success
        if not self.cov.shape[0] > 100000:
            return success
        # initiate and run polisher
        seqpol = Polisher(backbone_header=self.header,
                          backbone_seq=self.seq,
                          atoms=self.atoms,
                          read_sources=read_sources)
        polished_seq = seqpol.run_polish()
        success = self.replace_polished_products(polished_seq)
        return success


    def replace_polished_products(self, polished_seq):
        # Check the difference in length between old and new
        orig_len = len(self.seq)
        new_len = len(polished_seq)
        len_diff = abs(orig_len - new_len)
        # dont use new seq if the length changed by a lot
        if len_diff > orig_len * 0.1:
            return 0
        # replace the sequence with polished one
        self.seq = polished_seq
        # adjust the coverage array to new length
        orig_cov = np.copy(self.cov)
        adjusted_arr = orig_cov
        if orig_len == new_len:
            pass
        elif orig_len > new_len:
            mask = np.ones(orig_len, np.bool)
            rem_pos = np.random.choice(orig_len, size=len_diff, replace=False)
            mask[rem_pos] = 0
            adjusted_arr = orig_cov[mask]
        elif orig_len < new_len:
            ins_pos = np.random.randint(low=0, high=orig_len, size=len_diff)
            ins_val = orig_cov[ins_pos]
            adjusted_arr = np.insert(arr=orig_cov, obj=ins_pos, values=ins_val)
        # make sure the new coverage array is the same length as the sequence
        assert adjusted_arr.shape[0] == len(polished_seq)
        # replace the coverage array
        self.cov = adjusted_arr
        # change the polishing threshold
        self.last_polish = self.next_polish
        self.next_polish += self.polish_step
        return 1






class SequencePool:

    def __init__(self, sequences=None, name="dummy", min_len=1000, out_dir="dummy", threads=48):
        # a unified pool for reads and contigs with persistent AVA
        self.min_len = min_len
        self.out_dir = out_dir
        self.sequences = dict()
        self.threads = threads

        if sequences:
            # the given sequences can be raw or Sequence objects
            input_type = type(list(sequences.values())[0])
            if input_type == str:
                self._ingest_dict(seqs=sequences)
            elif input_type == Sequence:
                self.sequences = sequences
            else:
                print("SequencePool input type not supported")

        self.polished = {}

        # filenames
        self.fa = f'{name}.fa'  # fasta of whole pool
        self.contig_fa = f'{name}.contig.fa'  # fasta of long sequences to map against
        self.ava = f'{name}.ava'  # ava in paf
        self.gfa = f'{name}.gfa'

        # executables
        self.exe_mm2 = None




    def seqdict(self):
        # convenience raw sequence dict
        return {header: seqo.seq for header, seqo in self.sequences.items()}


    def ingest(self, seqs):
        # add a pile of sequences to the pool
        # can read from a fastq, ingest a dict from a stream, or add an existing pool
        skipped = 0
        if type(seqs) == str:
            pass  # TODO live
            # skipped = self._ingest_file(seqs=seqs, covs=covs)

        elif type(seqs) == dict:
            skipped = self._ingest_dict(seqs=seqs)
            logging.info(f"ingested: {len(seqs) - skipped} pool size: {len(self.sequences.keys())}")

        elif type(seqs) == SequencePool:
            self._ingest_pool(new_pool=seqs)
            logging.info(f"ingested: {len(seqs.sequences)} pool size: {len(self.sequences.keys())}")

        else:
            logging.info("seqs need to be fq file, dict, or SequencePool")



    # def _ingest_file(self, seqs, covs):
    #     # ingest sequences from a file  # TODO live
    #     skipped = 0
    #     with open(seqs, 'r') as fqf:
    #         for desc, name, seq, _ in readfq(fqf):
    #             if len(seq) > self.args.min_len:
    #                 self.sequences[str(name)] = seq
    #                 self._ingest_coverage(rid=name, seq=seq, covs=covs)
    #             else:
    #                 skipped += 1
    #     return skipped


    def _ingest_dict(self, seqs):
        # ingest a dictionary of raw sequences
        skipped = 0
        for rid, seq in seqs.items():
            if len(seq) > self.min_len:
                # init sequence object without existing arrays
                seqo = Sequence(header=rid, seq=seq)
                self.sequences[rid] = seqo
            else:
                skipped += 1
        return skipped


    def _ingest_pool(self, new_pool):
        # if the new sequences are already in a pool
        # this is the case after merging sequences for example
        for rid, seqo in new_pool.sequences.items():
            if len(seqo.seq) > self.min_len:
                self.sequences[rid] = seqo



    def run_ava(self, sequences, fa, paf, base_level=False):
        # perform AVA
        # minimap2 -x ava-ont -t8 {input} {input} >{output}
        logging.info(f"Running ava for: {len(sequences)} queries")
        # write current pool to file first
        self.write_seq_dict(seq_dict=sequences, file=fa)
        # then perform all vs all
        if not self.exe_mm2:
            self.exe_mm2 = find_exe("minimap2")

        comm = f'{self.exe_mm2} -x ava-ont -t{self.threads} {fa} {fa} >{paf}'
        if base_level:
            comm = f'{self.exe_mm2} -cx ava-ont -t{self.threads} {fa} {fa} >{paf}'
        stdout, stderr = execute(comm)
        write_logs(stdout, stderr, f'{self.out_dir}/logs/ava')
        return paf



    @staticmethod
    def filter_seqs_length(sequence_dict, min_len):
        # UNUSED, now done when creating a sequence pool
        # filter sequences in a dictionary by length
        seq_dict_filt = {sid: seq for sid, seq in sequence_dict.items() if len(seq) > min_len}
        length_diff = len(sequence_dict) - len(seq_dict_filt)
        logging.info(f'length filter: {length_diff}')
        return seq_dict_filt





    def add2ava(self, new_sequences):
        # instead of rerunning complete ava, run ava for new sequences
        # and map new sequences onto existing ones
        # then merge: seqpool ava, new ava, new onto seqpool
        logging.info(f'adding to ava: {len(new_sequences.sequences)}')
        # write current pool to file
        self.write_seq_dict(seq_dict=self.seqdict(), file=self.fa)
        # declare filenames for new reads that will be added
        new_fa = f'{self.fa}.new'
        new_ava = f'{self.ava}.new'
        new_onto_pool = f'{self.fa}.new.onto_pool'
        # write new reads to file
        self.write_seq_dict(seq_dict=new_sequences.seqdict(), file=new_fa)
        # ava of new sequences
        if not self.exe_mm2:
            self.exe_mm2 = find_exe("minimap2")

        comm = f'{self.exe_mm2} -x ava-ont -t{self.threads} {new_fa} {new_fa} >{new_ava}'
        stdout, stderr = execute(comm)
        write_logs(stdout, stderr, f'{self.out_dir}/logs/ava_add')
        # mapping new sequences to previous pool
        comm = f'{self.exe_mm2} -x map-ont -t{self.threads} {self.fa} {new_fa} >{new_onto_pool}'
        stdout, stderr = execute(comm)
        write_logs(stdout, stderr, f'{self.out_dir}/logs/map2pool')
        # return filenames to be ingested as AVA
        return new_ava, new_onto_pool



    def remove_sequences(self, sequences):
        # given some ids, remove them from the readpool
        # e.g. after making new paths, we want to remove the sequences used to construct them
        # as soon as a read is used, it can not contribute to any other alignment
        for sid in sequences:
            self.sequences.pop(sid, None)
            # self.coverages.pop(sid, None)
            # self.borders.pop(sid, None)
            # self.atoms.pop(sid, None)
        logging.info(f'pool size: {len(self.sequences)}')




    def trim_sequences(self, trim_dict):
        # takes a dictionary of [sid: (start:stop, other_name)]
        # to trim down in order to create valid alignments
        trimmed_seqs = {}
        other_seqs = {}
        valid_ids = set()

        for sid, (start, stop, other) in trim_dict.items():
            try:
                nsid = sid + '%'
                trimmed_seqs[nsid] = deepcopy(self.sequences[sid])
                other_seqs[other] = self.sequences[other]
                valid_ids.add(nsid)
            except KeyError:
                logging.info("key for trimming not in sequence pool")
                continue

        for sid, (start, stop, other) in trim_dict.items():
            nsid = sid + '%'
            # skip sequences that were not found in previous step
            # TODO find out why this happens
            if nsid not in valid_ids:
                continue
            seqo = trimmed_seqs[nsid]
            # deselect the trimmed bit with a boolean mask
            mask = np.ones(shape=len(seqo.seq), dtype='bool')
            mask[start: stop] = 0
            seq_arr = np.array(list(seqo.seq))
            seq_arr_trim = seq_arr[mask]
            seq_trim = ''.join(seq_arr_trim)
            # replace attributes of the Sequence Obj
            seqo.seq = seq_trim
            seqo.cov = seqo.cov[mask]
            seqo.header = nsid


        trimmed_pool = SequencePool(sequences=trimmed_seqs)
        other_pool = SequencePool(sequences=other_seqs)
        # ingest pool of trimmed sequences
        self.ingest(seqs=trimmed_pool)

        # combine sequence dicts for mapping
        seq_dict = trimmed_pool.seqdict() | other_pool.seqdict()
        return seq_dict



    def increment(self, containment, overlaps, internals):
        # use the records of containment to increase the coverage counts & borders
        # containment: defaultdict with read_id: list of mappings
        # first check which ava to ignore
        for rid, rec_list in containment.items():
            for rec in rec_list:
                if rid == rec.qname:
                    other = rec.tname
                    other_start = rec.tstart
                    other_end = rec.tend
                    other_seqlen = rec.tlen
                    self_start = rec.qstart
                    self_end = rec.qend
                else:
                    other = rec.qname
                    other_start = rec.qstart
                    other_end = rec.qend
                    other_seqlen = rec.qlen
                    self_start = rec.tstart
                    self_end = rec.tend


                # add as constituent atom
                if '*' not in rid:
                    self.sequences[other].atoms.add(rid)

                # remove if other is also contained
                # if other in cont:
                #     continue
                # else:

                # TODO window: is done in saving to file at the moment, could be done earlier

                # check if we have coverage on the contained one already
                try:
                    self_len = self_end - self_start
                    other_len = other_end - other_start
                    cont_cov = self.sequences[rid].cov[self_start: self_end]
                    cont_cov += 1
                    # cont_borders = self.borders[rid][self_start: self_end]
                    # cont_borders[0] += 1
                    # cont_borders[-1] += 1
                    # make sure the arrays are the same length before adding the coverage
                    if self_len == other_len:
                        pass
                    elif self_len > other_len:
                        cont_cov = cont_cov[: other_len]
                        # cont_borders = cont_borders[: other_len]
                    elif self_len < other_len:
                        cont_cov = np.pad(cont_cov, (0, other_len - self_len), mode='edge')
                        # cont_borders = np.pad(cont_borders,
                        # (0, other_len - self_len), mode='constant', constant_values=0)

                    # account for reverse complement
                    if rec.rev:
                        cont_cov = cont_cov[::-1]
                        # cont_borders = cont_borders[::-1]
                    else:
                        pass

                    self.sequences[other].cov[other_start: other_end] += cont_cov
                    # limit coverage to 100 to prevent mess
                    self.sequences[other].cov[np.where(self.sequences[other].cov > 100)] = 100
                    # self.borders[other][other_start: other_end] += cont_borders
                # if the contained seq has no array for some reason
                except KeyError:
                    try:
                        self.sequences[other].cov[other_start: other_end] += 1
                        # self.borders[other][other_start] += 1
                        # self.borders[other][other_end] += 1
                    except KeyError:
                        other_cov = np.zeros(shape=other_seqlen, dtype='uint16')
                        # other_borders = np.zeros(shape=other_seqlen, dtype='uint16')
                        other_cov[other_start: other_end] += 1
                        # other_borders[other_start] += 1
                        # other_borders[other_end] += 1
                        self.sequences[other].cov = other_cov
                        # self.borders[other] = other_borders

        # also use increments from overlaps between reads
        other_inc = internals | overlaps
        # TODO simple for now, could incorporate some way of merging arrays
        for (query, target), rec in other_inc.items():
            try:
                qcov = np.copy(self.sequences[query].cov)
                # qbor = self.borders[query]
            except KeyError:
                qcov = np.zeros(shape=rec.qlen, dtype='uint16')
                # qbor = np.zeros(shape=rec.qlen, dtype='uint16')
            qcov[rec.qstart: rec.qend] += 1
            qcov[np.where(qcov > 100)] = 100
            self.sequences[query].cov = qcov
            # qbor[rec.qstart] += 1
            # qbor[rec.qend - 1] += 1

            try:
                tcov = np.copy(self.sequences[target].cov)
                # tbor = self.coverages[target]
            except KeyError:
                tcov = np.zeros(shape=rec.tlen, dtype='uint16')
                # tbor = np.zeros(shape=rec.tlen, dtype='uint16')
            tcov[rec.tstart: rec.tend] += 1
            tcov[np.where(tcov > 100)] = 100
            self.sequences[target].cov = tcov
            # tbor[rec.tstart] += 1
            # tbor[rec.tend - 1] += 1

            # rec.plot()





    def declare_contigs(self, min_len):
        # collect a subdict of sequences that are longer than some limit
        contigs = {header: seqo for header, seqo in self.sequences.items() if len(seqo.seq) > min_len}
        contig_pool = SequencePool(sequences=contigs)
        return contig_pool



    def polish_sequences(self, contigs, read_sources):
        # loop over the contig pool and polish them with racon
        polish_count = 0
        new_polished = []
        for contig_header, contig in contigs.sequences.items():
            if not self.time_to_polish(contig):
                continue
            success = contig.polish_sequence(read_sources)
            polish_count += success
            if success:
                self.polished[contig_header.split('*')[0]] = contig.last_polish
                new_polished.append(contig_header)
        logging.info(f"polished: {polish_count}")
        return new_polished


    def time_to_polish(self, contig):
        # check if a component of the contig has already been polished
        components = contig.components
        c_threshold = contig.next_polish
        cmp_polish_times = {self.polished.get(c.split('*')[0], None) for c in components}
        times = {c >= c_threshold for c in cmp_polish_times if c is not None}
        if any(times):
            return False
        else:
            return True


    def get_atoms(self, headers):
        # given a list of headers, get all atomic reads
        atoms = set()
        for h in headers:
            atm = self.sequences[h].atoms
            atoms.update(atm)
        return atoms



    def get_components(self, headers):
        # given a list of headers, get all component reads
        components = set()
        for h in headers:
            cmp = self.sequences[h].components
            components.update(cmp)
            components.add(h)
        return components


    @staticmethod
    def write_seq_dict(seq_dict, file):
        with open(file, 'w') as fasta:
            for sid, seq in seq_dict.items():
                fasta.write(f'>{sid}\n')
                fasta.write(f'{seq}\n')



    @staticmethod
    def contigs2gfa(gfa, contigs, node_size):
        # convert sequences to gfa with chunked up nodes
        # verify files exist and are empty
        empty_file(gfa)

        # init node counter
        node = 0
        n = node_size

        # check if there are any contigs
        if not contigs:
            return

        # record the source and position of the nodes
        node_sources = dict()
        node_positions = dict()

        # loop through contigs
        for header, seqo in contigs.sequences.items():
            # translate each sequence into gfa with fixed node size
            # number of first seg in this longer segment
            node_init = deepcopy(node)
            # write a single sequence to gfa file
            # chunk it up into nodes
            seq = seqo.seq
            cpos = 0
            seq_chunks = [seq[i: i + n] for i in range(0, len(seq), n)]
            # also chunk up the coverage of this segment
            cov = seqo.cov
            cov_chunks = [np.sum(cov[i: i + n]) for i in range(0, len(cov), n)]

            with open(gfa, 'a') as gfa_file:
                for c_idx in range(len(seq_chunks)):
                    gfa_file.write(f'S\t{node}\t{seq_chunks[c_idx]}\tc:i:{cov_chunks[c_idx]}\n')
                    node_sources[node] = header
                    node_positions[node] = (cpos, cpos + n - 1)
                    node += 1
                    cpos += n

                for i in range(node_init, node - 1):
                    gfa_file.write(f'L\t{i}\t+\t{i + 1}\t+\t0M\n')
                    gfa_file.write(f'L\t{i + 1}\t-\t{i}\t-\t0M\n')

        return node_sources, node_positions




    @staticmethod
    def parse_gfa(infile):
        with open(infile, 'r') as gfa_file:
            for line in gfa_file:
                if line.startswith('S'):
                    ll = line.split('\t')
                    header = ll[1]
                    seq = ll[2]
                    tags = SequencePool._parse_tags('\t'.join(ll[3:]))
                    yield header, seq, tags


    @staticmethod
    def _parse_tags(tag_string):
        # parse the tags from a gaf segment
        tags = tag_string.strip().split('\t')
        tag_dict = dict()
        for t in tags:
            tt = t.split(':')
            tag_dict[tt[0]] = tt[-1]
        return tag_dict


    def is_intra(self, seq1, seq2):
        # takes two sequence names and runs euclidean distance of tetramers
        # this is to check whether the sequences would be classified as intraspecific
        s1 = self.sequences[seq1]
        s2 = self.sequences[seq2]
        euc = euclidean_dist(s1, s2)
        return euc <= euclidean_threshold





class FastqFile:

    def __init__(self):
        # this is for the live version when we read actual files instead of
        # getting data from a stream
        pass


    def read_batch(self, fq_files, channels):
        # read sequencing data from all new files
        read_sequences = dict()

        for fq in fq_files:
            rseq = self._read_single_batch(fastq_file=fq, channels=channels)
            read_sequences.update(rseq)

        self.read_sequences = read_sequences
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = {rid: len(seq) for rid, seq in read_sequences.items()}
        self.total_bases = np.sum(list(self.read_lengths.values()))
        logging.info(f'total new reads: {len(read_sequences)}')



    def _read_single_batch(self, fastq_file, channels=None):
        # get the reads from a single fq file
        logging.info(f"reading file: {fastq_file}")
        read_sequences = {}

        # to make sure its a path object, not a string
        if type(fastq_file) is str:
            fastq_file = Path(fastq_file)

        # check whether fastq is gzipped
        if fastq_file.name.endswith(('.gz', '.gzip')):
            fh = gzip.open(fastq_file, 'rt')
        else:
            fh = open(fastq_file, 'rt')

        # loop over all reads in the fastq file
        # if we don't do any filtering
        if not channels:
            for desc, name, seq, qual in readfq(fh):
                read_sequences[str(name)] = seq

        else:
            # if we filter the incoming batch by the channel that the read comes from
            for desc, name, seq, qual in readfq(fh):
                # find the source channel
                try:
                    # regex to get the channel number from the header
                    # \s=whitespace followed by 'ch=' and then any amount of numeric characters
                    curr_channel = re.search("\sch=[0-9]*", desc).group()
                    ch_num = int(curr_channel.split('=')[1])
                except AttributeError:
                    # if the pattern is not in the header, skip the read
                    logging.info("ch= not found in header of fastq read")
                    continue

                if ch_num in channels:
                    # check if the read comes from a channel that is in the set of selected channels
                    read_sequences[str(name)] = seq
        fh.close()

        logging.info(f"processing {len(read_sequences)} reads in this batch")
        return read_sequences





##############################


def setup_parser():
    parser = MyArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--fq', dest='fq', type=str, default=None, help='path to fastq for streaming')
    parser.add_argument('--name', dest='name', type=str, default="test", help='name for sequencing run')
    parser.add_argument('--maxb', dest='maxb', type=int, default=4, help='maximum batches')
    parser.add_argument('--bsize', dest='bsize', type=int, default=10, help='num reads per batch')
    parser.add_argument('--binit', dest='binit', type=int, default=1, help='loading batches')
    parser.add_argument('--lowcov', dest='lowcov', type=int, default=1, help='limit for strategy rejection')
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed for shuffling input')
    parser.add_argument('--dumptime', dest='dumptime', type=int, default=1, help='interval for dumping batches')
    parser.add_argument('--remote', dest='remote', type=int, default=0,
                        help='whether running local or remote, determines fastqstream access')

    parser.add_argument('--preload', dest='preload', type=str, default=None, help='path to fasta for pre-loading sequences')
    parser.add_argument('--hybrid', dest='hybrid', type=int, default=0, help='hybrid assembly, changes loading of prebuilt contigs')
    parser.add_argument('--tetra', dest='tetra', type=int, default=0, help='adds a test for tetramer freq dist before overlapping')
    parser.add_argument('--polish', dest='polish', type=int, default=0, help='whether to run contig polishing (not for scaffold mode)')

    parser.add_argument('--min_len_ovl', dest='min_len_ovl', type=int, default=10000, help='add length filter before overlapping')
    # snakemake pipeline (& redotable)
    parser.add_argument('--ref', dest='ref', type=str, default="", help='reference used in quast evaluation and redotable')
    # parser.add_argument('--snake', dest='snake', type=int, default=0, help='launch snakemake evaluation')
    # parser.add_argument('--gsize', dest='gsize', type=str, default="0", help='genome size estimate for assemblies')
    # parser.add_argument('--meta', dest='meta', type=int, default=0, help='metagenome snakemake pipe')



    # TODO for live version later
    parser.add_argument('--live', dest='live', type=int, default=0, help='flag to trigger live run')
    parser.add_argument('--device', default=None, type=str, help="employed device/sequencing position")
    parser.add_argument('--host', default='localhost', type=str, help="host of sequencing device")
    parser.add_argument('--port', default=None, type=str, help="port of sequencing device")
    parser.add_argument('--quadrants', default=0, type=int, help="assign channels to conditions with channels.toml")
    return parser





def average_chunk(arr, f):
    padding = (0, 0 if arr.size % f == 0 else f - arr.size % f)
    arr_pad = np.pad(arr.astype(float), padding, mode='constant', constant_values=np.NaN)
    arr_red = np.nanmean(arr_pad.reshape(-1, f), axis=1)
    return arr_red





def filter_matrix(mat, filter_indices):
    # remove elements from a sparse matrix by setting them to 0
    # first we use the indices we want to filter for the rows
    for row in filter_indices:
        mat.data[mat.indptr[row]: mat.indptr[row + 1]] = 0
    mat.eliminate_zeros()

    # for the columns we transform to CSC for faster indexing
    mat_csc = mat.tocsc()
    for col in filter_indices:
        mat_csc.data[mat_csc.indptr[col]: mat_csc.indptr[col + 1]] = 0
    mat_csc.eliminate_zeros()
    # finally transform back to csr
    mat = mat_csc.tocsr()
    return mat




def chunks(lst, n):
    """
    Yield successive (n+1)-sized, overlapping chunks from lst.
    explain overlap
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + (n + 1)]


def pickle_mem(inp):
    import pickle
    size_estimate = len(pickle.dumps(inp))
    print(size_estimate / 1e6)


def sparse_mem(m):
    # return the approximate memory size of a sparse matrix
    m = csr_matrix(m)
    mem = m.data.nbytes + m.indices.nbytes + m.indptr.nbytes
    print(f"{mem / 1e6} Mb")


def replace(a1, ind, x):
    a2 = np.copy(a1)
    a2.put(ind, x)
    return(a2)


def unq(a):
    unique_vals, counts = np.unique(a.round(decimals=5), return_counts=True)
    return unique_vals, counts




def approx_ccl(ccl, eta=11):
    apprx_ccl = np.zeros(eta, dtype='int32')

    i = 0
    for part in range(eta - 1):
        prob = 1 - (part + 0.5) / (eta - 1)
        while ccl[i] > prob:
            i += 1
        apprx_ccl[part] = i
    # approx_ccl[0] gives the length that reads reach with probability 1
    # approx_ccl[1] length that reads reach with prob 0.9
    # etc

    partition_lengths = np.diff(np.concatenate((np.zeros(1), apprx_ccl)))[:-1]

    return partition_lengths




def size(graph):
    print(f'v: {graph.num_vertices()}, e: {graph.num_edges()}')
    return graph.num_vertices(), graph.num_edges()


# def is_symm(a, rtol=1e-05, atol=1e-05):
#     return np.allclose(a, a.T, rtol=rtol, atol=atol)
#




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

#
# def nonzero_indices(sparse_mat):
#     return np.where(np.diff(sparse_mat.indptr) != 0)[0]

#
# def iterate_sparse(x):
#     cx = x.tocoo()
#     contents = []
#     for i, j, v in zip(cx.row, cx.col, cx.data):
#         contents.append((i, j, v))
#     return np.array(contents)
#
#
# def allequal(a, b):
#     # compare two sparse matrices
#     content_a = iterate_sparse(a)
#     content_b = iterate_sparse(b)
#
#     r = np.allclose(content_a, content_b)
#     print(r)




def dfs(graph, node):
    visited = []
    queue = [node]

    while len(queue) > 0:
        node = queue.pop()

        if node not in set(visited):
            visited.append(node)

            neighbours = graph[node, :].nonzero()[1]

            for n in neighbours:
                queue.insert(0, n)
    return visited


def find_all_paths(graph, culdesac):
    # save all paths
    # this is done by traversing the graph starting from culdesacs
    paths = []
    for c in culdesac:
        # check if the start is already in a path
        if any([c in p for p in paths]):
            continue

        cpath = path_search(graph=graph, node=c, culdesac=culdesac)
        paths.append(cpath)

    return paths


def path_search(graph, node, culdesac):
    visited = []
    queue = [node]
    init_node = deepcopy(node)
    found_ends = []
    subpaths = []

    cset = set(culdesac)
    cset.remove(node)
    # neighbours = graph[node, :].nonzero()[1]

    # for n in neighbours:
    #     queue.insert(0, n)

    while len(queue) > 0:
        node = queue.pop()

        if node not in set(visited):
            visited.append(node)
            if node in culdesac and node not in found_ends:
                subpaths.append(visited)
                queue = [init_node]
                visited = []
                found_ends.append(node)
                continue

            neighbours = graph[node, :].nonzero()[1]

            for n in neighbours:
                if n not in found_ends:
                    queue.insert(0, n)

    # choose the longest subpath
    path_lengths = [len(p) for p in subpaths]
    max_len = np.argmax(path_lengths)
    path = subpaths[max_len]
    return path





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
        the third and fourth row are the two orientations of the second edge
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
    numedges = int(adj.getnnz())
    numnodes = adj.shape[0]

    # containers that hold the mapping of
    # edgeindex, n1, n2
    # source node, edge index
    # target node, edge index
    edge_mapping = np.zeros(shape=(numedges, 3), dtype="int64")
    src_pairs = np.zeros(shape=(numedges, 2))
    tgt_pairs = np.zeros(shape=(numedges, 2))

    # loop over the edges and get the node indices of them
    for idx, (node1, node2) in enumerate(zip(adj.row, adj.col)):
        edge_mapping[idx, :] = (idx, node1, node2)
        src_pairs[idx] = (node1, idx)
        tgt_pairs[idx] = (node2, idx)

    # construct the sparse incidence matrices
    data = np.ones(numedges)
    src_coo = coo_matrix((data, list((src_pairs[:, 0], src_pairs[:, 1]))), shape=(numnodes, numedges))
    tgt_coo = coo_matrix((data, list((tgt_pairs[:, 0], tgt_pairs[:, 1]))), shape=(numnodes, numedges))
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





def simple_read_batch(fastq_file):
    # fill the dicts read_lengths and read_sequences with all reads
    read_lengths = {}
    read_sequences = {}

    # to make sure its a path object, not a string
    if type(fastq_file) is str:
        fastq_file = Path(fastq_file)

    fh = open(fastq_file, 'rt')

    for desc, name, seq, qual in readfq(fh):
        bases_in_read = len(seq)
        read_lengths[str(name)] = bases_in_read
        read_sequences[str(name)] = seq

    fh.close()

    return read_lengths, read_sequences





def append_row(frame_dict, new_row):
    # this is used by save metrics
    # takes 2 dicts as input: one is a metric frame,
    # the other is a row to append to that
    for colname, rowval in new_row.items():
        frame_dict[colname].append(rowval)
    return frame_dict








class PlaceHolder:



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

        # for i, j, v in zip(cx.row, cx.col, cx.data):
        #     edges.append((str(i), str(j), v))

        # kmer_indices = gtg.add_edge_list(edges, hashed=True, hash_type="string", eprops=[gtg.ep.weights])#, gtg.ep.strat])
        kmer_indices = gtg.add_edge_list(edges, hashed=True, hash_type="string",
                                         eprops=[gtg.ep.weights, gtg.ep.strat])

        ki_list = []
        for ki in kmer_indices:
            ki_list.append(int(ki))
        ki = np.array(ki_list)

        gtg.vp["ind"] = kmer_indices

        # argsort the kmer indices to then associate the properties
        ki_sorting = np.argsort(ki)

        # add scores as property for plotting
        scprop = gtg.new_vertex_property("float")
        # scprop.a[ki_sorting] = self.scores_filt.data
        scprop.a[ki_sorting] = self.scores.data
        gtg.vp["scores"] = scprop

        # property for betweenness
        btwprop = gtg.new_vertex_property("float")
        # btwprop.a[ki_sorting] = self.between_filt.data  # [1:]
        btwprop.a[ki_sorting] = self.between.data#[1:]
        gtg.vp["btw"] = btwprop

        # property for coverage
        covprop = gtg.new_vertex_property("float")
        covprop.a[ki_sorting] = self.coverage.data  # [1:]
        gtg.vp["cov"] = covprop

        self.gtg = gtg
        return gtg


