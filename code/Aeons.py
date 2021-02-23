#! /usr/bin/env python3
# standard library imports
from copy import copy, deepcopy
from pathlib import Path
import numpy as np
import gzip
import subprocess
import shlex
from difflib import SequenceMatcher

# scipy
from scipy.special import gammaln, digamma
# from scipy.stats import dirichlet
from scipy.sparse import csr_matrix, coo_matrix  # diags
# from scipy import sparse
from scipy.sparse.sparsetools import csr_scale_rows
# from itertools import combinations

# from BossRuns import reverse_complement # , CCL_ApproxConstant
from BossRuns import readfq
import simulateReads

import graph_tool as gt
from graph_tool.all import graph_draw
from graph_tool.topology import label_components
import khmer
import matplotlib.pyplot as plt
from matplotlib import cm
# plt.switch_backend("GTK3cairo")
# plt.switch_backend("Qt5cairo")

# import line_profiler
# import sys

# cd "./23_seq2graph"



#%%
def plot_gt(graph, vcolor=None, ecolor=None, hcolor=None, comp=None):
    comp_label, _ = label_components(graph, directed=False)  # can be used for vertex_fill_color
    # print(set(comp_label))


    _, ax = plt.subplots()

    # transform edge weights to float for plotting
    if ecolor is not None:
        ecolor = ecolor.copy(value_type="float")

    vcol = vcolor if vcolor is not None else "grey"
    hcol = hcolor if hcolor is not None else [0, 0, 0, 0, 0]
    ecol = ecolor if ecolor is not None else ""

    # overwrite vcol with components if set
    if comp is not None:
        vcol = comp_label
        print(len(set(comp_label)))

    # if color is not None:
    #     obj_col.a = color
    # else:
    #     obj_col = "grey"

    a = graph_draw(graph, mplfig=ax,
                   vertex_fill_color=vcol, vcmap=cm.coolwarm,        # vertex fill is used to show score/util
                   vertex_halo=True, vertex_halo_color=hcol,
                   vertex_text=graph.vertex_index,                         # just indices
                   edge_text=ecol, edge_text_distance=0,                   # show edge weights as text
                   # edge_text=graph.edge_index, edge_text_distance=0,     # edge indices
                   # edge_color="grey", vertex_size=2)#, ecmap=cm.coolwarm,                  # color edge weights
                   output_size=(3000, 3000), vertex_size=1)
    plt.show(block=a)

#%%
def plot_benefit(graph, direc):
    # separate function to plot the edge-centric benefit
    ecol = graph.ep.util_zig if direc == 0 else graph.ep.util_zag
    e_weights = graph.ep.edge_weights.copy(value_type="float")

    _, ax = plt.subplots()
    a = graph_draw(graph, mplfig=ax,
                   edge_color=ecol, ecmap=cm.coolwarm, edge_pen_width=1.5,
                   vertex_fill_color="grey", vertex_size=0.1,
                   edge_text=e_weights, edge_text_distance=1, edge_font_size=2)  #,
                   # output_size=(3000, 3000),
                   # vertex_text=graph.vertex_index)  # just indices
                   # edge_text=graph.edge_index, edge_text_distance=0,     # edge indices
    plt.show(block=a)




#%%


def size(graph):
    print(f'v: {graph.num_vertices()}, e: {graph.num_edges()}')
    return graph.num_vertices(), graph.num_edges()


def is_symm(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def reverse_complement(dna):
    '''
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

    '''
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



def kmers(read, k=59):
    for i in range(len(read) - k + 1):
        yield read[i: i + k]


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


#%%



def name2vertex(graph, km):
    # takes a kmer and returns the vertex index
    # NOT PROPERLY IMPLEMENTED YET - WILL BE NEEDED FOR MAPPING NEW READS
    node_dict = {graph.vp.nodes[i]: i for i in range(graph.num_vertices())}
    return node_dict[km]


def kmer2edge(kmer, dbg):
    # takes kmer and returns edge number

    # get the vertex indices of the left and right node
    lmer = kmer[:-1]
    rmer = kmer[1:]
    lnode = name2vertex(dbg, lmer)
    rnode = name2vertex(dbg, rmer)

    # use the node to get the edge
    edge = dbg.edge(s=lnode, t=rnode)


#%%

def construct(readpool, k, genome_estimate):
    # second version of constructor that does two passes over all kmers
    # first to just read all kmers, then iterate over bloom to construct with edge weights
    # readpool is a dict of {id:seq}

    # init an empty graph
    dbg = gt.Graph(directed=False)
    # init edge weights
    edge_weights = dbg.new_edge_property("short")
    edge_weights.a.fill(0)
    dbg.ep["edge_weights"] = edge_weights

    # first pass to fill the bloom filter
    read_kmers, bloom_reads = consume_and_return(reads=readpool, k=k)
    # we fill another bloom filter with (k+1)-mers to find the paths through each node for score calc
    read_kmers_plusone, bloom_reads_plusone = consume_and_return(reads=readpool, k=(k + 1))

    # find the threshold for solid kmers
    threshold = solid_kmers(read_kmers=read_kmers, bloom_reads=bloom_reads, genome_estimate=genome_estimate)
    print(f"using threshold {threshold}")

    # construct the graph with k
    # second loop through the kmers
    present_edges = khmer.Nodetable(k, 5e6, 4)
    edge_list = []
    edge_weight_list = []
    read_kmers = set(read_kmers)

    for km in read_kmers:
        # km = read_kmers[140]
        present = present_edges.get(km)
        count = bloom_reads.get(km)

        # ignore kmer if edge is already in the graph
        if present:
            continue
        # also ignore if count is smaller than t
        elif count < threshold:
            continue

        # only if the edge is not present yet and counter is over t, add it
        else:
            # construct the vertices
            lmer = km[:-1]
            rmer = km[1:]
            edge_list.append((lmer, rmer))
            edge_weight_list.append(count)
            # set edge as present
            present_edges.add(km)

    # add all edges in one swoop
    vertex_names = dbg.add_edge_list(edge_list, hashed=True, string_vals=True)
    # set the edge weights
    dbg.ep.edge_weights.a = edge_weight_list

    # internalize the vertex names
    dbg.vp["nodes"] = vertex_names

    # initialise BOSS properties (prior, scores, util, smu, timecost)

    return dbg, read_kmers, bloom_reads, bloom_reads_plusone, threshold


#%%

def update_dbg(dbg, read_kmers, bloom_reads, bloom_reads_plusone, threshold, add_seqs):
    # UPDATE DBG
    # here we update the graph after mapping a batch
    # make_decision returns a set of sequences that the graph will be extended by
    # (either adding counts or adding newly observed kmers - the bloom filter takes care of that)
    # TODO for now this function just constructs a new graph - maybe in the future we want to update the
    # graph instead of creating a new one
    # or actually patch gt out of the code anyway and create the hashimoto etc. from the bloom directly

    for s in add_seqs:
        for kmer in bloom_reads.get_kmers(s):
            # TODO for now we don't care if the kmer has already been counted
            bloom_reads.consume(kmer)
            read_kmers.add(kmer)

        for kmer in bloom_reads_plusone.get_kmers(s):
            bloom_reads_plusone.consume(kmer) # this is for the paths


    # init an empty graph TODO for now this creates a new graph instead of updating
    dbg = gt.Graph(directed=False)
    # init edge weights
    edge_weights = dbg.new_edge_property("short")
    edge_weights.a.fill(0)
    dbg.ep["edge_weights"] = edge_weights

    # loop through all kmers TODO in future update instead of regenerate
    # i.e. second loop through the bloom filter
    present_edges = khmer.Nodetable(bloom_reads.ksize(), 5e6, 4)
    edge_list = []
    edge_weight_list = []

    for km in read_kmers:
        # km = read_kmers[140]
        present = present_edges.get(km)
        count = bloom_reads.get(km)

        # ignore kmer if edge is already in the graph
        if present:
            continue
        # also ignore if count is smaller than t
        elif count < threshold:
            continue

        # only if the edge is not present yet and counter is over t, add it
        else:
            # construct the vertices
            lmer = km[:-1]
            rmer = km[1:]
            edge_list.append((lmer, rmer))
            edge_weight_list.append(count)
            # set edge as present
            present_edges.add(km)

    # add all edges in one swoop
    vertex_names = dbg.add_edge_list(edge_list, hashed=True, string_vals=True)
    # set the edge weights
    dbg.ep.edge_weights.a = edge_weight_list

    # internalize the vertex names
    dbg.vp["nodes"] = vertex_names

    # initialise BOSS properties (prior, scores, util, smu, timecost)

    return dbg, read_kmers, bloom_reads, bloom_reads_plusone, threshold

#%%


# all of these init functions set some property map but don't return anything!
def init_prior(graph, prior):
    # save the prior as graph property, for access
    gprior = graph.new_graph_property("float")
    # internalise property
    graph.gp["prior"] = gprior
    # set value for internal property
    graph.gp.prior = prior


def init_edge_weights(graph):
    # init edge weights
    edge_weights = graph.new_edge_property("short")
    edge_weights.a.fill(1)  # temporary, need to implement edge merging if there are already parallel ones
    # internalise the property
    graph.ep["edge_weights"] = edge_weights


def init_scores(graph):
    # initialise a score vector for all nodes
    scores = graph.new_vertex_property("float")
    graph.vp["scores"] = scores
    # fill the scores with pseudocounts
    scores.a = graph.gp.prior
    # a new read can only do one thing here
    potential_counts = copy(scores.a)
    potential_counts[0] += 1
    # calculate KL divergence once and fill the array with that
    s = kl_diri(a=potential_counts, b=scores.a)
    scores.a.fill(s)


def init_utility(graph):
    # initialise 2 utility vectors for all edges and their reciprocal
    util_zig = graph.new_edge_property("float")
    util_zag = graph.new_edge_property("float")
    util_zig.a.fill(0) # gets overwritten when updating anyways
    util_zag.a.fill(0)  # gets overwritten when updating anyways
    # internalise
    graph.ep["util_zig"] = util_zig
    graph.ep["util_zag"] = util_zag


def init_timeCost(graph, lam, mu, rho, alpha):
    # init time cost per EDGE
    # for now this is uniform
    # no prior about chromosome ends
    # add Fhat at some point
    timeCost_zig = graph.new_edge_property("float")
    timeCost_zag = graph.new_edge_property("float")
    tc = (lam - mu - rho)
    timeCost_zig.a.fill(tc)
    timeCost_zag.a.fill(tc)
    # internalise
    graph.ep["timeCost_zig"] = timeCost_zig
    graph.ep["timeCost_zag"] = timeCost_zag

    # all edges leading to nodes with out-degree of 1 should have 0 timeCost
    # we always allow expansion of the graph
    # this should be done according to read length dist at some point TODO
    culdesac = np.nonzero(graph.get_out_degrees(graph.get_vertices()) == 1)[0]
    _, edge_targets = get_edge_mapping(graph)
    edge_targets_zig = edge_targets[:len(edge_targets)//2]
    edge_targets_zag = edge_targets[len(edge_targets) // 2:]
    graph.ep.timeCost_zig.a[np.isin(edge_targets_zig, culdesac)] = 1e-20
    graph.ep.timeCost_zag.a[np.isin(edge_targets_zag, culdesac)] = 1e-20

    # add t0 - cost irrespective of accept or reject
    t0 = graph.new_graph_property("float")
    graph.gp["t0"] = t0
    graph.gp.t0 = (alpha + mu + rho)


def init_smu(graph):
    # initialise an edge property for S_mu, gets updated together with U
    s_mu_zig = graph.new_edge_property("float")
    s_mu_zag = graph.new_edge_property("float")
    graph.ep["s_mu_zig"] = s_mu_zig
    graph.ep["s_mu_zag"] = s_mu_zag
    # fill the scores with pseudocounts
    s_mu_zig.a.fill(0)  # gets updated together with U
    s_mu_zag.a.fill(0)  # gets updated together with U





def updateS(graph, bloom_paths, t_solid):
    # loop over nodes - there is probably a more efficient way
    for i in range(graph.num_vertices()):
        # get corresponding k-1mer and loop through the k+1-mers to get the paths through the node
        current_node = graph.vp.nodes[i]
        counts = count_paths(node=current_node, bloom_paths=bloom_paths, t_solid=t_solid)
        # actual calculation
        node_score = nodeScore(counts=counts, prior=graph.gp.prior)
        graph.vp.scores[i] = node_score

    # all nodes with out-degree of 1 have adjusted scores to make sense in combo
    # with the absorbers
    culdesac = np.nonzero(graph.get_out_degrees(graph.get_vertices()) == 1)[0]
    graph.vp.scores.a[culdesac] = max(graph.vp.scores.a) / 4

    # return array of scores (pointer to vertex property)
    scores = graph.vp.scores.a
    return scores


def count_paths(node, bloom_paths, t_solid):
    # count the paths spanning a node
    # - instead of estimating them from combinations of incident edges
    # - actual k+1 mers are stored in a separate bloom filter and can be checked
    # special treatment needed for palindromic kmers, otherwise they return each count twice
    counts = []
    if is_palindrome(node):
        for km in kmer_neighbors_palindromic(node):
            c = bloom_paths.get(km)
            if c >= t_solid:
                counts.append(c)
    else:
        for km in kmer_neighbors(node):
            c = bloom_paths.get(km)
            if c >= t_solid:
                counts.append(c)
    return counts


# function to calculate the potential increase of KL
def nodeScore(counts, prior):
    # append a 0 for a potential new link
    count_arr = np.append(counts, 0).astype("float")
    # add the prior values
    np.add(count_arr, prior, out=count_arr)
    # calculate potential information increase by increasing every path by 1
    # or establishing a new link and average over all D_KL
    # potential info increase is weighted by already existing links, cf Polya Urn model
    potential_counts = np.tile(count_arr, (len(counts) + 1, 1))
    np.add(potential_counts, np.identity(n=len(counts) + 1), out=potential_counts)
    # observation probabilities as weights
    p_obs = count_arr / np.sum(count_arr)
    # KL divergence with every potential change after observing new data
    score = 0
    for row in range(potential_counts.shape[0]):
        score += p_obs[row] * kl_diri(a=potential_counts[row,], b=count_arr)
    return score




def kl_diri(a, b):
    # Kullback-Leibler divergence of two dirichlet distributions
    # a, b are arrays of parameters of the dirichlets
    a0 = sum(a)
    b0 = sum(b)

    D = sum((a - b) * (digamma(a) - digamma(a0))) +\
        sum(gammaln(b) - gammaln(a)) +\
        gammaln(a0) - gammaln(b0)
    return D




# pr = 1e-6
# a = csr_matrix(np.array([5, 4, 1, 0.0, 0,0,0,0]))
# nodeScore(a, prior=pr)
# b = csr_matrix(np.array([2, 2, 0, 0, 0, 0.0, 0, 0, 0, 0, 0]))
# nodeScore(b, prior=pr)
#
#
# dirichlet(a).entropy()
# dirichlet(b).entropy()
# kl_diri(a, b)




#%%
# create a better "random" assembly for testing
def generate_dummy_assembly(length):
    # init empty graph
    assembly = gt.Graph(directed=False) # undirected to account for forward and reverse
    base_edges = np.array([ np.arange(0,length), np.arange(1,length + 1)]).transpose()
    assembly.add_edge_list(edge_list=base_edges)
    # base_edges_rev = np.array([ np.arange(1,length + 1), np.arange(0,length)]).transpose()
    # assembly.add_edge_list(edge_list=base_edges_rev)
    return assembly


def generate_bubbles(graph):
    # introduce some bubbles of different size and length
    # every node has a probability of spawning a bubble
    bubbles = np.nonzero(np.random.choice(a=[0,1], size=graph.num_vertices()-3, p=[0.95,0.05]))[0]
    # bubble_sizes = np.random.choice(a=[2,3,4], size=len(bubbles), p=[0.85,0.1,0.05])
    bubble_lengths = np.random.choice(a=[3,4,5], size=len(bubbles), p=[0.6,0.3,0.1])

    for b in range(len(bubbles)):
        if bubble_lengths[b] == 1:
            bv = graph.add_vertex()
            graph.add_edge(bubbles[b] - 1, bv)
            graph.add_edge(bv, bubbles[b] + 1)
        elif bubble_lengths[b] == 2:
            bv_list = list(graph.add_vertex(2))
            graph.add_edge(bubbles[b] - 1, bv_list[0])
            graph.add_edge(bv_list[0], bv_list[1])
            graph.add_edge(bv_list[1], bubbles[b] + 2)
        elif bubble_lengths[b] == 3:
            bv_list = list(graph.add_vertex(3))
            graph.add_edge(bubbles[b] - 1, bv_list[0])
            graph.add_edge(bv_list[0], bv_list[1])
            graph.add_edge(bv_list[1], bv_list[2])
            graph.add_edge(bv_list[2], bubbles[b] + 3)

    # set the weight of new edges to 1
    graph.ep.edge_weights.a[np.nonzero(graph.ep.edge_weights.a == 0)] = 1
    return graph


def generate_tips(graph):
    # every node can spawn tips
    tips = np.nonzero(np.random.choice(a=[0,1], size=graph.num_vertices(), p=[0.9,0.1]))[0]
    tip_lengths = np.random.choice(a=[1, 2, 3], size=len(tips), p=[0.6, 0.3, 0.1])

    for t in range(len(tips)):
        if tip_lengths[t] == 1:
            tv = graph.add_vertex()
            graph.add_edge(tips[t], tv)

        elif tip_lengths[t] == 2:
            tv_list = list(graph.add_vertex(2))
            graph.add_edge(tips[t], tv_list[0])
            graph.add_edge(tv_list[0], tv_list[1])

        elif tip_lengths[t] == 3:
            tv_list = list(graph.add_vertex(3))
            graph.add_edge(tips[t], tv_list[0])
            graph.add_edge(tv_list[0], tv_list[1])
            graph.add_edge(tv_list[1], tv_list[2])
    # set the weight of new edges to 1
    graph.ep.edge_weights.a[np.nonzero(graph.ep.edge_weights.a == 0)] = 1
    return graph


def add_random_edges(graph):
    # add some random edges to a graph by increasing the edge weights
    n_edges = graph.num_edges()
    rand_indices = np.random.randint(low=0, high=n_edges, size=n_edges)
    graph.ep.edge_weights.a[rand_indices] += 1



def trunc_normal(mu, sd, lam):
    # input parameters and output dist are in base unit
    # get the maximum read length
    lastl = int(lam + 10 * sd)
    # prob density of normal distribution
    x = np.arange(lastl, dtype='int')
    ldist = np.exp(-((x - lam + 1) ** 2) / (2 * (sd ** 2))) / (sd * np.sqrt(2 * np.pi))
    # exclude reads shorter than mu
    ldist[:mu] = 0.0
    # normalise
    ldist /= sum(ldist)
    return ldist


def comp_cml_dist(ldist):
    #complement of cumulative distribtuion of read lengths
    ccl = np.zeros(len(ldist) + 1)
    ccl[0] = 1
    # subtract cml sum to get ccl
    ccl[1:] = 1 - ldist.cumsum()
    # correct numerical errors, that should be 0.0
    ccl[ccl < 1e-10] = 0
    # cut distribution off at some point to reduce complexity of calculating U
    # or just trim zeros
    ccl = np.trim_zeros(ccl, trim='b')
    return ccl



def sim_reads(N, mu, sd, lam, batchSize, err, errD, perc, write):
    genome = simulateReads.simGenome(N=N, perc=perc)
    N = len(genome)
    # generate read length dist
    L = simulateReads.readLengthDist_TruncNormal(N=N, mu=mu, sd=sd, lam=lam)
    # distribution for read start positions
    F = simulateReads.readStartDist(N=N, oscill=1, amp=0)
    reads = simulateReads.createFastq(genome=genome, num=0, L=L, F=F,
                                      batchSize=batchSize, err=err, errD=errD, write=write)
    genome = simulateReads.basify(genome)
    # calc coverage
    bases = 0
    for seq in reads.values():
        bases += len(seq)
    cov = bases / N
    print(f"average coverage is {cov}")

    return reads, genome






def fq2readpool(fq):
    # init container for lengths and seqs
    read_lengths = {}
    read_sequences = {}

    # fill the dicts read_lengths and read_sequences
    _fq = Path(fq).expanduser()
    print(f"Processing file: {_fq}")
    # check whether fastq is gzipped - there might be a better way here
    if _fq.name.endswith(('.gz', '.gzip')):
        fh = gzip.open(_fq, 'rt')
    else:
        fh = open(_fq, 'rt')

    # loop over all reads in the fastq file
    for desc, name, seq, qual in readfq(fh):
        bases_in_read = len(seq)
        read_lengths[str(name)] = bases_in_read
        read_sequences[str(name)] = seq
    fh.close()

    return read_lengths, read_sequences





def consume_and_return(reads, k):
    # take a bunch of reads, initialise a bloom filter and return all consumed kmers
    # init the bloom filter
    target_table_size = 5e6
    num_tables = 5
    bloom_reads = khmer.Counttable(k, target_table_size, num_tables)

    # consume all reads and save the kmers at the same time
    # each kmer has a reverse complement twin which are counted in a single bin
    read_kmers = []
    for id, seq in reads.items():
        kmers = bloom_reads.get_kmers(seq)
        bloom_reads.consume(seq)
        read_kmers.extend(kmers)

    return read_kmers, bloom_reads



def solid_kmers(read_kmers, bloom_reads, genome_estimate):
    # from the current state of the bloom filter get threshold t
    # for different vals of t - filter and count
    # if number of k-mers < genome_estimate: return t-1
    for t in range(0, 40):
        counter = 0
        # filter read kmers by abundance
        for km in read_kmers:
            if bloom_reads.get(km) >= t:
                counter += 1

        if counter > (genome_estimate):
            # counter = 0
            continue

        else:
            threshold = t - 1
            threshold = 2  # temp TODO
            return threshold



def probability_mat(mat, edge_weights):
    # transform a matrix to probabilities with edge weights
    # multiply by edge weights. They are repeated because hashimoto expands the edge set to 2E
    # ew = np.repeat(edge_weights, 2) # this version is for "consecutive" ordering of the hashimoto
    ew = np.concatenate((edge_weights, edge_weights))  # this version is for "blocks" ordering of hashimoto
    mat_weighted = csr_matrix(mat.multiply(np.array(ew)))
    # normalise to turn into probabilities
    mat_prob = normalize_matrix_rowwise(mat=mat_weighted)
    # filter very low probabilities, e.g. edges escaping absorbers
    mat_prob = filter_low_prob(prob_mat=mat_prob)
    return mat_prob


def filter_low_prob(prob_mat):
    # filter very low probabilities by setting them to 0
    # this prevents the probability matrix from getting denser and denser because
    # of circular structures and the absorbers
    # indices of (nonzero < 0.01) relative to all nonzero
    threshold_mask = np.array(prob_mat[prob_mat.nonzero()] < 0.01)[0]
    # row and col indices of filtered items
    threshold_rows = prob_mat.nonzero()[0][threshold_mask]
    threshold_cols = prob_mat.nonzero()[1][threshold_mask]
    prob_mat[threshold_rows, threshold_cols] = 0
    prob_mat.eliminate_zeros()
    # then normalise again
    prob_mat = normalize_matrix_rowwise(mat=prob_mat)
    return prob_mat


def normalize_matrix_rowwise(mat):
    factor = mat.sum(axis=1)
    nnzeros = np.where(factor > 0)
    factor[nnzeros] = 1 / factor[nnzeros]
    factor = np.array(factor)[0]

    if not mat.format == "csr":
        raise ValueError("csr only")
    # using csr_scale_rows from scipy.sparse.sparsetools
    csr_scale_rows(mat.shape[0], mat.shape[1], mat.indptr,
                   mat.indices, mat.data, factor)
    return mat


def source_and_targets(graph, eso):
    # get the source and target nodes of the edges in the hashimoto 2|E| edge set
    # sort the edge index first
    edges = graph.get_edges()[eso]
    # generate the reverse edge, just like the hashimoto does
    edges_rev = np.fliplr(edges)
    # fill a bigger array that lets us index into the source and targets
    edge_indices = np.empty(shape=(edges.shape[0] * 2, 2), dtype="int")
    edge_indices[0::2] = edges
    edge_indices[1::2] = edges_rev
    edge_sources = edge_indices[:, 0]
    edge_targets = edge_indices[:, 1]
    # index as in hashimoto - just for debugging
    edge_indices = np.concatenate((edge_indices, np.array([np.arange(0,edges.shape[0]*2)]).T), axis=1)
    return edge_sources, edge_targets, edge_indices


def get_edge_mapping(graph):
    # returns a list of source and target nodes for each edge
    # to use with the "blocks" ordering of the improved hashimoto implementation
    edge_mapping = np.array([(graph.vertex_index[node1], graph.vertex_index[node2])
                             for (node1, node2) in graph.edges()])

    nedges = graph.num_edges()
    edge_mapping_rev = np.fliplr(edge_mapping)
    # fill a bigger array that lets us index into the source and targets
    edge_indices = np.empty(shape=(nedges * 2, 2), dtype="int")
    edge_indices[0:nedges] = edge_mapping
    edge_indices[nedges:] = edge_mapping_rev
    edge_sources = edge_indices[:, 0]
    edge_targets = edge_indices[:, 1]
    return edge_sources, edge_targets


def edge_target_scores(graph, edge_targets):
    # use the edge targets to generate a score vector
    edge_target_scores = graph.vp.scores.a[edge_targets]
    return edge_target_scores




def merge(source_node, target_node):
    m = SequenceMatcher(None, source_node, target_node)
    for o, i1, i2, j1, j2 in m.get_opcodes():
        if o == 'equal':
            yield source_node[i1:i2]
        elif o == 'delete':
            yield source_node[i1:i2]
        elif o == 'insert':
            yield target_node[j1:j2]
        elif o == 'replace':
            yield source_node[i1:i2]
            yield target_node[j1:j2]


def verify_paths(graph, bloom_paths, edge_sources, edge_targets, absorber_tri_edges):
    # we want to verify which paths actually exist in the data
    hashimoto = fast_hashimoto(graph=graph)
    num_edges = int(hashimoto.shape[0] / 2)
    # hash_py, ordering = fast_hashimoto(graph=graph, ordering="blocks", return_ordering=True)
    # h_dense = densify(hashimoto)
    # plot_gt(graph, ecolor=graph.ep.edge_weights)

    # we might be able to use the k+1 mer graph to do this instead
    # so that we do not have to rebuild the seqs
    # dbg_pone_adj = gt.spectral.adjacency(dbg_pone).T
    # dbg_pone_adj2 = dbg_pone_adj @ dbg_pone_adj

    rowsums = np.squeeze(np.array(hashimoto.sum(axis=1)))
    # find edges which have more than one next possible step
    multistep = np.where(rowsums > 1)[0]
    # exception for absorer tri-edges
    multistep = multistep[~np.isin(multistep, absorber_tri_edges)]

    # if multistep empty: return empty or something, probably not necessary
    mask = np.zeros(shape=hashimoto.shape, dtype=bool)
    # masklist = []
    # edge = multistep[0] # tmp
    for edge in multistep:
        # rebuild the edge kmer
        path_source = edge_sources[edge]
        path_source_seq = graph.vp.nodes[path_source]
        path_mid = edge_targets[edge]
        path_mid_seq = graph.vp.nodes[path_mid]
        path_source_mid_seq = ''.join(merge(path_source_seq, path_mid_seq))
        # next possible edges and nodes
        nextsteps = hashimoto[edge, :].nonzero()[1]
        path_targets = edge_targets[nextsteps]
        for t in range(len(path_targets)):
            # t = 1 # tmp
            path_target = path_targets[t]
            edge_target = nextsteps[t]
            # this is for using the k+1 mer adjacency trick
            # print(f'{path_source} -> {path_target}: {dbg_pone_adj2[path_source, path_target]}')

            path_target_seq = graph.vp.nodes[path_target]
            path_seq = ''.join(merge(path_source_mid_seq, path_target_seq))
            # check if the constructed sequence has been observed
            observation_count = bloom_paths.get(path_seq)

            # print(f'{path_source} -> {path_target}: {observation_count}')
            # if not, set the mask at that point to 1
            if observation_count == 0:
                mask[edge, edge_target] = 1
                # also add the reciprocal
                edge_recip = reciprocal_edge(edge, num_edges)
                target_recip = reciprocal_edge(edge_target, num_edges)
                mask[target_recip, edge_recip] = 1
                # masklist.append((path_source, path_target))

    # print(masklist)
    # print(np.nonzero(mask))
    # plot_gt(graph, ecolor=graph.ep.edge_weights)
    return mask


def reciprocal_edge(edge, num_edges):
    # returns the reciprocal edge under the "blocks" ordering of the hashimoto
    # i.e. two reciprocal edges have the indices i & i + num_edges
    if edge < num_edges:
        return edge + num_edges
    else:
        return edge - num_edges


def updateU(graph, bloom_paths, ccl, mu):
    # to get consecutive scores from each node, we would use the transition matrix up to the lth power
    # but since the graph is bidirectional we need to use the non-backtracking operator instead
    # which needs a bit more calculation to arrive at transition probabilities
    # we calculate both the probability of transitioning to a node and arriving at that node
    # the first is done with hashimoto turned into a probability matrix
    # and the second with compl. cml. distribution of read lengths
    # Another point to consider are dead-ends. Because ccl can not extend further at these positions,
    # they will have a biased, lowered benefit. Therefore we add some absorbing triangles
    # to the graph that will propagate ccl at dead-ends

    # add triangular absorbers
    # we will use this graph for calculating the hashimoto, later trim back to original edge set
    graph_absorbers, culdesac, absorb_edges = add_absorbers(graph)
    # original_v = graph.num_vertices()
    # calc the hashimoto
    hashimoto = fast_hashimoto(graph=graph_absorbers)

    # source and targets for edges in the hashimoto ("blocks" ordering)
    edge_sources, edge_targets = get_edge_mapping(graph_absorbers)
    edge_scores = edge_target_scores(graph=graph_absorbers, edge_targets=edge_targets)

    # the edges involved in the absorbers should not be deleted, despite not actually being observed
    absorber_tri_edges = np.nonzero(np.isin(edge_targets, culdesac))

    # verify paths across hubs by the (k+1)-mer data observed
    zero_mask = verify_paths(graph=graph_absorbers, bloom_paths=bloom_paths,
                             edge_sources=edge_sources, edge_targets=edge_targets,
                             absorber_tri_edges=absorber_tri_edges)

    # set 0 for the non-observed paths
    hashimoto[zero_mask] = 0

    # turn into a probability matrix (rows sum to 1)
    # this normalizes per row and filters low probabilities
    hp = probability_mat(mat=hashimoto, edge_weights=graph_absorbers.ep.edge_weights.a)
    hp_base = deepcopy(hp) # save a copy for mat mult
    # hp_dense = densify(hp)

    # first transition with H^1
    arrival_scores = hp.multiply(edge_scores) # * ccl[0]
    # as_dense = densify(arrival_scores)
    # In this function we calculate both utility and S_mu at the same time
    s_mu = csr_matrix(deepcopy(arrival_scores))
    # for all consecutive steps
    for i in range(1, len(ccl)):
        # increment step of hashimoto_probs (multiplication instead of power for efficiency)
        hp = hp @ hp_base

        # reduce the density of the probability matrix
        # if i % 5 == 0:
        hp = filter_low_prob(hp)
        # hp_dense = densify(hp)

        # multiply by scores and add - this is element-wise per row
        transition_score = hp.multiply(edge_scores)
        # trans_dense = densify(transition_score)

        if i <= mu:
            s_mu += transition_score

        # element-wise multiplication of coo matrix and float
        # trans_intermed_dense = densify(transition_score * ccl[i])
        arrival_scores += transition_score * ccl[i]
        # as_dense2 = densify(arrival_scores)


    # row sums are utility for each edge
    utility = np.squeeze(np.array(arrival_scores.sum(axis=1)))
    s_mu_vec = np.squeeze(np.array(s_mu.sum(axis=1)))
    # add back the original score of the starting node
    utility += edge_scores
    s_mu_vec += edge_scores

    # get rid of the utility of edges within the absorbers
    # using masked array
    util_mask = np.ma.array(utility, mask=False)
    s_mu_mask = np.ma.array(s_mu_vec, mask=False)
    util_mask[absorb_edges] = np.ma.masked
    s_mu_mask[absorb_edges] = np.ma.masked

    # assign utility to edge properties
    graph.ep.util_zig.a = util_mask[~util_mask.mask][:len(util_mask[~util_mask.mask]) // 2]
    graph.ep.util_zag.a = util_mask[~util_mask.mask][len(util_mask[~util_mask.mask]) // 2:]
    graph.ep.s_mu_zig.a = s_mu_mask[~s_mu_mask.mask][:len(s_mu_mask[~s_mu_mask.mask]) // 2]
    graph.ep.s_mu_zag.a = s_mu_mask[~s_mu_mask.mask][len(s_mu_mask[~s_mu_mask.mask]) // 2:]
    return None





#%%

def add_absorbers(graph):
    # create a copy of the graph
    absorb_graph = graph.copy()
    original_v = absorb_graph.num_vertices()
    original_e = absorb_graph.num_edges()
    # find all dead-ends
    culdesac = np.nonzero(absorb_graph.get_out_degrees(absorb_graph.get_vertices()) == 1)[0]
    culdesac_scores = absorb_graph.vp.scores.a[culdesac]
    if len(culdesac) == 0:
        return absorb_graph

    # add absorber for each end
    # - this could probably be more efficient without the loop
    # with some smarter indexing and a single add_edge_list
    for c in culdesac:
        # get current number of vertices
        num_v = absorb_graph.num_vertices()
        # 2 new absorbing vertices
        abs1 = num_v
        abs2 = num_v + 1
        # 3 new edges
        edge_list = [(c, abs1), (c, abs2), (abs1, abs2)]
        absorb_graph.add_edge_list(edge_list)

    # set edge weights to a high number for absorbers
    absorb_graph.ep.edge_weights.a[original_e:] = 999
    # transfer scores to absorbers
    culdesac_scores = np.repeat(culdesac_scores, 2)
    absorb_graph.vp.scores.a[original_v:] = culdesac_scores
    # indices of the new edges
    new_e = absorb_graph.num_edges()
    absorb_edges = list(range(original_e, new_e))
    absorb_edges_recip = [i + new_e for i in absorb_edges]
    absorb_edges += absorb_edges_recip

    return absorb_graph, culdesac, absorb_edges



#%%
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


def fast_hashimoto(graph, ordering='blocks', return_ordering=False):
    """Make the Hashimoto (aka Non-Backtracking) matrix.
    Params
    ------
    graph (gt graph)
    ordering (str): Ordering used for edges (see `half_incidence`).
    return_ordering (bool): If True, return the edge ordering used (see
    `half_incidence`).  If False, only return the matrix.
    Returns
    -------
    A sparse (csr) matrix.
    """
    if return_ordering:
        sources, targets, ord_func = half_incidence(graph, ordering, return_ordering)
    else:
        sources, targets = half_incidence(graph, ordering, return_ordering)
    temp = np.dot(targets.T, sources).asformat('coo')
    temp_coords = set(zip(temp.row, temp.col))

    coords = [(r, c) for r, c in temp_coords if (c, r) not in temp_coords]
    data = np.ones(len(coords))
    shape = 2*graph.num_edges()
    hashimoto = coo_matrix((data, list(zip(*coords))), shape=(shape, shape))

    if return_ordering:
        return hashimoto.asformat('csr'), ord_func
    else:
        return hashimoto.asformat('csr')


def detect_cycles(hashimoto):
    edges = int(hashimoto.shape[0] / 2)
    h_base = deepcopy(hashimoto)
    # count the cycles
    cycles = []
    trace = hashimoto.diagonal().sum()
    cycles.append(trace)
    for i in range(edges * 3):
        hashimoto = hashimoto @ h_base
        trace = hashimoto.diagonal().sum()
        cycles.append(trace)
    return cycles


def has_cycles(hashimoto):
    return hashimoto.diagonal().sum()



#%%

def find_strat_aeons(graph):

    # utility_cont is the expected benefit from keeping reading a fragment
    utility_cont = np.array(graph.vp.util.a - graph.vp.s_mu.a)
    # timeCost is the expected cost increase by keeping reading a fragment
    timeCost = np.array(graph.vp.timeCost.a)


    # vector of ratios
    uot = utility_cont / timeCost
    # if time cost is negative (or almost zero), always accept.
    # Does not do anything now, maybe adjust graph ends with this
    uot = np.where(timeCost <= 0.000000000001, np.inf, uot)
    # argsort the u/t array
    # "I" contains the list of genome positions from the most valuable for the strategy to the least valuable.
    forwarded_i = np.argsort(uot)[::-1]

    # average benefit of strategy, initialized to the case that all fragments are rejected
    # (like U^0 at the end of Supplementary Section 2).
    # ubar0 = np.sum(Fhat[:, 0] * S_mu[:, 0]) + np.sum(Fhat[:, 1] * S_mu[:, 1])
    ubar0 = np.average(graph.vp.s_mu.a)

    # NICOLA: this is more operations than the old version,
    # but by using numpy functions it should be almost always faster.
    # Find the best strategy (or equivalently, the number of positions stratSize to accept).
    # here we start with a strategy that rejects all fragments and increase its size one by one.
    # in some scenario it might be faster to start from a strategy that
    # accepts everything and then remove fragments one by one.
    ordered_benefit = utility_cont[forwarded_i]
    cumsum_u = np.cumsum(ordered_benefit) + ubar0
    cumsum_t = np.cumsum(timeCost[forwarded_i]) + graph.gp.t0
    # stratSize is the total number of accepted positions (the number of 1's) of the current strategy
    strat_size = np.argmax(cumsum_u / cumsum_t) + 1
    # print(f'stratsize: {strat_size}')
    # threshold = ordered_benefit[strat_size-1]

    # put strat in an array
    strategy = np.zeros(graph.num_vertices(), dtype=bool)
    np.put(strategy, forwarded_i[:strat_size], True)
    # make a property map
    strat = graph.new_vertex_property("float")
    graph.vp["strat"] = strat
    graph.vp.strat.a = strategy

    print(f'Accepting nodes: {np.sum(strategy)}, {np.sum(strategy) / graph.num_vertices()}')
    return strategy



#%%
# FUNCTIONS FOR READING AND MAPPING
# we need functions to write the current graph to GFA format
# and to add the updates to the graph to the GFA (if de novo writing takes too long)
def toGFA(graph, k, file):
    # write the graph to GFA format
    overlap = k - 2

    with open(file, "w") as gfa:
        gfa.write(f'H\tVN:Z:1.0\n')

        # write nodes
        for name, seq in enumerate(graph.vp.nodes):
            gfa.write(f'S\t{name}\t{seq}\n')

        # write edges
        for source, target in graph.get_edges():
            gfa.write(f'L\t{source}\t+\t{target}\t+\t{overlap}M\n')

    return None


# function to discover all sequence files in a directory
def discover_seq_files(directory):
    # return list of files
    file_extensions = [".fq.gz", ".fastq", ".fq", ".fastq.gz"]
    direc = Path(directory).expanduser()

    if direc.is_dir():
        file_list = [x for x in direc.iterdir() if "".join(x.suffixes).lower() in file_extensions and 'trunc' not in x.name]
        return file_list
    else:
        return []


def execute(command, cmd_out=subprocess.PIPE, cmd_err=subprocess.PIPE, cmd_in=subprocess.PIPE):
    args = shlex.split(command)
    running = subprocess.Popen(args, stdout=cmd_out, stderr=cmd_err, stdin=cmd_in,
                               encoding='utf-8')
    stdout, stderr = running.communicate()
    return stdout, stderr


def truncate_fq(fq, mu):
    # truncates the seqs in a fastq file to some length
    # using awk and subprocess
    fq_base = Path(fq.stem).with_suffix('')
    fq_trunc = str(fq_base) + "_trunc" + ''.join(fq.suffixes)
    fq_trunc_name = fq.with_name(fq_trunc)

    # case of gzipped file - gunzip | awk | gzip
    if '.gz' in fq.suffixes:
        fq_trunc_file = gzip.open(fq_trunc_name, 'w')
        uncompress = subprocess.Popen(['gunzip', '-c', fq], stdout=subprocess.PIPE)
        awk_cmd = f"awk '{{if ( NR%2 == 0 ) print substr($0,1,{mu}); else print}}'"
        truncate = subprocess.Popen(shlex.split(awk_cmd), stdin=uncompress.stdout, stdout=subprocess.PIPE)
        stdout, stderr = execute("gzip -", cmd_in=truncate.stdout, cmd_out=fq_trunc_file)

    # uncompressed case - awk
    else:
        fq_trunc_file = open(fq_trunc_name, 'w')
        cmd = f"awk '{{if ( NR%2 == 0 ) print substr($0,1,{mu}); else print}}' {fq}"
        stdout, stderr = execute(command=cmd, cmd_out=fq_trunc_file)

    fq_trunc_file.close()
    return fq, fq_trunc_name





# read the next batch of reads from a fastq file
def map2graph(fastq_dir, processed_files, mu, exe_path, ref_path):
    # get list of all sequence files
    file_list = discover_seq_files(directory=fastq_dir)

    # loop over fastqs; select next one
    for _fq in file_list:
        if _fq not in processed_files:
            print(f"Processing file: {_fq}")

            # first we truncate the file, deals with compressed and uncompressed
            full_file, trunc_file = truncate_fq(fq=_fq, mu=mu)
            full_gaf = full_file.with_suffix('.gaf')
            trunc_gaf = trunc_file.with_suffix('.gaf')

            # map the full file TODO tune parameters for mapping
            graphaligner_full = f"{exe_path} -g {ref_path} -a {full_gaf} -f {full_file}" \
                                f" -x dbg --seeds-minimizer-length 7 --seeds-minimizer-windowsize 11"

            # map the truncated file
            graphaligner_trunc = f"{exe_path} -g {ref_path} -a {trunc_gaf} -f {trunc_file}" \
                                 f" -x dbg --seeds-minimizer-length 7 --seeds-minimizer-windowsize 11"

            stdout_full, stderr_full = execute(command=graphaligner_full)
            stdout_trunc, stderr_trunc = execute(command=graphaligner_trunc)

            # add the fastq file to the processed ones
            processed_files.append(_fq)

            # return the file descriptors for gaf files and the stdout of the two runs
            return full_gaf, trunc_gaf, stdout_full, stdout_trunc, processed_files
    # return (None, None, None) # if file list is empty, return Nones



def parse_GA_stdout(GAout):
    # this function parses the stdout from GraphAligner runs
    # the stdout contains some useful info: total bases, mapped reads

    # total amount of bases
    basesTOTAL_str = [x for x in GAout.split('\n') if x.startswith("Input reads:")][0].split(" ")[-1]
    basesTOTAL = int(''.join(c for c in basesTOTAL_str if c.isdigit()))

    # number of reads
    n_reads_str = [x for x in GAout.split('\n') if x.startswith("Input reads:")][0].split(" ")[-2]
    n_reads = int(''.join(c for c in n_reads_str if c.isdigit()))

    # number of mapped reads
    n_mapped = [x for x in GAout.split('\n') if x.startswith("Reads with an")][0].split(" ")[-1]
    n_mapped = int(''.join(c for c in n_mapped if c.isdigit()))

    # unmapped reads
    n_unmapped = n_reads - n_mapped

    return basesTOTAL, n_reads, n_mapped, n_unmapped


def _conv_type(s, func):
    # Generic converter, to change strings to other types
    try:
        return func(s)
    except ValueError:
        return s


def parseGAF(gaf_file):
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


    for record in gaf_file:
        record = record.strip().split("\t")
        record_dict = {fields[x]:_conv_type(record[x], int) for x in range(12)}
        yield record_dict


def make_decision(gaf, strat, read_seqs, mu):
    # decide accept/reject for each read
    add_seqs = set()

    # loop over gaf entries with generator function
    gaf_file = open(gaf, 'r')
    # with open(gaf, "r") as gaf_file:
    for record in parseGAF(gaf_file):
        # records = list(parseGAF(gaf_file))
        # print(record)

        # filter by mapping quality
        if record["mapping_quality"] < 55:
            continue

        # decision process
        strand = 0 if record['strand'] == '+' else 1
        node = _conv_type([x for x in record['path'].split('>') if x != ''][0], int)
        # ACCEPT
        if strat[strand][node]:
            record_seq = read_seqs[record["qname"]]
        # REJECT
        else:
            record_seq = read_seqs[record["qname"]][:mu]  # TODO truncating again here after awk, efficiency loss

        add_seqs.add(record_seq)

        # this snippet extracts the alignment path without considering the exact sequence
        # could be used if the graph should not be extended, but simply edge weights are added
        # mapping = [_conv_type(x, int) for x in record['path'].split('>') if x != '']
        # for edge in zip(mapping, mapping[1:]):
        #     print(edge)

    gaf_file.close()
    return add_seqs





