import numpy as np


# def integerize(seq):
#     trans = str.maketrans('ACGT', '1234')
#     intseq = seq.translate(trans)
#     return intseq



# version 1 of the node score function
# does not account for forward and reverse reads
def nodeScore(counts, prior):
    # transform sparse matrix to row vector
    count_arr = np.squeeze(np.array(counts.todense()))
    print(count_arr)
    # convenience range
    nodes = np.arange(len(count_arr))
    # find where this node links to
    link = (count_arr != 0)

    link_ind = nodes[link]
    nonlink_ind = nodes[~link]

    # add the prior values
    np.add(count_arr, prior, out=count_arr)
    # calculate potential information increase by increasing every linkage by 1
    # and establishing a new link (times missing link) and average over all D_KL
    # potential info increase is weighted by already existing links, cf Polya Urn model

    potential_counts = np.tile(count_arr, (len(link_ind) + 1, 1))
    potential_counts[np.arange(len(link_ind)), link_ind] += 1
    # increment first index of non linking node
    potential_counts[-1, nonlink_ind[0]] += 1

    # observation probabilities as weights
    p_obs = count_arr / np.sum(count_arr)
    obs_index = np.append(link_ind, nonlink_ind[0])
    p_obs = p_obs[obs_index]

    # KL divergence with every potential change after observing new data
    score = 0
    for row in range(potential_counts.shape[0]):
        score += p_obs[row] * kl_diri(a=potential_counts[row,], b=count_arr)

    return score



# version 1 of the construction function
# less efficient and does not account for forward and reverse reads
def construct_dbg(readpool, k=61, filt=False, genome_estimate=0):
    # main constructor of the de bruijn graph
    # can either use all kmer or only a bloom filtered set to construct A-Bruijn
    # init an empty graph
    dbg = gt.Graph()


    if filt:
        # load up the bloom filter and get threshold
        read_kmers, bloom_reads = consume_and_return(readpool, k)
        threshold = solid_kmers(read_kmers, bloom_reads, genome_estimate)
        # threshold, bloom_reads = solid_kmers(reads=readpool, genome_estimate=genome_estimate, k=k)
        print(f"using threshold {threshold}")

        # filter kmers down to solid kmers
        edges = []
        for id, seq in readpool.items():
            for km in bloom_reads.get_kmers(seq):
                if bloom_reads.get(km) >= threshold:
                    lmer = km[:-1]
                    rmer = km[1:]
                    edges.append((lmer, rmer))
                    bloom_reads.add(km)


    else:
        # construct unfiltered edge set
        edges = []
        for id, seq in readpool.items():
            for km in kmers(seq, k):
                lmer = km[:-1]
                rmer = km[1:]
                edges.append((lmer, rmer))
            # for km in kmers(reverse_complement(read), k):
            #     lmer = km[:-1]
            #     rmer = km[1:]
            #     edges.append((lmer, rmer))

    # add edges
    added_vertices = dbg.add_edge_list(edges, hashed=True, string_vals=True)
    # internalize the vertex names
    dbg.vp["nodes"] = added_vertices
    return dbg


# procedure to get probability from hashimoto
hashi = gt.spectral.hashimoto(udemo)#, index=np.arange(0,20))

# hashi2 = hashi.dot(hashi)
hashi_de = deinterleave(hashi)
hashi_de_dense = densify(hashi_de)
# multiply hashi by edge weights, then normalize to make probabilities
hash_weights = hashi_de_dense * np.array(udemo.ep.edge_weights.a)
hash_prob = hash_weights / hash_weights.sum(axis=1, keepdims=True)


# normalise to turn into probabilities
# but using a trick to avoid division:
# first get a diagonal matrix of reciprocals of rowsums
# then matrix multiplic of diagonal with original
reciproc_sums = diags(1 / mat_weighted.sum(axis=1).A.ravel())
mat_prob = reciproc_sums @ mat_weighted


# functions attempting to get from the hashimoto back to a node centered matrix
def hashimoto_bidir_prob(mat, edge_weights):
    # deinterleave the matrix
    # mat_bidir = deinterleave(mat)
    # multiply by edge weights
    # mat_weighted = csr_matrix(mat_bidir.multiply(np.array(edge_weights)))
    ew = np.repeat(edge_weights, 2)
    mat_weighted = csr_matrix(mat.multiply(np.array(ew)))
    # normalise to turn into probabilities
    mat_prob = normalize_matrix_rowwise(mat=mat_weighted)

    return mat_prob


def node_centric(hashimoto, graph, edge_sortorder):
    # transform the hashimoto to be node centered instead of edge centered
    # first we need to get the source and target of each edge
    edges = graph.get_edges()[edge_sortorder,:]
    edges_rev = np.fliplr(edges)

    edge_indices = np.empty(shape=(edges.shape[0] * 2, 2), dtype="int")
    edge_indices[0::2] = edges
    edge_indices[1::2] = edges_rev
    edge_sources = edge_indices[:, 0]
    edge_targets = edge_indices[:, 1]
    print(edge_indices)
    accum = accumulate_edges(hashimoto, edge_sources, edge_targets)
    return accum

def accumulate_edges(hashimoto, edge_sources, edge_targets):
    # then we do a bin count of the sources and targets to get the node matrix
    # accumulate the rows first
    accum_rows = np.zeros(shape=(np.max(edge_sources) + 1, edge_sources.shape[0]))
    np.add.at(accum_rows, edge_sources, hashimoto)
    # then accumulate the cols
    accum = np.zeros(shape=(np.max(edge_targets) + 1, np.max(edge_targets) + 1))
    np.add.at(accum, edge_targets, accum_rows.T)
    accum = accum.T
    return accum

# hashimoto goes from E to 2E by repeating each edge
# this function untangles the hashimoto my collapsing them back together
def deinterleave(mat):
    # cols
    deinter_cols = np.add(mat[:, ::2], mat[:, 1::2])
    # rows
    deinter = np.add(deinter_cols[::2, :], deinter_cols[1::2, :])
    return deinter



# update U before the graph was bidirectional
# this one uses the transition matrix
def updateU(graph, ccl, mu):
    # to get consecutive scores from each node, we use the transition matrix up to the lth power
    # we calculate both the probability of transitioning to a node and arriving at that node
    # fist with transition matrix and the second with compl. cml. distribution of read lengths
    scores = np.array(graph.vp.scores.a)
    trans = gt.spectral.transition(graph, weight=graph.ep.edge_weights).transpose()
    # In this function we calculate both utility and S_mu, for efficiency
    # first transition
    arrival_scores = trans.multiply(scores) # * ccl[0]
    s_mu = copy(arrival_scores)
    # for all consecutive steps
    for i in range(2, len(ccl)):
        # increment step of transition matrix (use multiplication instead of power for efficiency)
        trans = trans.dot(trans)
        # tr_dense = densify(trans)
        # multiply by scores and add
        transition_score = trans.multiply(scores)
        if i <= mu:
            s_mu += transition_score
        arrival_scores += transition_score * ccl[i]

    # row sums are utility for each node
    utility = np.squeeze(np.array(np.sum(arrival_scores, axis=1)))
    s_mu_vec = np.squeeze(np.array(np.sum(s_mu, axis=1)))
    # add back the original score of the starting node
    utility += scores
    s_mu_vec += scores
    graph.vp.util.a = utility
    graph.vp.s_mu.a = s_mu_vec
    return utility





#%%

# old version of calculating score
# - using adjacency matrix to find the neighbors
# - and then combining incident edges to estimate path counts
def updateS_adjacency(graph):
    # adjacency matrix as measure of certainty
    adj = gt.spectral.adjacency(graph, weight=graph.ep.edge_weights).transpose()

    # loop over nodes - there is probably a more efficient way
    for row in range(adj.shape[0]):
        node_score = nodeScore_adjacency(counts=adj[row, ], prior=graph.gp.prior)
        graph.vp.scores[row] = node_score

    # all nodes with out-degree of 1 have adjusted scores to make sense in combo
    # with the absorbers
    culdesac = np.nonzero(graph.get_out_degrees(graph.get_vertices()) == 1)[0]
    graph.vp.scores.a[culdesac] = max(graph.vp.scores.a) / 5
    graph.vp.scores.a[culdesac] = 0

    # return array of scores (pointer to vertex property)
    scores = graph.vp.scores.a
    return scores, adj



# function to calculate the potential increase of KL
def nodeScore_adjacency(counts, prior):
    # transform sparse matrix to row vector
    count_arr = np.squeeze(np.array(counts.todense()))
    # print(count_arr)
    # convenience range
    nodes = np.arange(len(count_arr))
    # find where this node links to
    link = (count_arr != 0)
    link_ind = nodes[link]
    # nonlink_ind = nodes[~link]

    # path counts -
    if len(link_ind) == 1:
        path_counts = np.array(count_arr[link_ind])
    else:
        path_combos = np.array(list(combinations(count_arr[link], 2)))
        path_counts = np.min(path_combos, axis=1)
    # append a 0 for a potential new link
    count_arr = np.append(path_counts, 0)

    # add the prior values
    np.add(count_arr, prior, out=count_arr)
    # calculate potential information increase by increasing every linkage by 1
    # and establishing a new link and average over all D_KL
    # potential info increase is weighted by already existing links, cf Polya Urn model
    potential_counts = np.tile(count_arr, (len(path_counts) + 1, 1))
    np.add(potential_counts, np.identity(n=len(path_counts) + 1), out=potential_counts)
    # observation probabilities as weights
    p_obs = count_arr / np.sum(count_arr)
    # KL divergence with every potential change after observing new data
    score = 0
    for row in range(potential_counts.shape[0]):
        score += p_obs[row] * kl_diri(a=potential_counts[row,], b=count_arr)
    return score










#%%
# eigenvalue fun
# could be used to get the length spectrum more efficiently but not immediately useful I think

# compute the pseudo or compressed hashimoto
def pseudo_hashimoto(graph):
    """Return the pseudo-Hashimoto matrix.
    The pseudo Hashimoto matrix of a graph is the block matrix defined as
    B' = [0  D-I]
         [-I  A ]
    Where D is the degree-diagonal matrix, I is the identity matrix and A
    is the adjacency matrix.  The eigenvalues of B' are always eigenvalues
    of B, the non-backtracking or Hashimoto matrix.
    Params
    ------
    graph
    Returns
    -------
    A sparse matrix in csr format.
    """
    # Note: the rows of nx.adjacency_matrix(graph) are in the same order as
    # the list returned by graph.nodes().
    degrees = graph.get_total_degrees(graph.get_vertices())
    degrees = sparse.diags(degrees)
    adj = gt.spectral.adjacency(graph)
    ident = sparse.eye(graph.num_vertices())
    pseudo = sparse.bmat([[None, degrees - ident], [-ident, adj]])
    return pseudo.asformat('csr')


# shave the graph down to nodes with degree at least 2,
# because with less than that (i.e. at tips) we can never have cycles
def shave(graph):
    """Return the 2-core of a graph.
    Iteratively remove the nodes of degree 0 or 1, until all nodes have
    degree at least 2.
    """
    core = copy(graph)
    while True:

        to_remove = [v for v in core.get_vertices() if len(core.get_all_neighbors(v)) < 2]

        core.remove_vertex(to_remove)
        if len(to_remove) == 0:
            break
    return core



# calculate eigenvalues of hashimoto
def hashi_eigenvals(graph, topk='automatic', batch=100, fmt='complex', tol=1e-5):
    """Compute the largest-magnitude non-backtracking eigenvalues.
    Params
    ------
    graph (gt graph)
    topk (int or 'automatic'): The number of eigenvalues to compute.  The
    maximum number of eigenvalues that can be computed is 2*n - 4, where n
    is the number of nodes in graph.  All the other eigenvalues are equal
    to +-1. If 'automatic', return all eigenvalues whose magnitude is
    larger than the square root of the largest eigenvalue.
    batch (int): If topk is 'automatic', compute this many eigenvalues at a
    time until the condition is met.  Must be at most 2*n - 4; default 100.
    fmt (str): The format of the return value.  If 'complex', return a list
    of complex numbers, sorted by increasing absolute value.  If '2D',
    return a 2D array of shape (topk, 2), where the first column contains
    the real part of each eigenvalue, and the second column, the imaginary
    part.  If '1D', return an array of shape (2*topk,) made by concatenaing
    the two columns of the '2D' version into one long vector.
    tol (float): Numerical tolerance.  Default 1e-5.
    Returns
    -------
    A list or array with the eigenvalues, depending on the value of fmt.
    """
    if not isinstance(topk, str) and topk < 1:
        return np.array([[], []])

    # The eigenvalues are left untouched by removing the nodes of degree 1.
    # Moreover, removing them makes the computations faster.  This
    # 'shaving' leaves us with the 2-core of the graph.
    core = shave(graph)
    matrix = pseudo_hashimoto(core)
    if not isinstance(topk, str) and topk > matrix.shape[0] - 1:
        topk = matrix.shape[0] - 2
        print('Computing only {} eigenvalues'.format(topk))

    if topk == 'automatic':
        batch = min(batch, 2*core.num_vertices() - 4)
        if 2*core.num_vertices() - 4 < batch:
            print('Using batch size {}'.format(batch))
        topk = batch
    eigs = lambda k: sparse.linalg.eigs(matrix, k=k, return_eigenvectors=False, tol=tol)
    count = 1
    while True:
        vals = eigs(topk*count)
        largest = np.sqrt(abs(max(vals, key=abs)))
        if abs(vals[0]) <= largest or topk != 'automatic':
            break
        count += 1
    if topk == 'automatic':
        vals = vals[abs(vals) > largest]

    # The eigenvalues are returned in no particular order, which may yield
    # different feature vectors for the same graph.  For example, if a
    # graph has a + ib and a - ib as eigenvalues, the eigenvalue solver may
    # return [..., a + ib, a - ib, ...] in one call and [..., a - ib, a +
    # ib, ...] in another call.  To avoid this, we sort the eigenvalues
    # first by absolute value, then by real part, then by imaginary part.
    vals = sorted(vals, key=lambda x: x.imag)
    vals = sorted(vals, key=lambda x: x.real)
    vals = np.array(sorted(vals, key=np.linalg.norm))

    if fmt.lower() == 'complex':
        return vals

    vals = np.array([(z.real, z.imag) for z in vals])
    if fmt.upper() == '2D':
        return vals

    if fmt.upper() == '1D':
        return vals.T.flatten()



#%%

def init_utility_nodes(graph):
    # initialise utility vector for all nodes
    util = graph.new_vertex_property("float")
    util.a.fill(0) # gets overwritten when updating anyways
    # internalise
    graph.vp["util"] = util

def init_timeCost_nodes(graph, lam, mu, rho, alpha):
    # for now this is uniform
    # no prior about chromosome ends
    # add Fhat at some point
    timeCost = graph.new_vertex_property("float")
    tc = (lam - mu - rho)
    timeCost.a.fill(tc)
    # internalise
    graph.vp["timeCost"] = timeCost

    # all nodes with out-degree of 1 should have 0 timeCost
    # we always allow expansion of the graph
    culdesac = np.nonzero(graph.get_out_degrees(graph.get_vertices()) == 1)[0]
    graph.vp.timeCost.a[culdesac] = 1e-20

    # add t0 - cost irrespective of accept or reject
    t0 = graph.new_graph_property("float")
    graph.gp["t0"] = t0
    graph.gp.t0 = (alpha + mu + rho)



def utility_edge2node(utility, s_mu_vec, edge_sources, graph_absorbers, graph, original_v):
    # deprecated way of transferring the benefit back from edges to nodes
    # we actually want to keep separate benefit for edges going in both directions
    # as they can be wildly different

    # turn edge utility back to node utility
    # using source nodes of each edge
    # bincount to sum over variable sized chunks: i.e. all edges going out of a node
    node_utility = np.bincount(edge_sources, weights=utility)
    node_smu = np.bincount(edge_sources, weights=s_mu_vec)
    # divide by degree - averaging over all incident edges of each vertex
    indegree = graph_absorbers.get_total_degrees(graph_absorbers.get_vertices())
    np.divide(node_utility, indegree, out=node_utility)
    np.divide(node_smu, indegree, out=node_smu)

    # trim util and s_mu to the set of original vertices
    node_utility = node_utility[:original_v]
    node_smu = node_smu[:original_v]
    # assign benefit and s_mu to a node property
    graph.vp.util.a = node_utility
    graph.vp.s_mu.a = node_smu
    return


# original way was to read batch into memory before mapping and then feed it as stdin:
# read the next batch of reads from a fastq file
def read_batch(fastq_dir, processed_files):
    # get list of all sequence files
    file_list = discover_seq_files(directory=fastq_dir)

    # init container for lengths and seqs
    read_lengths = {}
    read_sequences = {}

    # loop over fastq files; select the next one to process
    # fill the dicts read_lengths and read_sequences
    # from one of the fastqs that has not been processed already
    for _fq in file_list:
        if str(_fq) not in processed_files:
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
                # basesTOTAL += bases_in_read # replace this with a call to sum(d.values()) later in the code
            fh.close()

            # add the fastq file to the processed ones
            processed_files.append(str(_fq))
            break

    return read_lengths, read_sequences, processed_files


def map2graph(
    read_sequences,
    exe_path,
    reference_path,
    mu,
    truncate):

    # batch of reads is input as dict of seqs
    # generate a fasta string of all sequences in the current batch
    # GraphAligner can read fasta or fastq
    if truncate:
        # truncate all reads to mu
        fasta_list = [f">{k}\n{v[:mu]}" for k, v in read_sequences.items()]
        fasta_string = "\n".join(fasta_list)
    else:
        fasta_string = "\n".join([f">{k}\n{v}" for k, v in read_sequences.items()])

    # base command for mapping with GraphAligner
    # TODO tune parameters for mapping
    cmd = f"{exe_path} -g {reference_path} -a aln.gaf -x dbg --seeds-minimizer-length 7 --seeds-minimizer-windowsize 11 -f /dev/stdin"
    # print(cmd)

    proc = subprocess.Popen(
        cmd.split(),
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        stdin = subprocess.PIPE,
        universal_newlines = True)

    # use the sequences in string/fasta format as input
    (out, err) = proc.communicate(input=fasta_string)
    # if err:
        # print(err)

    # print(f"Finished minimap2. Truncated: {truncate}")

    # check how many reads mapped and how many bases that is
    # used to keep track of sequencing time for RU mode
    mappings = 0
    readIDs = set()

    # split the output into lines
    for line in out.split('\n'):
        # split the line into its PAF fields
        paf_line = line.split('\t')
        # be aware of empty lines
        if len(paf_line) < 2:
            continue
        if paf_line[0] in readIDs:
            continue
        else:
            mappings += 1
            readIDs.add(paf_line[0])

    # calc how many reads did not map
    n_unmapped = len(read_sequences) - mappings

    print(f"Number of mapped reads: {mappings}")
    # Get the results of the Paf output
    paf_output = out

    return paf_output



# first version of updating dbg - actually just a new construction of it
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
            bloom_reads_plusone.consume(kmer)  # this is for the paths

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


# before overwriting the functions with keyless methods
 def add_novel_kmers(self, observed, new_kmers):
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
            edge_list.append((lmer_hash, rmer_hash))

            # also the reverse edge
            mk = reverse_complement(km)
            lmer = mk[:-1]
            rmer = mk[1:]
            # hash the reverse kmers
            lmer_hash = str(self.bloomf_m1.hash(lmer))
            rmer_hash = str(self.bloomf_m1.hash(rmer))

            edge_list.append((lmer_hash, rmer_hash))

            edge_weights.append(count)
            edge_weights.append(count)
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



class ImplicitGraph:
    def __init__(self, bloom, kmer):
        self.bloom = bloom
        self.kmer = kmer


    def query_kmer(self):
        self.neighbors = [self.bloom.bloomf.get(x) for x in self.kmer.n]


    def traverse(self):
        pass


class Kmer:
    def __init__(self, string):
        self.string = string

    def neighbors(self):
        self.n = set()
        for x in "ACGT":
            self.n.add(self.string[1:] + x)
            self.n.add(x + self.string[:-1])

    def iter_neighbors(self):
        pass


# LEGACY

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


# LEGACY update U

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



# ORIGINAL


# from Torres 2018
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




# from Torres 2018
def half_incidence(graph, ordering='blocks', return_ordering=False):
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
    numnodes = graph.num_vertices()
    numedges = graph.num_edges()

    if ordering == 'blocks':
        src_pairs = lambda i, u, v: [(u, i), (v, numedges + i)]
        tgt_pairs = lambda i, u, v: [(v, i), (u, numedges + i)]
    if ordering == 'consecutive':
        src_pairs = lambda i, u, v: [(u, 2 * i), (v, 2 * i + 1)]
        tgt_pairs = lambda i, u, v: [(v, 2 * i), (u, 2 * i + 1)]

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


# LEGACY


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


# LEGACY


# LEGACY TESTING
bloom.legacy_fill(reads=reads)
ass = Assembly(bloom=bloom)
ass.reconstruct()

graph_draw(ass.dbg)
graph=ass.dbg



def probability_mat(mat, edge_weights):
    # transform a matrix to probabilities by multiplying with edge weights & normalising
    # weights are repeated because hashimoto expands the edge set to 2E
    # ew = np.repeat(edge_weights, 2) # this version is for "consecutive" ordering of the hashimoto
    ew = np.concatenate((edge_weights, edge_weights))  # this version is for "blocks" ordering of hashimoto
    mat_weighted = csr_matrix(mat.multiply(np.array(ew)))
    # normalise to turn into probabilities
    mat_prob = normalize_matrix_rowwise(mat=mat_weighted)
    # filter very low probabilities, e.g. edges escaping absorbers
    mat_prob = filter_low_prob(prob_mat=mat_prob)
    return mat_prob
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
# from scipy.sparse.sparsetools import csr_scale_rows
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



def edge_target_scores(graph, edge_targets):
    # use the edge targets to generate a score vector
    edge_target_scores = graph.vp.scores.a[edge_targets]
    return edge_target_scores




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
# def reciprocal_edge(edge, num_edges):
#     # returns the reciprocal edge under the "blocks" ordering of the hashimoto
#     # i.e. two reciprocal edges have the indices i & i + num_edges
#     if edge < num_edges:
#         return edge + num_edges
#     else:
#         return edge - num_edges
#
#


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




# LEGACY
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


# non vectorised version
from scipy.special import gammaln, digamma

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


# works with non vectorised version of kl diri
def node_score(counts, prior):
    # TODO this probably needs some tuning
    # transform to array
    count_arr = np.array(counts, dtype='float')
    count_arr = np.sort(count_arr, axis=1)
    # get rid of all 0 counts to avoid redundant calculation?

    # append a 0 for a potential new link
    count_arr = np.append(count_arr, 0)
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




# TODO legacy implementation using gt directly
# TODO banished!
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



# LEGACY graph mapping
 # # read the next batch of reads from a fastq file
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


    def parse_output(GAout):
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


# POST HYBRID APPROACH IMPLEMENTATION

 # add the edges to both the kmer and k+1mer graph
self.dbg.update_graph(updated_kmers=self.bloom.updated_kmers)  # TODO tmp
self.dbg.update_graph_p(updated_kmers=self.bloom.updated_kmers_p)  # TODO tmp


allequal(self.dbg.adjacency, self.dbg2.adjacency)

indices = np.where(self.dbg.adjacency.data != self.dbg2.adjacency.data)

# TODO a few places in the matrix have higher values now. why?
# maybe that is actually correct

self.dbg.adjacency.data[indices]
self.dbg2.adjacency.data[indices]


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



def kmer_neighbors(km):
    for x in 'ACGT':
        for y in 'ACGT':
            yield x + km + y


def kmer_neighbors_palindromic(km):
    pal_tuples = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "GA", "GC", "TA"]
    for t in pal_tuples:
        yield t[0] + km + t[1]



def is_palindrome(seq):
    if seq == reverse_complement(seq):
        return True
    else:
        return False



# MORE EFFICIENT PATH COUNTING
# vectorised an indexing operation that was used within a loop


def path_counting(self, updated_edges, incr_nodes, threshold):
    # for all edges, which were either incremented or novel we need to check the path counts
    # function to get path counts for all updated kmers in a batch

    # gather all node indices for which we check the path counts
    updated_nodes = np.concatenate((np.unique(updated_edges), incr_nodes), dtype="int")

    # 17 slots, one for the index, rest for the path counts
    updated_paths = np.zeros(shape=(updated_nodes.shape[0], 17), dtype='int64')

    for i in range(len(updated_nodes)):
        # now check the possible paths for both the lmer & rmer
        node = updated_nodes[i]
        paths = self.count_paths(node=node, t_solid=threshold)

        # collect the indices and counts for the paths
        updated_paths[i, 0] = node
        updated_paths[i, 1:] = paths

    return updated_paths




def count_paths(self, node, t_solid):
    # count the paths spanning a node
    # - get the index of the node
    # - check the adjacency for all existing neighbors
    # - make all pairwise combinations of the existing neighbors
    # - check the existance of paths in the adjacency_p
    # can be used with a k-1mer or its index directly
    if type(node) == str:
        node_index = self.kmer2index(kmer=node, m=True)
    elif type(node) == int or type(node) == np.int64:
        node_index = node
    else:
        print("node neither str nor int")
        return

    neighbors = self.adjacency[node_index, :].nonzero()[1]

    neighbor_combos = list(combinations(neighbors, 2))
    counts = [0] * 16

    for i in range(len(neighbor_combos)):
        n1, n2 = neighbor_combos[i]
        c = self.adjacency_p[n1, n2]
        if c >= t_solid:
            counts[i] = c

    return counts



# less efficient version of normalising matrix row wise

#
# def normalize_matrix_rowwise(mat):
#     # rowsums for normalisation
#     rowsums = csr_matrix(mat.sum(axis=1))
#     rowsums.data = 1 / rowsums.data
#     # find the diagonal matrix to scale the rows
#     rowsums = rowsums.transpose()
#     scaling_matrix = diags(rowsums.toarray()[0])
#     norm = scaling_matrix.dot(mat)
#     return norm

# @profile
# def filter_low_prob(prob_mat, threshold=0.001):
#     # filter very low probabilities by setting them to 0
#     # this prevents the probability matrix from getting denser and denser because
#     # of circular structures and the absorbers
#     # simply index into the data, where it is smaller than some threshold
#     prob_mat.data[np.where(prob_mat.data < threshold)] = 0
#     # unset the 0 elements
#     prob_mat.eliminate_zeros()
#     # normalise the matrix again by the rowsums
#     prob_mat = normalize_matrix_rowwise(mat=prob_mat)
#     return prob_mat




