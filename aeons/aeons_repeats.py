from copy import deepcopy
import logging
import numpy as np
from collections import defaultdict

from .aeons_mapper import LinearMapper
from .aeons_utils import ascii_hist_values


class Repeat:

    def __init__(self, rid=None, side=None):
        self.rid = rid
        self.side = side
        self.starts = []
        self.ends = []
        self.start = 0
        self.end = 0
        self.degree = 0
        self.strands = []


    def calc_ranges(self):
        # get the means of starts and ends to define the repeat
        self.start = int(np.mean(self.starts))
        self.end = int(np.mean(self.ends))


    def get_sequence(self, seqpool):
        # index into seqpool and trim
        seq = seqpool[self.rid].seq[self.start: self.end]
        return seq


    def fasta(self, seqpool):
        # after collecting all edges
        # calc mean of ranges
        self.calc_ranges()
        # then extract the sequence
        seq = self.get_sequence(seqpool)
        self.length = len(seq)
        if not self.length:
            return ""
        # construct fasta entry
        header = f'{self.rid}-{self.side}-{self.start}:{self.end}'
        fa = f'>{header}\n{seq}\n'
        return fa


class RepeatFilter:
    """
    A class to:
     - extract repeats
     - generate a fasta
     - map batches to repeat library for filtering

    """
    def __init__(self, name, ava_dict, seqpool, filters):
        # constant values for PafLine filtering
        self.filters = filters
        # name for library file
        self.library = f'{name}.repeat_lib.fa'
        lim = self.repeat_degree_limit(ava_dict)
        # collect repeat overlaps in the given ava_dict
        repeats = self.collect_repeats(ava_dict, lim=lim)
        # write out the fasta file
        self.write_library(repeats, seqpool)
        # initialise a LinearMapper object
        self.lm = LinearMapper(ref=self.library)
        # get the ids of the affected reads
        self.affected_sids = self.filter_construction_seqs(repeats)


    def repeat_degree_limit(self, ava_dict):
        # check the degree of all nodes in the graph
        degree = self.node_degree(ava_dict)
        # make little histogram plot
        logging.info("degree distribution of initial graph")
        logging.info("\n" + ascii_hist_values(degree))
        # detect limit
        lim = np.quantile(degree, 0.95)
        if lim < 3:
            lim = 3
        logging.info(f"detecting repeats at degree >{lim}")
        return lim


    def node_degree(self, ava_dict):
        # automatic way of finding the degree to declare repeats
        degree = []
        # indexing ava_dict returns node, dict for each end
        for node, edge_dict in ava_dict.items():
            for side, avas in edge_dict.items():
                degree.append(len(avas))
        return degree


    def collect_repeats(self, ava_dict, lim=4):
        n_edges = []
        repeats = defaultdict(Repeat)
        # indexing ava_dict returns node, dict for each end
        for node, edge_dict in ava_dict.items():
            for side, avas in edge_dict.items():
                # only nodes with more than lim edges are considered
                n = len(avas)
                n_edges.append(n)
                if n < lim:
                    continue
                # loop through edges from the query end
                for rec in avas.values():
                    # grab the repeats
                    repo = repeats[f'{node}-{side}']
                    repo.rid, repo.side = node, side

                    # ATTENTION node is not always the query!
                    if rec.qname == node:
                        start = rec.qstart
                        end = rec.qend
                        if rec.rev:
                            strand = 1
                        else:
                            strand = 0
                    elif rec.tname == node:
                        start = rec.tstart
                        end = rec.tend
                        strand = 0
                    else:
                        # should never be the case
                        print("node name not in paf record")
                        continue

                    # apply repo attr
                    repo.starts.append(start)
                    repo.ends.append(end)
                    repo.strands.append(strand)
                    repo.degree = n

        return repeats


    def write_library(self, repeats, seqpool):
        # after collecting all repeats, extract their sequences
        n_seqs = 0
        total_seq = 0
        with open(self.library, 'w') as repfa:
            for node, repeat in repeats.items():
                fa = repeat.fasta(seqpool)
                repfa.write(fa)
                n_seqs += 1
                total_seq += repeat.length
        logging.info(f'{n_seqs} repeat sequences with {total_seq} bases total')


    def filter_construction_seqs(self, repeats):
        # after initialising the RepeatFilter we want to get rid of the affected sequences
        repeat_sids = set()
        for node, repeat in repeats.items():
            repeat_sids.add(repeat.rid)
        logging.info(f"filtering {len(repeat_sids)} sequences after constructing repeat library")
        return repeat_sids


    def filter_batch(self, seq_dict):
        # takes as input a dict of sequences
        logging.info("repeat filtering this batch of reads")
        # first map them to the library
        paf_dict = self.lm.map_sequences(seq_dict)
        # then classify the mappings
        filter_ids = self.filter_pafdict(paf_dict)
        # return the filtered sequence dict
        filt_dict = deepcopy(seq_dict)
        for fid in filter_ids:
            filt_dict.pop(fid, None)
        return filt_dict


    def filter_pafdict(self, paf_dict):
        # THIS IS MODELED AFTER SequenceAVA.load_ava()
        filter_ids = set()
        for rid, record_list in paf_dict.items():
            for rec in record_list:
                # check if this mapping passes the filters
                is_filtered = rec.filter(filters=self.filters)
                if is_filtered:
                    continue

                # classify the alignment
                rec.classify()

                if rec.c == 1:
                    # internal match
                    continue
                elif rec.c == 2:
                    # first contained
                    filter_ids.add(rec.qname)
                elif rec.c == 3:
                    # second contained
                    # if the repeat is contained within the query, we want that read!
                    continue
                elif rec.c in {4, 5}:
                    # overlaps[(f'{rec.qname}-{rec.qside}', f'{rec.tname}-{rec.tside}')] = rec
                    filter_ids.add(rec.qname)
                else:
                    pass
        logging.info(f"repeat filtering {len(filter_ids)} sequences from this batch")
        return filter_ids


    def update_library(self):
        # TODO maybe implement an aupdate to the repeat library
        pass


    def cluster_sequences(self):
        # TODO cluster together the repeats to decrease the library size
        pass