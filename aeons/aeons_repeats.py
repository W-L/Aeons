from copy import deepcopy
import logging
import numpy as np
from collections import defaultdict

from .aeons_mapper import LinearMapper


class Repeat:

    def __init__(self, rid=None, side=None):
        self.rid = rid
        self.side = side
        self.starts = []
        self.ends = []
        self.start = 0
        self.end = 0


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
    def __init__(self, name, ava_dict, seqpool):
        self.library = f'{name}.repeat_lib.fa'
        # collect repeat overlaps in the given ava_dict
        repeats = self.collect_repeats(ava_dict)
        # write out the fasta file
        self.write_library(repeats, seqpool)
        # initialise a LinearMapper object
        self.lm = LinearMapper(ref=self.library)
        # get the ids of the affected reads
        self.affected_sids = self.filter_construction_seqs(repeats)



    def collect_repeats(self, ava_dict, lim=4):
        n_edges = []
        repeats = defaultdict(Repeat)
        # indexing ava_dict returns node, dict for each end
        for node, edge_dict in ava_dict.items():
            for side, avas in edge_dict.items():
                # only nodes with more than 3 edges are considered
                n = len(avas)
                n_edges.append(n)
                if n < lim:
                    continue
                # loop through edges from the query end
                for (tnode, tside), pafline in avas.items():
                    # grab the repeats at the query and target sides
                    qr = repeats[f'{node}-{side}']
                    tr = repeats[f'{tnode}-{tside}']
                    qr.rid, qr.side = node, side
                    tr.rid, tr.side = tnode, tside
                    # extract the ranges of this overlap
                    qstart, qend, tstart, tend = pafline.get_ranges()
                    qr.starts.append(qstart)
                    qr.ends.append(qend)
                    tr.starts.append(tstart)
                    tr.ends.append(tend)
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
        # first map them to the library
        paf_dict = self.lm.map_sequences(seq_dict)
        # then classify the mappings
        overlaps, containments, filter_ids = self.filter_pafdict(paf_dict)
        # return the filtered sequence dict
        filt_dict = deepcopy(seq_dict)
        for fid in filter_ids:
            filt_dict.pop(fid, None)
        return filt_dict


    def filter_pafdict(self, paf_dict):
        # THIS IS MODELED AFTER SequenceAVA.load_ava()
        overlaps = {}
        containments = defaultdict(list)
        filter_ids = set()
        for rid, record_list in paf_dict.items():
            for rec in record_list:
                # classify the alignment
                rec.classify()

                if rec.c == 0:
                    # short or self-alignment
                    continue
                if rec.c == 1:
                    # internal match
                    continue
                elif rec.c == 2:
                    # first contained
                    containments[rec.qname].append(rec)
                    filter_ids.add(rec.qname)
                elif rec.c == 3:
                    # second contained
                    # and qprox? should be overlap then?
                    # containments[rec.tname].append(rec)
                    continue
                elif rec.c in {4, 5}:
                    overlaps[(f'{rec.qname}-{rec.qside}', f'{rec.tname}-{rec.tside}')] = rec
                    filter_ids.add(rec.qname)
                else:
                    pass
        logging.info(f"repeat filtering {len(filter_ids)} sequences from this batch")
        return overlaps, containments, filter_ids


    def update_library(self):
        # TODO maybe implement an aupdate to the repeat library
        pass

