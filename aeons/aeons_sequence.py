import logging
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np

from .aeons_paf import PafLine
from .aeons_utils import execute, find_exe, write_logs, empty_file
from .aeons_polisher import Polisher
from .aeons_kmer import euclidean_dist, euclidean_threshold


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
                    # grab next target - this requires that there is only 1 overlap
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


    def ava2gt(self):
        # indexing ava_dict returns node, dict for each end
        edges = []
        processed_edges = set()
        for node, edge_dict in self.ava_dict.items():
            for side, avas in edge_dict.items():
                # no overlaps on the end of this node
                if not avas:
                    continue

                for (tnode, tside), _ in avas.items():
                    source = f'{node}-{side}'
                    target = f'{tnode}-{tside}'
                    edge = (source, target)
                    edge_r = (target, source)
                    if edge in processed_edges:
                        continue
                    elif edge_r in processed_edges:
                        continue
                    else:
                        edges.append(edge)
                        processed_edges.add(edge)
                        processed_edges.add(edge_r)
        return edges


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

