import logging
import os
import sys
from collections import defaultdict, Counter
from copy import deepcopy
from pathlib import Path
from shutil import copy

import numpy as np

from .aeons_paf import PafLine, Paf
from .aeons_utils import execute, find_exe, write_logs, empty_file, random_id, find_blocks_generic
from .aeons_polisher import Polisher
from .aeons_kmer import euclidean_dist, euclidean_threshold
from .aeons_mapper import Indexer
from .aeons_benefit import benefit_bins, calc_fragment_benefit, score_array



class Dependencies:
    def __init__(self):
        self.mm2 = find_exe("minimap2")
        # logging.info(f'mm2: {self.mm2}')
        if not self.mm2:
            sys.exit("Dependency minimap2 not found in path")
        self.paf2gfa = find_exe("paf2gfa")
        # logging.info(f'paf2gfa: {self.paf2gfa}')
        if not self.paf2gfa:
            sys.exit("Dependency paf2gfa (gfatools) not found in path")





class SequenceAVA:

    def __init__(self, paf, filters, tetra=False):
        self.paf = paf
        self.gfa = f'{paf}.gfa'
        self.filters = filters
        self.tetra = tetra    # whether to use the tetramer distance to filter overlaps
        # container to keep overlaps
        # save paf lines as qname-tname with alphanum sort
        self.links = defaultdict(lambda: defaultdict(PafLine))
        self.paf_links = f"{paf}.links.paf"

        self.dep = Dependencies()  # TODO


    def load_dependencies(self):
        self.dep = Dependencies()


    def load_ava(self, paf, seqpool):
        # load all entries from a paf file as PafLines
        # filter the entries while loading
        self.trims = []  # used for trimming
        self.overlaps = {}  # used for trimming
        containments = {}  # collect, used for coverage increments
        overlappers = set()  # used to track temperature of sequences
        ovl = 0
        inter = 0

        records, skip = Paf.parse_filter_classify_records(paf=paf, filters=self.filters)

        for rec in records:
            if rec.c == 2:
                # first contained
                containments[(rec.qname, rec.tname)] = rec
            elif rec.c == 3:
                # second contained
                containments[(rec.tname, rec.qname)] = rec
            elif rec.c in {4, 5}:
                # if applicable, check for tetramer dist
                if self.tetra:
                    intra = seqpool.is_intra(rec.qname, rec.tname)
                    if not intra:
                        inter += 1
                        continue

                # append the alignment to both the query and the target
                ovl += 1
                self.overlaps[(rec.qname, rec.tname)] = rec
                # TODO new
                if rec.tname in self.links[rec.qname].keys():
                    if rec.s1 < self.links[rec.qname][rec.tname].s1:
                        # we already have an overlap between the two
                        # and the previous recorded one had higher s1
                        continue

                self.links[rec.qname][rec.tname] = rec
                self.links[rec.tname][rec.qname] = rec
                overlappers.add(rec.qname)
                overlappers.add(rec.tname)
            elif rec.c == 6:
                self.trims.append(rec)
            else:
                pass

        contained_ids = set([i for (i, j) in containments.keys()])
        # remove contained ones from further containment search
        skip_filt = [s for s in skip if s.qname not in contained_ids and s.tname not in contained_ids]
        # run multiline containment detection
        mc = MultilineContainments(records=skip_filt)
        containments.update(mc.containments)

        logging.info(f"ava load: skip {len(skip)}, cont {len(contained_ids)} cont multi: {len(mc.containments)} ovl: {ovl} inter: {inter}")
        return containments, overlappers




    def remove_links(self, sequences):
        # remove overlaps of certain sequences
        for sid in sequences:
            # targets for overlaps where sid is "query"
            targets = self.links[sid].keys()
            # remove the overlaps where sid is "query"
            self.links.pop(sid, None)
            # remove overlaps from targets
            for t in targets:
                self.links[t].pop(sid, None)

    def remove_specific_link(self, s1, s2):
        # just remove one specific link from the overlaps
        self.links[s1].pop(s2, None)
        self.links[s2].pop(s1, None)


    def to_be_trimmed(self):
        # after classification use the records that need to be trimmed
        # and find the coordinates for trimming
        trim = self.trims
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



    def check_occupancy(self, avas, occupied):
        # check whether targets of some node are already occupied
        not_occ = {(tname, tside): rec for (tname, tside), rec in avas.items()
                   if (tname, tside) not in occupied}
        return not_occ



    def links2paf(self, paf_out):
        # write overlaps to paf file
        written = set()
        with open(paf_out, 'w') as fh:
            for node, target_dict in self.links.items():
                if not target_dict:
                    continue
                # go through all links
                for target, rec in target_dict.items():
                    if rec.line in written:
                        continue
                    else:
                        fh.write(rec.line)
                        written.add(rec.line)



    def paf2gfa_fpa(self, paf_in, gfa_out):
        # transform to GFA file for further processing
        if not os.path.getsize(paf_in):
            logging.info("no overlaps for merging")
            return False

        comm = f"fpa -i {paf_in} -o /dev/null gfa -o {gfa_out}"
        stdout, stderr = execute(comm)
        if stderr:
            logging.info(f"stderr: \n {stderr}")
        return True



    def paf2gfa_gfatools(self, paf, fa, gfa=None):
        # use gfatools to generate the graph and unitigs
        if not os.path.getsize(paf):
            logging.info("no overlaps for merging")
            return False

        comm = f"{self.dep.paf2gfa} -i {fa} -u -c {paf}"
        stdout, stderr = execute(comm)
        # writing is not necessary if we use stdout of process
        if gfa:
            with open(gfa, 'w') as fh:
                fh.write(stdout)
        # if stderr:
        #     logging.info(f"stderr: \n {stderr}")
        return stdout


    @staticmethod
    def source_union(edges0, edges1):
        # given some sets of edge dicts, return the union of source nodes
        sources0 = zip(*edges0.keys())
        sources1 = zip(*edges1.keys())
        try:
            set0 = set(list(sources0)[0])
        except IndexError:
            set0 = set()

        try:
            set1 = set(list(sources1)[0])
        except IndexError:
            set1 = set()

        source_union = set0 | set1
        return source_union




class Sequence:

    def __init__(self, header, seq, cov=None, merged_components=None, merged_atoms=None, cap_l=False, cap_r=False):
        self.header = header
        self.seq = seq

        if cov is None:
            self.cov = np.ones(shape=len(seq), dtype='float')
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
        # self.kmers = 0
        # temperature for ignoring reads
        # if temperature reaches 0, reads gets frozen
        self.temperature = 30
        self.cap_l = cap_l
        self.cap_r = cap_r



    def check_temperature(self):
        if self.temperature <= 0:
            return 0
        return 1


    def polish_sequence(self, read_sources):
        success = 0
        # don't polish raw reads, only derived sequences
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
        # don't use new seq if the length changed by a lot
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


    def chunk_up_coverage(self, n):
        # for generating strategies, we need to chunk the coverage
        cov = self.cov
        self.cov_chunked = np.array([np.sum(cov[i: i + n]) for i in range(0, len(cov), n)])
        # init an empty array to record nodes of interest
        self.noi = np.zeros(shape=self.cov_chunked.shape[0], dtype="bool")
        self.scores = np.zeros(shape=self.cov_chunked.shape[0], dtype="float")
        self.benefit = np.zeros(shape=self.cov_chunked.shape[0], dtype="float")


    def contig_scores(self, score_vec, n):
        sc = score_array(score_vec=score_vec, cov_arr=self.cov_chunked, node_size=n)
        self.scores = sc
        assert self.cov_chunked.shape[0] == self.scores.shape[0]


    def contig_benefits(self, mu, ccl, node_size):
        benefit, smu_sum = calc_fragment_benefit(
            scores=self.scores,
            mu=mu,
            approx_ccl=ccl,
            node_size=node_size,
            e1=self.noi[0],
            e2=self.noi[-1]
        )
        self.benefit = benefit
        self.smu_sum = smu_sum
        assert self.cov_chunked.shape[0] == self.benefit.shape[1]



    def set_contig_ends(self, n, lim=50):
        cc = self.cov_chunked
        if cc[0] > lim * n:
            pass
        elif self.cap_l:
            pass
        # only set as interesting if every check fails
        # for either model: mark as noi or give max score
        else:
            self.noi[0] = 1
            self.scores[0] = 1

        if cc[-1] > lim * n:
            pass
        elif self.cap_r:
            pass
        else:
            self.noi[-1] = 1
            self.scores[-1] = 1


    def find_low_cov(self, n, lim):
        # find where the coverage is too low
        cc = self.cov_chunked
        dropout_lim = find_dropout_threshold(cc)
        lc = np.where((cc > dropout_lim * n) & (cc < lim * n))[0]
        # filter single windows of low coverage
        # using the difference between adjacent indices of lowcov windows
        # EXCLUDING blocks of coverage 1
        lc_diff = np.diff(lc)
        lc_blocks = find_blocks_generic(lc_diff, 1, 3)
        lc_filt = set()
        for start, end in lc_blocks:
            lc_filt.update(lc[start: end + 1])
        lc_arr = np.array(list(lc_filt), dtype="int")
        n_unfilt = lc.shape[0]
        n_filt = lc_arr.shape[0]
        # multi-index to set the nois
        self.noi[lc_arr] = 1
        return n_unfilt, n_filt


    def find_strat(self, ccl, n):
        # cover X% of ccl
        n_steps = int(ccl[-3] / n)
        # spread the nodes of interest in both directions to get final binary arr
        fwd = roll_boolean_array(arr=self.noi.copy(), steps=n_steps, direction=0)
        rev = roll_boolean_array(arr=self.noi.copy(), steps=n_steps, direction=1)
        self.strat = np.column_stack((fwd, rev))
        return self.strat



    def find_strat_m0(self, threshold):
        strat = np.where(self.benefit >= threshold, True, False)
        return strat.transpose()






class SequencePool:

    def __init__(self, sequences=None, name="dummy", min_len=3000, out_dir="dummy", threads=24):
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

        self.dep = Dependencies()  # TODO



    def load_dependencies(self):
        self.dep = Dependencies()


    def headers(self):
        return set(self.sequences.keys())


    def seqdict(self):
        # convenience raw sequence dict
        return {header: seqo.seq for header, seqo in self.sequences.items()}


    def count_coverage(self):
        # check the coverage of all sequences in the pool
        cov_counts = np.zeros(shape=200)
        cov_means = np.zeros(shape=1000)
        i = 0
        for header, seqo in self.sequences.items():
            seqo_counts = np.bincount(seqo.cov.astype('int'))
            cov_counts[:seqo_counts.shape[0]] += seqo_counts
            seqo_mean = np.mean(seqo.cov)
            cov_means[i] = seqo_mean
            i += 1
            if header.startswith("SRR"):
                continue
            # print(header)
            # print(seqo_mean)
        cov_counts_t = np.trim_zeros(cov_counts, trim='b')
        cov_means_t = np.trim_zeros(cov_means, trim='b')
        return cov_counts_t, cov_means_t


    def ingest(self, seqs):
        # add a pile of sequences to the pool
        # can read from a dict, or add an existing pool
        if type(seqs) == dict:
            skipped = self._ingest_dict(seqs=seqs)
            logging.info(f"ingested: {len(seqs) - skipped} pool size: {len(self.sequences.keys())}")

        elif type(seqs) == SequencePool:
            self._ingest_pool(new_pool=seqs)
            logging.info(f"ingested: {len(seqs.sequences)} pool size: {len(self.sequences.keys())}")

        else:
            logging.info("seqs need to be dict, or SequencePool")



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
        if not self.dep.mm2:
            logging.info("could not find minimap2")

        comm = f'{self.dep.mm2} -x ava-ont -t{self.threads} {fa} {fa} >{paf}'
        if base_level:
            comm = f'{self.dep.mm2} -cx ava-ont -t{self.threads} {fa} {fa} >{paf}'
        import time
        tic = time.time()
        stdout, stderr = execute(comm)
        toc = time.time()
        logging.info(f"timing: {toc - tic}")
        if os.path.exists(f'{self.out_dir}/logs'):
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
        if not self.dep.mm2:
            logging.info("could not find minimap2")

        comm = f'{self.dep.mm2} -x ava-ont -t{self.threads} {new_fa} {new_fa} >{new_ava}'
        stdout, stderr = execute(comm)
        write_logs(stdout, stderr, f'{self.out_dir}/logs/ava_add')
        # mapping new sequences to previous pool
        comm = f'{self.dep.mm2} -x map-ont -w5 -e0 -m100 -r2k -t{self.threads} {self.fa} {new_fa} >{new_onto_pool}'
        stdout, stderr = execute(comm)
        write_logs(stdout, stderr, f'{self.out_dir}/logs/map2pool')
        # return filenames to be ingested as AVA
        return new_ava, new_onto_pool


    def remove_sequences(self, sequences):
        # given some ids, remove them from the readpool
        # e.g. after making new paths, we want to remove the sequences used to construct them
        # as soon as a read is used, it can not contribute to any other alignment
        popped = 0
        for sid in sequences:
            self.sequences.pop(sid, None)
            popped += 1
        logging.info(f'pool size: {len(self.sequences)}')
        logging.info(f'popped {popped}')




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
        # seq_dict = trimmed_pool.seqdict() | other_pool.seqdict()  # TODO
        seq_dict = dict(trimmed_pool.seqdict(), **other_pool.seqdict())
        return seq_dict


    def get_next_increment_edges(self, edges, previous_edges=None):
        # if no argument given, get the edges with in-degree of 0
        if not previous_edges:
            sources, targets = zip(*edges)
            next_sources = set(sources) - set(targets)
        # otherwise grab the edges starting at the previous targets
        else:
            next_sources = [t for (s, t) in previous_edges]
        # get the next edges to increment
        next_edges = {(s, t) for (s, t) in edges if s in next_sources}
        # remove the next edges
        for e in next_edges:
            edges.remove(e)
        return edges, next_edges


    def affect_increment(self, source, target, rec, edge_multiplicity):
        # relevant coordinates of this containment
        ostart, oend, olen, cstart, cend, clen = rec.grab_increment_coords()

        # grab the source coverage
        cont_cov = self.sequences[source].cov[cstart: cend].copy()

        # adjust length of coverage array
        if clen == olen:
            pass
        elif clen > olen:
            cont_cov = cont_cov[: olen]
        elif clen < olen:
            cont_cov = np.pad(cont_cov, (0, olen - clen), mode='edge')

        # account for reverse complement
        if rec.rev:
            cont_cov = cont_cov[::-1]
        else:
            pass

        # adjust for edge multiplicity
        cont_cov /= edge_multiplicity

        # add the source coverage to target coverage
        self.sequences[target].cov[ostart: oend] += cont_cov
        # limit coverage to 100 to prevent mess
        self.sequences[target].cov[np.where(self.sequences[target].cov > 100)] = 100

        # add the source as constituent of target
        if '*' not in source:
            self.sequences[target].atoms.add(source)


    def affect_increments(self, next_edges, containment, edge_multiplicity):
        # loop over the next set of edges to affect the collected increments
        for (source, target) in next_edges:
            rec = containment[(source, target)]
            # grab edge multiplicity modifier, in case of parallel edges
            em = edge_multiplicity[source]
            self.affect_increment(source, target, rec, em)


    @staticmethod
    def find_edge_multiplicity(edges):
        sources, targets = zip(*edges)
        source_counts = Counter(sources)
        return source_counts


    def increment(self, containment):
        # use the records of containment to increase the coverage counts & borders
        # containment = (contained, container) : rec
        edges = set(containment.keys())
        # if there are no increments to do
        if not edges:
            return []

        # debugging
        # import sys
        # sys.path.insert(0, "/home/lukas/Desktop/Aeons/code/plot")
        # from gt_plot import containment_graph
        # containment_graph(edges, 0)

        # get the first edges to increment, i.e. those with 0 in-degree
        edges, next_edges = self.get_next_increment_edges(edges, previous_edges=None)
        if not next_edges:
            return []
        edge_multiplicity = self.find_edge_multiplicity(next_edges)
        self.affect_increments(next_edges, containment, edge_multiplicity)
        previous_edges = next_edges

        while len(edges) > 0:
            # get the next edges to deal with
            edges, next_edges = self.get_next_increment_edges(edges, previous_edges=previous_edges)
            if not next_edges:
                return []
            edge_multiplicity = self.find_edge_multiplicity(next_edges)
            self.affect_increments(next_edges, containment, edge_multiplicity)

            # circular containment relationships can trap us here
            if len(next_edges) == len(previous_edges):
                break
            previous_edges = next_edges

        # for removal: return the ids of the contained sequences
        contained_ids = set([s for (s, t) in containment.keys()])
        return contained_ids


    def reset_temperature(self, sids, t=50):
        # give active reads a boost in temperature
        for s in sids:
            try:
                seqo = self.sequences[s]
                seqo.temperature = t
            except KeyError:
                pass


    def decrease_temperature(self, lim):
        # decrease the temperature of all reads by 1
        # if a read is longer than lim, we never ignore it
        frozen_seqs = set()
        for header, seqo in self.sequences.items():
            if len(seqo.seq) < lim:
                seqo.temperature -= 1
                hot = seqo.check_temperature()
                if not hot:
                    frozen_seqs.add(header)
        logging.info(f"frozen seqs: {len(frozen_seqs)}")
        return frozen_seqs


    def declare_contigs(self, min_contig_len):
        # collect a subdict of sequences that are longer than some limit
        contigs = {header: seqo for header, seqo in self.sequences.items() if len(seqo.seq) > min_contig_len}
        if not contigs:
            return False
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


    # TODO depr
    # @staticmethod
    # def contigs2gfa(gfa, contigs, node_size):
    #     # convert sequences to gfa with chunked up nodes
    #     # verify file exists and is empty
    #     empty_file(gfa)
    #     # init node counter
    #     node = 0
    #     n = node_size
    #     # check if there are any contigs
    #     if not contigs:
    #         return
    #
    #     # record the source and position of the nodes
    #     node_sources = dict()
    #     node_positions = dict()
    #
    #     # loop through contigs
    #     for header, seqo in contigs.sequences.items():
    #         # translate each sequence into gfa with fixed node size
    #         # number of first seg in this longer segment
    #         node_init = deepcopy(node)
    #         # write a single sequence to gfa file
    #         # chunk it up into nodes
    #         seq = seqo.seq
    #         cpos = 0
    #         seq_chunks = [seq[i: i + n] for i in range(0, len(seq), n)]
    #         # also chunk up the coverage of this segment
    #         cov = seqo.cov.copy()
    #         cov_chunks = [np.sum(cov[i: i + n]) for i in range(0, len(cov), n)]
    #
    #         with open(gfa, 'a') as gfa_file:
    #             for c_idx in range(len(seq_chunks)):
    #                 gfa_file.write(f'S\t{node}\t{seq_chunks[c_idx]}\tc:i:{cov_chunks[c_idx]}\n')
    #                 node_sources[node] = header
    #                 node_positions[node] = (cpos, cpos + n - 1)
    #                 node += 1
    #                 cpos += n
    #
    #             for i in range(node_init, node - 1):
    #                 gfa_file.write(f'L\t{i}\t+\t{i + 1}\t+\t0M\n')
    #                 gfa_file.write(f'L\t{i + 1}\t-\t{i}\t-\t0M\n')
    #
    #     return node_sources, node_positions




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



    @staticmethod
    def load_unitigs(gfa):
        # load unitigs after graph cleaning with gfatools
        # read either a gfa file or a string
        # split the lines
        # this makes sure that the unitigs are split
        # and that we can filter the link lines
        # first check if there are any unitigs
        if not gfa:
            return []

        if gfa.startswith('S\t'):
            gfat = gfa
        else:
            with open(gfa, 'r') as fh:
                gfat = fh.read()
        # string operations
        gfatt = gfat.split('\nS\t')
        # first and last need extra treatment
        gfatt[0] = gfatt[0][2:]  # skip the first "S\t"
        gfatt[-1] = gfatt[-1].replace('\nx', '\nL').split('\nL')[0]  # exclude all L and x lines

        # get the x lines
        gfax = gfat.split('\nx\t')[1:]
        # make sure we have the same number of S_A lines and x lines
        assert len(gfatt) == len(gfax)

        # parse into unitig objects
        unitigs = []
        for sa_lines, x_line in zip(gfatt, gfax):
            unitigs.append(Unitig(sa_lines.split('\n'), x_line))
        return unitigs



    def is_intra(self, seq1, seq2):
        # takes two sequence names and runs euclidean distance of tetramers
        # this is to check whether the sequences would be classified as intraspecific
        s1 = self.sequences[seq1]
        s2 = self.sequences[seq2]
        euc = euclidean_dist(s1, s2)
        return euc < euclidean_threshold


class ContigPool(SequencePool):

    """
    a SequencePool specifically for contigs
    - simply initialise as ContigPool(sequences=contigs)
    - can do some additional things necessary for contigs
    """

    def process_contigs(self, node_size, lim, ccl, out_dir, batch, write=False):
        # WRAPPER
        logging.info("finding new strategies.. ")
        # chunk up contigs
        logging.info("chunking contigs")
        self._chunk_up_contigs(node_size=node_size)
        # process ends and low cov regions
        logging.info("processing ends")
        self._process_contig_ends(node_size=node_size)
        logging.info("processing low cov nodes")
        self._process_low_cov_nodes(node_size=node_size, lim=lim)
        # find and write new strategies
        logging.info("finding strategies")
        contig_strats = self._find_contig_strategies(node_size=node_size, ccl=ccl)
        if write:
            logging.info("writing new strategies")
            self._write_contig_strategies(out_dir=out_dir, contig_strats=contig_strats)
            self._write_index_file(out_dir=out_dir, batch=batch)
        return contig_strats


    def process_contigs_m0(self, score_vec, node_size, ccl, out_dir, mu, lam, batch, write=False):
        # WRAPPER
        logging.info("finding new strategies.. ")
        # chunk up contigs
        logging.info("chunking contigs")
        self._chunk_up_contigs(node_size=node_size)
        # find scores
        self._contigs_scores(score_vec=score_vec, node_size=node_size)
        # process ends and low cov regions
        self._process_contig_ends(node_size=node_size)
        # find benefit
        self._contigs_benefits(ccl=ccl, mu=mu, node_size=node_size)
        # find threshold
        t = self.find_threshold(mu=mu, lam=lam, node_size=node_size)
        # find and write new strategies
        logging.info("finding strategies - m0")
        contig_strats = self._find_contig_strategies(node_size=node_size, ccl=ccl, t=t, m0=True)
        if write:
            logging.info("writing new strategies")
            self._write_contig_strategies(out_dir=out_dir, contig_strats=contig_strats)
            self._write_index_file(out_dir=out_dir, batch=batch)
        return contig_strats


    def _chunk_up_contigs(self, node_size):
        # first thing to do for contigs
        # for a collection of contigs, give them a chunked representation
        n_comp = 0
        lengths = []
        for header, seqo in self.sequences.items():
            seqo.chunk_up_coverage(n=node_size)
            n_comp += 1
            lengths.append(len(seqo.seq))
        # report some info
        lengths_sort = np.sort(lengths)[::-1]
        logging.info(f'num components: {n_comp}')
        logging.info(f'total comp length: {lengths_sort.sum()}')
        logging.info(f'longest components: {lengths_sort[:10]}')


    def _contigs_scores(self, score_vec, node_size):
        # get the scores for each contig
        for header, seqo in self.sequences.items():
            seqo.contig_scores(score_vec=score_vec, n=node_size)



    def _contigs_benefits(self, ccl, mu, node_size):
        # get the benefit for each contig
        for header, seqo in self.sequences.items():
            seqo.contig_benefits(mu=mu, ccl=ccl, node_size=node_size)


    def find_threshold(self, mu, lam, node_size):
        # flatten all benefit values
        benefit = np.column_stack([seqo.benefit for seqo in self.sequences.values()]).ravel()
        smu_sum = np.sum([seqo.smu_sum for seqo in self.sequences.values()])

        # find acceptance threshold
        alpha = 200 // node_size
        rho = 300 // node_size
        tc = (lam - mu - 300) // node_size

        benefit_bin, counts = benefit_bins(benefit)

        # average benefit of strategy in the case that all fragments are rejected
        ubar0 = smu_sum
        tbar0 = alpha + rho + (mu // node_size)
        # cumsum of the benefit (bins multiplied by how many sites are in the bin)
        cs_u = np.cumsum(benefit_bin * counts) + ubar0
        cs_t = np.cumsum(tc * counts) + tbar0
        peak = cs_u / cs_t
        strat_size = np.argmax(peak) + 1
        # plt.plot(cs_u)
        # plt.plot(cs_t)
        # plt.plot(peak)
        # plt.show()

        # calculate threshold exponent and where values are geq
        try:
            threshold = benefit_bin[strat_size]
        except IndexError:
            threshold = benefit_bin[-1]

        return threshold




    def _process_contig_ends(self, node_size):
        # set contig ends so that they are picked up by the strategy
        for header, seqo in self.sequences.items():
            seqo.set_contig_ends(n=node_size)


    def _process_low_cov_nodes(self, node_size, lim):
        # find low coverage nodes that we want to target
        unfilt = 0
        filt = 0
        for header, seqo in self.sequences.items():
            n_unfilt, n_filt = seqo.find_low_cov(n=node_size, lim=lim)
            unfilt += n_unfilt
            filt += n_filt
        logging.info(f'low coverage nodes: {unfilt}, after filtering: {filt}')


    def _find_contig_strategies(self, node_size, ccl, t=0, m0=False):
        # find the boolean strategy for each contig
        contig_strats = {}
        for header, seqo in self.sequences.items():
            if not m0:
                cstrat = seqo.find_strat(ccl=ccl, n=node_size)
            else:
                cstrat = seqo.find_strat_m0(threshold=t)
            contig_strats[header] = cstrat
        return contig_strats


    def _write_contig_strategies(self, out_dir, contig_strats):
        # write the strategies for all contigs to a single file
        cpath = f'{out_dir}/masks/aeons'
        np.savez(cpath, **contig_strats)
        # container = np.load(f'{cpath}.npz')
        # data = {key: container[key] for key in container}
        # place a marker that the strategies were updated
        markerfile = f'{out_dir}/masks/masks.updated'
        Path(markerfile).touch()



    def _write_index_file(self, out_dir, batch):
        # write new index file to map against
        # and place marker file to tell readfish to reload
        fa_path = f'{out_dir}/contigs/aeons.fa'
        mmi_path = f'{out_dir}/contigs/aeons.mmi'
        # save the contigs to fasta
        with open(fa_path, 'w') as fasta:
            for sid, seqo in self.sequences.items():
                fasta.write(f'>{sid}\n')
                fasta.write(f'{seqo.seq}\n')
        # generate and save index with mappy
        Indexer(fasta=fa_path, mmi=mmi_path)
        # copy previous contigs
        if batch % 10 == 0:
            copy(fa_path, f'{out_dir}/contigs/prev/aeons_{batch}.fa')
        # place a marker that the contigs were updated
        markerfile = f'{out_dir}/contigs/contigs.updated'
        Path(markerfile).touch()


class Unitig:

    def __init__(self, sa_line_list, x_line):
        # takes a list of lines from a gfa
        # i.e. the S & A lines corresponding to a single unitig
        self.name = random_id()
        self._format_sa(sa_line_list)
        self._format_x(x_line)
        # dummy init
        self.cov = None


    def _format_x(self, x_line):
        xl = x_line.split("\t")
        assert xl[0].startswith('utg')
        try:
            cl = int(xl[3])
            cr = int(xl[4])
        except IndexError:
            self.cap_l = False
            self.cap_r = False
            return

        self.cap_l = True if cl > 0 else False
        self.cap_r = True if cr > 0 else False



    def _format_sa(self, line_list):
        self._format_sline(sline=line_list[0])
        self._format_atoms(alines=line_list[1:])


    def _format_sline(self, sline):
        sls = sline.split('\t')
        self.seq = sls[1]
        self.length = int(sls[2].split(':')[-1])
        circ = sls[0][-1]
        self.circ = True if circ == 'c' else False
        # self.n_atoms = int(sls[3].split(':')[-1])
        assert sls[0].startswith('utg')
        assert self.length == len(self.seq)


    def _format_atoms(self, alines):
        atoms = []
        for line in alines:
            assert line.startswith('A')
            atom = {}
            al = line.split('\t')
            atom['pos'] = int(al[2])
            atom['strand'] = al[3]
            if atom['strand'] == '-':
                atom['rev'] = 1
            elif atom['strand'] == '+':
                atom['rev'] = 0
            else:
                print(line)
                print(al)
                print(atom)
                logging.info("wrong strand spec of unitig")
                exit(1)
            atom['name'] = al[4]
            atoms.append(atom)
        # loop a second time to get the ends
        cpos = 0
        for i in range(len(alines) - 1):
            line = alines[i + 1]
            al = line.split('\t')
            pos = int(al[2])
            to_add = pos - cpos
            atoms[i]['n'] = to_add
            cpos = pos
        # mark last atom
        atoms[-1]['n'] = -1

        self.atoms = atoms
        self.atom_headers = [a['name'] for a in atoms]


    def to_seqo(self, seqpool):
        # transform the unitig to sequence object
        # this can only be done after merging coverage array
        # check if that has been done
        assert self.cov is not None
        # grab atoms of atoms
        merged_atoms = seqpool.get_atoms(headers=self.atom_headers)
        merged_components = seqpool.get_components(headers=self.atom_headers)
        seqo = Sequence(header=self.name, seq=self.seq, cov=self.cov,
                        merged_components=merged_components, merged_atoms=merged_atoms,
                        cap_l=self.cap_l, cap_r=self.cap_r)
        return seqo



class UnitigPool:

    def __init__(self, unitigs):
        self.unitigs = unitigs


    def get_unitig_coverage_arrays(self, seqpool):
        # for each of the unitigs, perform the cov array merging
        for u in self.unitigs:
            cm = CoverageMerger(u, seqpool)
            cov_arr = cm.cov_arr
            u.cov = cov_arr


    def unitigs2seqpool(self, seqpool, min_seq_len):
        # transform all unitigs to seq objects
        # and get the read ids to remove
        seqos = {}
        used_sids = set()
        for u in self.unitigs:
            unitig_seqo = u.to_seqo(seqpool)
            seqos[u.name] = unitig_seqo
            used_sids.update(u.atom_headers)
        # construct a new pool
        new_pool = SequencePool(sequences=seqos, min_len=min_seq_len)
        return new_pool, used_sids



class CoverageMerger:

    def __init__(self, unitig, seqpool):
        # create the merged coverage array for an unitig
        self.unitig = unitig
        cov_arr = self._create_merged_arr(atoms=unitig.atoms, seqpool=seqpool)
        assert unitig.length == cov_arr.shape[0]
        self.cov_arr = cov_arr

    def _create_merged_arr(self, atoms, seqpool):
        arr_parts = []
        cpos = 0
        for a in atoms:
            assert a['pos'] >= cpos
            name = a['name']
            atom_arr = seqpool[name].cov.copy()
            # TODO not sure if reverse first or trim first
            atom_arr = atom_arr[::-1] if a['rev'] else atom_arr
            if not a['n'] == -1:
                atom_arr = atom_arr[: a['n']]  # TODO maybe + 1?
            else:
                # add last atom (goes until end of atom)
                if not self.unitig.circ:
                    atom_arr = atom_arr
                else:
                    diff = self.unitig.length - a['pos']
                    atom_arr = atom_arr[:diff]
            arr_parts.append(atom_arr)
            cpos = a['pos']
        return np.concatenate(arr_parts)




class MultilineContainments:


    def __init__(self, records):
        # records is a list of PafLine objects
        self.records = records
        # collect all partners with multiple mappings
        multidict = self._fill_multidict()
        # get a dict of (contained, container): [recs]
        containments = self._get_multiline_containments(multidict=multidict)
        self.containments = containments


    def _fill_multidict(self):
        multidict = defaultdict(list)
        for rec in self.records:
            multidict[rec.keygen()].append(rec)
        multidict = {k: recs for k, recs in multidict.items() if len(recs) > 1}
        return multidict


    def _get_multiline_containments(self, multidict):
        # check for containment from multiple internal match mappings
        containments = {}

        for k, recs in multidict.items():
            cont = self.multiline_containment(recs)
            if cont:
                containments.update(cont)
        return containments


    @staticmethod
    def multiline_containment(records, n=100):
        qlen = records[0].qlen // n
        tlen = records[0].tlen // n
        qarr = np.zeros(shape=qlen , dtype="bool")
        tarr = np.zeros(shape=tlen , dtype="bool")

        if len(records) > 10:
            return False

        for r in records:
            qarr[r.qstart // n: r.qend // n] = 1
            tarr[r.tstart // n: r.tend // n] = 1

        # if more than 0.9 are covered by mappings
        if sum(qarr) > qlen * 0.9:
            # if overhang is smaller than 0.15 of len
            q_low, q_high = np.nonzero(qarr)[0][[0, -1]]
            if (q_high - q_low) > qlen * 0.85:
                # return containment tuple
                t_low, t_high = np.nonzero(tarr)[0][[0, -1]]
                ctd_low, ctd_high, ctr_low, ctr_high = q_low, q_high, t_low, t_high
                cont = MultilineContainments.generate_paf_cont(
                    records, 'q', 't', ctd_low, ctd_high, ctr_low, ctr_high, n
                )
                return cont

        if sum(tarr) > tlen * 0.9:
            t_low, t_high = np.nonzero(tarr)[0][[0, -1]]
            if (t_high - t_low) > tlen * 0.85:
                q_low, q_high = np.nonzero(qarr)[0][[0, -1]]
                ctd_low, ctd_high, ctr_low, ctr_high = t_low, t_high, q_low, q_high
                cont = MultilineContainments.generate_paf_cont(
                    records, 't', 'q', ctd_low, ctd_high, ctr_low, ctr_high, n
                )
                return cont

        # if neither q nor t are contained
        return False


    @staticmethod
    def generate_paf_cont(records, ctd, ctr, ctd_low, ctd_high, ctr_low, ctr_high, n):
        # given a list of records that make up a multic
        # generate some paf that describes the containment
        # so that we can use it for incrementing
        ctd_name = getattr(records[0], f'{ctd}name')
        ctr_name = getattr(records[0], f'{ctr}name')
        ctd_len = getattr(records[0], f'{ctd}len')
        ctr_len = getattr(records[0], f'{ctr}len')
        # use maximum span
        ctd_span = ctd_high - ctd_low
        ctr_span = ctr_high - ctr_low
        # except if span on container longer than 2.2x of contained
        if ctr_span > 2.2 * ctd_span:
            r = 0
            maplen = 0
            for i in range(len(records)):
                # then use the longest alignment between the two
                if records[i].maplen > maplen:
                    maplen = records[i].maplen
                    r = i
            ctr_low = getattr(records[r], f'{ctr}start') // n
            ctr_high = getattr(records[r], f'{ctr}end') // n
        # generate paf entry
        paf = f'{ctd_name}\t{ctd_len}\t{ctd_low * n}\t{ctd_high * n}\t+' \
              f'\t{ctr_name}\t{ctr_len}\t{ctr_low * n}\t{ctr_high * n}\t0\t0\t0'
        rec = PafLine(paf)
        # mark as first contained
        rec.c = 2
        return {(ctd_name, ctr_name): rec}



def roll_boolean_array(arr, steps, direction):
    assert arr.dtype == "bool"
    # rolling direction is opposite of input
    if direction == 0:
        d = -1
    elif direction == 1:
        d = 1
    else:
        raise ValueError("direction must be in {0, 1}")
    # roll array to spread truthy values
    for i in range(steps):
        arr += np.roll(arr, d)
    return arr




def find_dropout_threshold(coverage, mod=800):
    '''
    If there are sites that have not had much coverage after some time
    we don't expect them to gain much after that and want to ignore them

    Parameters
    ----------
    coverage: np.array
        coverage depth at each position
    mod: int
        threshold modifier

    Returns
    -------
    dropout: np.array
        array of positions that will most likely not get much more coverage

    '''
    cov_mean = np.mean(coverage)
    # ignore threshold is dependent on mean coverage
    threshold = int(cov_mean / mod)
    # dropout = np.where(coverage <= threshold)[0]
    return threshold

