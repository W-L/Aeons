# CUSTOM
from .aeons_const import Constants, Filters
from .aeons_utils import execute, random_id, empty_file,\
    init_logger, read_fa, readfq, MyArgumentParser # , spawn, redotable
from .aeons_sampler import FastqStream_mmap # , OverlapSampler
from .aeons_readlengthdist import ReadlengthDist
from .aeons_paf import Paf, choose_best_mapper
from .aeons_mapper import LinearMapper # , ValidationMapping
from .aeons_sequence import SequencePool, SequenceAVA, UnitigPool, ContigPool
from .aeons_benefit import init_scoring_vec
from .aeons_repeats import RepeatFilter2


# STANDARD LIBRARY
import os
import gzip
import sys
import glob
import time
import logging
import re
from sys import exit
from collections import defaultdict
from pathlib import Path
from io import StringIO


# NON STANDARD LIBRARY
import pandas as pd
import numpy as np
import toml

from minknow_api.manager import Manager
from minknow_api import __version__ as minknow_api_version



# TODO tmp imports
# import matplotlib.pyplot as plt
# backend for interactive plots
# plt.switch_backend("GTK3cairo")
# plt.switch_backend("Qt5cairo")
# import line_profiler
# import memory_profiler
# import gt_plot






class AeonsRun:

    def __init__(self, args):
        # initialise constants and put into arguments
        # constants overwrite args with same name
        const = Constants()
        for c, cval in const.__dict__.items():
            setattr(args, c, cval)

        self.args = args
        self.name = args.name

        self.read_sources = dict()
        # initial strategy is to accept
        self.strat = 1
        # for plotting afterwards, we keep a list of rows
        self.metrics = defaultdict(list)
        self.metrics_sep = defaultdict(list)

        # make sure the run name does not have any spaces
        assert ' ' not in args.name

        args.out_dir = f'./out_{args.name}'
        if not os.path.exists(args.out_dir):
            os.mkdir(args.out_dir)
            os.mkdir(f'{args.out_dir}/masks')
            os.mkdir(f'{args.out_dir}/fq')
            os.mkdir(f'{args.out_dir}/logs')
            os.mkdir(f'{args.out_dir}/contigs')

        # initialise a log file in the output folder
        init_logger(logfile=f'{args.out_dir}/{args.name}.aeons.log', args=args)

        self.filt = Filters()
        self.pool = SequencePool(name=args.name, min_len=self.filt.min_seq_len, out_dir=args.out_dir)
        self.ava = SequenceAVA(paf=f'{args.name}.ava', tetra=args.tetra, filters=self.filt)
        # load dependencies for the Pool and AVA
        self.pool.load_dependencies()
        self.ava.load_dependencies()
        self.rl_dist = ReadlengthDist(mu=args.mu)
        # init scoring func
        self.score_vec = init_scoring_vec(lowcov=args.lowcov)


        if not args.live:
            # for keeping track of the sequencing time
            self.time_naive = 0
            self.time_readfish = 0
            self.time_aeons = 0
            # for writing reads to file
            self.cache_naive = dict()
            self.cache_readfish = dict()
            self.cache_aeons = dict()
            # after how much time should sequences be written to file
            # dump time is incremented every time a batch is written, which happens once that is overcome
            self.dump_every = args.dumptime
            self.dump_number_naive = 1
            self.dump_number_readfish = 1
            self.dump_number_aeons = 1
            self.dump_time = self.dump_every
            # INITS FOR SIM ONLY
            # for storing the batches of reads for snakesembly
            if not os.path.exists('./00_reads'):
                os.mkdir('./00_reads')
            empty_file(f'00_reads/{self.args.name}_0_naive.fa')
            empty_file(f'00_reads/{self.args.name}_0_readfish.fa')
            empty_file(f'00_reads/{self.args.name}_0_aeons.fa')

            # init fastq stream - continous blocks for local (i.e. for reading from usb)
            self.stream = FastqStream_mmap(source=self.args.fq, batchsize=self.args.bsize,
                                           maxbatch=self.args.maxb, seed=self.args.seed)
            # self.stream = FastqStream(source=self.args.fq, bsize=self.args.bsize,
            #                           seed=self.args.seed, workers=self.args.workers)

            # if self.args.ovl:
            #     sampler_paf = f'{self.args.fq}.paf'
            #     self.overlapDB = OverlapSampler(paf=sampler_paf)
            #     self.sample_pool = SequencePool(name=f'{args.name}.sample', min_len=self.filt.min_seq_len, out_dir=f'{args.out_dir}_sample')
            #     self.sample_pool.load_dependencies()

            # load a mapper for some reference
            # used to validate mergers in testing
            # if args.ref:
            #     self.reference_mapper = LinearMapper(ref=args.ref)

            if args.preload:
                self.load_init_contigs(preload=self.args.preload)

                # initialise strategy for readfish
                self.readfish = Readfish(seqpool=self.pool, ref_path=args.preload, mu=self.args.mu, node_size=self.args.node_size)
            else:
                self.readfish = None


            # load some initial batches
            if self.args.binit:
                self.load_init_batches(binit=self.args.binit)
            # if binit is set to 0, we calculate how many batches it takes to cover the genome x times
            else:
                binit = self.wait_for_batches(bsize=self.args.bsize, cov=self.args.cov_wait, gsize=self.args.gsize)
                logging.info(f"loading {binit} batches...")
                self.load_init_batches(binit=binit)
            # increment time after preloading
            self.update_times(read_sequences=self.pool.seqdict(),
                              reads_decision=self.pool.seqdict(),
                              reads_decision_readfish=self.pool.seqdict())


            # set the batch counter for the run
            self.batch = self.stream.batch

            # fill the initial AVA
            self.prep_first_ava()

            # initialise a RepeatFilter from first AVA
            if self.args.filter_repeats:
                self.repeat_filter = RepeatFilter2(name=args.name, seqpool=self.pool)

            # create first asm
            self.assemble_add_and_filter_contigs()

            # cov_counts, cov_means = self.pool.count_coverage()


        else:
            # LIVE RUN INITIALISATION
            # initialisation waits until X files are available
            self.init_live(args=args)

            contigs = False
            new_fastq = []
            while not contigs:
                # waiting until we have some data
                logging.info("no contigs yet. Waiting for data ... ")
                new_fastq = []
                while len(new_fastq) < 2:
                    logging.info("waiting for data ... ")
                    time.sleep(5)
                    new_fastq = LiveRun.scan_dir(fq=self.args.fq, processed_files=set())
                # transform files into batch
                fq_batch = FastqBatch(fq_files=new_fastq, channels=self.channels)
                # reset these in case the loop has to run
                self.pool = SequencePool(name=args.name, min_len=self.filt.min_seq_len, out_dir=args.out_dir)
                self.ava = SequenceAVA(paf=f'{args.name}.ava', tetra=args.tetra, filters=self.filt)
                self.pool.ingest(seqs=fq_batch.read_sequences)
                # fill the initial AVA
                self.prep_first_ava()
                # create first asm
                contigs = self.assemble_add_and_filter_contigs()
            # once there are contigs, record used files
            self.processed_files = set()
            self.processed_files.update(new_fastq)
            self.n_fastq = len(new_fastq)
            logging.info("Initial asm completed\n\n")



    def init_live(self, args):
        # initialise seq device dependent things
        # - find the output path where the fastq files are placed
        # - and where the channels toml is if we need that
        self.processed_files = set()
        self.batch = 0

        # connect to sequencing machine and grab the output directory
        # allows also to specify at cl for debugging
        if not self.args.fq:
            out_path = LiveRun.connect_sequencer(device=args.device, host=args.host, port=args.port)
            self.args.fq = f'{out_path}/fastq_pass'
        else:
            # DEBUGGING
            out_path = self.args.fq

        # grab channels of the condition - irrelevant if not splitting flowcell
        if args.split_flowcell:
            channels = LiveRun.split_flowcell(out_path=out_path, run_name=args.name)
        else:
            # if we use a whole flowcell, use all channels
            channels = set(np.arange(1, 512 + 1))
        self.channels = channels






    def wait_for_batches(self, bsize, gsize=12e6, cov=2):
        # how many batches of reads do we need to wait for until the estimated genome size is
        # covered ~cov times?
        read_lengths = self.stream.prefetch()
        self.rl_dist.update(read_lengths=read_lengths, recalc=True)
        mean_rld = self.rl_dist.lam
        x = (cov * gsize) / bsize / mean_rld
        return int(np.ceil(x))


    def load_init_batches(self, binit):
        # this is to load several batches from which to make an initial assembly
        for i in range(binit):
            self.stream.read_batch()
            self.pool.ingest(seqs=self.stream.read_sequences)
            # save the source file name for the reads
            for header in self.stream.read_sequences.keys():
                self.read_sources[header] = self.stream.source  # TODO live version


    def load_init_contigs(self, preload):
        # this is to load already built contigs
        # either finished or in construction
        prebuilt = dict()
        with open(preload, 'r') as contigs:
            for header, seq in read_fa(contigs):
                header = str(header)
                header = header.replace('>', '')
                # header = header.replace('_', '-')  # TODO why did we do this?
                prebuilt[header] = seq

        # save the source file name for the contigs (for polishing)
        for header in prebuilt.keys():
            self.read_sources[header] = preload

        contig_pool = SequencePool(sequences=prebuilt, min_len=self.filt.min_seq_len)
        # preload also some coverage, if we trust these contigs already
        # i.e. if we want to focus on component ends, not on covering everything all over again
        oz = 10000  # allow overlap zone of this size
        for header, seqo in contig_pool.sequences.items():
            seqo.cov[oz: -oz] = self.args.preload_cov
        # add to general pool
        self.pool.ingest(seqs=contig_pool)


    def prep_first_ava(self):
        # write out the current readpool & run complete all versus all
        logging.info("running first AVA")
        paf = self.pool.run_ava(sequences=self.pool.seqdict(), fa=self.pool.fa, paf=self.pool.ava)
        # load paf into ava object - includes filtering
        containments, ovl = self.ava.load_ava(paf=paf, seqpool=self.pool)
        contained_ids = self.pool.increment(containment=containments)
        self.remove_seqs(sequences=contained_ids)
        self.pool.reset_temperature(ovl)




    def assemble_unitigs(self):
        # WRAPPER
        # - write current links and pool sequences
        # - transform to gfa, do some cleaning and merge unitigs
        # - parse unitigs & coverage arrays
        # - transform unitigs to sequence pool and remove used sequences
        # write current links to paf
        self.ava.links2paf(paf_out=self.ava.paf_links)
        # write pool to file
        SequencePool.write_seq_dict(seq_dict=self.pool.seqdict(), file=self.pool.fa)
        # create gfa and unitigs
        gfa = self.ava.paf2gfa_gfatools(paf=self.ava.paf_links, fa=self.pool.fa)
        # load the new unitigs
        unitigs = SequencePool.load_unitigs(gfa=gfa)
        # put them into a collection
        unitig_pool = UnitigPool(unitigs)
        # get the coverage arrays
        unitig_pool.get_unitig_coverage_arrays(seqpool=self.pool.sequences)
        # transform into a sequence pool
        new_pool, used_sids = unitig_pool.unitigs2seqpool(
            seqpool=self.pool, min_seq_len=self.filt.min_seq_len)
        self.remove_seqs(used_sids)
        return new_pool


    def assemble_add_and_filter_contigs(self):
        # WRAPPER
        # - assemble the current graph and extract new unitigs
        # - add them to the seqpool
        # - extract the large contigs for mapping against
        logging.info("assembling new unitigs.. ")
        new_pool = self.assemble_unitigs()
        # add new sequences to the dict and to the ava
        logging.info("loading and overlapping new unitigs.. ")
        self.add_new_sequences(sequences=new_pool, increment=False)
        # write the current pool to file for mapping against
        logging.info("finding contigs to map against.. ")
        contigs = self.pool.declare_contigs(min_contig_len=self.filt.min_contig_len)
        if not contigs:
            return False
        SequencePool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)
        return contigs



    def make_decision_paf(self, paf_out, read_sequences, strat):
        # decide accept/reject for each read
        # first, transform paf output into dictionary
        # filtering here is for alignment_block_length, not sequence length
        # i.e. at least half of the initial mu-sized fragment has to map
        paf_dict = Paf.parse_PAF(StringIO(paf_out), min_len=int(self.args.mu / 2))

        # if nothing mapped, just return. Unmapped = accept
        if len(paf_dict.items()) == 0:
            logging.info("nothing mapped")
            self.reject_count = 0
            self.accept_count = 0
            self.unmapped_count = 0
            self.reject_ids = set()
            self.accept_ids = set()
            self.unmapped_ids = set()
            return read_sequences

        reads_decision = dict()

        reject_count = 0
        accept_count = 0
        unmapped_count = 0

        reject_ids = set()
        accept_ids = set()
        unmapped_ids = set()

        # loop over paf dictionary
        for record_id, record_list in paf_dict.items():
            # record_id, record_list = list(gaf_dict.items())[0]
            if len(record_list) > 1:
                # should not happen often since we filter secondary mappings
                rec = choose_best_mapper(record_list)[0]
            else:
                rec = record_list[0]

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
                decision = strat[str(rec.tname)][rec.c_start // self.args.node_size][rec.rev]
            except TypeError:
                # if we don't have a strategy yet, it's an integer so except this and accept all
                decision = 1

            # uncomment to make random decisions
            # decision = np.random.choice((0, 1))

            # ACCEPT
            if decision:
                record_seq = read_sequences[rec.qname]
                accept_count += 1
                accept_ids.add(rec.qname)

            # REJECT
            else:
                record_seq = read_sequences[rec.qname][: self.args.mu]
                reject_count += 1
                reject_ids.add(rec.qname)

            # append the read's sequence to a new dictionary of the batch after decision-making
            reads_decision[rec.qname] = record_seq

        # all unmapped reads also need to be accepted, i.e. added back into the dict
        mapped_ids = set(reads_decision.keys())

        for read_id, seq in read_sequences.items():
            if read_id in mapped_ids:
                continue
            else:
                reads_decision[read_id] = seq
                unmapped_count += 1
                unmapped_ids.add(read_id)

        logging.info(f'decisions - rejecting: {reject_count} accepting: {accept_count} unmapped: {unmapped_count}')
        self.reject_count = reject_count
        self.accept_count = accept_count
        self.unmapped_count = unmapped_count
        self.reject_ids = reject_ids
        self.accept_ids = accept_ids
        self.unmapped_ids = unmapped_ids
        return reads_decision





    def update_times(self, read_sequences, reads_decision, reads_decision_readfish=None):
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
        rejection_cost = n_reject * self.args.rho
        self.time_aeons += (bases_aeons + acquisition + rejection_cost)
        logging.info(f"time aeons: {self.time_aeons}")

        # for readfish: fully sequenced reads (acc & unmapped)
        # and ((mu + rho) *  no. of rejected reads) + (alpha * batch_size)
        if not self.readfish:
            return
        else:
            read_lengths_readfish = np.array([len(seq) for seq in reads_decision_readfish.values()])
            n_reject_readfish = np.sum(np.where(read_lengths_readfish == self.args.mu, 1, 0))
            bases_readfish = np.sum(read_lengths_readfish)
            rejection_cost_readfish = n_reject_readfish * self.args.rho
            self.time_readfish += (bases_readfish + acquisition + rejection_cost_readfish)
            logging.info(f"time readfish: {self.time_readfish}")


    def _execute_dump(self, cond, dump_number, cache):
        # write out the next cumulative batch file
        logging.info(f'dump {cond} #{dump_number}. # of reads {len(list(cache.keys()))}')
        filename = f'00_reads/{self.args.name}_{dump_number}_{cond}.fa'
        # copy previous file to make cumulative
        previous_filename = f'00_reads/{self.args.name}_{dump_number - 1}_{cond}.fa'
        try:
            execute(f"cp {previous_filename} {filename}")
        except FileNotFoundError:
            # at the first batch, create empty 0th and copy to 1st file
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



    def _prep_dump(self, cond):
        # grab the attributes of the condition
        dump_time = self.dump_time
        curr_time = getattr(self, f'time_{cond}')
        dump_number = getattr(self, f'dump_number_{cond}')
        cache = getattr(self, f'cache_{cond}')
        # check if it's time to write out the next file
        if curr_time > (dump_time * dump_number):
            self._execute_dump(cond=cond, dump_number=dump_number, cache=cache)


    def write_batch(self, read_sequences, reads_decision, reads_decision_readfish=None):
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

        if self.readfish:
            add_to_cache(seqs=reads_decision_readfish, cache=self.cache_readfish)
            self._prep_dump(cond='readfish')



    def collect_metrics(self):
        # save a few things after each batch
        # add it as row into a list that can be transformed to a pandas frame and saved for plotting
        logging.info("saving metrics")

        def append_row(frame_dict, new_row):
            # this is used by save metrics
            # takes 2 dicts as input: one is a metric frame,
            # the other is a row to append to that
            for colname, rowval in new_row.items():
                frame_dict[colname].append(rowval)
            return frame_dict

        bsize = len(self.stream.read_ids)

        row = {'name': self.args.name,
               'batch': self.batch,
               'n_mapped': self.mapped_count_lm / bsize,
               'n_unmapped': self.unmapped_count_lm / bsize,
               'n_reject': self.reject_count / bsize,
               'n_accept': self.accept_count / bsize,
               'time_aeons': self.time_aeons,
               'time_naive': self.time_naive,
               'pool_size': len(self.pool.sequences.keys())}

        if self.readfish:
            row['n_reject_readfish'] = self.readfish.reject_count / bsize
            row['n_accept_readfish'] = self.readfish.accept_count / bsize
            row['n_unmapped_readfish'] = self.readfish.unmapped_count / bsize
            row['time_readfish'] = self.time_readfish


        self.metrics = append_row(self.metrics, row)
        # write to file
        df = pd.DataFrame(self.metrics)
        df_csv = f"{self.args.name}_metrics.csv"
        with open(df_csv, 'w'):
            pass
        df.to_csv(df_csv)

        # separate file separated by sources
        source_counts_aeons, source_counts_readfish = self.check_sources(
            read_sources=self.stream.read_sources)


        def append_rows_sep(source_counts, cond):
            stypes = ['accepted', 'rejected', 'unmapped']
            for i in range(3):
                scount = source_counts[i]
                for name, count in scount.items():
                    row = {'name': self.args.name,
                           'batch': self.batch,
                           'ref': name,
                           'count': count / bsize,
                           'dec': stypes[i],
                           'cond': cond}


                    self.metrics_sep = append_row(self.metrics_sep, row)


        def append_totals(accept_count, reject_count, unmapped_count, cond):
            # next 3 are for total counts across all sources
            row_acc = {'name': self.args.name,
                       'batch': self.batch,
                       'ref': 'total',
                       'count': accept_count / bsize,
                       'dec': 'accepted',
                       'cond': cond}
            row_rej = {'name': self.args.name,
                       'batch': self.batch,
                       'ref': 'total',
                       'count': reject_count / bsize,
                       'dec': 'rejected',
                       'cond': cond}
            row_unm = {'name': self.args.name,
                       'batch': self.batch,
                       'ref': 'total',
                       'count': unmapped_count / bsize,
                       'dec': 'unmapped',
                       'cond': cond}

            self.metrics_sep = append_row(self.metrics_sep, row_acc)
            self.metrics_sep = append_row(self.metrics_sep, row_rej)
            self.metrics_sep = append_row(self.metrics_sep, row_unm)

        append_rows_sep(source_counts=source_counts_aeons, cond="aeons")
        append_totals(
            accept_count=self.accept_count,
            reject_count=self.reject_count,
            unmapped_count=self.unmapped_count_lm,
            cond='aeons')

        if self.readfish:
            append_rows_sep(source_counts=source_counts_readfish, cond="readfish")
            append_totals(
                accept_count=self.readfish.accept_count,
                reject_count=self.readfish.reject_count,
                unmapped_count=self.readfish.unmapped_count,
                cond='readfish')

        # write to file
        df = pd.DataFrame(self.metrics_sep)
        df_csv = f"{self.args.name}_metrics_sep.csv"
        with open(df_csv, 'w'):
            pass
        df.to_csv(df_csv)





    def strat_csv(self, strat, node2pos):
        # write the strategy to csv file, so it can be displayed in bandage
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
        self.ava.remove_links(sequences=sequences)
        self.pool.remove_sequences(sequences=sequences)



    def add_new_sequences(self, sequences, increment=True):
        # WRAPPER
        logging.info('')
        logging.info("adding new seqs")
        ava_new, ava_onto_pool = self.pool.add2ava(sequences)
        # ingest the new sequences
        self.pool.ingest(seqs=sequences)
        # load new alignments
        cont_new, ovl_new = self.ava.load_ava(ava_new, seqpool=self.pool)
        if increment:
            self.pool.increment(containment=cont_new)
        cont_onto, ovl_onto = self.ava.load_ava(ava_onto_pool, seqpool=self.pool)
        if increment:
            self.pool.increment(containment=cont_onto)
        cont = SequenceAVA.source_union(edges0=cont_new, edges1=cont_onto)
        self.remove_seqs(sequences=cont)
        # affect the overlappers
        ovl = ovl_new | ovl_onto
        self.pool.reset_temperature(ovl, t=self.args.temperature)



    def overlap_pool(self):
        # WRAPPER
        # run AVA for the pool to find overlaps and remove contained sequences
        logging.info('')
        logging.info("ava pool")
        contigs = self.pool.declare_contigs(min_contig_len=self.filt.min_contig_len)
        if not contigs:
            return
        pool_paf = self.pool.run_ava(sequences=contigs.seqdict(), fa=self.pool.fa, paf=self.pool.ava)
        pool_contained, pool_ovl = self.ava.load_ava(paf=pool_paf, seqpool=self.pool)
        self.pool.increment(containment=pool_contained)
        cont = SequenceAVA.source_union(edges0=pool_contained, edges1={})
        if cont:
            logging.info(f'removing {len(cont)} contained sequences from pool')
            self.remove_seqs(sequences=cont)
        self.pool.reset_temperature(pool_ovl)


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
        # trim and ingest
        trimmed_seqs = self.pool.trim_sequences(trim_dict=trim_dict)
        trim_paf = self.pool.run_ava(sequences=trimmed_seqs,
                                     fa=f'{self.pool.fa}.trim',
                                     paf=f'{self.pool.ava}.trim')
        trim_contained, _ = self.ava.load_ava(paf=trim_paf, seqpool=self.pool)
        to_remove = self.ava.trim_success(trim_dict=trim_dict, overlaps=self.ava.overlaps)
        # remove original sequences & failed mergers
        self.remove_seqs(sequences=to_remove)






    # @profile
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
        paf_trunc = lm.mappy_batch(sequences=self.stream.read_sequences,
                                   truncate=True,
                                   out=f'{self.args.name}.lm_out.paf')
        # for metrics collection
        self.mapped_count_lm = lm.mapped_count
        self.unmapped_count_lm = lm.unmapped_count
        # make decisions
        # logging.info("making decisions")
        reads_decision = self.make_decision_paf(paf_out=paf_trunc,
                                                read_sequences=self.stream.read_sequences,
                                                strat=self.strat)

        if not self.readfish:
            return reads_decision, dict()

        # READFISH ACTION
        paf_trunc_readfish = Readfish.map_truncated_reads(
            mapper=self.readfish.mapper,
            reads=self.stream.read_sequences
        )

        reads_decision_readfish = self.readfish.make_decision_paf_readfish(
            paf_out=paf_trunc_readfish,
            read_sequences=self.stream.read_sequences,
            mu=self.args.mu, node_size=self.args.node_size,
            readfish_strat=self.readfish.strat
        )

        return reads_decision, reads_decision_readfish



    def cleanup(self):
        # after the run, move temporary files into the run dir
        tmpdir = f'{self.args.out_dir}/tmp'
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        execute(f'mv {self.name}.* {tmpdir}')



    def check_sources(self, read_sources):
        # check the sources of the accepted/rejected/unmapped reads
        def fetch_sources(ids, source_dict):
            source_counts = defaultdict(int)
            for name in ids:
                try:
                    source = source_dict[name]
                except KeyError:
                    source = 'na'
                source_counts[source] += 1
            return source_counts

        def fetch_id_sets(accept_ids, reject_ids, unmapped_ids):
            id_sets = [accept_ids, reject_ids, unmapped_ids]
            res = []
            for set_idx in range(3):
                sources = fetch_sources(id_sets[set_idx], read_sources)
                res.append(sources)
            return res

        res_aeons = fetch_id_sets(
            self.accept_ids, self.reject_ids, self.unmapped_ids)

        if not self.readfish:
            return res_aeons, []

        res_readfish = fetch_id_sets(
            self.readfish.accept_ids, self.readfish.reject_ids, self.readfish.unmapped_ids)
        return res_aeons, res_readfish


    def process_batch_live(self):
        # LIVE version
        logging.info(f"\n\n\n Next batch ---------------------------- # {self.batch}")
        tic = time.time()

        # find new fastq files
        new_fastq = LiveRun.scan_dir(
            fq=self.args.fq, processed_files=self.processed_files)
        if not new_fastq:
            logging.info("no new files, deferring update ")
            return self.args.wait

        # add the new files to the set of processed files
        self.processed_files.update(new_fastq)
        self.n_fastq += len(new_fastq)
        fq_batch = FastqBatch(fq_files=new_fastq, channels=self.channels)
        self.rl_dist.update(read_lengths=fq_batch.read_lengths, recalc=True)
        new_reads = fq_batch.read_sequences

        # ---------- from here analogous to sim

        # filter sequences with repeats at the end
        if self.args.filter_repeats:
            reads_filtered = self.repeat_filter.filter_batch(seq_dict=new_reads)
        else:
            reads_filtered = new_reads

        # load new sequences, incl length filter
        sequences = SequencePool(sequences=reads_filtered, min_len=self.filt.min_seq_len)
        # add new sequences to AVA
        self.add_new_sequences(sequences=sequences)
        # check for overlaps and containment in pool
        self.overlap_pool()
        # trim sequences that might lead to overlaps
        self.trim_sequences()

        # call wrapper to update assembly
        contigs = self.assemble_add_and_filter_contigs()
        contig_pool = ContigPool(sequences=contigs.sequences)

        if self.args.polish:
            cpolished = self.pool.polish_sequences(contigs=contigs, read_sources=fq_batch.read_sources)
            contigs = self.pool.declare_contigs(min_contig_len=self.filt.min_contig_len)
            self.ava.remove_links(sequences=cpolished)

        # write the current pool to file for mapping against
        self.pool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)

        # check if we have any frozen sequences
        frozen_ids = self.pool.decrease_temperature(lim=self.filt.min_contig_len)
        self.remove_seqs(sequences=frozen_ids)

        # self.strat = contig_pool.process_contigs(
        #     node_size=self.args.node_size,
        #     lim=self.args.lowcov,
        #     ccl=self.rl_dist.approx_ccl,
        #     out_dir=self.args.out_dir,
        #     write=True)


        self.strat = contig_pool.process_contigs_m0(
            score_vec=self.score_vec,
            node_size=self.args.node_size,
            ccl=self.rl_dist.approx_ccl,
            out_dir=self.args.out_dir,
            mu=self.args.mu,
            lam=self.rl_dist.lam,
            write=True)

        # collect metrics
        # self.collect_metrics()   # TODO do we want any?

        # final bit for timing next scan
        toc = time.time()
        passed = toc - tic
        next_update = int(self.args.wait - passed)
        logging.info(f"batch took: {passed}")
        logging.info(f"finished updating masks, waiting for {next_update} ... \n")
        self.batch += 1
        return next_update



    # @profile
    def process_batch(self):
        logging.info(f'\n NEW BATCH #############################  {self.batch}')
        tic = time.time()

        reads_decision, reads_decision_readfish = self.sim_batch()
        # update read length dist, time recording
        self.rl_dist.update(read_lengths=self.stream.read_lengths, recalc=True)
        self.update_times(read_sequences=self.stream.read_sequences,
                          reads_decision=reads_decision,
                          reads_decision_readfish=reads_decision_readfish)
        self.write_batch(read_sequences=self.stream.read_sequences,
                         reads_decision=reads_decision,
                         reads_decision_readfish=reads_decision_readfish)

        #  -------------------------------- POST DECISIONS
        logging.info("")

        # if self.batch % 5 == 0:
        #     print("breakpoint")

        # filter sequences with repeats at the end
        if self.args.filter_repeats:
            reads_filtered = self.repeat_filter.filter_batch(seq_dict=reads_decision)
        else:
            reads_filtered = reads_decision
        # load new sequences, incl length filter
        sequences = SequencePool(sequences=reads_filtered, min_len=self.filt.min_seq_len)
        # add new sequences to AVA
        self.add_new_sequences(sequences=sequences)
        # check for overlaps and containment in pool
        self.overlap_pool()
        # trim sequences that might lead to overlaps
        self.trim_sequences()

        # call wrapper to update assembly
        contigs = self.assemble_add_and_filter_contigs()
        contig_pool = ContigPool(sequences=contigs.sequences)


        if self.args.polish:
            cpolished = self.pool.polish_sequences(contigs=contigs, read_sources=self.read_sources)
            contigs = self.pool.declare_contigs(min_contig_len=self.filt.min_contig_len)
            self.ava.remove_links(sequences=cpolished)

        # write the current pool to file for mapping against
        self.pool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)


        # check if we have any frozen sequences
        frozen_ids = self.pool.decrease_temperature(lim=self.filt.min_contig_len)
        self.remove_seqs(sequences=frozen_ids)

        # SIMPLER MODEL WITH ROLLING TARGET STRATEGY
        # self.strat = contig_pool.process_contigs(
        #     node_size=self.args.node_size,
        #     lim=self.args.lowcov,
        #     ccl=self.rl_dist.approx_ccl,
        #     out_dir=self.args.out_dir,
        #     write=True)

        self.strat = contig_pool.process_contigs_m0(
            score_vec=self.score_vec,
            node_size=self.args.node_size,
            ccl=self.rl_dist.approx_ccl,
            out_dir=self.args.out_dir,
            mu=self.args.mu,
            lam=self.rl_dist.lam,
            write=True)

        # self.strat_csv(self.strat, node2pos)  # this is for bandage viz

        # collect metrics
        self.collect_metrics()

        # if self.batch % 5 == 0:
        #     redotable(fa=self.pool.contig_fa,
        #               out=f'{self.pool.contig_fa}.{self.batch}.redotable.png',
        #               ref=self.args.ref,
        #               prg=self.args.redotable,
        #               logdir=f'{self.args.out_dir}/logs/redotable')

        self.batch += 1
        logging.info(f"batch took: {time.time() - tic}")

        # cov_counts, cov_means = self.pool.count_coverage()

    def __repr__(self):
        return str(self.__dict__)



class Readfish:

    def __init__(self, seqpool, ref_path, mu, node_size):
        self.strat = Readfish.init_strat(seqpool=seqpool, node_size=node_size)
        self.mapper = Readfish.init_persistent_mapper(ref_path=ref_path, mu=mu)


    @staticmethod
    def init_strat(seqpool, node_size):
        strat = {}
        oz = 10000 // node_size
        for header, seqo in seqpool.sequences.items():
            cstrat = np.zeros(shape=((len(seqo.seq) // node_size) + 1, 2))
            cstrat[-oz:, 0] = 1
            cstrat[:oz, 1] = 1
            strat[header] = cstrat
        return strat


    @staticmethod
    def init_persistent_mapper(ref_path, mu):
        m = LinearMapper(ref=ref_path, mu=mu, default=False)
        return m


    @staticmethod
    def map_truncated_reads(mapper, reads):
        paf_trunc = mapper.mappy_batch(sequences=reads, truncate=True)
        return paf_trunc



    def make_decision_paf_readfish(self, paf_out, read_sequences, mu, node_size, readfish_strat):
        # decide accept/reject for each read
        # THIS IS FOR READFISH - MODIFIED VERSION
        paf_dict = Paf.parse_PAF(StringIO(paf_out), min_len=int(mu / 2))

        # if nothing mapped, just return. Unmapped = accept
        if len(paf_dict.items()) == 0:
            logging.info("nothing mapped - readfish")
            return read_sequences

        reads_decision = dict()
        reject_count = 0
        accept_count = 0
        unmapped_count = 0

        reject_ids = set()
        accept_ids = set()
        unmapped_ids = set()

        # loop over paf dictionary
        for record_id, record_list in paf_dict.items():
            if len(record_list) > 1:
                # should not happen often since we filter secondary mappings
                rec = choose_best_mapper(record_list)[0]
            else:
                rec = record_list[0]

            # find the start and end position relative to the whole linearised genome
            if rec.strand == '+':
                rec.c_start = rec.tstart
            elif rec.strand == '-':
                rec.c_start = rec.tend - 1
            else:
                continue

            # index into strategy to find the decision
            try:
                decision = readfish_strat[str(rec.tname)][rec.c_start // node_size][rec.rev]
            except KeyError:
                decision = 1  # if the mapping is not on a preloaded contig

            # ACCEPT
            if decision:
                record_seq = read_sequences[rec.qname]
                accept_count += 1
                accept_ids.add(rec.qname)

            # REJECT
            else:
                record_seq = read_sequences[rec.qname][: mu]
                reject_count += 1
                reject_ids.add(rec.qname)

            # append the read's sequence to a new dictionary of the batch after decision-making
            reads_decision[rec.qname] = record_seq

        # all unmapped reads also need to be accepted, i.e. added back into the dict
        mapped_ids = accept_ids | reject_ids

        for read_id, seq in read_sequences.items():
            if read_id in mapped_ids:
                continue
            else:
                reads_decision[read_id] = seq
                unmapped_count += 1
                unmapped_ids.add(read_id)

        logging.info(f'decisions - rejecting: {reject_count} accepting: {accept_count} unmapped: {unmapped_count}  - readfish ')
        self.reject_count = reject_count
        self.accept_count = accept_count
        self.unmapped_count = unmapped_count
        self.reject_ids = reject_ids
        self.accept_ids = accept_ids
        self.unmapped_ids = unmapped_ids
        return reads_decision




class LiveRun:


    @staticmethod
    def split_flowcell(out_path, run_name):
        # out_path = "/nfs/research/goldman/lukasw/BR/data/zymo_all_live/20211124_boss_runs_log_live_001/20211124_1236_X2_FAQ09307_bfb985c5"
        channel_path = f'{out_path}/channels.toml'
        logging.info(f'looking for channels specification at : {channel_path}')
        channels_found = False
        channels = []
        while not channels_found:
            if not os.path.isfile(channel_path):
                logging.info("channels file does not exist (yet), waiting for 30s")
                time.sleep(30)
            else:
                channels = LiveRun._grab_channels(channels_toml=channel_path, run_name=run_name)
                channels_found = True
        # channels successfully found
        logging.info(f"found channels specification: Using {len(channels)} channels.")
        return channels



    @staticmethod
    def connect_sequencer(device, host, port):
        try:
            out_path = LiveRun._grab_output_dir(device=device, host=host, port=port)
            logging.info(f"grabbing Minknow's output path: \n{out_path}\n")
        except:
            logging.info("Minknow's output dir could not be inferred from device name. Exiting.")
            logging.info(f'\n{device}\n{host}\n{port}')
            out_path = ""  # dummy
            # out_path = "/home/lukas/Desktop/BossRuns/playback_target/data/pb01/no_sample/20211021_2209_MS00000_f1_f320fce2"
            exit()
        return out_path



    @staticmethod
    def _grab_output_dir(device, host='localhost', port=None):
        '''
        Capture the output directory of MinKNOW,
        i.e. where fastq files are deposited during sequencing
        host and port should be optional if BOSS-RUNS is run on the sequencing machine

        Parameters
        ----------
        device: str
            device name of the 'position' in the sequencing machine
        host: str
            hostname to connect to for MinKNOW
        port: int
            override default port to connect to

        Returns
        -------

        '''
        logging.info(f"minknow API Version {minknow_api_version}")
        # minknow_api.manager supplies Manager (wrapper around MinKNOW's Manager gRPC)
        if minknow_api_version.startswith("5"):
            if not port:
                port = 9502
            manager = Manager(host=host, port=int(port))
        elif minknow_api_version.startswith("4"):
            if not port:
                port = 9501
            manager = Manager(host=host, port=int(port), use_tls=False)
        else:
            logging.info("unsupported version of minknow_api")
            sys.exit()
        # Find a list of currently available sequencing positions.
        positions = list(manager.flow_cell_positions())
        pos_dict = {pos.name: pos for pos in positions}
        # index into the dict of available devices
        try:
            target_device = pos_dict[device]
        except KeyError:
            logging.info(f"Error: target device {device} not available")
            logging.info("Error: Please make sure to supply correct name of sequencing position in MinKNOW.")
            sys.exit()
        # connect to the device and navigate api to get output path
        device_connection = target_device.connect()
        current_run = device_connection.protocol.get_current_protocol_run()
        run_id = current_run.run_id
        logging.info(f"connected to run_id: {run_id}")
        out_path = current_run.output_path
        return out_path


    @staticmethod
    def _grab_channels(channels_toml, run_name):
        # parse the channels TOML file
        toml_dict = toml.load(channels_toml)
        # find the corresponding condition
        correct_key = ''
        for key in toml_dict["conditions"].keys():
            name = toml_dict["conditions"][key]["name"]
            if name == run_name:
                correct_key = key
                break
        try:
            selected_channels = set(toml_dict["conditions"][correct_key]["channels"])
            logging.info("grabbing channel numbers ...")
            return selected_channels
        except UnboundLocalError:
            logging.info("--name in .params not found in channel-specification toml. Exiting")
            exit()


    @staticmethod
    def scan_dir(fq, processed_files):
        # preiodically scanning Minknow's output dir
        # create new batch from all NEW files
        patterns = ["*.fq.gz", "*.fastq.gz", "*.fastq.gzip", "*.fq.gzip", "*.fastq", "*.fq"]
        all_fq = set()
        for p in patterns:
            all_fq.update(glob.glob(f'{fq}/{p}'))

        # which files have we not seen before?
        new_fq = all_fq.difference(processed_files)
        logging.info(f"found {len(new_fq)} new fq files: \n {new_fq}")
        new_fq_list = [f for f in new_fq]
        return new_fq_list












class FastqBatch:

    def __init__(self, fq_files, channels):
        # this is for the live version when we read actual files
        self.read_batch(fq_files=fq_files, channels=channels)




    def read_batch(self, fq_files, channels):
        # read sequencing data from all new files
        read_sequences = {}
        read_sources = {}

        for fq in fq_files:
            rseq = self._read_single_batch(fastq_file=fq, channels=channels)
            read_sequences.update(rseq)
            read_sources.update({header: fq for header in rseq.keys()})

        self.read_sequences = read_sequences
        self.read_sources = read_sources
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = {rid: len(seq) for rid, seq in read_sequences.items()}
        self.total_bases = np.sum(list(self.read_lengths.values()))
        logging.info(f'total new reads: {len(read_sequences)}')



    def _read_single_batch(self, fastq_file, channels=None):
        # get the reads from a single fq file
        logging.info(f"reading file: {fastq_file}")
        read_sequences = {}

        # to make sure it's a path object, not a string
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
        return read_sequences





##############################


def setup_parser():
    parser = MyArgumentParser(fromfile_prefix_chars='@')
    parser.add_argument('--name', dest='name', type=str, default="test", help='name for sequencing run')
    parser.add_argument('--bsize', dest='bsize', type=int, default=4000, help='num reads per batch')                                                    # OPT
    parser.add_argument('--lowcov', dest='lowcov', type=int, default=10, help='limit for strategy rejection')                                           # OPT
    parser.add_argument('--preload', dest='preload', type=str, default=None, help='path to fasta for pre-loading sequences')                            # OPT
    parser.add_argument('--preload_cov', dest='preload_cov', type=int, default=11, help='how much coverage to assign to preloaded seqs')                # OPT
    parser.add_argument('--tetra', dest='tetra', type=int, default=0, help='adds a test for tetramer freq dist before overlapping')                     # OPT
    parser.add_argument('--polish', dest='polish', type=int, default=0, help='whether to run contig polishing (not for scaffold mode)')                 # OPT
    parser.add_argument('--filter_repeats', dest='filter_repeats', type=int, default=0, help='whether to run repeat filtering')                         # OPT
    parser.add_argument('--fq', dest='fq', type=str, default=None, help='path to fastq for streaming')                                          # SIM
    parser.add_argument('--maxb', dest='maxb', type=int, default=4, help='maximum batches')                                                     # SIM
    parser.add_argument('--binit', dest='binit', type=int, default=1, help='loading batches')                                                   # SIM
    parser.add_argument('--dumptime', dest='dumptime', type=int, default=1, help='interval for dumping batches')                                # SIM
    parser.add_argument('--seed', dest='seed', type=int, default=1, help='seed for shuffling input')                                            # SIM   # OPT
    parser.add_argument('--gsize', dest='gsize', type=float, default=12e6, help='genome size estimate')                                         # SIM   # OPT
    # for auto snake
    parser.add_argument('--snake', dest='snake', type=str, default=None, help='path to snakemake config')                                       # SIM   # OPT
    # parser.add_argument('--ref', dest='ref', type=str, default="", help='reference used in quast evaluation and redotable')                   # SIM   # OPT
    # TODO live version
    parser.add_argument('--live', default=False, action="store_true", help="internally used switch")
    parser.add_argument('--device', default=None, type=str, help="employed device/sequencing position")
    parser.add_argument('--host', default='localhost', type=str, help="host of sequencing device")
    parser.add_argument('--port', default=None, type=int, help="port of sequencing device")
    parser.add_argument('--split_flowcell', default=False, action="store_true", help="assign channels to conditions with channels.toml")
    return parser










