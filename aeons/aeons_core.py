import argparse
import os
import gzip
import sys
import glob
import time
import logging
import re
from sys import exit
from pathlib import Path
from io import StringIO
from typing import Set, Dict, List
import inspect

import numpy as np
from numpy.typing import NDArray
import rtoml
from minknow_api.manager import Manager
from minknow_api import __version__ as minknow_api_version

from .aeons_utils import execute, random_id, empty_file, readfq, spawn
from .aeons_sampler import FastqStream_mmap
from .aeons_paf import Paf, choose_best_mapper
from .aeons_mapper import LinearMapper
from .aeons_sequence import SequencePool, SequenceAVA, UnitigPool, ContigPool, Benefit, ReadlengthDist
from .aeons_repeats import RepeatFilter



class AeonsRun:

    def __init__(self, args: argparse.Namespace):
        """
        Main initialisation of experiment

        :param args: Configuration of the experiment
        """
        self.args = args
        self.name = args.name
        # initial strategy is to accept
        self.strat = 1
        # initialise some folders and files
        self.init_file_struct()
        # initialise central objects
        self.pool = SequencePool(name=self.args.name,
                                 min_len=self.args.min_seq_len,
                                 out_dir=self.out_dir)
        self.ava = SequenceAVA(paf=f'{self.args.name}.ava',
                               tetra=self.args.tetra,
                               filters=self.args)
        self.rl_dist = ReadlengthDist(mu=self.args.mu)
        # init scoring func
        self.score_vec = Benefit.init_scoring_vec(lowcov=args.lowcov)

        if self.args.sim_run:
            self.init_sim()
        elif self.args.live_run:
            self.launch_readfish()
            self.init_live()
            self.first_live_asm()
        else:
            raise ValueError("Neither sim nor live run")
        logging.info("Finished initial setup")



    def init_file_struct(self) -> None:
        """
        Set up the required directory structure

        :return:
        """
        # make sure the run name does not have any spaces
        assert ' ' not in self.args.name

        self.out_dir = f'./out_{self.args.name}'
        out_path = Path(self.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        (out_path / "masks").mkdir(parents=True, exist_ok=True)
        (out_path / "fq").mkdir(parents=True, exist_ok=True)
        (out_path / "logs").mkdir(parents=True, exist_ok=True)
        (out_path / "contigs").mkdir(parents=True, exist_ok=True)
        (out_path / "contigs" / "prev").mkdir(parents=True, exist_ok=True)
        (out_path / "contigs" / "init").mkdir(parents=True, exist_ok=True)


    def init_sim(self) -> None:
        """
        Initialisation wrapper for simulated experiments

        :return:
        """
        self.sim_run = SimRun(args=self.args)

        # load some initial batches
        init_pool = SequencePool(name="init_pool", out_dir=self.out_dir)
        for i in range(self.args.binit):
            self.sim_run.stream.read_batch()
            init_pool.ingest(seqs=self.sim_run.stream.read_sequences)
        logging.info(f"total bases in pool: {init_pool.total_bases()}")
        # increment time after preloading
        self.sim_run.update_times(read_sequences=init_pool.seqdict(),
                                  reads_decision=init_pool.seqdict())
        # set the batch counter for the run
        self.batch = self.sim_run.stream.batch
        # initialise a repeat filter from the raw initial reads
        if self.args.filter_repeats:
            self.repeat_filter = RepeatFilter(name=self.args.name, seqpool=init_pool)

        # run first asm
        logging.info("Running assembly of initial data..")
        init_contigs = init_pool.initial_asm_miniasm()
        self.pool.ingest(init_contigs)
        has_contig = self.pool.has_min_one_contig(min_contig_len=self.args.min_contig_len)
        ncontigs = len(self.pool.sequences)
        logging.info(f'initial contigs: {ncontigs}')
        if len(self.pool.sequences) == 0 or not has_contig:
            print("no contigs of sufficient length, need more data")
            sys.exit()  # hard exit from simulation
        self.pool.write_seq_dict(seq_dict=self.pool.seqdict(), file=self.pool.contig_fa)


    def launch_readfish(self) -> None:
        """
        Wrapper to launch readfish into the background
        Used during live runs only, not for simulations

        :return:
        """
        # check that the readfish.toml that aeons wrote exists
        if not Path(self.args.toml_readfish).exists():
            raise FileNotFoundError("readfish toml does not exist. Something went wrong loading configs")
        # launch readfish into the background
        module_path = inspect.getfile(AeonsRun)
        logging.info(module_path)
        script_path = Path(module_path).parent / "aeons_readfish.py"
        if not script_path.is_file():
            raise FileNotFoundError("aeons_readfish.py not found. Something went wrong..")
        readfish_comm = f'python {script_path} {self.args.toml_readfish} {self.args.device} {self.args.name} 2>&1 | tee -a readfish.log'
        logging.info("Launching readfish")
        logging.info(readfish_comm)
        spawn(readfish_comm)


    def init_live(self) -> None:
        """
        Wrapper to initialise a live experiment, incl. device related things
        find output path where the fastq files are placed
        find channels toml for runs with multiple regions

        :return:
        """
        self.processed_files = set()
        self.batch = 0
        # connect to sequencing machine and grab the output directory
        out_path = LiveRun.connect_sequencer(device=self.args.device,
                                             host=self.args.host,
                                             port=self.args.port)
        self.args.fq = f'{out_path}/fastq_pass'
        # grab channels of the condition
        # for this readfish needs to be up and running already
        if self.args.split_flowcell:
            channels = LiveRun.split_flowcell(out_path=out_path, run_name=self.args.name)
        else:
            # if we use a whole flowcell, leave channels empty to use all data
            channels = set()
        self.channels = channels



    def first_live_asm(self) -> None:
        """
        Construct a first assembly from some initial data.
        Scan directory for data and wait until args.data_wait megabases accumulate
        Loop until some contigs have been generated

        :return:
        """
        while True:
            logging.info(f"waiting for {self.args.data_wait} Mbases of data ... \n")
            time.sleep(30)
            new_fastq = LiveRun.scan_dir(fq=self.args.fq, processed_files=set())
            fq_batch = FastqBatch(fq_files=new_fastq, channels=self.channels)
            logging.info(f"available so far: {fq_batch.total_bases / 1e6} Mbases")
            if fq_batch.total_bases / 1e6 < self.args.data_wait:
                continue
            else:
                # try first asm
                logging.info("Attempting initial assembly")
                init_pool = SequencePool(name="init_pool",
                                         min_len=self.args.min_seq_len,
                                         out_dir=self.out_dir)
                init_pool.ingest(seqs=fq_batch.read_sequences)

                init_contigs = init_pool.initial_asm_miniasm()
                ncontigs = len(init_contigs.sequences)
                has_contig = init_pool.has_min_one_contig(min_contig_len=self.args.min_contig_len)

                if not ncontigs or not has_contig:
                    continue
                else:
                    self.pool = SequencePool(name=self.args.name,
                                             min_len=self.args.min_seq_len,
                                             out_dir=self.out_dir)
                    self.ava = SequenceAVA(paf=f'{self.args.name}.ava',
                                           tetra=self.args.tetra,
                                           filters=self.args)
                    self.pool.ingest(init_contigs)
                    # set up a repeat filter from the initial set of reads
                    if self.args.filter_repeats:
                        self.repeat_filter = RepeatFilter(name=self.args.name, seqpool=init_pool)
                    break

        # once there are contigs, record used files
        self.processed_files = set()
        self.processed_files.update(new_fastq)
        self.n_fastq = len(new_fastq)
        logging.info("Initial asm completed\n\n")


    def assemble_unitigs(self) -> SequencePool:
        """
        Wrapper to construct new contigs from the current data.
        Removes used sequences in the process

        :return: SequencePool of the new unitigs
        """
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
        unitig_pool.get_unitig_coverage_arrays(seqpool=self.pool)
        # transform into a sequence pool
        new_pool, used_sids = unitig_pool.unitigs2seqpool(
            seqpool=self.pool, min_seq_len=self.args.min_seq_len
        )
        # remove used sequences from current pool
        self.remove_seqs(used_sids)
        return new_pool


    def assemble_add_and_filter_contigs(self) -> SequencePool:
        """
        Main wrapper to update the current contigs.
        Assembles the current graph and extracts new unitigs,
        adds them to the seqpool and extracts contigs for mapping against

        :return: SequencePool of current contigs (old and new)
        """
        logging.info("assembling new unitigs.. ")
        new_pool = self.assemble_unitigs()
        # add new sequences to the dict and to the ava
        logging.info("loading and overlapping new unitigs.. ")
        self.add_new_sequences(sequences=new_pool, increment=False)
        # write the current pool to file for mapping against
        logging.info("finding contigs to map against.. ")
        contigs = self.pool.declare_contigs(min_contig_len=self.args.min_contig_len)
        SequencePool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)
        return contigs



    def make_decision_paf(
        self,
        paf_out: str,
        read_sequences: Dict,
        strat: NDArray | int,
        node_size: int = 100
    ) -> Dict:
        """
        Decision function for simulations only.
        In real experiments readfish performs this functionality

        :param paf_out: String output of mapping reads to contigs
        :param read_sequences: Dictionary of new read sequences
        :param strat: Mask array for look-ups.
        :param node_size: Resolution reduction factor of sequencing masks
        :return: Dict of read sequences after decisions, i.e. rejected reads are truncated
        """
        # transform paf output to dictionary
        # filtering here is for alignment_block_length, not sequence length
        # i.e. at least half of the initial mu-sized fragment has to map
        paf_dict = Paf.parse_PAF(StringIO(paf_out), min_len=int(self.args.mu / 2))
        # if nothing mapped, just return. Unmapped = accept
        if len(paf_dict.items()) == 0:
            logging.info("nothing mapped")
            self.reject_count = 0
            self.accept_count = 0
            self.unmapped_count = 0
            return read_sequences

        reads_decision = dict()
        reject_count = 0
        accept_count = 0
        unmapped_count = 0
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
                decision = strat[str(rec.tname)][rec.c_start // node_size][rec.rev]
            except TypeError:
                # if we don't have a strategy yet, it's an integer so except this and accept all
                decision = 1

            # ACCEPT
            if decision:
                record_seq = read_sequences[rec.qname]
                accept_count += 1

            # REJECT
            else:
                record_seq = read_sequences[rec.qname][: self.args.mu]
                reject_count += 1

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

        logging.info(f'decisions - rejecting: {reject_count} accepting: {accept_count} unmapped: {unmapped_count}')
        self.reject_count = reject_count
        self.accept_count = accept_count
        self.unmapped_count = unmapped_count
        return reads_decision



    def remove_seqs(self, sequences: Set[str]) -> None:
        """
        Wrapper to remove sequences from pool, ava, coverage etc.

        :param sequences: Set of read IDs
        :return:
        """
        if not sequences:
            return
        self.ava.remove_links(sequences=sequences)
        self.pool.remove_sequences(sequences=sequences)



    def add_new_sequences(self, sequences: SequencePool, increment: bool = True) -> None:
        """
        Wrapper to add new sequences to the main pool

        :param sequences: Sequences to add
        :param increment: Whether to increment coverage when adding new sequences
        :return:
        """
        logging.info("Adding new sequences")
        ava_new, ava_onto_pool = self.pool.add2ava(sequences)
        self.pool.ingest(seqs=sequences)
        # load new alignments
        cont_new, ovl_new = self.ava.load_ava(ava_new, seqpool=self.pool)
        if increment:
            self.pool.increment(containment=cont_new)
        cont_onto, ovl_onto = self.ava.load_ava(ava_onto_pool, seqpool=self.pool)
        if increment:
            self.pool.increment(containment=cont_onto)
        # remove contained sequences
        cont = SequenceAVA.source_union(edges0=cont_new, edges1=cont_onto)
        self.remove_seqs(sequences=cont)
        # raise temp for overlappers (new links are saved as class attr in load_ava)
        ovl = ovl_new | ovl_onto
        self.pool.reset_temperature(ovl, t=self.args.temperature)



    def overlap_pool(self) -> None:
        """
        Wrapper to run AVA for pool to find overlaps and remove contained sequences

        :return:
        """
        logging.info("Running all-versus-all of sequence pool")
        contigs = self.pool.declare_contigs(min_contig_len=self.args.min_contig_len)
        if contigs.is_empty():
            return
        pool_paf = self.pool.run_ava(sequences=contigs.seqdict(), fa=self.pool.fa, paf=self.pool.ava)
        pool_contained, pool_ovl = self.ava.load_ava(paf=pool_paf, seqpool=self.pool)
        self.pool.increment(containment=pool_contained)
        cont = SequenceAVA.source_union(edges0=pool_contained, edges1={})
        if cont:
            logging.info(f'Removing {len(cont)} contained sequences from pool')
            self.remove_seqs(sequences=cont)
        self.pool.reset_temperature(pool_ovl)



    def trim_sequences(self) -> None:
        """
        Wrapper to find reads that need trimming for potential overlaps
        done after load_ava where mappings are marked with c=6
        i.e. only trim stuff already in the pool

        :return:
        """
        logging.info('')
        trim_dict = self.ava.to_be_trimmed()
        logging.info(f"Trimming {len(trim_dict.keys())} sequences")
        # trim and ingest
        trimmed_seqs = self.pool.trim_sequences(trim_dict=trim_dict)
        trim_paf = self.pool.run_ava(sequences=trimmed_seqs,
                                     fa=f'{self.pool.fa}.trim',
                                     paf=f'{self.pool.ava}.trim')
        trim_contained, _ = self.ava.load_ava(paf=trim_paf, seqpool=self.pool)
        to_remove = self.ava.trim_success(trim_dict=trim_dict, overlaps=self.ava.overlaps)
        # remove original sequences & failed mergers
        self.remove_seqs(sequences=to_remove)


    def sim_batch(self) -> Dict:
        """
        Wrapper for simulation of a new batch of reads.
        Grab reads from a file, map them to current contigs, and make decisions

        :return: Dict of processed reads, i.e. rejected sequences are truncated
        """
        self.sim_run.stream.read_batch()
        # initialise a new LinearMapper for the current contigs
        lm = LinearMapper(ref=self.pool.contig_fa, mu=self.args.mu, default=False)
        paf_trunc = lm.mappy_batch(sequences=self.sim_run.stream.read_sequences,
                                   truncate=True,
                                   out=f'{self.args.name}.lm_out.paf')
        reads_decision = self.make_decision_paf(paf_out=paf_trunc,
                                                read_sequences=self.sim_run.stream.read_sequences,
                                                strat=self.strat)
        return reads_decision



    def cleanup(self) -> None:
        """
        Move temp files after a simulation

        :return:
        """
        tmpdir = f'{self.out_dir}/tmp'
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        execute(f'mv {self.args.name}.* {tmpdir}')


    def process_batch_live(self) -> float:
        """
        Process new batch of live reads. Scan directory for new data,
        and launch the common steps of processing

        :return: Time to wait until next update
        """
        # LIVE version
        logging.info(f"\n\n\n Next batch ---------------------------- # {self.batch}")
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
        self.rl_dist.update(read_lengths=fq_batch.read_lengths)
        new_reads = fq_batch.read_sequences
        # run remaining steps of update process
        next_update = self.process_batch_common(new_reads=new_reads)
        return next_update



    def process_batch_sim(self) -> None:
        """
        Process new batch of simulated data. Get new reads, make decisions,
        update times and write to files before launching the common steps of
        processing new data. Does not need to return waiting time during sims

        :return:
        """
        logging.info(f'\n NEW BATCH #############################  {self.batch}')
        reads_decision = self.sim_batch()
        # update read length dist, time recording
        self.rl_dist.update(read_lengths=self.sim_run.stream.read_lengths)
        self.sim_run.update_times(read_sequences=self.sim_run.stream.read_sequences,
                                  reads_decision=reads_decision)
        self.sim_run.write_batch(read_sequences=self.sim_run.stream.read_sequences,
                                 reads_decision=reads_decision)
        # run the rest of the steps to update
        _ = self.process_batch_common(new_reads=reads_decision)



    def process_batch_common(self, new_reads: Dict[str, str]) -> float:
        """
        The part of updates that is equal between sims and live runs

        :param new_reads: Dictionary of new header, sequence pairs
        :return: Waiting time until next update
        """
        tic = time.time()
        # filter sequences with repeats at the end
        if self.args.filter_repeats:
            reads_filtered = self.repeat_filter.filter_batch(seq_dict=new_reads)
        else:
            reads_filtered = new_reads

        # load new sequences, incl length filter
        sequences = SequencePool(sequences=reads_filtered, min_len=self.args.min_seq_len)
        # add new sequences to AVA
        self.add_new_sequences(sequences=sequences)
        # check for overlaps and containment in pool
        self.overlap_pool()
        # trim sequences that might lead to overlaps
        self.trim_sequences()
        # call wrapper to update assembly
        contigs = self.assemble_add_and_filter_contigs()
        contig_pool = ContigPool(sequences=contigs.sequences)
        # write the current pool to file for mapping against
        self.pool.write_seq_dict(seq_dict=contigs.seqdict(), file=self.pool.contig_fa)
        # check if we have any frozen sequences
        frozen_ids = self.pool.decrease_temperature(lim=self.args.min_contig_len)
        self.remove_seqs(sequences=frozen_ids)
        # update the sequencing masks
        self.strat = contig_pool.process_contigs(
            score_vec=self.score_vec,
            ccl=self.rl_dist.approx_ccl,
            out_dir=self.out_dir,
            mu=self.args.mu,
            lam=self.rl_dist.lam,
            batch=self.batch,
            write=True)
        # calc waiting time until next update
        # only used for live runs
        toc = time.time()
        passed = toc - tic
        next_update = int(self.args.wait - passed)
        logging.info(f"batch took: {passed}")
        logging.info(f"finished update, waiting for {next_update}s ... \n")
        self.batch += 1
        return next_update


    def __repr__(self):
        return str(self.__dict__)





class LiveRun:


    @staticmethod
    def split_flowcell(out_path: str, run_name: str) -> Set:
        """
        Perform the necessary steps if running multiple regions on a flowcell
        We wait until we find a channels TOML spec that contains the channel
        numbers assigned to the BOSS region on the flowcell.

        :param out_path: General MinKNOW output path
        :param run_name: Experiment name from config TOML
        :return: Set of channel IDs to consider data from
        """
        # DEBUG
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
    def connect_sequencer(device: str, host: str = 'localhost', port: int = None) -> str:
        """
        Connect to the running sequencer to get the path
        to its output directory

        :param device: Device name on sequencing machine
        :param host: Host to connect to
        :param port: Possibility to overwrite default port
        :return: Path of MinKNOW output directory
        """
        try:
            out_path = LiveRun._grab_output_dir(device=device, host=host, port=port)
            logging.info(f"grabbing Minknow's output path: \n{out_path}\n")
        except:
            logging.info("Minknow's output dir could not be inferred from device name. Exiting.")
            logging.info(f'\n{device}\n{host}\n{port}')
            if device == "TESTING":
                out_path = "."
                if not os.path.exists(f'{out_path}/fastq_pass'):
                    os.mkdir(f'{out_path}/fastq_pass')
            else:
                exit()
        return out_path



    @staticmethod
    def _grab_output_dir(device: str, host: str = 'localhost', port: int = None) -> str:
        """
        Capture the output directory of MinKNOW,
        i.e. where fastq files are deposited during sequencing
        host and port should be optional if run on the sequencing machine

        :param device: device name of the 'position' in the sequencing machine
        :param host: hostname to connect to for MinKNOW
        :param port: override default port to connect to
        :return: String of path where sequencing data is put my MinKNOW
        """
        logging.info(f"minknow API Version {minknow_api_version}")
        # minknow_api.manager supplies Manager (wrapper around MinKNOW's Manager gRPC)
        if minknow_api_version.startswith("5"):
            if not port:
                port = 9502
            manager = Manager(host=host, port=int(port))
        elif minknow_api_version.startswith("4"):
            if not port:
                port = 9501
            manager = Manager(host=host, port=int(port))  # , use_tls=False)
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
    def _grab_channels(channels_toml: str, run_name: str) -> Set:
        """
        Look into the channels toml that readfish writes
        This toml contains lists of channels assigned to each region
        Grab the channel numbers for the BOSS region

        :param channels_toml: Path to channels TOML from readfish
        :param run_name: experiment name of BOSS region
        :return: Set of channel numbers from which to consider data
        """
        toml_dict = rtoml.load(Path(channels_toml))
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
            logging.info("Experiment name in .toml not found in channel-specification toml. Exiting")
            exit()


    @staticmethod
    def scan_dir(fq: str, processed_files: set) -> List[str]:
        """
        Periodically scan Minknow's output dir and
        create new batch of sequences from all NEW files

        :param fq: Path of MinKNOW's output (fastq_pass)
        :param processed_files: Set of already processed files
        :return: List of new, previously unprocessed files
        """
        patterns = ["*.fq.gz", "*.fastq.gz", "*.fastq.gzip", "*.fq.gzip", "*.fastq", "*.fq"]
        all_fq = set()
        for p in patterns:
            all_fq.update(glob.glob(f'{fq}/{p}'))
        # which files have we not seen before?
        new_fq = all_fq.difference(processed_files)
        logging.info(f"found {len(new_fq)} new fq files")
        new_fq_list = [f for f in new_fq]
        return new_fq_list



class SimRun:

    def __init__(self, args: argparse.Namespace):
        """
        Initialisation of a simulation run. Few bits that are
        needed here, but not in a real sequencing experiment

        :param args: Config arguments Namespace
        """
        self.args = args
        # for keeping track of the sequencing time
        self.time_control = 0
        self.time_aeons = 0
        # for writing reads to file
        self.cache_control = dict()
        self.cache_aeons = dict()
        # after how much time should sequences be written to file
        # dump time is incremented every time a batch is written, which happens once that is overcome
        self.dump_every = self.args.dumptime
        self.dump_number_control = 1
        self.dump_number_aeons = 1
        # for storing the batches of reads for snakesembly
        if not os.path.exists('./00_reads'):
            os.mkdir('./00_reads')
        empty_file(f'00_reads/{self.args.name}_0_control.fa')
        empty_file(f'00_reads/{self.args.name}_0_aeons.fa')
        # init fastq stream - continous blocks for local (i.e. for reading from usb)
        self.stream = FastqStream_mmap(
            source=self.args.fq,
            batchsize=self.args.bsize,
            maxbatch=self.args.maxb,
        )
        # self.stream = FastqStream(
        #   source=self.args.fq,
        #   bsize=self.args.bsize
        # )



    def update_times(
        self,
        read_sequences: Dict[str, str],
        reads_decision: Dict[str, str],
        alpha: int = 200,
        rho: int = 300,
    ) -> None:
        """
        Increment pseudotime for control and aeons regions
        on a flowcell during simulations

        :param read_sequences: Dict of raw sequences, accepting everything
        :param reads_decision: Dict of processed sequences after decisions
        :param alpha: acquisition time constant
        :param rho: rejection time
        :return:
        """
        # for control: all reads as they come out of the sequencer
        # total bases + (#reads * alpha)
        bases_total = np.sum([len(seq) for seq in read_sequences.values()])
        acquisition = self.args.bsize * alpha
        self.time_control += (bases_total + acquisition)
        logging.info(f"time control: {self.time_control}")

        # for aeons: bases of the fully sequenced reads (accepted & unmapped) and of the truncated reads
        read_lengths_decision = np.array([len(seq) for seq in reads_decision.values()])
        n_reject = np.sum(np.where(read_lengths_decision == self.args.mu, 1, 0))
        bases_aeons = np.sum(read_lengths_decision)
        rejection_cost = n_reject * rho
        self.time_aeons += (bases_aeons + acquisition + rejection_cost)
        logging.info(f"time aeons: {self.time_aeons}")



    def write_batch(self, read_sequences: Dict[str, str], reads_decision: Dict[str, str]) -> None:
        """
        Write batches of simulated reads for convenient processing after the experiment
        Either only adds reads to a cache, or dumps them to file if it's time

        :param read_sequences: Dict of raw sequences
        :param reads_decision: Dict of sequences after BOSS decisions
        :return:
        """
        # helper function for both conditions
        def add_to_cache(seqs, cache):
            for rid, seq in seqs.items():
                cache[rid] = seq
        # add the current sequences to the cache
        add_to_cache(seqs=read_sequences, cache=self.cache_control)
        add_to_cache(seqs=reads_decision, cache=self.cache_aeons)
        # check if time to dump and execute
        self._prep_dump(cond='control')
        self._prep_dump(cond='aeons')



    def _prep_dump(self, cond: str) -> None:
        """
        Check if it's time to write cache to file, and execute if it is

        :param cond: String to determine which attributes to grab
        :return:
        """
        # grab the attributes of the condition
        curr_time = getattr(self, f'time_{cond}')
        dump_number = getattr(self, f'dump_number_{cond}')
        cache = getattr(self, f'cache_{cond}')
        # check if it's time to write out the next file
        if curr_time > (self.dump_every * dump_number):
            self._execute_dump(cond=cond, dump_number=dump_number, cache=cache)



    def _execute_dump(self, cond: str, dump_number: int, cache: Dict[str, str]) -> None:
        """
        Write out the next cumulative batch file from the cache.
        This is for simulations only

        :param cond: String indicating which region on the flowcell
        :param dump_number: Running number of the dump to execute
        :param cache: Cache of read sequences
        :return:
        """
        logging.info(f'dump {cond} #{dump_number}. # of reads {len(list(cache.keys()))}')
        filename = f'00_reads/{cond}_{dump_number}.fa'
        # copy previous file to make cumulative
        previous_filename = f'00_reads/{cond}_{dump_number - 1}.fa'
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




class FastqBatch:

    def __init__(self, fq_files: List[str], channels: Set):
        """
        Initialise a new batch of sequencing reads using their filepaths

        :param fq_files: Paths of new files
        :param channels: Set of channels of the BOSS region
        """
        self.fq_files = fq_files
        self.channels = channels
        self.read_batch()



    def read_batch(self) -> None:
        """
        read sequencing data from all new files

        :return:
        """
        read_sequences = {}
        for fq in self.fq_files:
            rseq = self._read_single_batch(fastq_file=fq)
            read_sequences.update(rseq)
        # set attributes of the batch
        self.read_sequences = read_sequences
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = {rid: len(seq) for rid, seq in read_sequences.items()}
        self.total_bases = np.sum(list(self.read_lengths.values()))
        logging.info(f'total new reads: {len(read_sequences)}')


    def _read_single_batch(self, fastq_file: str) -> Dict[str, str]:
        """
        Get the reads from a single fq file and put into dictionary

        :param fastq_file: One of the new fastq files
        :return: Dictionary of new sequences as header, sequence pairs
        """
        logging.info(f"Reading file: {fastq_file}")
        read_sequences = {}
        # to make sure it's a path object, not a string
        if type(fastq_file) is str:
            fpath = Path(fastq_file)
        elif type(fastq_file) is Path:
            fpath = fastq_file
        else:
            raise TypeError("New FASTQ paths need to be string or Path")

        # check whether fastq is gzipped
        if fpath.name.endswith(('.gz', '.gzip')):
            fh = gzip.open(fpath, 'rt')
        else:
            fh = open(fpath, 'rt')

        # loop over all reads in the fastq file
        # if we consider all channels
        if not self.channels:
            for desc, name, seq, qual in readfq(fh):
                read_sequences[str(name)] = seq
        else:
            # consider source channel
            for desc, name, seq, qual in readfq(fh):
                try:
                    # regex to get the channel number from the header
                    # \s=whitespace followed by 'ch=' and then any amount of numeric characters
                    curr_channel = re.search("\sch=[0-9]*", desc).group()
                    ch_num = int(curr_channel.split('=')[1])
                except AttributeError:
                    # if the pattern is not in the header, skip the read
                    logging.info("ch= not found in header of fastq read")
                    continue
                # check if read comes from a BOSS channel
                if ch_num in self.channels:
                    read_sequences[str(name)] = seq
        fh.close()
        return read_sequences








