# STANDARD LIBRARY
from io import StringIO
from concurrent.futures import ThreadPoolExecutor as TPE
import logging
# NON STANDARD LIBRARY
import mappy
# CUSTOM
from .aeons_paf import Paf



class Indexer:
    """
    simple indexing wrapper around mappy
    """

    def __init__(self, fasta, mmi, t=6):
        self.aligner = mappy.Aligner(fn_idx_in=fasta, fn_idx_out=mmi, preset="map-ont", n_threads=t)



class LinearMapper:
    """
    mapping class for minimap2
    """

    def __init__(self, ref, mu=450, workers=8, default=True):
        self.mu = mu
        self.workers = workers
        # self.aligner = mappy.Aligner(fn_idx_in=ref, preset="map-ont")  # , extra_flags=0x400)
        if default:
            self.aligner = mappy.Aligner(fn_idx_in=ref, preset="map-ont")
        else:
            self.aligner = mappy.Aligner(fn_idx_in=ref, fn_idx_out=f'{ref}.mmi', preset="map-ont",
                                         k=13, w=5, min_cnt=2, min_chain_score=20)


    def map_sequences(self, sequences):
        # sequences is a dict
        # output is a dict with PafLine objects
        # before ingesting reads, check for overlaps and containment
        # also used to check if any reads from the readpool are contained
        paf_raw = self.mappy_batch(sequences=sequences, truncate=False)
        paf_dict = Paf.parse_PAF(StringIO(paf_raw), min_len=int(self.mu / 2))
        return paf_dict


    def mappy_batch(self, sequences, out=None, truncate=False, log=True):
        """
        Wrapper function that maps a full batch of reads. Can also be used for the in silico exp
        where we truncate each read to mu bases
        Parameters
        ----------
        sequences: dict
            dict of read_id: sequence items loaded as instance of current batch
        out: str
            name for a file to output mappings
        truncate: bool
            for in silico experiment we truncate the input reads. not needed for use during sequencing

        Returns
        -------
        alignments: str
            PAF formatted mapping hits of all mapped input reads
        """
        # container to hold the hits from all reads
        batch_alignments = []
        # for in silico experiments, truncate the reads
        if truncate:
            sequences = {read_id: seq[:self.mu] for read_id, seq in sequences.items()}

        unmapped_count = 0
        mapped_count = 0
        # loop over all sequences and map them one by one
        with TPE(max_workers=self.workers) as executor:
            results = executor.map(self.map_query, sequences.items())

        for result in results:
            res = '\n'.join(result)
            # prevent appending an empty list if the read was not mapped
            if len(res) > 0:
                batch_alignments.append(res)
                mapped_count += 1
            else:
                unmapped_count += 1

        # transform to a single string
        alignments = '\n'.join(batch_alignments)

        # tmp write to file too
        if out:
            with open(out, 'w') as lm_out:
                lm_out.write(alignments)

        # counting the number of mapped and unmapped fragments
        if log:
            logging.info(f"MAPPY: mapped queries: {mapped_count}, unmapped queries: {unmapped_count} ")
        self.mapped_count = mapped_count
        self.unmapped_count = unmapped_count
        return alignments


    def map_query(self, query):
        """
        Fast mapper that takes a query and returns the mapped query.
        Parameters
        ----------
        query: dict item
            key:value pair of read_id and query
        Returns
        -------
        list of str
            List of Paf formatted mapping hits
        """
        results = []
        read_id, seq = query

        thr_buf = mappy.ThreadBuffer()
        # For each alignment to be mapped against, returns a PAF format line
        for hit in self.aligner.map(seq, buf=thr_buf):
            # if hit.is_primary:
            results.append(f"{read_id}\t{len(seq)}\t{hit}")
        return results


class ValidationMapping:

    def __init__(self, mapper, merged_seq, seqpool):
        # merged_seq is newly merged sequence that we want to validate - Sequence()
        self.merged_seq = merged_seq
        # persistent linear mapper - created in AeonsRun
        self.mapper = mapper
        # we need the seqpool to grab the component sequences
        self.seqpool = seqpool
        # grab component sequences
        component_sequences = self.grab_components(self.merged_seq)
        # map the component seqs
        self.component_paf_dict = self.map_components_to_reference(component_sequences)
        self.component_plot = self.plot_components(self.component_paf_dict, "comp")
        # map the components onto the new merged sequence
        self.merger_paf_dict = self.map_components_to_merged(component_sequences)
        self.merger_plot = self.plot_components(self.merger_paf_dict, "merge")


    def grab_components(self, merged_seq):
        # grab the ids of components
        merger_component_ids = merged_seq.components
        logging.info(f'{len(merger_component_ids)} components in this merger')
        # grab the sequences
        component_sequences = dict()
        for comp_id in merger_component_ids:
            try:
                comp_seqo = self.seqpool.sequences[comp_id]
            except KeyError:
                logging.info("component sequence should have been found")
                continue
            component_sequences[comp_id] = comp_seqo.seq
        return component_sequences

    def map_components_to_reference(self, component_sequences):
        # map the components of the merged sequences to the reference
        comp_paf_dict = self.mapper.map_sequences(sequences=component_sequences)
        return comp_paf_dict

    def map_components_to_merged(self, component_sequences):
        # map components to their newly formed merged sequence
        # make fasta from merged sequence
        merged_fasta = f'>{self.merged_seq.header}\n{self.merged_seq.seq}'
        with open("tmp.fa", "w") as merged_fasta_file:
            merged_fasta_file.write(merged_fasta)
        merge_mapper = LinearMapper(ref="tmp.fa")
        merger_paf_dict = merge_mapper.map_sequences(sequences=component_sequences)
        return merger_paf_dict


    def plot_components(self, paf_dict, out):
        # use generic paf_dict plotting method from aeons_paf
        component_plot = Paf.plot_pafdict(paf_dict, out)
        return component_plot

