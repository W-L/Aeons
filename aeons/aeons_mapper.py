# STANDARD LIBRARY
from io import StringIO
from concurrent.futures import ThreadPoolExecutor as TPE
import logging
# NON STANDARD LIBRARY
import mappy
# CUSTOM
from .aeons_paf import Paf




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
        # before ingesting reads, check for overlaps and containment
        # also used to check if any reads from the readpool are contained
        paf_raw = self.mappy_batch(sequences=sequences, truncate=False)
        paf_dict = Paf.parse_PAF(StringIO(paf_raw), min_len=int(self.mu / 2))
        return paf_dict


    def mappy_batch(self, sequences, out="lm_out.paf", truncate=False, log=True):
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

        unmapped_ids = 0
        mapped_ids = 0
        # loop over all sequences and map them one by one
        with TPE(max_workers=self.workers) as executor:
            results = executor.map(self.map_query, sequences.items())

        for result in results:
            res = '\n'.join(result)
            # prevent appending an empty list if the read was not mapped
            if len(res) > 0:
                batch_alignments.append(res)
                mapped_ids += 1
            else:
                unmapped_ids += 1

        # transform to a single string
        alignments = '\n'.join(batch_alignments)

        # tmp write to file too
        with open(out, 'w') as lm_out:
            lm_out.write(alignments)

        # counting the number of mapped and unmapped fragments
        if log:
            logging.info(f"MAPPY: mapped queries: {mapped_ids}, unmapped queries: {unmapped_ids} ")
        self.mapped_ids = mapped_ids
        self.unmapped_ids = unmapped_ids
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

