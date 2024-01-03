import multiprocessing
from itertools import repeat
import logging
import os
import time
import mmap
import gzip
import sys
from typing import Tuple, Dict, Any

import numpy as np


# module for random read sampling in aeons with two versions: mmap, random access without mmap



class FastqStream:

    def __init__(self, source: str = None, bsize: int = 4000, workers: int = 8, seed: int = 0):
        """
        Initialize FastqStream with random sampling without mmap
        Uses random jump into file instead of pre-recorded offsets
        This one can sample the same read multiple times
        Seems slower on cluster arch
        FOR TESTING, NOT USED IN AEONS

        :param source: Source sequence file.
        :param bsize: Batch size (default: 4000).
        :param workers: Number of workers.
        :param seed: Seed for random number generation.
        """
        self.source = source
        self.bsize = bsize
        self.workers = workers
        self.batch = 0
        np.random.seed(seed)



    def read_batch(self) -> None:
        """
        Get a batch of randomly sampled reads without memory mapping.

        :return: None
        """
        read_lengths, read_sequences, read_sources, total_bases = self._parallel_batches()
        # assign attributes with fetched data
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = read_lengths
        self.read_sequences = read_sequences
        self.read_sources = read_sources
        self.total_bases = total_bases
        self.batch += 1
        logging.info(f'got new batch of {len(read_sequences)} reads')



    def _parallel_batches(self) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str], int]:
        """
        Generate batches of reads in parallel.

        :return: Tuple containing dictionaries of read lengths, sequences, sources, and the total number of bases.
        """
        pool = multiprocessing.Pool(processes=self.workers)
        # prep arg lists
        fqs = [self.source] * self.workers
        seeds = np.random.randint(10000, size=self.workers)
        # map the jobs to the workers
        batches = pool.starmap(self._get_random_batch, zip(fqs, repeat(int(self.bsize / self.workers)), seeds))
        # merge the output of all workers
        batches_concat = ''.join(batches)
        # parse the batch (long string) into dicts
        read_lengths, read_sequences, read_sources, basesTOTAL = FastqStream.parse_batch(batch_string=batches_concat)
        return read_lengths, read_sequences, read_sources, basesTOTAL



    @staticmethod
    def parse_batch(batch_string: str) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str], int]:
        """
        Parse a batch in string format into dictionaries.

        :param batch_string: Batch in string format.
        :return: Tuple containing dictionaries of read lengths, sequences, sources, and the total number of bases.
        """
        read_lengths = {}
        read_sequences = {}
        read_sources = {}

        batch_lines = batch_string.split('\n')
        n_lines = len(batch_lines)

        i = 0
        # since we increment i by 4 (lines for read in fq), loop until n_lines - 4
        while i < (n_lines - 4):
            # grab the name of the read. split on space, take first element, trim the @
            desc = batch_lines[i].split(' ')
            name = str(desc[0][1:])
            source = str(desc[-1])
            seq = batch_lines[i + 1]
            read_len = len(seq)
            # update the containers
            read_lengths[name] = read_len
            read_sequences[name] = seq
            read_sources[name] = source
            i += 4
        # get the total length of reads in this batch
        total_bases = int(np.sum(np.array(list(read_lengths.values()))))
        return read_lengths, read_sequences, read_sources, total_bases


    def _get_random_batch(self, fq: str, bsize: int, seed: int) -> str:
        """
        Generate a random batch of reads.

        :param fq: Filepath.
        :param bsize: Batch size.
        :param seed: Seed for random number generation.
        :return: Batch of reads.
        """
        batch = ''
        np.random.seed(seed)
        for b in range(bsize):
            read = self._get_random_read(filepath=fq)
            batch += read
        return batch


    @staticmethod
    def _get_random_read(filepath: str) -> str:
        """
        Get a random read from the file.

        :param filepath: Filepath.
        :return: Random read.
        """
        file_size = os.path.getsize(filepath)
        fq_lines = b''
        proper_line = b''

        with open(filepath, 'rb') as f:
            while True:
                pos = np.random.randint(1, file_size)
                f.seek(pos)  # seek to random position
                f.readline()  # skip possibly incomplete line
                # skip lines until we are at a fastq header
                while not proper_line.decode().startswith('@'):
                    proper_line = f.readline()
                # add the fastq header to the read
                fq_lines += proper_line
                for i in range(3):  # add the other 3 fastq lines as well
                    fq_lines += f.readline()
                if fq_lines:
                    return fq_lines.decode()





class FastqStream_mmap:

    def __init__(self, source: str, seed: int = 1, shuffle: bool = False, batchsize: int = 1, maxbatch: int = 1):
        """
        Stream reads from a fastq file (4 lines per read) during simulations
        This implementation won't sample the same read twice

        :param source: Path to the file-like object.
        :param seed: Seed for random number generation.
        :param shuffle: Whether to shuffle the offsets.
        :param batchsize: Batch size.
        :param maxbatch: Maximum number of batches.
        """
        self.source = source
        # check if file is gzipped. Not very good
        suffix = source.split('.')[-1]
        if suffix == 'gz':
            self.gzipped = True
        else:
            self.gzipped = False

        self.log_each = int(int(1e5))  # log frequency of scanning for offsets
        self.filesize = int(os.stat(source).st_size)
        logging.info(f"Representing {self.filesize / 1e6} Mbytes of data from source: {self.source}")
        # scan the offsets if the file does not exist
        if not os.path.exists(f'{source}.offsets.npy'):
            logging.info("scanning offsets")
            self._scan_offsets()
        logging.info("loading offsets")
        self._load_offsets(seed=seed, shuffle=shuffle, batchsize=batchsize, maxbatch=maxbatch)
        self.batch = 0



    def _scan_offsets(self, k: int = 4, limit: float = 1e9) -> None:
        """
        Scan the file to find byte offsets of each sequencing read.

        :param k: Number of lines per chunk, default chink size is 4 for fastq
        :param limit: Maximum number of reads to scan.
        :return: None.
        """
        tic = time.time()
        tmp_offsets = []
        read_num = 0

        with open(self.source, 'rb') as f:
            k_tmp = 1
            # memory-map the file; lazy eval-on-demand via POSIX filesystem
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if self.gzipped:
                mm = gzip.GzipFile(mode="rb", fileobj=mm)

            for _ in iter(mm.readline, b''):
                if k_tmp % k == 0:
                    pos = mm.tell()
                    tmp_offsets.append(pos)
                    k_tmp = 1
                    read_num += 1
                    # status update
                    if read_num % self.log_each == 0:
                        logging.info(f"{read_num} reads scanned")
                else:
                    k_tmp += 1

                if read_num >= limit:
                    break

        toc = time.time()
        offsets = np.asarray(tmp_offsets, dtype='uint64')
        del tmp_offsets
        # write the offsets to a file
        np.save(f'{self.source}.offsets', offsets)
        logging.info(f"DONE scanning {read_num} reads")
        logging.info(f'wrote {len(offsets)} offsets to {self.source}.offsets.npy')
        logging.info(f"{round(toc - tic, 4)} seconds elapsed scanning file for offsets")



    def _load_offsets(self, seed: int = 1, shuffle: bool = False, batchsize: int = 1, maxbatch: int = 1) -> None:
        """
        Load offsets of sequencing reads in a fastq file.

        :param seed: Seed for random number generation.
        :param shuffle: Whether to shuffle the offsets.
        :param batchsize: Batch size.
        :param maxbatch: Maximum number of batches.
        :return: None.
        """
        if seed == 0:
            seed = np.random.randint(low=0, high=int(1e6))
        np.random.seed(seed)
        offsets = np.load(f'{self.source}.offsets.npy')
        # add one batch for initialising length dist
        maxbatch = maxbatch + 1

        if shuffle:
            np.random.shuffle(offsets)
            logging.info(f"offsets shuffled using random seed: {seed}")

        # shorten the offsets to number of reads we need
        len_offsets = len(offsets)
        logging.info(f"available batches: {len_offsets / batchsize}")
        n_reads = batchsize * maxbatch
        if n_reads < len_offsets:
            offsets = offsets[: n_reads]
        else:
            logging.info("requested more reads than there are available in the fastq")
            sys.exit(1)

        # restructure the offsets into 2D array to represent batches (rows)
        offsets = offsets.reshape((maxbatch, batchsize))
        self.offsets = offsets



    def read_batch(self) -> None:
        """
        Get a batch of randomly sampled reads without memory mapping.

        :return: None
        """
        read_lengths, read_sequences, read_sources, total_bases = self._get_batch()
        # assign attributes with fetched data
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = read_lengths
        self.read_sequences = read_sequences
        self.read_sources = read_sources
        self.total_bases = total_bases
        self.batch += 1
        logging.info(f'got new batch of {len(read_sequences)} reads')



    def _get_batch(self, delete: bool = True) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, str], int]:
        """
        Return a batch of reads from the fastq file.

        :param delete: Whether to delete the batch offsets after retrieval.
        :return:
        """
        # check if offsets are empty
        if self.offsets.shape[0] == 0:
            logging.info("no more batches left")
            sys.exit(1)

        with open(self.source, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            if self.gzipped:
                mm = gzip.GzipFile(mode="rb", fileobj=mm)
            # the first row of the offsets are the next batch
            batch_offsets = self.offsets[0, :]
            # initialise list instead of string concat
            batch = [''] * len(batch_offsets)
            # possibly LINUX specific and not POSIX
            # here we tell the kernel to preload pages of the mapped memory
            # magically makes it even faster to access
            # pagesize is a LINUX (system)-specific constant of 4096 bytes per "page"
            pagesize = 4096
            # the start of WILLNEED needs to be a multiple of pagesize
            # we take the modulo and move the start of the offset a bit earlier if needed
            new_offsets = batch_offsets - (batch_offsets % pagesize)

            for new_offset in new_offsets:
                # we preload 20 pages of data following each read start
                # 20 pages = 80 kbytes (read of up to ~40 kbases, I think)
                mm.madvise(mmap.MADV_RANDOM)
                mm.madvise(mmap.MADV_WILLNEED, int(new_offset), 20)

            batch_offsets = np.sort(batch_offsets)
            for i in range(len(batch_offsets)):
                try:
                    # jump to position in file and return the next 4 lines
                    chunk = self._get_single_read(mm=mm, offset=int(batch_offsets[i]))
                    batch[i] = chunk
                except:
                    logging.info(f"Error at location: {batch_offsets[i]}")
                    continue
                if len(chunk) == 0:
                    continue

            # add call to close memory map, only file itself is under with()
            mm.close()

        if not batch[0].startswith('@') and not batch[0].startswith('>'):
            logging.info("The batch is broken")

        if delete:
            # remove the row from the offsets, so it does not get sampled again
            new_offsets = np.delete(self.offsets, 0, 0)
            self.offsets = new_offsets
        # parse the batch (long string) into dicts
        batch_string = ''.join(batch)
        read_lengths, read_sequences, read_sources, basesTOTAL = FastqStream.parse_batch(batch_string=batch_string)
        return read_lengths, read_sequences, read_sources, basesTOTAL


    @staticmethod
    def _get_single_read(mm: Any, offset: int) -> str:
        """
        Return 4 lines from a memory-mapped fastq file given a byte-wise position.

        :param mm: The memory-mapped fastq file.
        :param offset: The byte-wise position of the read.
        :return: The read chunk.
        """
        mm.seek(offset)
        chunk_size = 4
        chunk = b''
        # read 4 lines of the fastq entry
        for _ in range(chunk_size):
            chunk += mm.readline()
        chunk = chunk.decode("utf-8")
        return chunk






