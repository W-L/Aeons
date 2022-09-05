import multiprocessing
from itertools import repeat
import logging
import os
import time
import mmap
import gzip

# non-std lib
import numpy as np

"""
module for random read sampling in aeons
two version:
    -- mmap
    -- random access without mmap
"""



class FastqStream:
    """
    this one samples randomly, i.e. can sample the same read multiple times
    seems to be slower on the cluster as well?
    FALLBACK FOR TESTING, NOT ACTUALLY USED
    """

    def __init__(self, source=None, bsize=4000, workers=10, seed=0):
        # need source file, batchsize (default 4k), workers
        self.source = source
        self.bsize = bsize
        self.workers = workers
        self.batch = 0
        np.random.seed(seed)


    def read_batch(self):
        # get a batch of randomly sampled reads without mem mapping - 3rd generation in BR
        read_lengths, read_sequences, read_sources, total_bases = self._parallel_batches()
        # assign attributes with fetched data
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = read_lengths
        self.read_sequences = read_sequences
        self.read_sources = read_sources
        self.total_bases = total_bases
        self.batch += 1
        logging.info(f'got new batch of {len(read_sequences)} reads')


    def _parallel_batches(self):
        pool = multiprocessing.Pool(processes=self.workers)
        # prep arg lists
        fqs = [self.source] * self.workers
        seeds = np.random.randint(10000, size=self.workers)
        # map the jobs to the workers
        batches = pool.starmap(self._get_random_batch, zip(fqs, repeat(int(self.bsize / self.workers)), seeds))
        # merge the output of all workers
        batches_concat = ''.join(batches)
        # parse the batch, which is just a long string into dicts
        read_lengths, read_sequences, read_sources, basesTOTAL = FastqStream.parse_batch(batch_string=batches_concat)
        return read_lengths, read_sequences, read_sources, basesTOTAL


    @staticmethod
    def parse_batch(batch_string):
        # take a batch in string format and parse it into some containers. Imitates reading from an actual fq
        read_lengths = {}
        read_sequences = {}
        read_sources = {}  # if fastq is speciallt formatted, grab the source

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
        total_bases = np.sum(np.array(list(read_lengths.values())))
        return read_lengths, read_sequences, read_sources, total_bases


    def _get_random_batch(self, fq, bsize, seed):
        batch = ''
        np.random.seed(seed)

        for b in range(bsize):
            read = self._get_random_read(filepath=fq)
            batch += read
        return batch


    def _get_random_read(self, filepath):

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
    """
    Stream reads from a fastq file (4 lines per read).
    Class only used for in silico experiments when we have a large fastq file that we randomly sample reads from
    advantage of this one is that we won't read the same read twice
    """
    def __init__(self, source, seed=1, shuffle=False, batchsize=1, maxbatch=1):
        self.source = source  # path to file-like object

        # check if file is gzipped. Not very good
        suffix = source.split('.')[-1]
        if suffix == 'gz':
            self.gzipped = True
        else:
            self.gzipped = False

        self.log_each = int(int(1e5))  # defining logging frequency of scanning for offsets
        self.filesize = int(os.stat(source).st_size)
        logging.info(f"Representing {self.filesize / 1e6} Mbytes of data from source: {self.source}")
        # scan the offsets if the file does not exist
        if not os.path.exists(f'{source}.offsets.npy'):
            logging.info("scanning offsets")
            self._scan_offsets()

        logging.info("loading offsets")
        self._load_offsets(seed=seed, shuffle=shuffle, batchsize=batchsize, maxbatch=maxbatch)
        self.batch = 0



    def _scan_offsets(self, k=4, limit=1e9):
        """
        Scan file to find byte offsets. Offsets are created for chunks of k lines each (4 for fastq)
        """
        tic = time.time()
        tmp_offsets = []
        read_num = 0

        # f = open(self.source, 'rb')
        with open(self.source, 'rb') as f:
            k_tmp = 1
            # memory-map the file; lazy eval-on-demand via POSIX filesystem
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # handle gzipped seemlessly
            if self.gzipped:
                mm = gzip.GzipFile(mode="rb", fileobj=mm)

            for _ in iter(mm.readline, b''):
                if k_tmp % k == 0:
                    pos = mm.tell()
                    tmp_offsets.append(pos)
                    k_tmp = 1
                    read_num += 1
                    # status update in case there are many reads
                    if read_num % self.log_each == 0:
                        logging.info(f"{read_num} reads scanned")
                else:
                    k_tmp += 1

                if read_num >= limit:
                    break

        toc = time.time()
        # convert to numpy array
        # uint32 for small and medium corpora?
        offsets = np.asarray(tmp_offsets, dtype='uint64')
        del tmp_offsets
        # write the offsets to a file
        np.save(f'{self.source}.offsets', offsets)
        logging.info(f"DONE scanning {read_num} reads")
        logging.info(f'wrote {len(offsets)} offsets to {self.source}.offsets.npy')
        logging.info(f"{round(toc - tic, 4)} seconds elapsed scanning file for offsets")


    def _load_offsets(self, seed, shuffle=False, batchsize=1, maxbatch=1):
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
            # sys.exit()

        # restructure the offsets into 2D array to represent batches (rows)
        offsets = offsets.reshape((maxbatch, batchsize))
        self.offsets = offsets



    def read_batch(self):
        # get a batch of randomly sampled reads without mem mapping - 3rd generation in BR
        read_lengths, read_sequences, read_sources, total_bases = self._get_batch()
        # assign attributes with fetched data
        self.read_ids = set(read_sequences.keys())
        self.read_lengths = read_lengths
        self.read_sequences = read_sequences
        self.read_sources = read_sources
        self.total_bases = total_bases
        self.batch += 1
        logging.info(f'got new batch of {len(read_sequences)} reads')



    def _get_batch(self, delete=True):
        """
        return a batch of reads from the fastq file
        """
        batch = ''
        # check if offsets are empty
        if self.offsets.shape[0] == 0:
            logging.info("no more batches left")
            # sys.exit()

        # f = open(self.source, 'rb')
        with open(self.source, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            # handle gzipped files
            if self.gzipped:
                mm = gzip.GzipFile(mode="rb", fileobj=mm)
            # the first row of the offsets are the next batch
            batch_offsets = self.offsets[0, :]

            # this is probably LINUX specific and not POSIX
            # here we tell the kernel what to do with the mapped memory
            # we tell it to preload specific "pages" of the file into memory
            # which magically makes it even faster to access
            # pagesize is a LINUX (system)-specific constant of 4096 bytes per "page"
            pagesize = 4096 #
            # the start of WILLNEED needs to be a multiple of pagesize
            # so we take the modulo and move the start of the offset a little bit earlier if needed
            new_offsets = batch_offsets - (batch_offsets % pagesize)

            for new_offset in new_offsets:
                # we preload 20 pages of data following each read start
                # 20 pages = 80 kbytes (read of up to ~40 kbases, I think..)
                mm.madvise(mmap.MADV_RANDOM)
                mm.madvise(mmap.MADV_WILLNEED, int(new_offset), 20)

            # offset = batch_offsets[0]
            batch_offsets = np.sort(batch_offsets)
            for offset in batch_offsets:
                try:
                    # here's the magic. Use the offset to jump to the position in the file
                    # then return the next 4 lines, i.e. one read
                    chunk = self._get_single_read(mm=mm, offset=offset)
                    # append the fastq entry to the batch
                    batch += chunk
                except:
                    logging.info(f"Error at location: {offset}")
                    continue
                if len(chunk) == 0:
                    continue

            # add call to close memory map, only file itself is under with()
            mm.close()

        if not batch.startswith('@') and not batch.startswith('>'):
            logging.info("The batch is broken")

        if delete:
            # remove the row from the offsets so it does not get sampled again
            new_offsets = np.delete(self.offsets, 0, 0)
            self.offsets = new_offsets
        # parse the batch, which is just a long string into dicts
        read_lengths, read_sequences, read_sources, basesTOTAL = FastqStream.parse_batch(batch_string=batch)
        return read_lengths, read_sequences, read_sources, basesTOTAL



    def _get_single_read(self, mm, offset):
        # return 4 lines from a memory-mapped fastq file given a byte-wise position
        mm.seek(offset)
        chunk_size = 4
        chunk = b''
        # read the 4 lines of the fastq entry
        for _ in range(chunk_size):
            chunk += mm.readline()

        chunk = chunk.decode("utf-8")
        return chunk


    def prefetch(self):
        # this is to look into a batch without actually using it
        # this way we can have a first idea about the read length dist
        # in simulation runs before using any data
        read_lengths, _, _, _= self._get_batch(delete=False)
        return read_lengths