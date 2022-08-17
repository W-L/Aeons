import subprocess
from shutil import which
from sys import executable
from itertools import groupby
import numpy as np
import argparse


class MyArgumentParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        return arg_line.split()




def read_args_fromfile(parser, file):
    with open(file, 'r') as f:
        arguments = [ll for line in f for ll in line.rstrip().split()]
    args = parser.parse_args(args=arguments)
    return args



def empty_file(path):
    with open(path, 'w'): pass
    return


def execute(command):
    # create the unix process
    running = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding='utf-8', shell=True)
    # run on a shell and wait until it finishes
    stdout, stderr = running.communicate()
    return stdout, stderr


def write_logs(stdout, stderr, basename):
    # write stdout and stderr from subprocess to file
    with open(f'{basename}.out', 'a') as outf:
        outf.write(stdout)
        outf.write('\n')
    with open(f'{basename}.err', 'a') as errf:
        errf.write(stderr)
        errf.write('\n')


def find_exe(name):
    # shutil.which seems to work mostly but is still not completely portable
    exe = which(name, path='/'.join(executable.split('/')[0:-1]))
    if not exe:
        exe = which(name)
    if not exe:
        exe = subprocess.run(f'which {name}', shell=True, capture_output=True, universal_newlines=True).stdout
    if not exe:
        return
    return exe.strip()


def conv_type(s, func):
    # Generic converter, to change strings to other types
    try:
        return func(s)
    except ValueError:
        return s




def readfq(fp):
    """
    GENERATOR FUNCTION
    Read a fastq file and return the sequence
    Parameters
    ----------
    fp: _io.IO
        File handle for the fastq file.

    Yields
    -------
    desc: str
        The fastq read header
    name: str
        The read ID
    seq: str
        The sequence

    """
    last = None  # this is a buffer keeping the last unprocessed line
    while True:  # mimic closure; is it a bad idea?
        if not last:  # the first record or a record following a fastq
            for ll in fp:  # search for the start of the next record
                if ll[0] in ">@":  # fasta/q header line
                    last = ll[:-1]  # save this line
                    break
        if not last:
            break
        desc, name, seqs, last = last[1:], last[1:].partition(" ")[0], [], None
        for ll in fp:  # read the sequence
            if ll[0] in "@+>":
                last = ll[:-1]
                break
            seqs.append(ll[:-1])
        if not last or last[0] != "+":  # this is a fasta record
            yield desc, name, "".join(seqs), None  # yield a fasta record
            if not last:
                break
        else:  # this is a fastq record
            seq, leng, seqs = "".join(seqs), 0, []
            for ll in fp:  # read the quality
                seqs.append(ll[:-1])
                leng += len(ll) - 1
                if leng >= len(seq):  # have read enough quality
                    last = None
                    yield desc, name, seq, "".join(seqs)  # yield a fastq record
                    break
            if last:  # reach EOF before reading enough quality
                yield desc, name, seq, None  # yield a fasta record instead
                break








def init_logger(logfile):
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(f"{logfile}"), logging.StreamHandler()])



def read_fa(fh):
    # iterator for all headers in the file
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        # drop the ">"
        headerStr = header.__next__().strip().split(' ')[0]
        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())
        yield headerStr, seq



def simple_read_fa(fa):
    read_sequences = {}

    with open(fa, 'r') as fa_file:
        for header, seq in read_fa(fa_file):
            read_sequences[header] = seq
    return read_sequences



def find_blocks_generic(arr, x, min_len):
    # find run starts
    x_pos = np.where(arr == x)[0]

    if x_pos.shape[0] == 0:
        return np.array([])

    # diff between neighboring loc
    x_diff = np.diff(x_pos)
    # if diff > 1: new block
    big_dist = np.where(x_diff > 1)[0]
    # the first entry is a block start and then all other where a big change happens
    # also each change is a block end, and the last entry of course as well
    block_ranges = np.concatenate((np.array([x_pos[0]]), x_pos[big_dist + 1],
                                   x_pos[big_dist] + 1, np.array([x_pos[-1] + 1])))
    blocks = block_ranges.reshape(big_dist.shape[0] + 1, 2, order='F')
    # only report blocks longer than min_len
    blocks_filt = blocks[np.where(blocks[:, 1] - blocks[:, 0] > min_len)[0], :]
    return blocks_filt


def range_intersection(r1, r2):
    return len(range(max(r1.start, r2.start), min(r1.stop, r2.stop)))


def reverse_complement(dna):
    """
    Return the reverse complement of a dna string. Used when parsing the cigar
    of a read that mapped on the reverse strand.

    Parameters
    ----------
    dna: str
        string of characters of the usual alphabet

    Returns
    -------
    rev_comp: str
        the reverse complemented input dna

    """
    trans = str.maketrans('ATGC', 'TACG')
    rev_comp = dna.translate(trans)[::-1]
    return rev_comp



def random_sparse(size, dens):
    from scipy.sparse import random
    # create a random matrix with some existing elements
    m = random(m=size, n=size, density=dens, format="csr")
    # how many are filled?
    n_vals = m.data.shape[0]
    # produce the filled 1s with random floats
    vals = np.random.random(size=n_vals).astype("float")
    vals = np.array([round(v, 2) for v in vals])
    m.data = vals
    return m


def random_id(k=20):
    import random, string
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


def window_sum(arr, w):
    # sums of non-overlapping windows
    sumw = np.sum(arr[: (len(arr) // w) * w].reshape(-1, w), axis=1)
    return sumw



def redotable(
    fa, out,
    prg='/home/lukas/software/redotable/redotable_v1.1/redotable',
    ref='/home/lukas/Desktop/Aeons/data/ecoli/ecoli_k12_U00096_3.fa',
    size=1000,
    logdir='redotable_log'):
    # run redotable to create a dotplot compared to a reference
    comm = f"{prg} --width {size} --height {size} --reordery {ref} {fa} {out}"
    print(comm)
    stdout, stderr = execute(comm)
    write_logs(stdout, stderr, logdir)



# might be unused
def chunk_data(data, window_size, overlap_size=0):
    from numpy.lib.stride_tricks import as_strided as ast
    # put into column-vector
    data = data.reshape((-1,1))
    # get number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)
    # if there's overhang, need an extra window and a zero pad on the data
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(data,
              shape=(num_windows,window_size*data.shape[1]),
              strides=((window_size-overlap_size)*data.shape[1]*sz,sz))
    return ret



# unused in aeons directly
def separate_by_species(paf):
    from contextlib import ExitStack
    # these need to be in the headers of the references mapped against
    ref_names = ["lm", "pa", "bs", "sc", "ec", "se", "lf", "ef", "cn", "sa"]
    # function to open a file for each ref_name
    def open_files(ref_names):
        fhs = {}
        with ExitStack() as cm:
            for name in ref_names:
                fhs[name] = cm.enter_context(open(f'zymo_even_on_accurate-{name}.paf', 'a'))
            cm.pop_all()
            return fhs
    fhs = open_files(ref_names)
    # loop through the inputfile
    # and separate out into individual files
    with open(paf, 'r') as inpf:
        for line in inpf:
            ll = line.split('\t')
            ref = ll[5].split('_')[0]
            # write line to correct file
            fhs[ref].write(line)
