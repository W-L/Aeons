import subprocess
from shutil import which
from sys import executable
from itertools import groupby
from argparse import Namespace
from typing import Tuple, TextIO, Dict

import numpy as np
from numpy.typing import NDArray



def empty_file(path: str) -> None:
    """
    Create an empty file at the specified path.

    :param path: The path to the file.
    """
    with open(path, 'w'):
        pass
    return


def execute(command: str) -> Tuple[str, str]:
    """
    Execute a command in a shell and return the stdout and stderr.

    :param command: The command to execute.
    :return: The stdout and stderr as a tuple.
    """
    # create the unix process
    running = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding='utf-8', shell=True)
    # run on a shell and wait until it finishes
    stdout, stderr = running.communicate()
    return stdout, stderr



def spawn(comm: str) -> subprocess.Popen:
    """
    Spawn a subprocess with the specified command.

    :param comm: The command to execute.
    :return: The spawned subprocess.
    """
    running = subprocess.Popen(comm, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               encoding='utf-8', shell=True)
    return running


def write_logs(stdout: str, stderr: str, basename: str) -> None:
    """
    Write the stdout and stderr from a subprocess to files.

    :param stdout: The stdout output.
    :param stderr: The stderr output.
    :param basename: The base name for the log files.
    """
    # write stdout and stderr from subprocess to file
    with open(f'{basename}.out', 'a') as outf:
        outf.write(stdout)
        outf.write('\n')
    with open(f'{basename}.err', 'a') as errf:
        errf.write(stderr)
        errf.write('\n')


def find_exe(name: str):
    """
    Find the executable file with the specified name.

    :param name: The name of the executable.
    :type name: str
    :return: The path to the executable, or None if not found.
    :rtype: str | None
    """
    # shutil.which seems to work mostly but is still not completely portable
    exe = which(name, path='/'.join(executable.split('/')[:-1]))
    if not exe:
        exe = which(name)
    if not exe:
        exe = subprocess.run(f'which {name}', shell=True, capture_output=True, universal_newlines=True).stdout
    if not exe:
        return
    return exe.strip()



def readfq(fp: TextIO):
    """
    Read a fastq file and yield the entries.

    :param fp: File handle for the fastq file.
    :yield: A tuple containing the fastq read header, read ID, and sequence.
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



def init_logger(logfile: str, args: Namespace) -> None:
    """
    Initialize the logger with the given logfile and log the arguments.

    :param logfile: The path to the logfile.
    :param args: The arguments to log.
    """
    empty_file(logfile)
    import logging
    logging.basicConfig(format='%(asctime)s %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(f"{logfile}"), logging.StreamHandler()])

    logging.info("AEONS")
    logging.info('\n')
    for a, aval in args.__dict__.items():
        logging.info(f'{a} {aval}')
    logging.info('\n')



def read_fa(fh: TextIO):
    """
    Generator for fasta files: yields all headers and sequences in the file.

    :param fh: The file handle.
    :yield: A tuple containing the header and sequence.
    """
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))
    for header in faiter:
        headerStr = header.__next__().strip().split(' ')[0]  # drop the ">"
        # join all sequence lines to one
        seq = "".join(s.strip() for s in faiter.__next__())
        yield headerStr, seq



def find_blocks_generic(arr: NDArray, x: int, min_len: int) -> NDArray:
    """
    Find blocks in the array that match x.

    :param arr: The input array.
    :param x: The value to find blocks of.
    :param min_len: The minimum length of blocks to report.
    :return: An array containing the start and end positions of the blocks.
    """
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



def find_blocks_ge(arr: NDArray, x: int, min_len: int) -> NDArray:
    """
    Find blocks in the array where the value is greater than or equal to x.

    :param arr: The input array.
    :param x: The value to compare against.
    :param min_len: The minimum length of blocks to report.
    :return: An array containing the start and end positions of the blocks.
    """
    # find run starts
    x_pos = np.where(arr >= x)[0]

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



def reverse_complement(dna: str) -> str:
    """
    Return the reverse complement of a DNA string.

    :param dna: The DNA sequence.
    :type dna: str
    :return: The reverse complement of the input DNA.
    :rtype: str
    """
    trans = str.maketrans('ATGC', 'TACG')
    rev_comp = dna.translate(trans)[::-1]
    return rev_comp



def random_id(k: int = 20) -> str:
    """
    Generate a random alphanumeric ID of length k.

    :param k: The length of the ID.
    :type k: int
    :return: The generated random ID.
    :rtype: str
    """
    import random
    import string
    x = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return x


def load_gfa(gfa_path: str) -> Dict[str, str]:
    """
    Load a GFA file and return a dictionary of sequences.

    :param gfa_path: The path to the GFA file.
    :return: A dictionary mapping header strings to sequence strings.
    """
    def _load_gfa(infile: str):
        with open(infile, 'r') as gfa_file:
            for line in gfa_file:
                if line.startswith('S'):
                    ll = line.split('\t')
                    header = ll[1]
                    seq = ll[2]
                    yield header, seq

    sequences = {}
    for hd, sq in _load_gfa(gfa_path):
        sequences[hd] = sq
    return sequences



# def redotable(fa, out, prg='/home/lukas/software/redotable/redotable_v1.1/redotable',
#               ref='/home/lukas/Desktop/Aeons/38_synmix/mod/ec.fa', size=1000,
#               logdir='redotable_log'):
#     # run redotable to create a dotplot compared to a reference
#     comm = f"{prg} --width {size} --height {size} --reordery {ref} {fa} {out}"
#     print(comm)
#     stdout, stderr = execute(comm)
#     write_logs(stdout, stderr, logdir)




