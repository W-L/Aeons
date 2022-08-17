from collections import defaultdict
import os
import pathlib
import sys

import pyfastx

from .aeons_mapper import LinearMapper
from .aeons_utils import execute


class Polisher:

    def __init__(self, backbone_header, backbone_seq, atoms, read_sources):
        # grab atom sequences
        atom_dict = self.grab_atoms(atoms, read_sources)
        # write components to file
        self.to_file(backbone_header, backbone_seq, atom_dict)
        # perform mapping with mappy
        self.map_atoms(atom_dict)
        # ready to run racon


    def grab_atoms(self, atoms, read_sources):
        # this uses a dictionary of read_ids: file_sources
        # to find the file where the read is on disk
        sources = defaultdict(list)
        for h in atoms:
            if '*' in h:
                continue
            r_source = read_sources[h]
            sources[r_source].append(h)

        # open each file in turn to grab sequences
        atom_dict = dict()

        fq_patterns = {".fq", ".fastq"}
        fa_patterns = {".fa", ".fasta"}

        for filename, headers in sources.items():
            # load the fastx object for random access
            # load either as fasta or fastq depending on suffix
            file_suffixes = set(pathlib.Path(filename).suffixes)
            if fq_patterns.intersection(file_suffixes):
                fx = pyfastx.Fastq(filename)
            elif fa_patterns.intersection(file_suffixes):
                fx = pyfastx.Fasta(filename)
            else:
                FileNotFoundError(f'unsupported file format or file not found {filename}')
                sys.exit()

            for h in headers:
                # grab sequence
                h = h.replace('-', '_')  # TODO temp for preloaded contigs
                seq = fx[h].seq
                atom_dict[h] = seq
        return atom_dict


    def to_file(self, backbone_header, backbone_seq, atom_dict):
        # save the backbone and sequences to file
        self.contig_fa = f'{backbone_header.replace("*", "")}.fa'
        with open(self.contig_fa, 'w') as bb:
            bb.write(f'>{backbone_header}\n')
            bb.write(f'{backbone_seq}\n')

        self.read_fa = f'{backbone_header.replace("*", "")}.atoms.fa'
        with open(self.read_fa, 'w') as ba:
            for header, seq in atom_dict.items():
                ba.write(f'>{header}\n')
                ba.write(f'{seq}\n')


    def map_atoms(self, atom_dict):
        # init a linear mapper
        lm = LinearMapper(ref=self.contig_fa, workers=6)
        self.map_paf = f'{self.read_fa}.paf'
        lm.mappy_batch(sequences=atom_dict, out=self.map_paf, log=False)


    def run_polish(self):
        comm = f'racon -u --no-trimming {self.read_fa} {self.map_paf} {self.contig_fa}'
        stdout, stderr = execute(comm)
        # get just sequence
        outlines = stdout.split('\n')
        seq = outlines[1]
        self.cleanup()
        return seq


    def cleanup(self):
        # after polishing, remove the intermediate files
        # this is the reference contig, the reads, and mappings
        os.remove(self.contig_fa)
        os.remove(self.read_fa)
        os.remove(self.map_paf)

