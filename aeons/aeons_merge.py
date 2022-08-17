from gfapy.graph_operations.linear_paths import LinearPaths
import gfapy
from types import SimpleNamespace
import numpy as np
import line_profiler

"""
wrapper function takes a gfa object and uses logic from the gfapy module


"""



class ArrayMerger:

    def __init__(self, gfa):
        self.gfa = gfa
        # gfapy function to fetch all paths to be merged
        self.paths = LinearPaths.linear_paths(self=self.gfa, redundant_junctions=False)


    def merge_linear_arrs(self, seq_pool):
        # input dict
        self.seq_pool = seq_pool
        # output is dict
        merged_arrs = dict()

        for segpath in self.paths:
            # if the path is too short, we don't do anything
            if len(segpath) < 2:
                print("path length <2")
                return

            # make sure the elements are the correct gfapy object
            segpath = [gfapy.SegmentEnd(s) for s in segpath]
            name, seq, arr = self.create_merged_arr(segpath=segpath)
            merged_arrs[name] = arr
        return merged_arrs



    def create_merged_arr(self, segpath):
        # initialise a merged object from the first segment
        # merged = self.gfa.try_get_segment(segpath[0].segment).clone()
        merged = SimpleNamespace(name=[], sequence=[], arr=[])
        # the start becomes a
        a = segpath[0]
        first_reversed = (a.end_type == "L")

        # add the first segment to merged
        segment = self.gfa.segment(a.segment)
        self.add_array_to_merged(merged=merged, segment=segment, is_reversed=first_reversed, cut=0, init=True)

        for i in range(len(segpath) - 1):
            b = gfapy.SegmentEnd(segpath[i + 1]).inverted()
            ls = self.gfa.segment(a.segment).end_relations(a.end_type, b, "dovetails")
            if len(ls) != 1:
                print(f"A single link was expected between {a} and {b}. {len(ls)} were found")

            l = ls[0]
            if not l.overlap:
                cut = 0
            elif all(op.code in ["M", "="] for op in l.overlap):
                cut = sum([len(op) for op in l.overlap])
            else:
                cut = 0
                print(f"Merging is only allowed if all operations are M/=")
                exit()

            last_reversed = (b.end_type == "R")
            self.add_array_to_merged(merged=merged, segment=self.gfa.segment(b.segment),
                                     is_reversed=last_reversed, cut=cut, init=False)
            a = gfapy.SegmentEnd(b).inverted()


        merged.name = "_".join(merged.name)
        merged.sequence = "".join(merged.sequence)
        merged.arr = np.concatenate(merged.arr)
        # TODO compress array if recorded with some reduction in resolution
        return merged.name, merged.sequence, merged.arr




    def add_array_to_merged(self, merged, segment, is_reversed, cut, init):
        n = segment.name
        # grab the array of this sequence
        arr = self.seq_pool[n].cov
        # TODO: if coverage is not recorded at same length: do the expansion from BR
        # TODO using the length of the sequence and some reduction factor to stretch array

        if is_reversed:
            n = self._reverse_segment_name(segment.name, "_")
            s = gfapy.sequence.rc(segment.sequence)[cut:]   # TODO this is my own injected function
            c = arr[::-1][cut:]
        else:
            s = segment.sequence[cut:]
            c = arr[cut:]

        if init:
            merged.sequence = [s]
            merged.arr = [c]
            merged.name = [n]
        else:
            merged.sequence.append(s)
            merged.arr.append(c)
            merged.name.append(n)


    @staticmethod
    def _reverse_segment_name(name, separator):
        retval = []
        for part in name.split(separator):
            # cycle through parts and change the reverse indicator
            if part[-1] == "^":
                part = part[:-1]
            else:
                part += "^"
            retval.append(part)
        n = separator.join(reversed(retval))
        return n















