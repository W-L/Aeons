from io import StringIO
from collections import defaultdict

import numpy as np



class PafLine:
    '''
    @DynamicAttrs
    parse a single alignment from a PAF into a flexible container
    '''

    def __init__(self, line, tags=True):
        self.line = line
        # parsing into shape
        fields = ['qname', 'qlen', 'qstart', 'qend',
                  'strand', 'tname', 'tlen', 'tstart', 'tend',
                  'num_matches', 'alignment_block_length',
                  'mapq']
        core = 12
        record = line.strip().split("\t")

        f = format_records(record[:core])
        for i in range(core):
            setattr(self, fields[i], f[i])

        # make sure query and target name are strings
        self.qname = str(self.qname)
        self.tname = str(self.tname)

        self.rev = 0 if self.strand == '+' else 1
        # parse the tags only if needed
        if tags:
            tags_parsed = parse_tags(record[core:])
            self.align_score = int(tags_parsed.get("AS", 0))
            self.cigar = tags_parsed.get("cg", None)
            self.s1 = tags_parsed.get("s1", 0)
            prim = tags_parsed.get("tp", None)
            self.primary = 1 if prim == 'P' else 0
            # self.dv = tags_parsed.get('dv', 0)

        # markers for trimming
        self.qprox = False
        self.tprox = False
        # dummy inits
        self.maplen = None
        self.min_length_pair = None


    def filter(self, filters):
        # like classify, pack all conditions in here
        if self._self_aligned():
            return True
        if self.map_length() < filters.min_map_len:
            return True
        if self.s1 < filters.min_s1:
            return True
        if self.min_length_in_pair() < filters.min_seq_len:
            return True


    def min_length_in_pair(self):
        if not self.min_length_pair:
            self.min_length_pair = min(self.qlen, self.tlen)
        return self.min_length_pair


    def overhang(self):
        # sum of the smallest overhangs on both sequences
        # used in classification of mapping
        if not self.rev:
            overhang = min(self.qstart, self.tstart) +\
                       min(self.qlen - self.qend, self.tlen - self.tend)
        else:
            overhang = min(self.qstart, self.tlen - self.tend) +\
                       min(self.tstart, self.qlen - self.qend)
        return overhang


    def map_length(self):
        if not self.maplen:
            self.maplen = min(self.qend - self.qstart, self.tend - self.tstart)
        return self.maplen


    def classify(self):
        # classify the alignment according to miniasm alg 5
        # these are already filtered
        c = -1

        if self._internal_match():
            c = 1
        elif self._first_contained():
            c = 2
        elif self._second_contained():
            c = 3
        elif self._first_contained_fallback():
            c = 2
        elif self._second_contained_fallback():
            c = 3
        else:
            pass

        # if still unclassified -> overlap
        if c < 0:
            c, qside, tside = self._overlap_orientation()
            self.qside = qside
            self.tside = tside

        # second chance for internal matches
        # consider them as ovl under special circumstances
        if c == 1:
            if self._first_contained_mostly():
                c = 2
            elif self._second_contained_mostly():
                c = 3
            elif self.internal_match_is_overlap():
                c = 6  # class 6: needs trimming
            else:
                pass

        return c


    def _internal_match(self):
        if self.overhang() > (self.map_length() * 0.15):
            return True
        else:
            return False


    def _first_contained(self):
        # FORWARD
        if not self.rev:
            if (self.qstart <= self.tstart) and ((self.qlen - self.qend) < (self.tlen - self.tend)):
                return True
            else:
                return False
        # REVERSE
        else:
            if (self.qstart <= (self.tlen - self.tend)) and ((self.qlen - self.qend) < self.tstart):
                return True
            else:
                return False


    def _second_contained(self):
        # FORWARD
        if not self.rev:
            if (self.qstart >= self.tstart) and ((self.qlen - self.qend) > (self.tlen - self.tend)):
                return True
            else:
                return False
        # REVERSE
        else:
            if (self.qstart >= (self.tlen - self.tend)) and ((self.qlen - self.qend) > self.tstart):
                return True
            else:
                return False


    def _first_contained_fallback(self):
        qcov = self.qend - self.qstart
        if (qcov / self.qlen) >= 0.90:
            return True
        else:
            return False


    def _second_contained_fallback(self):
        tcov = self.tend - self.tstart
        if (tcov / self.tlen) >= 0.90:
            return True
        else:
            return False


    def _first_contained_mostly(self):
        qcov = self.qend - self.qstart
        if (qcov / self.qlen) >= 0.50 and self.qlen > 20000:
            return True
        else:
            return False

    def _second_contained_mostly(self):
        tcov = self.tend - self.tstart
        if (tcov / self.tlen) >= 0.50 and self.qlen > 20000:
            return True
        else:
            return False


    def _overlap_orientation(self):
        if not self.rev:
            if self.qstart > self.tstart:
                # 'A' overlaps 'B'
                # a + b +
                return 4, 'R', 'L'
            else:
                # B overlaps A
                # b + a +
                return 5, 'L', 'R'
        elif self.qstart > (self.qlen - self.qend):
            if self.qstart > (self.tlen - self.tend):
                # 'A' overlaps 'B'
                # a + b -
                return 4, 'R', 'R'    # should this be LR?
            else:
                # B overlaps A
                # b + a -
                return 5, 'R', 'R'
        elif (self.qlen - self.qstart) > self.tend:
            # 'A' overlaps 'B'
            # a - b +
            return 4, 'L', 'L'   # should this be RL?
        else:
            # B overlaps A
            # b - a +
            return 5, 'L', 'L'


    def _self_aligned(self):
        if self.qname == self.tname:
            return True
        else:
            return False


    def _is_merged(self):
        if '*' in self.qname and '*' in self.tname:
            return True
        else:
            return False


    @staticmethod
    def _is_prox(start, end, length, lim=1000.0):
        # check if a record has a mapped region close to one of its ends
        # masm definition would be lim = 1000
        # a fixed basepair limit does not work for us, we need to trim off more
        # i.e. if limit is given as percentage, calc a limit
        if lim < 1:
            limit = lim * length
        else:
            limit = lim
        overhang = min(start, length - end)
        ovl = True if overhang < limit else False
        return ovl


    def _im_ovl_restrictions(self, min_seqlen=15000, min_maplen=5000):
        # check some other restrictions to count internal match as OVL
        maplen = self.map_length()
        if self.qlen > min_seqlen:
            if self.tlen > min_seqlen:
                if maplen > min_maplen:
                    if self._is_merged():
                        return True
        return False


    def internal_match_is_overlap(self):
        # due to overlapping untrimmed reads,
        # we reconsider internal matches that might be overlaps
        # if one record has a true dovetail, check if the other has a relaxed dovetail
        lim = 0.15
        if self._is_prox(start=self.qstart, end=self.qend, length=self.qlen):
            self.qprox = True  # mark which side the true prox is for trimming
            if self._is_prox(start=self.tstart, end=self.tend, length=self.tlen, lim=lim):
                if self._im_ovl_restrictions():
                    # check a few more restrictions
                    # relaxed dovetail
                    return True
        elif self._is_prox(start=self.tstart, end=self.tend, length=self.tlen):
            self.tprox = True  # marker for trimming
            if self._is_prox(start=self.qstart, end=self.qend, length=self.qlen, lim=lim):
                if self._im_ovl_restrictions():
                    # relaxed dovetail
                    return True
        else:
            # neither sequence has a mapping close to the end
            # i.e. true internal match
            return False
        return False


    @staticmethod
    def _find_coords(start, end, length):
        # which side is closer to the end?
        min_overhang_idx = np.argmin([start, length - end])
        if min_overhang_idx == 0:
            # trim_start, trim_stop = start, None
            trim_start, trim_stop = 0, start
        else:
            # trim_start, trim_stop = 0, end
            trim_start, trim_stop = end, None
        return trim_start, trim_stop


    def find_trim_coords(self):
        # if this alignment has been identified to be useful when trimmed
        # find which of the sequences we want to trim
        # and which coordinates
        if self.qprox:
            # if q is a real prox, trim t
            sid = self.tname
            trim_start, trim_stop = self._find_coords(start=self.tstart, end=self.tend, length=self.tlen)
            other = self.qname
            other_len = self.qlen
            orig_len = self.tlen
        else:
            sid = self.qname
            trim_start, trim_stop = self._find_coords(start=self.qstart, end=self.qend, length=self.qlen)
            other = self.tname
            other_len = self.tlen
            orig_len = self.qlen

        # check that a potential trim would actually be longer than the original
        if trim_stop is None:
            stop = orig_len
        else:
            stop = trim_stop
        trimmed_bit = stop - trim_start
        # potential length of a merged sequence: original length minus trimmed bits
        # plus the length of the newly added sequence, minus the overlap length
        new_len = orig_len - trimmed_bit + other_len - self.alignment_block_length
        if new_len < orig_len:
            sid = 0

        return sid, trim_start, trim_stop, other


    def grab_increment_coords(self):
        # grab the coordinates of containment
        if self.c == 2:
            ostart = self.tstart
            oend = self.tend
            cstart = self.qstart
            cend = self.qend
        elif self.c == 3:
            ostart = self.qstart
            oend = self.qend
            cstart = self.tstart
            cend = self.tend
        else:
            print("this method should only be called on contained reads")
            return

        olen = oend - ostart
        clen = cend - cstart
        return ostart, oend, olen, cstart, cend, clen


    def keygen(self):
        if self.qname < self.tname:
            return f'{self.qname}-{self.tname}'
        else:
            return f'{self.tname}-{self.qname}'



    # for debugging
    # def plot(self, save=None):
    #     # visualize alignments as blocks
    #     import plotnine as p9
    #     import pandas as pd
    #     # this puts together the coordinates for the polygon
    #     # 5 points, starting at the lower left end
    #     if self.strand == '+':
    #         cols = ['qstart', 'qend', 'tend', 'tstart', 'qstart']
    #     else:
    #         cols = ["qend", "qstart", "tend", "tstart", "qend"]
    #
    #     pos = [getattr(self, c) for c in cols]
    #     # the query is always on top
    #     seqn = [2, 2, 1, 1, 2]
    #     cdat = pd.DataFrame({'pos': pos, 'seqn': seqn})
    #     # coordinates of the read rectangles
    #     seqlens = pd.DataFrame({'seq': [self.qname[:10], self.tname[:10]],
    #                             'start': [0, 0],
    #                             'end': [self.qlen, self.tlen],
    #                             'bottoms': [2.05, 0.8],
    #                             'tops': [2.2, 0.95]})
    #
    #     xpos = 100
    #     qpos = 1.9
    #     tpos = 1.1
    #     p = (p9.ggplot() +
    #          p9.geom_polygon(data=cdat,
    #                          mapping=p9.aes(x="pos", y="seqn"),
    #                          fill="grey", colour="black") +
    #          p9.geom_rect(data=seqlens,
    #                       mapping=p9.aes(xmin="start", xmax="end",
    #                                      ymin="bottoms", ymax="tops"),
    #                       colour="black", fill=None) +
    #          p9.annotate(x=xpos, label=self.qname[:10], y=2.12, geom='text', ha="left", va="center") +
    #          p9.annotate(x=xpos, label=self.tname[:10], y=0.87, geom='text', ha="left", va="center") +
    #          p9.annotate(x=self.qstart, label=self.qstart, y=qpos, color="darkred",
    #                      geom='text', ha="left", va="center") +
    #          p9.annotate(x=self.qend, label=self.qend, y=qpos, color="darkred",
    #                      geom='text', ha="left", va="center") +
    #          p9.annotate(x=self.tstart, label=self.tstart, y=tpos, color="darkblue",
    #                      geom='text', ha="left", va="center") +
    #          p9.annotate(x=self.tend, label=self.tend, y=tpos, color="darkblue",
    #                      geom='text', ha="left", va="center") +
    #          p9.annotate(x=xpos, label=self.strand, y=1.4, color="darkgoldenrod",
    #                      geom='text', ha="left", va="center", size=30) +
    #          p9.ylab("") +
    #          p9.xlab("") +
    #          p9.theme_minimal() +
    #          p9.theme(axis_text_y=p9.element_blank(),
    #                   axis_ticks_major_y=p9.element_blank(),
    #                   plot_background=p9.element_rect(fill="white", color="white")))
    #     if save:
    #         p.save(save)
    #     else:
    #         return p




class Paf:

    def __init__(self):
        pass


    @staticmethod
    def parse_PAF(paf_file, min_len):
        if isinstance(paf_file, str):
            with open(paf_file, 'r') as paff:
                paf_dict = Paf._parse_content(fh=paff, min_len=min_len)
        elif isinstance(paf_file, StringIO):
            paf_dict = Paf._parse_content(fh=paf_file, min_len=min_len)
        else:
            paf_dict = dict()
            print("need file path or StringIO")
        return paf_dict


    @staticmethod
    def _parse_content(fh, min_len):
        paf_dict = defaultdict(list)

        for record in fh:
            paf = PafLine(record)
            # FILTERING of PAF ENTRIES
            if paf.alignment_block_length < min_len:
                continue
            if not paf.primary:
                continue

            paf_dict[str(paf.qname)].append(paf)
        return paf_dict


    @staticmethod
    def parse_filter_classify_records(paf, filters):
        # parse a paf file by converting each line to Paf object
        # filtering and classifying mappings
        records = []
        skip = []
        with open(paf, 'r') as fh:
            for record in fh:
                rec = PafLine(record)
                # check if this mapping passes the filters
                is_filtered = rec.filter(filters=filters)
                if is_filtered:
                    # skip += 1
                    continue

                # classify the alignment
                rec.c = rec.classify()

                if rec.c == 1:
                    # internal match
                    skip.append(rec)
                    continue
                # if not filtered or skipped
                records.append(rec)
        return records, skip


def format_records(record):
    # Helper function to make fields the right type
    return [conv_type(x, int) for x in record]


def parse_tags(tags):
    """Convert a list of SAM style tags, from a PAF file, to a dict

    Parameters
    ----------
    tags : list
        A list of SAM style tags

    Returns
    -------
    dict
        Returns dict of SAM style tags
    """
    c = {"i": int, "A": str, "f": float, "Z": str}
    return {
        key: conv_type(val, c[tag])
        for key, tag, val in (x.split(":") for x in tags)
    }


def conv_type(s, func):
    # Generic converter, to change strings to other types
    try:
        return func(s)
    except ValueError:
        return s


def choose_best_mapper(records):
    # structured array to decide between ties, by using the score of the DP algorithm
    mapq = [(record.mapq, record.align_score) for record in records]
    custom_dtypes = [('q', int), ('dp', int)]
    mapping_qualities = np.array(mapq, dtype=custom_dtypes)
    sorted_qual = np.argsort(mapping_qualities, order=["q", "dp"])
    record = [records[sorted_qual[-1]]]
    return record

