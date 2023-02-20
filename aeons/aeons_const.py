"""
Constants used throughout aeons importable in other modules
"""


class Constants:
    def __init__(self):
        self.workers = 4                # so far used in fastqstream read retrieval
        self.mu = 400                   # initial bit of the read for decision making
        self.node_size = 100            # length of a single node in the graph in bases, determines strategy compression, hard-coded in readfish
        self.rho = 300
        self.alpha = 200
        self.wait = 60                  # waiting time in live version
        self.cov_wait = 2
        self.temperature = 30
        # self.redotable = "/hps/software/users/goldman/lukasw/redotable/redotable_v1.1/redotable"
        self.redotable = "/home/lukas/software/redotable/redotable_v1.1/redotable"

    def __repr__(self):
        return str(self.__dict__)


class Filters:
    def __init__(self):
        self.min_seq_len = 1500  # 3_000  # everything shorter than this will not even make it into the SequencePool
        self.min_contig_len = 10_000  # what to write out for mapping against
        self.min_s1 = 150  # 200
        self.min_map_len = 1000  # 2_000

    def __repr__(self):
        return str(self.__dict__)

