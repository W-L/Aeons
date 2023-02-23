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
        # self.cov_wait = 2
        # self.redotable = "/hps/software/users/goldman/lukasw/redotable/redotable_v1.1/redotable"
        # self.redotable = "/home/lukas/software/redotable/redotable_v1.1/redotable"

    def __repr__(self):
        return str(self.__dict__)


class Filters:
    def __init__(self, args):
        self.min_seq_len = 0
        self.min_contig_len = 0
        self.min_s1 = 0
        self.min_map_len = 0

        filters = {'min_seq_len', 'min_contig_len', 'min_s1', 'min_map_len'}
        # grab values from arguments
        for f in filters:
            setattr(self, f, getattr(args, f))

    def __repr__(self):
        return str(self.__dict__)

