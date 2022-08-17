import logging

# non-std lib
import numpy as np

"""
read length dist used in aeons
"""




class ReadlengthDist:

    def __init__(self, mu, sd=4000, lam=6000, eta=11):
        # initialise as truncated normal dist
        self.sd = sd
        self.lam = lam
        self.eta = eta
        self.mu = mu
        self.read_lengths = np.zeros(shape=int(1e6), dtype='uint16')
        # get the maximum read length
        longest_read = int(lam + 10 * sd)
        # prob density of normal distribution
        x = np.arange(longest_read, dtype='int')
        L = np.exp(-((x - lam + 1) ** 2) / (2 * (sd ** 2))) / (sd * np.sqrt(2 * np.pi))
        # exclude reads shorter than mu
        # L[:mu] = 0.0
        # normalise
        L /= sum(L)
        self.L = L
        # transform lambda to read length from distribution mode
        # mean_length = np.average(x, weights=L) + 1
        # get the stepwise approx
        self.approx_ccl = self.ccl_approx_constant()



    def update(self, read_lengths, recalc=False):
        # loop through the reads to keep track of their lengths
        for rid, length in read_lengths.items():
            # assign length of this read to the recording array
            # this is to ignore rejected reads for the read length dist
            # might overestimate the length slightly
            if length > self.mu * 2:
                self.read_lengths[length] += 1
            else:
                continue

        # calc current stats of read lengths
        if recalc:
            observed_read_lengths = np.nonzero(self.read_lengths)
            length_sum = np.sum(observed_read_lengths * self.read_lengths[observed_read_lengths])
            self.lam = length_sum / np.sum(self.read_lengths[observed_read_lengths])
            self.longest_read = np.max(np.where(self.read_lengths))
            self.L = np.copy(self.read_lengths[:self.longest_read]).astype('float64')
            self.L /= sum(self.L)
            # update approx CCL
            self.approx_ccl = self.ccl_approx_constant()
            logging.info(f'rld: {self.approx_ccl}')
            # update time cost
            # self.timeCost = self.lam - mu - rho



    def ccl_approx_constant(self):
        '''
        CCL is 1-CL, with CL the cumulative distribution of read lengths (L).
        \tilde CL in the manuscript section 0.1.3
        CCL starts at 1 and decreases as one increases the considered length.
        CCL[i] represents the probability that a random fragment (read) is at least i+1 long.
        CLL is then replaced by a piece-wise constant function.
        This is explained in the manuscript section 0.1.4 part "1) piecewise constant function".
        eta determines how many pieces to have; having 30 pieces
        makes updating the scores 3 times slower than having 10 pieces (at least I suspect)
        so careful not to use too high eta, however higher eta
        could improve the strategy.
        approx_CCL[i] tells you that the probability of reads being at least
        approx_CCL[i] long, is approximated by probability 1 - i / (eta-1),
        while the probability of reads being at least approx_CCL[i]+1 long,
        is approximated by probability 1-(i+1)/(eta-1).
        This is the same as saying that approx_CCL[i] is the point of the i-th
        change of value for the piece-wise constant function.
        approx_CCL contains eta - 1 values because the first and the last
        approximating probabilities are always 1 and 0 respectively.

        Parameters
        ----------
        L: np.array
            array containing the distribution of read lengths
        eta: int
            how many pieces to have in the piece-wise approximation. More = slower
            updating of the scores

        Returns
        -------
        approx_CCL: np.array
            array of length eta - 1, with each value as size of that
            partition

        '''
        # complement of cumulative distribtuion of read lengths
        ccl = np.zeros(len(self.L) + 1)
        ccl[0] = 1
        # instead of a loop we use numpy.cumsum, which generates an array itself
        ccl = 1 - self.L[1:].cumsum()
        # cut distribution off at some point to reduce complexity
        ccl[ccl < 1e-6] = 0
        # or just trim zeros
        ccl = np.trim_zeros(ccl, trim='b')
        self.ccl = ccl
        # to approx. U with a piecewise constant function
        # more partitions = more accurate, but slower (see update_U())
        approx_ccl = np.zeros(self.eta - 1, dtype='int32')
        i = 0
        for part in range(self.eta - 1):
            prob = 1 - (part + 0.5) / (self.eta - 1)
            while ccl[i] > prob:
                i += 1
            approx_ccl[part] = i
        return approx_ccl

