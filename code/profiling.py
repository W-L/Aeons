import line_profiler
lp = line_profiler.LineProfiler()

# run aeons until the dbg is initialised and updated once

# wrap functions
up = lp(SparseGraph.update_benefit)

# run code
read_lengths, read_sequences, basesTOTAL = fq.get_batch()
self.bloom.fill(reads=read_sequences)
self.rld.record(reads=read_sequences)
self.update_graph(updated_kmers=self.bloom.updated_kmers)
self.update_graph_p(updated_kmers=self.bloom.updated_kmers_p)
self.update_scores()
# self.update_benefit()
# this is the function that is replaced by the wrapped version to be profiled
up(self=self)

lp.print_stats()