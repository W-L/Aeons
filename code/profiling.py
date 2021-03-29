import line_profiler
lp = line_profiler.LineProfiler()
import longreadsim
import numpy as np
import os
os.chdir("/home/lukas/Desktop/Aeons/24_gfa")
cwd = os.getcwd()

#%%
# wrap functions
# update = lp(SparseGraph.update_graph)
# pc = lp(SparseGraph.path_counting)
cp = lp(SparseGraph.count_paths)

#%%
# run code

paths = cp(self=ar.dbg, node=node, t_solid=threshold)


lp.print_stats()