import numpy as np
import graph_tool as gt
from graph_tool.draw import graph_draw
from graph_tool.topology import label_components
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize as colnorm
from matplotlib.colorbar import ColorbarBase


"""
library of plotting code for aeons

- mix of generic and special functions
- either for plots of the entire graph or subgraphs

"""


"""
# OLD SKELETON OF PLOTTING FUNCTION
def plot_gt(graph, vcolor=None, ecolor=None, hcolor=None, comp=None):
    # initialise figure to plot on to
    _, ax = plt.subplots()

    # transform edge weights to float for plotting
    if ecolor is not None:
        ecolor = ecolor.copy(value_type="float")

    vcol = vcolor if vcolor is not None else "grey"
    hcol = hcolor if hcolor is not None else [0, 0, 0, 0, 0]
    ecol = ecolor if ecolor is not None else ""

    # overwrite vcol with components if set
    if comp is not None:
        comp_label, _ = label_components(graph, directed=False)  # can be used for vertex_fill_color
        # print(set(comp_label))
        vcol = comp_label
        print(len(set(comp_label)))

    # if color is not None:
    #     obj_col.a = color
    # else:
    #     obj_col = "grey"

    a = graph_draw(graph, mplfig=ax,
                   #vertex_fill_color="grey",# vcmap=cm.coolwarm,        # vertex fill is used to show score/util
                   vertex_halo=True, vertex_halo_color=hcol,
                   vertex_text=vcol,                         # just indices
                   edge_text=ecol, edge_text_distance=0,                   # show edge weights as text
                   # edge_text=graph.edge_index, edge_text_distance=0,     # edge indices
                   #edge_color=ecol,#, vertex_size=2)#, ecmap=cm.coolwarm,                  # color edge weights
                   output_size=(3000, 3000), vertex_size=1)
    # return a
    plt.show(block=True)
"""


def create_layout(graph):
    """
    compute a layout that looks half-way decent for most applications
    passed to graph_draw & recyclable if topology is the same
    """
    pos = gt.draw.sfdp_layout(graph, multilevel=True, coarse_method="hybrid", max_iter=100)
    return pos



def plot_index(graph, name, layout=None, vsize=3):
    """
    plot with index numbers as node text
    """
    ind = graph.vp.ind.copy(value_type="float")

    pos = create_layout(graph) if not layout else layout

    graph_draw(graph, pos=pos,
               vertex_fill_color=ind,
               vcmap=cm.coolwarm,
               vertex_text=ind,
               vertex_size=vsize,
               output=f'{name}.pdf',
               output_size=(5000,5000))
    return pos





def plot_scores(graph, name, layout=None, vsize=10):
    scores = graph.vp.scores.copy(value_type="float")
    pos = create_layout(graph) if not layout else layout

    graph_draw(graph, pos=pos,
               vertex_fill_color=scores,
               vcmap=cm.coolwarm,
               vertex_text=graph.vp.npaths,
               vertex_size=vsize,
               output=f'{name}.pdf',
               output_size=(5000,5000))
    return pos



def plot_benefit(graph, name, layout=None, esize=10):
    w = graph.ep.weights.copy(value_type="float")
    pos = create_layout(graph) if not layout else layout
    graph_draw(graph, pos=pos,
               edge_color=w,
               ecmap=cm.coolwarm,
               edge_pen_width=5,
               edge_end_marker="arrow",
               edge_marker_size=esize,
               # vertex_fill_color="white",
               vertex_size=10,
               # vertex_color="white",
               vertex_shape="none",
               output=f'{name}.pdf',
               output_size=(10000,10000))
    return pos


def plot_strat(graph, name, layout=None):
    w = graph.ep.strat.copy(value_type="bool")
    pos = create_layout(graph) if not layout else layout

    cmap = plt.get_cmap("Set2").reversed()
    graph_draw(graph, pos=pos,
               edge_color=w,
               ecmap=cmap,
               edge_pen_width=5,
               edge_end_marker="arrow",
               edge_marker_size=10,
               # vertex_fill_color="white",
               vertex_size=10,
               # vertex_color="white",
               vertex_shape="none",
               output=f'{name}.pdf',
               output_size=(10000, 10000))
    return pos






def bfs(graph, node, maxsteps, track_edges=None, reverse=False):
    # graph is a sparse adjacency matrix
    # visited = set()
    visited = dict()
    visited_edges = set()
    queue = []
    steps = 0
    # add the starting node and put it in the queue
    visited[node] = steps
    queue.append(node)
    # iterate until either the queue is empty
    # or the maximum steps have been taken
    while steps < maxsteps and len(queue) > 0:
        s = queue.pop(0)
        # get the neighbors of the current node
        neighbours = graph[s, :].nonzero()[1]
        for neighbour in neighbours:

            if track_edges and reverse:
                visited_edges.add((s, neighbour))
                visited_edges.add((neighbour, s))

            if neighbour not in visited.keys():
                visited[neighbour] = steps
                queue.append(neighbour)

                # for filtering edges we save them in an extra set
                if track_edges:
                    visited_edges.add((s, neighbour))

                    # if we also need the reverse edge for filtering
                    if reverse:
                        visited_edges.add((neighbour, s))

                steps += 1
    return visited, visited_edges



def dfs(graph, node, maxsteps, targets=None):
    visited = dict()
    step = 0
    queue = [node]
    target_nodes = []

    while step <= maxsteps and len(queue) > 0:
        node = queue.pop()

        if node not in visited.keys():
            visited[node] = step
            step += 1

            # extra bit to get each node after 'maxsteps' steps from the source
            if targets and step == maxsteps:
                target_nodes.append(node)
                step = 1
                continue

            neighbours = graph[node, :].nonzero()[1]

            # if a node has only one neighbour we reset steps (i.e. at the loose end)
            if targets and len(neighbours) == 1:
                step = 0

            queue.extend(neighbours)
    return visited, target_nodes





def filter_from_node(mat, graph, ograph, steps, start, reverse):
    # BFS from one node to filter the graph
    # all visited edges will be preserved, all others eliminated
    # create a copy of the input sparse matrix
    # mat = dbg.adjacency
    mask = mat.copy()
    mask.data.fill(0)

    # breadth-first search that keeps track of visited nodes and traversed edges
    visited_nodes, visited_edges = bfs(graph=mat, node=start, maxsteps=steps, track_edges=True, reverse=reverse)

    # set the visited edges to True for filtering
    indices = np.array(list(visited_edges))
    mask[indices[:, 0], indices[:, 1]] = 1

    # iterate over the matrix to create a 1d filter
    cx = mask.tocoo()
    edge_prop = []
    for i, j, v in zip(cx.row, cx.col, cx.data):
        edge_prop.append(v)

    # create an edge prop from the mask
    emask = graph.new_edge_property("bool")
    emask.a = np.array(edge_prop)

    # create a vertex prop from the visited nodes
    # .vp.ind is created when the graph is initialised from the hashed kmer indices
    vertex_indices = list(ograph.vp.ind)
    # transform to integers
    vertex_int = np.array([int(i) for i in vertex_indices])
    vertex_walked = np.isin(vertex_int, list(visited_nodes))
    vmask = graph.new_vertex_property("bool")
    vmask.a = vertex_walked
    # finally apply the masks for edges and vertices
    graph_filt = gt.GraphView(graph, efilt=emask, vfilt=vmask)
    return graph_filt, mask






def plot_complex(dbg, totalsteps, junctiondist, name, start=None, layout=None):
    """
    this filters the graph
    - subplot of scores
    - subplot of benefits
    - n subplots for each node m steps away from the start junction

    all made with the same layout!
    """

    # given a starting node, get all nodes that are some distance away to check the metrics
    # for finding these starting points, we use a DFS
    if start:
        starting_point = start
        dfs_chain, dfs_targets = dfs(graph=dbg.adjacency, node=start, maxsteps=junctiondist, targets=True)

    # if the starting node is not given, use the most complex one
    else:
        paths = dbg.n_paths
        complex_nodes = np.where(paths.data == np.max(paths.data))
        pc = paths.tocoo()
        complex_indices = pc.row[complex_nodes]
        starting_point = complex_indices[0]
        dfs_chain, dfs_targets = dfs(graph=dbg.adjacency, node=starting_point, maxsteps=junctiondist, targets=True)


    # transform adjacency to gt format
    dbg.gt_format(mat=dbg.benefit)

    # filter the graph down to n nodes from the starting point
    gfilt, mask_filt = filter_from_node(mat=dbg.adjacency, graph=dbg.gtg, ograph=dbg.gtg, steps=totalsteps,
                                        start=starting_point, reverse=True)

    # check if a layout has been passed to the function
    pos = create_layout(gfilt) if not layout else layout


    # PLOT THE FILTERED GRAPH
    scores = gfilt.vp.scores.copy(value_type="float")
    benefit = gfilt.ep.weights.copy(value_type="float")
    strategy = gfilt.ep.strat.copy(value_type="float")



    graph_draw(gfilt, pos=pos,
               vertex_fill_color=scores,
               vcmap=cm.coolwarm,
               vertex_text=gfilt.vp.npaths,
               vertex_size=10,
               output=f'{name}_filt_scores.pdf',
               output_size=(1000, 1000))

    graph_draw(gfilt, pos=pos,
               # mplfig=ax1,
               edge_color=benefit,
               ecmap=cm.coolwarm,
               edge_pen_width=5,
               vertex_fill_color="white",
               vertex_size=8,
               vertex_color="white",
               vertex_shape="circle",
               edge_end_marker="arrow",
               nodesfirst=True,
               output=f'{name}_filt_benefit.pdf',
               output_size=(2000, 2000))

    graph_draw(gfilt, pos=pos,
               # mplfig=ax1,
               edge_color=strategy,
               ecmap=plt.get_cmap("Set2").reversed(),
               edge_pen_width=5,
               vertex_fill_color="white",
               vertex_size=8,
               vertex_color="white",
               vertex_shape="circle",
               edge_end_marker="arrow",
               nodesfirst=True,
               output=f'{name}_filt_strat.pdf',
               output_size=(2000, 2000))



    # now create a BFS walk from each of the DFS target nodes
    for i in range(len(dfs_targets)):
        gfilt1, mask = filter_from_node(mat=mask_filt, graph=gfilt, ograph=dbg.gtg,
                                        steps=totalsteps, start=dfs_targets[i], reverse=False)



        # weights contains whatever was passed to gt_format
        w = gfilt1.ep.weights.copy(value_type="float")

        vcolor = gfilt1.new_vertex_property("vector<double>")
        for v in gfilt1.vertices():
            vcolor[v] = [1, 1, 1, 1]
            if int(gfilt1.vp.ind[v]) == dfs_targets[i]:
                vcolor[v] = [0.807843137254902, 0.3607843137254902, 0.0, 1.0]


        graph_draw(gfilt1, pos=pos,
                   # mplfig=ax1,
                   edge_color=w,
                   ecmap=cm.coolwarm,
                   edge_pen_width=5,
                   vertex_fill_color=vcolor,
                   vertex_size=8,
                   vertex_color=vcolor,
                   vertex_shape="circle",
                   edge_end_marker="arrow",
                   nodesfirst=True,
                   output=f'{name}_{i}.pdf',
                   output_size=(1800, 1800))

    # draw a colorbar into an extra figure that spans the values of the other plot
    # plot_colorbar(np.min(w.a), np.max(w.a), f'{name.split(".")[0]}_cb.{name.split(".")[-1]}')
    # return the layout for reuse in other plots of the same graph
    return pos




def plot_colorbar(min, max, name):
    # create a standalone colormap
    # pasted onto the graph for now
    fig = plt.figure(figsize=(6, 1))
    ax = fig.add_subplot(111)

    cmap = cm.coolwarm
    norm = colnorm(vmin=min, vmax=max)

    # ColorbarBase derives from ScalarMappable and puts a colorbar in a specified axes
    # Many other possible args to explore
    ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    plt.tight_layout()
    plt.savefig(name)



def ps(ar):
    plt.plot(ar.dbg.scores.data, '.')
    plt.show()

def pb_raw(ar):
    plt.plot(ar.dbg.benefit_raw.data, '.')
    plt.show()

def pb(ar):
    plt.plot(ar.dbg.benefit.data, '.')
    plt.show()

def pa(ar):
    plt.plot(ar, '.')
    plt.show()



#%%



# def subset_graph(graph, mat, start, steps):
#     # to subset a graph for visualisation we perform a walk starting at some node
#     # the walk returns an edge mask and the set of visited nodes
#     edge_prop, visited_nodes = walk_graph(mat=mat, start=start, steps=steps)
#     # use the edge mask and the visited nodes to create a filtered version of the graph
#     graph_filt = apply_mask(graph=graph, mask=edge_prop, visited_nodes=visited_nodes)
#     return graph_filt
#
#
# def walk_graph(mat, start, steps):
#     # to create a visualisation of a subpart of a graph, we conduct a walk
#     # all visited edges will be preserved, all others eliminated
#     # create a copy of the input sparse matrix
#     mask = mat.copy()
#     mask.data.fill(0)
#
#     # breadth-first search that keeps track of visited nodes and traversed edges
#     visited_nodes, visited_edges = bfs(graph=mat, node=start, maxsteps=steps, track_edges=True)
#
#     # set the visited edges to True for filtering
#     indices = np.array(list(visited_edges))
#     mask[indices[:, 0], indices[:, 1]] = 1
#
#     # iterate over the matrix to create a 1d filter
#     cx = mask.tocoo()
#     edge_prop = []
#     for i, j, v in zip(cx.row, cx.col, cx.data):
#         edge_prop.append(v)
#
#     return np.array(edge_prop), set(visited_nodes.keys())
#
#
# def apply_mask(graph, mask, visited_nodes):
#     # create an edge prop from the mask
#     emask = graph.new_edge_property("bool")
#     emask.a = mask
#
#     # create a vertex prop from the visited nodes
#     # .vp.ind is created when the graph is initialised from the hashed kmer indices
#     vertex_indices = list(graph.vp.ind)
#     # transform to integers
#     vertex_int = np.array([int(i) for i in vertex_indices])
#     vertex_walked = np.isin(vertex_int, list(visited_nodes))
#     vmask = graph.new_vertex_property("bool")
#     vmask.a = vertex_walked
#     # finally apply the masks for edges and vertices
#     graph_filt = gt.GraphView(graph, efilt=emask, vfilt=vmask)
#     return graph_filt
