import random
from collections import deque

def computeMC(graph, S, R):
    ''' compute expected influence using MC under IC
        R: number of trials
    '''
    sources = set(S)
    inf = 0
    for _ in range(R):
        source_set = sources.copy()
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(child for child in graph.get_children(curr_node) \
                    if not(child in source_set) and random.random() <= graph.edges[(curr_node, child)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        inf += len(source_set)
        
    return inf / R

def workerMC(x):
    ''' for multiprocessing '''
    return computeMC(x[0], x[1], x[2])

def computeRR(graph, S, R, cache=None):
    ''' compute expected influence using RR under IC
        R: number of trials
        The generated RR sets are not saved; 
        We can save those RR sets, then we can use those RR sets
            for any seed set
        cache: maybe already generated list of RR sets for the graph
        l_c: a list of RR set covered, to compute the incremental score
            for environment step
    '''
    # generate RR set
    covered = 0
    generate_RR = False
    if cache is not None:
        if len(cache) > 0:
            # might use break for efficiency for large seed set size or number of RR sets
            return sum(any(s in RR for s in S) for RR in cache) * 1.0 / R * graph.num_nodes()
        else:
            generate_RR = True

    for i in range(R):
        # generate one set
        source_set = {random.randint(0, graph.num_nodes() - 1)}
        queue = deque(source_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                curr_source_set.update(parent for parent in graph.get_parents(curr_node) \
                    if not(parent in source_set) and random.random() <= graph.edges[(parent, curr_node)])
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            source_set |= curr_source_set
        # compute covered(RR) / number(RR)
        for s in S:
            if s in source_set:
                covered += 1
                break
        if generate_RR:
            cache.append(source_set)
    return covered * 1.0 / R * graph.num_nodes()

def workerRR(x):
    ''' for multiprocessing '''
    return computeRR(x[0], x[1], x[2])