import random
from collections import deque


def cp_pos(graph, S, R):        
    seed_nodes = set(S)
    inf = 0
    for _ in range(R):
        seed_nodes_set = seed_nodes.copy()
        queue = deque(seed_nodes_set)
        while True:
            curr_source_set = set()
            while len(queue) != 0:
                curr_node = queue.popleft()
                for child in graph.get_children(curr_node):
                    if not(child in seed_nodes_set) and random.random() <= graph.get_node_pro(child):
                    # if not(child in seed_nodes_set) and random.random() <= graph.edges[(curr_node, child)]:
                        curr_source_set.add(child)
            if len(curr_source_set) == 0:
                break
            queue.extend(curr_source_set)
            seed_nodes_set |= curr_source_set
        inf += len(seed_nodes_set)

    # G = cls.build_user_network2(account_info_list)

    # for node in G:
    #     G.nodes[node]['state'] = 0

    # seed_nodes = selected_id_nodes
    # for seed in seed_nodes:
    #     G.nodes[seed]['state'] = 1

    
    # start_influence_nodes = seed_nodes[:]
    # count_activated = len(seed_nodes)
    # activated_count = [count_activated]


    # for i in range(max_iter_num):
    #     for v in start_influence_nodes:
    #         for nbr in G.neighbors(v):
    #             if G.nodes[nbr]['state'] == 0:
    #                 if random.uniform(0, 1) < G.nodes[nbr]['w']:
    #                     G.nodes[nbr]['state'] = 1
                        
    #                     count_activated += 1
            
    #                 else:
    #                     G.nodes[nbr]['state'] = 2
    #     activated_count.append(count_activated)
    return inf / R

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