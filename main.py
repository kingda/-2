from heapq import heappop, heappush
from datetime import datetime
import time  


import networkx as nx

# Create a graph
G = nx.Graph()
# Define the coordinates of each node (for the heuristic)
coordinates = {
    'A': (0, 0),
    'B': (1, 1),
    'C': (1, -1),
    'D': (2, 2),
    'E': (2, 0),
    'F': (3, -2),
    'G': (1, -2),
    'H': (3, 1),
    'I': (3, 2),
    'J': (1, 2),
    'K': (1, 3)
}

# Define the heuristic function using the coordinates
def euclidean_distance_between_nodes(a, b):
    # return G[a][b]['weight'];
    return euclidean_distance(coordinates[a], coordinates[b])


# Add edges and their weights to the graph
G.add_edge('A', 'B', weight=euclidean_distance_between_nodes('A','B'))
G.add_edge('A', 'C', weight=euclidean_distance_between_nodes('A','C'))
G.add_edge('B', 'E', weight=euclidean_distance_between_nodes('B','E'))
G.add_edge('B', 'D', weight=euclidean_distance_between_nodes('B','D'))
G.add_edge('C', 'F', weight=euclidean_distance_between_nodes('C','F'))
G.add_edge('C', 'G', weight=euclidean_distance_between_nodes('C','G'))
G.add_edge('G', 'K', weight=euclidean_distance_between_nodes('G','K'))
G.add_edge('K', 'J', weight=euclidean_distance_between_nodes('K','J'))
G.add_edge('D', 'H', weight=euclidean_distance_between_nodes('D','H'))
G.add_edge('E', 'I', weight=euclidean_distance_between_nodes('E','I'))
G.add_edge('E', 'J', weight=euclidean_distance_between_nodes('E','J'))

#print(G.get_edge_data('A', 'B'))



def astar(start, goal, neighbors, heuristic):
    # start: starting node
    # goal: goal node
    # neighbors: function that returns the neighbors of a node
    # heuristic: function that estimates the distance between two nodes
    frontier = []
    heappush(frontier, (0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while frontier:
        _, current = heappop(frontier)
        if current == goal:
            break

        for next in neighbors(current):
            new_cost = cost_so_far[current] + 1  # assuming edges have a cost of 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                heappush(frontier, (priority, next))
                came_from[next] = current

    path = []
    current = goal
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()

    return path



# Example problem: find the shortest path from node A to node J
# The graph is represented by a dictionary of neighbors for each node
graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': ['H'],
    'E': ['I', 'J'],
    'F': [],
    'G': ['K'],
    'H': [],
    'I': [],
    'J': [],
    'K': ['J']
}

# Define the heuristic function (Euclidean distance in this case)
import math
def euclidean_distance(a, b):
    ax, ay = a
    bx, by = b
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)

# Define the neighbors function
def get_neighbors(node):
    return graph[node]

# Define the start and goal nodes
start = 'A'
goal = 'J'


# Define the heuristic function using the coordinates
def euclidean_distance_between_nodes2(a, b):
    diu1 = G.get_edge_data(a, b)
    diu2 = G.get_edge_data(b, a)
    if diu1 != None :
      return diu1['weight']
    if diu2 != None: 
      return diu2['weight']
    return 0
    # return G[a][b]['weight'];
    # return euclidean_distance(coordinates[a], coordinates[b])





# Find the shortest path using A*

pre_time = time.time();
path = astar(start, goal, get_neighbors, euclidean_distance_between_nodes2)
print('A*:',time.time() - pre_time)
# Print the path
# print(path)




import heapq

def dijkstra_shortest_path(graph, start, end):
    # initialize distances to all nodes as infinite except the starting node
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    # use a priority queue to keep track of nodes to visit
    pq = [(0, start)]
    
    # keep track of the shortest path to each node
    shortest_path = {start: []}
    
    while pq:
        # pop the node with the smallest distance
        curr_dist, curr_node = heapq.heappop(pq)
        
        # ignore already processed nodes
        if curr_dist > distances[curr_node]:
            continue
        
        # iterate over the neighbors of the current node and update their distances and shortest path
        for neighbor, weight in graph[curr_node].items():
            new_dist = curr_dist + weight['weight']
            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                shortest_path[neighbor] = shortest_path[curr_node] + [curr_node]
                heapq.heappush(pq, (new_dist, neighbor))
        
        # if the end node is reached, return the shortest path to it
        if curr_node == end:
            return shortest_path[curr_node] + [end]
    
    # if end node is not reachable from start node, return empty path
    return []



# Find the shortest path between two nodes using Dijkstra's algorithm
pre_time = time.time();
path = dijkstra_shortest_path(G, start, goal)
print("dijkstra:",time.time() - pre_time)

# Find the length of the shortest path
length = nx.dijkstra_path_length(G, start, goal)

# Print the results
# print("Shortest path:", path)
# print("Length of shortest path:", length)






from collections import deque

import heapq

def bfs_shortest_path_weighted(graph, start, end):
    queue = [(0, start)]
    heapq.heapify(queue)
    
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    
    parents = {node: None for node in graph}
    
    while queue:
        current_distance, current_node = heapq.heappop(queue)
        
        if current_node == end:
            path = []
            while current_node != start:
                path.append(current_node)
                current_node = parents[current_node]
            path.append(start)
            path.reverse()
            return path
        
        for neighbor, weight in graph[current_node].items():
            # print('diu')
            # print(current_distance)
            # print('diu1')
            # print(weight)
            distance = current_distance + weight['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                parents[neighbor] = current_node
                heapq.heappush(queue, (distance, neighbor))


graph = {'A': {'B': 2, 'C': 1},
         'B': {'A': 2, 'D': 4, 'C': 3},
         'C': {'A': 1, 'B': 3, 'D': 5},
         'D': {'B': 4, 'C': 5}}



graph = {
    'A': {'B': euclidean_distance_between_nodes('A','B'), 'C': euclidean_distance_between_nodes('A','C')},
    'B': {'E': euclidean_distance_between_nodes('B','E'), 'D': euclidean_distance_between_nodes('B','D')},
    'C': {'F': euclidean_distance_between_nodes('C','F'),'G':euclidean_distance_between_nodes('C','G')},
    'G':{'K':euclidean_distance_between_nodes('G','K')},
    'K': {'J':euclidean_distance_between_nodes('K','J')},
    'D':{'H':euclidean_distance_between_nodes('D','H')},
    'E':{'I':euclidean_distance_between_nodes('E','I'),'J':euclidean_distance_between_nodes('E','J')}

}


pre_time = time.time();
shortest_path = bfs_shortest_path_weighted(G, start, goal)
print('BFS:',time.time() - pre_time)
print('path:',shortest_path)  # Output: ['A', 'B', 'D']








