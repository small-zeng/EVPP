import collections  #Some better containers are provided
import heapq
import imp        #heap algorithm is provided
import itertools
import numpy as np
import torch
import time
    
class MinheapPQ:
    """
    A priority queue based on min heap, which takes O(logn) on element removal
    https://docs.python.org/3/library/heapq.html#priority-queue-implementation-notes
    """
    def __init__(self):
        self.pq = [] # list of the entries arranged in a heap
        self.nodes = set()
        self.entry_finder = {} # mapping of the item entries
        self.counter = itertools.count() # unique sequence count
        self.REMOVED = '<removed-item>'
    
    def put(self, item, priority):
        '''add a new task or update the priority of an existing item'''
        if item in self.entry_finder:
            self.check_remove(item)
        count = next(self.counter)
        entry = [priority, count, item]
        self.entry_finder[item] = entry
        heapq.heappush(self.pq, entry)
        self.nodes.add(item)

    def check_remove(self, item):
        entry = self.entry_finder.pop(item)
        entry[-1] = self.REMOVED
        self.nodes.remove(item)

    def get(self):
        """Remove and return the lowest priority task. Raise KeyError if empty."""
        while self.pq:
            priority, count, item = heapq.heappop(self.pq)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                self.nodes.remove(item)
                return item
        raise KeyError('pop from an empty priority queue')

    def top_key(self):
        return self.pq[0][0]
        
    def enumerate(self):
        return self.pq

    def allnodes(self):
        return self.nodes

class Weighted_A_star(object):
    def __init__(self, start, goal, env, resolution=0.1):
        self.Alldirec = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                        (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                        (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                        (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                        (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                        (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                        (1, 1, 1): np.sqrt(3), (-1, -1, -1) : np.sqrt(3), \
                        (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                        (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}
        self.settings = 'NonCollisionChecking' 
        self.start, self.goal = tuple(start), tuple(goal)
        self.heuristic_fun = env.get_gain
        self.env = env;        
        self.resolution = resolution  
        self.voxelSize = 0.05  
        self.lamda =  0.5     
        self.g = {self.start:0,self.goal:np.inf}
        self.Parent = {}
        self.Point = {self.start:[0,0,0]} # 点的位置：到起点的路径长度,路径增益和，点的数目
        self.CLOSED = set()
        self.V = []
        self.done = False
        self.Path = []
        self.ind = 0
        self.x0, self.xt = self.start, self.goal
        self.OPEN = MinheapPQ()  # store [point,priority]
        # self.OPEN.put(self.x0, self.g[self.x0] - env.get_gain_discret(list(self.x0)))  # item, priority = g + h
        self.OPEN.put(self.x0, self.getDist(self.x0, self.xt))
        self.lastpoint = self.x0
        self.query_num = 0
        self.query_time = 0
       

    def getDist(self, pos1, pos2):
        return np.sqrt(sum([(pos1[0] - pos2[0]) ** 2, (pos1[1] - pos2[1]) ** 2, (pos1[2] - pos2[2]) ** 2]))
    
    def children(self, x):
        all_child = []
        for direc in self.Alldirec:
            child = tuple(map(np.add, x, np.multiply(direc, self.voxelSize)))
            state , isvalid = self.env.tsdf.get_state_cpu(child) 
            if isvalid == False or state != 1:
                continue
            all_child.append(child)
        return all_child
            
            
    def run(self, N=None):
        xt = self.xt
        xi = self.x0
        while self.OPEN:  # while xt not reached and open is not empty
            xi = self.OPEN.get()
            # print(len(self.OPEN.pq))
            if xi not in self.CLOSED:
                self.V.append(np.array(xi))
            self.CLOSED.add(xi)  # add the point in CLOSED set
            if self.getDist(xi,xt) < self.resolution:
                break
            for xj in self.children(xi):
                # if xj not in self.CLOSED:
                if xj not in self.g:
                    self.g[xj] = np.inf
                    self.Point[xj] = [0,0,0]
                else:
                    pass

                dis = self.getDist(xi, xj)
                self.Point[xj][0] = self.Point[xi][0] + dis  # xj到起点路径长度
                t0 = time.time()
                self.Point[xj][1] = self.Point[xi][1] + self.heuristic_fun(list(xj)) # xj到起点路径总增益
                self.query_num += 1
                self.query_time += (time.time()-t0)
                self.Point[xj][2] = self.Point[xi][2] + 1 # xj到起点路径总点数
                
                
                a = self.Point[xj][0] - self.lamda*self.Point[xj][0]*self.Point[xj][1]/self.Point[xj][2]
                # a = self.g[xi] + dis- self.lamda*dis*self.gain[xj][0]/self.gain[xj][1]
                if a < self.g[xj]:
                    self.g[xj] = a
                    self.Parent[xj] = xi
                    # assign or update the priority in the open
                    self.OPEN.put(xj, a + self.getDist(xj, xt))
                    # print("  --- add a new point")

        self.lastpoint = xi
        # if the path finding is finished
        if self.lastpoint in self.CLOSED:
            self.done = True
            self.Path = self.path()
            print("local plan successfully")
            return self.Path
        return False

    def path(self):
        path = []
        x = self.lastpoint
        start = self.x0
        path.append(list(x))
        while x != start:
            path.append(list(self.Parent[x]))
            x = self.Parent[x]
        return path[::-1]
