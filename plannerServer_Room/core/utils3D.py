import numpy as np
from numpy.matlib import repmat
from numpy.random import uniform
from collections import deque
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

# np.random.seed(10)

def getRay(x, y):
    direc = [y[0] - x[0], y[1] - x[1], y[2] - x[2]]
    return np.array([x, direc])


def getAABB(blocks):
    AABB = []
    for i in blocks:
        AABB.append(np.array([np.add(i[0:3], -0), np.add(i[3:6], 0)]))  # make AABBs alittle bit of larger
    return AABB


def getDist(pos1, pos2):
    return np.sqrt(sum([(pos1[0] - pos2[0]) ** 2, (pos1[1] - pos2[1]) ** 2, (pos1[2] - pos2[2]) ** 2]))


''' The following utils can be used for rrt or rrt*,
    required param initparams should have
    env,      environement generated from env3D
    V,        node set
    E,        edge set
    i,        nodes added
    maxiter,  maximum iteration allowed
    stepsize, leaf growth restriction

'''



def sampleConstrain(initparams, bias = 0.0):
    '''biased sampling'''
    i = np.random.random()
    p = np.random.uniform()
    if 0 < p < 0.05:
        x = sampleUniformRing(np.array([4.3,5]),initparams.Phi)
    elif 0.05 < p < 0.3:
        x = sampleUniformRing(np.array([5,6]),initparams.Phi)
    elif 0.3 < p <0.9:
        x = sampleUniformRing(np.array([6,9]),initparams.Phi)
    else:
        x = sampleUniformRing(np.array([9,10]),initparams.Phi)
    x = tuple(x)

    if getDist(x, initparams.env.start) + getDist(x, initparams.env.goal) > initparams.cbest:
        return sampleConstrain(initparams, bias = 0.0)
    return x

 
def sampleUniformRing(R, Phi=None):
    r = np.random.uniform(R[0], R[1])
    phi = np.random.uniform(Phi[0],Phi[1])
    z = np.random.uniform(0,10)
    return np.array([r * np.cos(phi) + 10, r * np.sin(phi) + 10, z])

def smapleUniform(coordx, coordy, coordz):
    x_x = smapleUniform_D1(coordx)
    x_y = smapleUniform_D1(coordy)
    x_z = smapleUniform_D1(coordz)
    return np.array([x_x,x_y,x_z])

def smapleUniform_D1(coord):
    if len(coord) == 2:
       v = np.random.uniform(coord[0], coord[1])
       return v
    p = np.random.uniform()
    if 0 < p < 0.5:
        v = np.random.uniform(coord[0], coord[1])
    else:
        v = np.random.uniform(coord[2], coord[3])
    return v
# ---------------------- Collision checking algorithms
def isCollide(initparams, x, child, dist=None):
    '''see if line intersects obstacle'''
    '''specified for expansion in A* 3D lookup table'''
    if dist==None:
        dist = getDist(x, child)
    ## TODO 射线是否穿过障碍物，线段端点是否在约束内
    if 3<=getDist(x,initparams.object_center)<=4 and 3<=getDist(child,initparams.object_center):
        return False, dist
    else :
        return True, dist

def isinbound(i, x, mode = False, factor = 0, isarray = False):
    if mode == 'obb':
        return isinobb(i, x, isarray)
    if isarray:
        compx = (i[0] - factor <= x[:,0]) & (x[:,0] < i[3] + factor) 
        compy = (i[1] - factor <= x[:,1]) & (x[:,1] < i[4] + factor) 
        compz = (i[2] - factor <= x[:,2]) & (x[:,2] < i[5] + factor) 
        return compx & compy & compz
    else:    
        return i[0] - factor <= x[0] < i[3] + factor and i[1] - factor <= x[1] < i[4] + factor and i[2] - factor <= x[2] < i[5]

def isinobb(i, x, isarray = False):
    # transform the point from {W} to {body}
    if isarray:
        pts = (i.T@np.column_stack((x, np.ones(len(x)))).T).T[:,0:3]
        block = [- i.E[0],- i.E[1],- i.E[2],+ i.E[0],+ i.E[1],+ i.E[2]]
        return isinbound(block, pts, isarray = isarray)
    else:
        pt = i.T@np.append(x,1)
        block = [- i.E[0],- i.E[1],- i.E[2],+ i.E[0],+ i.E[1],+ i.E[2]]
        return isinbound(block, pt)

def isinball(i, x, factor = 0):
    if getDist(i[0:3], x) <= i[3] + factor:
        return True
    return False

def lineSphere(p0, p1, ball):
    # https://cseweb.ucsd.edu/classes/sp19/cse291-d/Files/CSE291_13_CollisionDetection.pdf
    c, r = ball[0:3], ball[-1]
    line = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
    d1 = [c[0] - p0[0], c[1] - p0[1], c[2] - p0[2]]
    t = (1 / (line[0] * line[0] + line[1] * line[1] + line[2] * line[2])) * (
                line[0] * d1[0] + line[1] * d1[1] + line[2] * d1[2])
    if t <= 0:
        if (d1[0] * d1[0] + d1[1] * d1[1] + d1[2] * d1[2]) <= r ** 2: return True
    elif t >= 1:
        d2 = [c[0] - p1[0], c[1] - p1[1], c[2] - p1[2]]
        if (d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2]) <= r ** 2: return True
    elif 0 < t < 1:
        x = [p0[0] + t * line[0], p0[1] + t * line[1], p0[2] + t * line[2]]
        k = [c[0] - x[0], c[1] - x[1], c[2] - x[2]]
        if (k[0] * k[0] + k[1] * k[1] + k[2] * k[2]) <= r ** 2: return True
    return False

def lineAABB(p0, p1, dist, aabb):
    # https://www.gamasutra.com/view/feature/131790/simple_intersection_tests_for_games.php?print=1
    # aabb should have the attributes of P, E as center point and extents
    mid = [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2]  # mid point
    # print("dist_lineAABB = ", dist)
    I = [(p1[0] - p0[0]) / dist, (p1[1] - p0[1]) / dist, (p1[2] - p0[2]) / dist]  # unit direction
    hl = dist / 2  # radius
    T = [aabb.P[0] - mid[0], aabb.P[1] - mid[1], aabb.P[2] - mid[2]]
    # do any of the principal axis form a separting axis?
    if abs(T[0]) > (aabb.E[0] + hl * abs(I[0])): return False
    if abs(T[1]) > (aabb.E[1] + hl * abs(I[1])): return False
    if abs(T[2]) > (aabb.E[2] + hl * abs(I[2])): return False
    # I.cross(s axis) ?
    r = aabb.E[1] * abs(I[2]) + aabb.E[2] * abs(I[1])
    if abs(T[1] * I[2] - T[2] * I[1]) > r: return False
    # I.cross(y axis) ?
    r = aabb.E[0] * abs(I[2]) + aabb.E[2] * abs(I[0])
    if abs(T[2] * I[0] - T[0] * I[2]) > r: return False
    # I.cross(z axis) ?
    r = aabb.E[0] * abs(I[1]) + aabb.E[1] * abs(I[0])
    if abs(T[0] * I[1] - T[1] * I[0]) > r: return False

    return True

def lineOBB(p0, p1, dist, obb):
    # transform points to obb frame
    res = obb.T@np.column_stack([np.array([p0,p1]),[1,1]]).T 
    # record old position and set the position to origin
    oldP, obb.P= obb.P, [0,0,0] 
    # calculate segment-AABB testing
    ans = lineAABB(res[0:3,0],res[0:3,1],dist,obb)
    # reset the position
    obb.P = oldP 
    return ans

def isCollide(initparams, x, child, dist=None):
    '''see if line intersects obstacle'''
    '''specified for expansion in A* 3D lookup table'''
    if dist==None:
        dist = getDist(x, child)
    
    return False, dist

# ---------------------- leaf node extending algorithms
def nearest(initparams, x, isset=False):
    V = np.array(initparams.V)
    if initparams.i == 0:
        return initparams.V[0]
    xr = repmat(x, len(V), 1)
    dists = np.linalg.norm(xr - V, axis=1)
    heuristic = dists - initparams.U
    return tuple(initparams.V[np.argmin(heuristic)])

def nearPoints(initparams, x):
    # s = np.array(s)
    V = np.array(initparams.V)
    if initparams.i == 0:
        return [initparams.V[0]]
    cardV = len(initparams.V)
    eta = initparams.eta
    gamma = initparams.gamma
    # min{γRRT∗ (log(card (V ))/ card (V ))1/d, η}
    r = min(gamma * ((np.log(cardV) / cardV) ** (1/3)), eta)
    if initparams.done: 
        r = 1
    xr = repmat(x, len(V), 1)
    inside = np.linalg.norm(xr - V, axis=1) < r
    nearpoints = V[inside]
    return np.array(nearpoints)

def steer(initparams, x, y, DIST=False):
    # steer from s to y
    if np.equal(x, y).all():
        return x, 0.0
    dist, step = getDist(y, x), initparams.stepsize
    # print("dist = ", dist)
    # if dist < 0.01:
    #     return None, dist
    step = min(dist, step)
    increment = ((y[0] - x[0]) / dist * step, (y[1] - x[1]) / dist * step, (y[2] - x[2]) / dist * step)
    xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])
    # direc = (y - s) / np.linalg.norm(y - s)
    # xnew = s + initparams.stepsize * direc
    if DIST:
        return xnew, dist
    return xnew, dist

def cost(initparams, x):
    '''here use the additive recursive cost function'''
    if x == initparams.x0:
        return 0
    return cost(initparams, initparams.Parent[x]) + getDist(x, initparams.Parent[x])

def cost_from_set(initparams, x):
    '''here use a incremental cost set function'''
    if x == initparams.x0:
        return 0
    return initparams.COST[initparams.Parent[x]] + getDist(x, initparams.Parent[x])

def path(initparams):
    path = []
    dist = 0.0
    x = initparams.xt
    path.append(x)
    while x != initparams.x0:
        x2 = initparams.Parent[x]
        # print(x2)
        path.append(list(x2))
        dist += getDist(x, x2)
        x = x2
    return path[::-1], dist

class edgeset(object):
    def __init__(self):
        self.E = {}

    def add_edge(self, edge):
        x, y = edge[0], edge[1]
        if x in self.E:
            self.E[x].add(y)
        else:
            self.E[x] = set()
            self.E[x].add(y)

    def remove_edge(self, edge):
        x, y = edge[0], edge[1]
        self.E[x].remove(y)

    def get_edge(self, nodes = None):
        edges = []
        if nodes is None:
            for v in self.E:
                for n in self.E[v]:
                    # if (n,v) not in edges:
                    edges.append((v, n))
        else: 
            for v in nodes:
                for n in self.E[tuple(v)]:
                    edges.append((v, n))
        return edges

    def isEndNode(self, node):
        return node not in self.E

#------------------------ use a linked list to express the tree 
class Node:
    def __init__(self, data):
        self.pos = data
        self.Parent = None
        self.child = set()

def tree_add_edge(node_in_tree, x):
    # add an edge at the specified parent
    node_to_add = Node(x)
    # node_in_tree = tree_bfs(head, xparent)
    node_in_tree.child.add(node_to_add)
    node_to_add.Parent = node_in_tree
    return node_to_add

def tree_bfs(head, x):
    # searches s in order of bfs
    node = head
    Q = []
    Q.append(node)
    while Q:
        curr = Q.pop()
        if curr.pos == x:
            return curr
        for child_node in curr.child:
            Q.append(child_node)

def tree_nearest(head, x):
    # find the node nearest to s
    D = np.inf
    min_node = None

    Q = []
    Q.append(head)
    while Q:
        curr = Q.pop()
        dist = getDist(curr.pos, x)
        # record the current best
        if dist < D:
            D, min_node = dist, curr
        # bfs
        for child_node in curr.child:
            Q.append(child_node)
    return min_node

def tree_steer(initparams, node, x):
    # steer from node to s
    dist, step = getDist(node.pos, x), initparams.stepsize
    increment = ((node.pos[0] - x[0]) / dist * step, (node.pos[1] - x[1]) / dist * step, (node.pos[2] - x[2]) / dist * step)
    xnew = (x[0] + increment[0], x[1] + increment[1], x[2] + increment[2])
    return xnew

def tree_print(head):
    Q = []
    Q.append(head)
    verts = []
    edge = []
    while Q:
        curr = Q.pop()
       # print(curr.pos)
        verts.append(curr.pos)
        if curr.Parent == None:
            pass
        else:
            edge.append([curr.pos, curr.Parent.pos])
        for child in curr.child:
            Q.append(child)
    return verts, edge

def tree_path(initparams, end_node):
    path = []
    curr = end_node
    while curr.pos != initparams.x0:
        path.append([curr.pos, curr.Parent.pos])
        curr = curr.Parent
    return path


#---------------KD tree, used for nearest neighbor search
class kdTree:
    def __init__(self):
        pass

    def R1_dist(self, q, p):
        return abs(q-p)

    def S1_dist(self, q, p):
        return min(abs(q-p), 1- abs(q-p))

    def P3_dist(self, q, p):
        # cubes with antipodal points
        q1, q2, q3 = q
        p1, p2, p3 = p
        d1 = np.sqrt((q1-p1)**2 + (q2-p2)**2 + (q3-p3)**2)
        d2 = np.sqrt((1-abs(q1-p1))**2 + (1-abs(q2-p2))**2 + (1-abs(q3-p3))**2)
        d3 = np.sqrt((-q1-p1)**2 + (-q2-p2)**2 + (q3+1-p3)**2)
        d4 = np.sqrt((-q1-p1)**2 + (-q2-p2)**2 + (q3-1-p3)**2)
        d5 = np.sqrt((-q1-p1)**2 + (q2+1-p2)**2 + (-q3-p3)**2)
        d6 = np.sqrt((-q1-p1)**2 + (q2-1-p2)**2 + (-q3-p3)**2)
        d7 = np.sqrt((q1+1-p1)**2 + (-q2-p2)**2 + (-q3-p3)**2)
        d8 = np.sqrt((q1-1-p1)**2 + (-q2-p2)**2 + (-q3-p3)**2)
        return min(d1,d2,d3,d4,d5,d6,d7,d8)




        
        


