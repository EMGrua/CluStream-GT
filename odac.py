import pandas as pd
import math

class Node():

    time_series = []
    children = []
    t = 0

    def __init__(self):
        self.time_series = []
        self.children = []
        t = 0

    def add_child(self, node):
        self.children.append(node)

    def remove_all_children(self):
        self.children = []

    def get_children(self):
        return self.children

    def get_time_series(self):
        return self.time_series

    def set_time_series(self, time_series):
        self.time_series = time_series

    def rnomc(self, A, P):

        # Does not seem to work for values lower than 3.....
        if self.t > 2:
            rnomc = pd.DataFrame(0, index=self.time_series, columns=self.time_series)
            d1 = 0
            d1_x1 = None
            d1_y1 = None
            d2 = 0
            d2_x2 = None
            d2_y2 = None
            d_min = float("inf")
            d_total = 0
            n = 0

            for i in range(0, len(self.time_series)):
                for j in range(i+1, len(self.time_series)):
                    n += 1
                    id1 = self.time_series[i]
                    id2 = self.time_series[j]
                    corr = (P.loc[id1, id2] - ((A.loc[id1, 'A']*A.loc[id2, 'A'])/float(self.t)))/(math.sqrt(A.loc[id1, 'A_2']-(math.pow(A.loc[id1, 'A'], 2)/float(self.t)))*math.sqrt(A.loc[id2, 'A_2']-(math.pow(A.loc[id2, 'A'], 2)/float(self.t))))
#                    print(self.t)
                    #the absolute score was not in the paper
                    rnomc_score = math.sqrt(abs((1-corr)/2))
                    d_total += rnomc_score
                    if rnomc_score > d1:
                        d2 = d1
                        d2_x2 = d1_x1
                        d2_y2 = d1_y1
                        d1 = rnomc_score
                        d1_x1 = id1
                        d1_y1 = id2
                    if rnomc_score < d_min:
                        d_min = rnomc_score
                    rnomc.loc[id1, id2] = rnomc_score
                    rnomc.loc[id2, id1] = rnomc_score
            return [rnomc, d1, d1_x1, d1_y1, d2, d2_x2, d2_y2, d_min, d_total/float(n)]
        else:
            return [None, None, None, None, None, None, None, None, None]


class Odac():

    delta = 0.95
    R = 1
    # The tree with clusters
    tree = None
    # number of time series
    n = 0
    # minimal statistics
    A = None
    P = None
    time_series = []
    t = 0
    clusters = []


    def __init__(self, ids):
        self.tree = Node()
        self.tree.set_time_series(ids)
        self.n = len(ids)
        self.time_series = ids
        self.A = pd.DataFrame(0, index=ids, columns=['A', 'A_2'])
        self.P = pd.DataFrame(0, index=ids, columns=ids)
        self.t = 0

    def reset_statistics(self, time_series):
        for id in time_series:
            self.A.loc[id,:] = 0
            self.P.loc[id,:] = 0
            self.P.loc[:,id] = 0

    def add_data(self, data):
        # Update our minimal statistics
        for id in self.time_series:
            self.A.loc[id, 'A'] = self.A.loc[id, 'A'] + data[id]
            self.A.loc[id, 'A_2'] = self.A.loc[id, 'A_2'] + math.pow(data[id], 2)

        # Note: part below not optimally efficient (can do it in half the time).
        for id1 in self.time_series:
            for id2 in self.time_series:
                self.P.loc[id1, id2] = self.P.loc[id1, id2] + data[id1] * data[id2]


        self.update_tree_time(self.tree)

        self.grow_tree(self.tree)
        self.aggregate_tree(self.tree)


        #print 'Printing tree....'
        #self.print_tree(self.tree, 1)

        self.clusters = []
        self.return_tree(self.tree, 1)

        return self.clusters

    def update_tree_time(self, node):
        node.t += 1
        for child in node.get_children():
            self.update_tree_time(child)

    def print_tree(self, node, level):
        for i in range(0,level):
            print '=',
        print ' '
        print node.time_series
        print '\n'

        if not (len(node.get_children()) == 0):
            for child in node.get_children():
                self.print_tree(child, level+1)

    def return_tree(self, node, level):
        self.clusters.append(node.time_series)
        if not (len(node.get_children()) == 0):
            for child in node.get_children():
                self.return_tree(child, level+1)

    def aggregate_tree(self, node):
        if len(node.get_children()) == 0:
            [rnomc, d1, d1_x1, d1_y1, d2, d2_x2, d2_y2, d0, d_avg] = node.rnomc(self.A, self.P)
            return d1
        else:
            children_d1 = []
            for child in node.get_children():
                children_d1.append(self.aggregate_tree(child))
                # There is a non leaf child
                if not (-1 in children_d1):
                    [rnomc, d1, d1_x1, d1_y1, d2, d2_x2, d2_y2, d0, d_avg] = node.rnomc(self.A, self.P)
                    if rnomc is not None:
                        delta_a = 2*d1 - (sum(children_d1))
                        epsilon = math.sqrt((math.pow(self.R, 2)*math.log(1/self.delta))/(2*node.t))
                        # Reset the children.....
                        if delta_a < epsilon:
                            print '++++++++ Aggregate! ++++++++'
                            node.remove_all_children()
                            node.t = 0
                            self.reset_statistics(node.time_series)
            return -1


    def grow_tree(self, node):
        # If this a leaf we can see whether we should split

        if len(node.get_children()) == 0:
            [rnomc, d1, d1_x1, d1_y1, d2, d2_x2, d2_y2, d0, d_avg] = node.rnomc(self.A, self.P)

            if rnomc is not None:
                epsilon = math.sqrt((math.pow(self.R, 2)*math.log(1/self.delta))/(2*node.t))

                # Check if we can split...
                if (d1-d2) > epsilon:
                    # Compute whether we want to split...
                    if ((d1-d0)*(abs(d1+d0-2*d_avg))) > epsilon:
                        # We split!
                        print '++++++++ Split! ++++++++'
                        node1 = Node()
                        node2 = Node()

                        for id in node.get_time_series():
                            if rnomc.loc[id, d1_x1] < rnomc.loc[id, d1_y1]:
                                node1.time_series.append(id)
                            else:
                                node2.time_series.append(id)
                        node.add_child(node1)
                        node.add_child(node2)
                        self.reset_statistics(node.time_series)
                        node.t = 0
        else:
            for child in node.get_children():
                self.grow_tree(child)
