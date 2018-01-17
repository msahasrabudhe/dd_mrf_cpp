#include <vector>
#include <queue> 
#include <utility>
#include <cstring>
#include <iostream>

/* 
 * Return all possible labellings of n_nodes, each of which has a possible number
 * of labels given by n_labels. 
 */
std::vector< std::vector<int> > generate_label_permutations(int n_nodes, int * n_labels)
{
    /* If there is only one node, return a vector of vectors of length one. */
    if(n_nodes == 1)
    {
        std::vector< std::vector<int> > ret;
        for(int i = 0; i < n_labels[0]; i ++)
        {
            std::vector<int> this_labels;
            this_labels.push_back(i);
            ret.push_back(this_labels);
        }

        return ret;
    }

    std::vector< std::vector<int> > labels_r = generate_label_permutations(n_nodes - 1, n_labels + 1);

    std::vector< std::vector<int> > new_labels_r;

    for(int i = 0; i < n_labels[0]; i ++)
    {
        for(unsigned int j = 0; j < labels_r.size(); j ++)
        {
            std::vector<int> this_labels;
            this_labels.push_back(i);
            this_labels.insert(this_labels.end(), labels_r[j].begin(), labels_r[j].end());
//            labels_r[j].insert(labels_r[j].begin(), i);
            new_labels_r.push_back(this_labels);
        }
    }
    return new_labels_r;
}

/*
 * _generate_tree_with_root -- Generate a tree using an adjacency matrix, starting at a root, 
 *  and with some maximum depth. 
 */
std::vector< std::pair<int,int> > _generate_tree_with_root(bool **adj_mat, int n_nodes, int root, int max_depth)
{
    /* BFS traversal. */
    std::queue< std::pair< int, int > > bfs;
   
    /* Visited array - mark whether a node has been visited. */
    bool * visited = new bool [n_nodes];
    memset(visited, 0, sizeof(bool)*n_nodes);
    /* Mark the root as visited. */
    visited[root]  = true;

    /* Push root and zero into the queue. */
    bfs.push(std::make_pair(root, 0));

    /* The edge list for this tree. */
    std::vector< std::pair<int,int> > edge_ends;

    /* Iterate while queue is not empty. */
    while(!bfs.empty())
    {
        /* Get next node. */
        std::pair<int, int> this_node = bfs.front();
        bfs.pop();

        int _v = this_node.first;
        int _d = this_node.second;

        /* If max depth has already been reached, do nothing. */
        if(_d == max_depth)
        {
            continue;
        }

        /* Find possible neighbours of this node. */
        for(int i = 0; i < n_nodes; i ++)
        {
            if(_v == i)
                continue;
            if(adj_mat[_v][i] && !visited[i])
            {
                /* Include this edge. */
                edge_ends.push_back(std::make_pair(_v, i));
                /* Push i into queue, but with degree _d + 1. */
                bfs.push(std::make_pair(i, _d + 1));
                /* Mark i as visited. */
                visited[i] = true;
            }
        }
    }

    delete [] visited;

    return edge_ends;
}
