#include "Slave.hpp"
#include <cmath> 

#define ABS(x) (x)<0?(-1*(x)):(x)

enum class OptimStrategy {
    STEP,
    STEP_SUBGRAD, 
    STEP_SQSUM,
    STEP_K,
    ADAPTIVE,
    ADAPTIVE_SDECAY,
    ADAPTIVE_LDECAY, 
    ADAPTIVE_KDECAY
};

class Graph
{
    private:
        /* Number of nodes and edges. */
        int n_nodes;
        int n_edges;

        /* Number of labels for every node. */
        int *                  n_labels;
        /* Number of labels for every edge. */
        int *                  e_labels; 

        /* Max number of labels in node energies and edge energies. */
        int                    max_n_label;
        int                    max_e_label;

        /* Number of slaves. */
        int                    n_slaves;

        /* V x V -> E map. */
        std::map< std::pair<int, int>, int > edge_id_from_node_ids;
        /* E -> V x V map. */
        std::map< int, std::pair<int, int> > node_ids_from_edge_id;

        /* TODO: Design choice - full adjacency matrix or sparse neighbours-only matrix? */
        bool **                adj_mat;         

        char                   decomposition[20];
        std::vector< Slave * > slave_list;
        /* Whether we already created slaves. */
        bool                   slaves_created;

        ValType **             node_energies;
        ValType **             edge_energies;

        /* Used to record the current edge count. This will be important for
           updating our V x V -> E and E - > V x V maps. */
        int current_edge_count;

        /* Keep track of which nodes and edges are in which slaves. */
        std::vector< std::vector<int> > nodes_in_slaves;
        std::vector< std::vector<int> > edges_in_slaves;

        /* Keep track of the number of slaves that contain each node or edge. */
        std::vector< int > n_slaves_node;
        std::vector< int > n_slaves_edge;

        /* Variables to hold marked updates. */
        ValType *** node_updates;
        ValType *** edge_updates;

        /* Whether to update a slave or not. */
        bool ** _mark_slave_updates;

        /* The multiplier for parameter udpates. */
        ValType alpha; 

        /* The norm of the subgradient at an iteration of optimisation. */
        ValType cur_subgradient;

        /* Whether to check a particular node for updates. This is an array of size 
           n_nodes with an element set to true if that particular node has conflicting
           labelling from the slave problems. In that case, that node should be 
           checked for updates. 
           Also, whether to check a particular edge for updates. Edge udpates are 
           marked if either of the end-points of the edges have updates marked 
           for them. 
           One array also to determine whether to check slaves. This tells us the
           number of subproblems being solved at each iteration. */
        bool * check_nodes;
        bool * check_edges;
        bool * check_slaves; 
        /* The number of nodes where we disagree on the labelling. */
        int    n_miss;

        /* The number of slaves to be solved. */
        int n_slaves_to_solve;

        /* The iteration during an optimisation. */
        int optim_it;
        /* Print optimisation report every ... */
        int _print_every;

        /* K in OptimStrategy::STEP_K and OptimStrategy::ADAPTIVE_KDECAY */
        float _decayk;

    public:
        /* The primal solution is stored in the memer labels. */
        /* Making labels public so that one can easily read the labelling
           after optimisation is complete. */
        int *                  labels;

        /* Debug mode. */
        bool debug;

        /* Verbose mode. */
        bool _verbose;

        /* Primal and dual costs. */
        ValType primal_cost;
        ValType dual_cost;
        /* Best primal and dual costs so far. */
        ValType best_primal_cost;
        ValType best_dual_cost;
        /* Primal and dual costs history. */
        std::vector< ValType > primal_hist;
        std::vector< ValType > dual_hist;
        /* n_miss history: number of disagreeing nodes. */
        std::vector< int > n_miss_hist;
        /* History of subgradient. */
        std::vector< ValType > subgrad_hist;
        /* History of alpha. */
        std::vector< ValType > alpha_hist;
        /* Best primal solution */
        std::vector< int > best_primal_solution;

        /* --------- Member functions --------- */
        
        /* Specify node energies for a node. */
        int set_node_energies(int, ValType *);

        /* Specify edge energies for a node. */
        int set_edge_energies(int, int, ValType *);

        /* Optimise the energy over the Graph using the current decomposition. */
        void optimise(ValType, int, OptimStrategy);

        /* Return the edge ID of an edge specified by its end points. */
        int get_edge_id(int, int);

        /* Return the node IDs of the ends of a specified edge. */
        std::pair<int, int> get_node_id(int);

        /* Add a cycle slave to the graph specified by nodes. The
           nodes are assumed to be in order when supplied. */
        void add_cycle_slave(std::vector<int>);

        /* Add tree slave. The node list and edge list need to be specified.
           Strictly speaking, the node list is not necessary to represent a 
           sub-tree of the Graph. But just to be safe, the node list 
           is asked as well. Can be easily modified to deduce the node
           list from the edge list. */
        void add_tree_slave(std::vector<int>, std::vector<int>);

        /* Find a spanning tree decomposition. Simply looks for spanning
           trees until all edges are accounted for. */
        void decompose_spanning_trees(void);

        /* Finalise Graph decomposition. This tells the Graph
           that no more slaves are to be added. 
           Energies are thus divided between slaves and 
           other necessary variables are set. */
        void finalise_decomposition(void);

        /* Check whether a decomposition is valid. The subproblems must
           sum to the PRIMAL. */
        bool check_decomposition(void);

        /* Ask the Graph to optimise slaves. This function iterates
           over all slaves and solves the ones that are marked to be solved. */
        void optimise_slaves(void);

        /* Estimate the PRIMAL from the current state. */
        void estimate_primal(void);

        /* Compute the PRIMAL cost for the current Graph labelling. */
        void compute_primal_cost(void);

        /* Compute the current DUAL cost. */
        void compute_dual(void);

        /* Find conflicting nodes and edges, i.e., nodes and edges that do not 
           receive the same labelling for contributing slaves. */
        void find_conflicts(void);

        /* Reset the updates variable to zero. */
        void _reset_updates(void);

        /* Compute parameter updates. */
        void compute_param_updates();

        /* Apply parameter updates. */
        void apply_param_updates(ValType, OptimStrategy);

        /* Print Graph state during a phase of optimisation. Useful
           for debugging. */
        void _print_state(int);

        /* Set verbosity */
        void verbose(bool);
        /* Specify how frequently to print optimisation status. */
        void print_every(int);
        /* Specify K for OptimStrategy::STEP_K and OptimStrategy::ADAPTIVE_KDECAY */
        void set_k_decayk(float);
        /* ------------------------------------ */

        /* Constructor. */
        Graph(int nn, int ne, int *nl)
        {
            n_nodes  = nn;
            n_edges  = ne;

            /* Set max_n_label and max_e_label to 0. */
            max_n_label = 0;
            max_e_label = 0;

            /* Allocate space for n_labels. */
            n_labels = new int [n_nodes];
            for(int i = 0; i < n_nodes; i ++)
            {
                n_labels[i] = nl[i];
                if(nl[i] > max_n_label)
                    max_n_label = nl[i];
            }

            /* Set slaves_created to false as they have not been created yet. */
            slaves_created = false;

            /* Allocate memory for e_labels. */
            e_labels = new int [n_edges];

            /* Used to record the current edge count. This will be important for
               updating our V x V -> E and E - > V x V maps. */
            current_edge_count = 0;

            /* Allocate memory for node energies. */
            node_energies = new ValType * [n_nodes];
            for(int i = 0; i < n_nodes; i ++)
                /* For every node, allocate space only for as many labels as required. */
                node_energies[i] = new ValType[n_labels[i]];

            /* Allocate energy for edge energies. */
            edge_energies = new ValType *[n_edges];
            /* Space for specific edges will be allocated later when the edge are specified. */

            /* Allocate space for adjacency matrix. */
            adj_mat = new bool *[n_nodes];
            for(int i = 0; i < n_nodes; i ++)
            {
                adj_mat[i] = (bool *)calloc(n_nodes, sizeof(bool));
            }

            /* Allocate space for the labelling. */
            labels = new int [n_nodes];

            /* Set the number of slaves to zero - we have not specified any decompositions yet. */
            n_slaves = 0;

            /* Allocate space for nodes_in_slaves and edges_in_slaves. */
            nodes_in_slaves.assign(n_nodes, std::vector<int>());
            edges_in_slaves.assign(n_edges, std::vector<int>());

            /* Allocate space for n_slaves_node and n_slaves_edge. */
            n_slaves_node.assign(n_nodes, 0);
            n_slaves_edge.assign(n_edges, 0);

            /* Allocate space for check_nodes and check_edges. */
            check_nodes  = new bool [n_nodes];
            check_edges  = new bool [n_edges];

            /* Initialise the best primal and dual costs to zero. */
            best_primal_cost = 0;
            best_dual_cost   = 0;

            /* Create empty best primal solution. */
            best_primal_solution.assign(n_nodes, 0);

            /* Set debug mode to false by default. */
            debug = false;
            
            /* Set verbose to true by default. */
            _verbose = true;
            /* By default, print every iteration. */
            _print_every = 1;
        }                       /* End of Graph(). */

        /* Destructor. */
        ~Graph()
        {
            for(int s = 0; s < n_slaves; s ++)
            {
                for(int i = 0; i < slave_list[s]->n_nodes; i ++)
                {
                    delete [] node_updates[s][i];
                }
                delete [] node_updates[s];
                for(int e = 0; e < slave_list[s]->n_edges; e ++)
                {
                    delete [] edge_updates[s][e];
                }
                delete [] edge_updates[s];
            }

            delete [] labels;

            delete [] _mark_slave_updates[0];
            delete [] _mark_slave_updates[1];
            delete [] _mark_slave_updates; 

            delete [] node_updates;
            delete [] edge_updates;

            delete [] n_labels;
            delete [] e_labels;
            for(int i = 0; i < n_nodes; i ++)
                free(adj_mat[i]);
            delete [] adj_mat;

            for(int i = 0; i < n_nodes; i ++)
                delete [] node_energies[i];
            delete [] node_energies;

            for(int i = 0; i < n_edges; i ++)
                delete [] edge_energies[i];
            delete [] edge_energies;

            for(int i = 0; i < n_slaves; i ++)
                delete slave_list[i];

            delete [] check_nodes;
            delete [] check_edges;
            delete [] check_slaves;

        }                       /* End of ~Graph(). */
};                      /* End of definition of class Graph. */
/* Public member functions' definitions for Graph to follow. */


/* 
 * Graph::add_cycle_slave -- Add a cycle slave. The order of the cycle
 * is given by the order in which the nodes are specified. All edges
 * should be existant. 
 */
void Graph::add_cycle_slave(std::vector<int> sl_nodes)
{
    /* Variables required to create slaves. */
    ValType ** slave_node_e;
    ValType ** slave_edge_e;
    int *      slave_node_l;
    int *      slave_edge_l;
    int *      slave_n_labels;

    /* The number of nodes and edges in this slave. */
    int slave_n_nodes;
    int slave_n_edges;

    /* The slave ID for this new slave. */
    int s_id = n_slaves;

    /* Increment the number of slaves. */
    ++ n_slaves;

    /* Get the node list and number of nodes and edges. */
    slave_n_nodes  = sl_nodes.size();
    slave_n_edges  = sl_nodes.size();

    slave_node_l   = new int [slave_n_nodes];
    slave_edge_l   = new int [slave_n_nodes];
    slave_n_labels = new int [slave_n_nodes];

    /* Allocate memory for node and edge energies. */
    slave_node_e      = new ValType * [slave_n_nodes];
    slave_edge_e      = new ValType * [slave_n_edges];

    for(int i = 0; i < slave_n_nodes; i ++)       /* Begin populating node energies, etc. for all nodes. */
    {
        /* Get the from/to edge. */
        int from = sl_nodes[i];
        int to   = sl_nodes[(i+1)%slave_n_nodes];

        /* Push the node ID into the node list for this slave. */
        slave_node_l[i]   = from;
        /* Push the edge ID into the edge list for this slave. */
        int edge_id       = edge_id_from_node_ids[std::make_pair(from, to)];
        slave_edge_l[i]   = edge_id;

        /* Push the number of labels for this node into the list of 
           number of labels. */
        slave_n_labels[i] = n_labels[from];

        /* Allocate space for node energies and edge energies for this node and edge. */
        slave_node_e[i]   = new ValType [n_labels[from]];
        slave_edge_e[i]   = new ValType [n_labels[from]*n_labels[to]];
        /* Push the node energies for this node into slave_node_e. */
        for(int j = 0; j < n_labels[from]; j ++)
        {
            slave_node_e[i][j] = node_energies[from][j];
        }
        /* Push the edge energies for this edge into slave_edge_e. */
        /* But first, determine which is the smaller node ID, because edges are 
           always assumed to go from the smaller node ID to the larger node ID. 
           However, in cycle slaves, node IDs for an edge always obey the order
           in which the node IDs appear in the node list. The node list, in turn, 
           always respects the way in which the cycle is built. */
        int _from, _to;
        _from = MIN(from, to);
        _to   = from + to - _from;

        int graph_lid;      /* Index to iterate over these edge energies in the Graph. */
        int slave_lid;      /* Index to iterate over these edge energies in the Slave. */
        int _from_nl = n_labels[_from];
        int _to_nl   = n_labels[_to];

        for(int j = 0; j < _from_nl; j ++)
        {
            for(int k = 0; k < _to_nl; k ++)
            {
                graph_lid = j*_to_nl + k;
                if(!(_from < _to))
                {
                    /* We store the transposed edge energies in the slave, so for this 
                       case, slave_lid and graph_lid are the same. */
                    slave_lid = graph_lid;
                }
                else
                {
                    slave_lid = k*_from_nl + j;
                }
                /* Finally, copy the values.  */
                slave_edge_e[i][slave_lid] = edge_energies[edge_id][graph_lid];
            }           /* Finished iterating over n_labels[_to]. */
        }               /* Finished iterating over n_labels[_from]. */

        /* For this node and edge, update nodes_in_slaves and edges_in_slaves. */
        nodes_in_slaves[from].push_back(s_id);
        edges_in_slaves[edge_id].push_back(s_id);
        /* Update slave counts for this node and edge. */
        ++ n_slaves_node[from];
        ++ n_slaves_edge[edge_id];
    }                   /* Finished iterating over the nodes in this slave. */

    /* Create a pointer to a slave. */
    Slave * this_slave = new Slave(s_id, slave_n_nodes, slave_n_edges, slave_node_l, slave_edge_l, 
            slave_n_labels, slave_node_e, slave_edge_e, 'c', 0, 0, 0);

    /* Add this slave to slave_list. */
    slave_list.push_back(this_slave);

    return;
}               /* End of Graph::add_cycle_slave */

void Graph::add_tree_slave(std::vector<int> nlist, std::vector<int> elist)
{
    /* Add a tree-structured slave. */

    /* The slave ID for this new slave. */
    int s_id = n_slaves;

    /* Increment the number of slaves. */
    ++ n_slaves;

    /* Get data on the subproblem's structure. */
    int slave_n_nodes = nlist.size();
    int slave_n_edges = elist.size();

    /* Allocate some memory to store node and edge energies. */
    ValType ** slave_node_e = new ValType *[slave_n_nodes];
    ValType ** slave_edge_e = new ValType *[slave_n_edges];

    /* Arrays to store node and edge lists. */
    int * slave_node_l   = new int [slave_n_nodes];
    int * slave_edge_l   = new int [slave_n_edges];
    int * slave_n_labels = new int [slave_n_nodes];

    /* Create two maps to store E -> V x V and V -> E x E maps 
       for this slave. */
    std::map< int, std::pair<int, int> > * n_from_e = new std::map< int, std::pair<int, int> >;
    std::map< std::pair<int, int>, int > * e_from_n = new std::map< std::pair<int, int>, int >;

    /* Node map to be used for this creation. */
    std::map<int, int> node_map;

    /* Adjacency matrix for this slave. */
    bool ** slave_adj = new bool * [slave_n_nodes];
    for(int i = 0; i < slave_n_nodes; i ++)
    {
        slave_adj[i] = (bool *)calloc(slave_n_nodes, sizeof(bool));
    }

    /* Start populating node energies. */
    for(int i = 0; i < slave_n_nodes; i ++)
    {
        /* The node ID. */
        int n_id = nlist[i];

        slave_node_l[i]   = n_id;

        /* The number of labels for this node. */
        int nl_node          = n_labels[n_id];
        slave_n_labels[i]    = nl_node;

        /* This node's node energy. */
        slave_node_e[i]   = new ValType [nl_node];

        /* Copy node energies. */
        for(int j = 0; j < nl_node; j ++)
        {
            slave_node_e[i][j] = node_energies[n_id][j];
        }

        /* Push this node into the node map. */
        node_map[n_id] = i;

        /* Update nodes_in_slaves for this node. */
        nodes_in_slaves[n_id].push_back(s_id);
        /* Update n_slaves count for this node. */
        ++ n_slaves_node[n_id];
    }

    /* Start populating edge energies and maps. */
    for(int e = 0; e < slave_n_edges; e ++)
    {
        /* The edge ID. */
        int e_id = elist[e];

        slave_edge_l[e] = e_id;

        /* The end points for this edge. */
        int _from, _to;
        _from = node_ids_from_edge_id[e_id].first;
        _to   = node_ids_from_edge_id[e_id].second;

        /* End points in this slave. */
        int _efrom, _eto;
        _efrom = node_map[_from];
        _eto   = node_map[_to];

        /* Whether to transpose the edge energies. They should be transposed
           if _efrom > _eto. */
        bool _transposed = (_efrom > _eto);

        /* Correct the indices :: the smaller one should be _efrom. */
        int _total = _efrom + _eto;
        _efrom     = MIN(_efrom, _eto);
        _eto       = _total - _efrom;

        /* Add this edge to the adjacency matrix. */
        slave_adj[_efrom][_eto] = true;
        slave_adj[_eto][_efrom] = true;

        /* Update maps to be used by the slave. */
        (*n_from_e)[e]                            = std::make_pair(_efrom, _eto);
        (*e_from_n)[std::make_pair(_efrom, _eto)] = e;
        (*e_from_n)[std::make_pair(_eto, _efrom)] = e;

        /* Allocate space for edge energies of this edge. */
        slave_edge_e[e]   = new ValType [e_labels[e_id]];

        int l_id;
        for(int j = 0; j < n_labels[_from]; j ++)
        {
            for(int k = 0; k < n_labels[_to]; k ++)
            {
                l_id = (_transposed) ? (k*n_labels[_from]+j) : (j*n_labels[_to] + k);
                slave_edge_e[e][l_id] = edge_energies[e_id][j*n_labels[_to] + k];
            }
        }

        /* Update edges_in_slaves for this edge. */
        edges_in_slaves[e_id].push_back(s_id);
        /* Update n_slaves count for this edge. */
        ++ n_slaves_edge[e_id];
    }

    /* Create this slave. */
    Slave *this_slave = new Slave(s_id, slave_n_nodes, slave_n_edges, slave_node_l, slave_edge_l, slave_n_labels, 
            slave_node_e, slave_edge_e, 't', slave_adj, n_from_e, e_from_n);

    /* Add this slave to slave_list. */
    slave_list.push_back(this_slave);

    return;
}               /* End of Graph::add_tree_slave */


/* 
 * Graph::decompose_spanning_trees -- Decompose the graph into spanning
 * trees. 
 */
void Graph::decompose_spanning_trees(void)
{
    /* Random seed. */
    srand(time(NULL));

    /* Array to say if all edges have been marked. */
    bool *marked_edges = new bool [n_edges];
    /* Set the array to zero. */
    memset(marked_edges, 0, sizeof(bool)*n_edges);

    bool _cont = true;

    /* Node degrees. */
    int *node_degrees = new int [n_nodes];
    int _this_deg;

    /* Node list. It is just the list of all nodes, because
       all the trees are spanning trees. We can reuse this. */
    std::vector< int > node_list;

    /* Create copy of adj_mat. */
    bool ** _adj = new bool *[n_nodes];
    for(int i = 0; i < n_nodes; i ++)
    {
        _this_deg = 0;
        _adj[i] = new bool [n_nodes];
        for(int j = 0; j < n_nodes; j ++)
        {
            _adj[i][j] = adj_mat[i][j];
            if(adj_mat[i][j])
                ++ _this_deg;
        }
        node_degrees[i] = _this_deg;
        /* Push i into node_list. */
        node_list.push_back(i);
    }

    while(_cont)
    {
        /* Find nodes with the maximum degree. */
        int _max_degree   = node_degrees[0];
        int _n_max_degree = 1;
        for(int i = 1; i < n_nodes; i ++)
        {
            if(node_degrees[i] == _max_degree)
            {
                ++ _n_max_degree;
            }
            else if(node_degrees[i] > _max_degree)
            {
                _max_degree   = node_degrees[i];
                _n_max_degree = 1;
            }
        }

        /* Choose a random node among these. */
        int rand_idx = rand() % _n_max_degree;
        int chosen_node = 0, _rand_it = 0;
        for(int i = 0; i < n_nodes; i ++)
        {
            if(node_degrees[i] == _max_degree)
            {
                if(_rand_it == rand_idx)
                {
                    chosen_node = i;
                    break;
                }
                ++ _rand_it;
            }
        }

        std::vector< std::pair<int,int> > edge_ends = _generate_tree_with_root(adj_mat, n_nodes, chosen_node, -1);
        std::vector< int > edge_list;

        //printf("Choosing node %d with degree %d, and %lu edges.\n", chosen_node, node_degrees[chosen_node], edge_ends.size());

        for(unsigned int e = 0; e < edge_ends.size(); e ++)
        {
            /* Remove this edge from the adjacency matrix. */
            int _from     = edge_ends[e].first;
            int _to       = edge_ends[e].second;
            int this_edge = edge_id_from_node_ids[edge_ends[e]];
            /* Mark this edge as recorded. */
            marked_edges[this_edge] = true;
            /* Push this edge into edge_list. */
            edge_list.push_back(this_edge);
            /* Remove these edges. */
            _adj[_from][_to]        = false;
            _adj[_to][_from]        = false;
            /* Reduce node degrees for _from and _to. */
            node_degrees[_from]    -= 1;
            node_degrees[_to]      -= 1;
        }

        /* Add this tree slave. */
        add_tree_slave(node_list, edge_list);

        /* Compute stopping condition. */
        _cont = false;
        for(int e = 0; e < n_edges; e ++)
        {
            if(!marked_edges[e])
            {
                _cont = true;
                break;
            }
        }
        /* Recompute node degrees. */
        for(int i = 0; i < n_nodes; i ++)
        {
            _this_deg = 0;
            for(int j = 0; j < n_nodes; j ++)
            {
                if(_adj[i][j])
                    ++ _this_deg;
            }
            node_degrees[i] = _this_deg;
        }
    }

    delete [] marked_edges;
    delete [] node_degrees;
    for(int i = 0; i < n_nodes; i ++)
        delete [] _adj[i];
    delete [] _adj;
}               /* End of Graph::decompose_spanning_trees */

void Graph::finalise_decomposition(void)
{
    /* Set slaves_created to true. */
    slaves_created = true;

    /* Now, we must divide the node and edge energies for every slave,
       so that the total gives us the corresponding node and edge energies in 
       the Graph. */
    std::vector<int> slaves_this_factor;
    int n_slaves_this_factor;
    int f_id_this_slave;

    /* Do it for nodes first. */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        /* Slave list for this node ::: IDs of all slaves that contain this node. */
        slaves_this_factor   = nodes_in_slaves[n_id];
        /* Number of slaves that contain this node. */
        n_slaves_this_factor = n_slaves_node[n_id];

        /* Iterate over all slaves to divide the energies appropriately. */
        for(int j = 0; j < n_slaves_this_factor; j ++)
        {
            /* Get the id for this node ID in this slave. */
            f_id_this_slave = slave_list[slaves_this_factor[j]]->node_map(n_id);
            /* Divide the energy for every label equally. */
            for(int k = 0; k < n_labels[n_id]; k ++)
            {
                slave_list[slaves_this_factor[j]]->node_energies[f_id_this_slave][k] /= n_slaves_this_factor;
            }
        }           /* Finished iterating over every slave containing n_id. */
    }               /* Finished dividing energies for nodes. */

    /* Do it for edges next. */
    for(int e_id = 0; e_id < n_edges; e_id ++)
    {
        /* Slave list for this edge ::: IDs of all slaves that contain this edge. */
        slaves_this_factor   = edges_in_slaves[e_id];
        /* Number of slaves that contain this edge. */
        n_slaves_this_factor = n_slaves_edge[e_id];

        /* Iterate over all slaves to divide the energies accordingly. */
        for(int j = 0; j < n_slaves_this_factor; j ++)
        {
            /* Get the ID for this edge ID in the Slave. */
            f_id_this_slave = slave_list[slaves_this_factor[j]]->edge_map(e_id);
            /* Divide the energy for every label equally. */
            for(int k = 0; k < e_labels[e_id]; k ++)
            {
                slave_list[slaves_this_factor[j]]->edge_energies[f_id_this_slave][k] /= n_slaves_this_factor;
            }
        }           /* Finished iterating over every slave containing e_id. */
    }               /* Finished dividing energies for edges. */

    /* Create memory for check flags on slaves. */
    check_slaves = new bool [n_slaves];

    /* Set the number of slaves to be solved to n_slaves initially (all of them are to be solved). */
    n_slaves_to_solve = n_slaves;
    /* Also initialise all elements in check_slaves to true. */
    for(int s = 0; s < n_slaves; s ++)
    {
        check_slaves[s] = true;
    }

    /* Allocate memory for updates. */
    node_updates = new ValType ** [n_slaves];
    edge_updates = new ValType ** [n_slaves];
    for(int s = 0; s < n_slaves; s ++)
    {
        int n_nodes_in_slave = slave_list[s]->n_nodes;
        int n_edges_in_slave = slave_list[s]->n_edges;
        node_updates[s] = new ValType * [n_nodes_in_slave];
        edge_updates[s] = new ValType * [n_edges_in_slave];

        for(int i = 0; i < n_nodes_in_slave; i ++)
        {
            node_updates[s][i]   = new ValType [slave_list[s]->n_labels[i]];
        }

        for(int e = 0; e < n_edges_in_slave; e ++)
        {
            edge_updates[s][e]   = new ValType [slave_list[s]->e_labels[e]];
        }
    }
    /* Allocate memory for flags: whether to update a slave's energies or not. */
    _mark_slave_updates    = new bool *[2];
    _mark_slave_updates[0] = new bool [n_slaves];
    _mark_slave_updates[1] = new bool [n_slaves];
}                       /* End of Graph::finalise_decomposition() */

bool Graph::check_decomposition(void)
{
    /* Check whether a decomposition is correct. 
       All slave energies should add up to the
       Graph energies. */
    std::vector<int> slaves_this_factor;
    int n_slaves_this_factor;

    /* Check nodes first. Node energies should be correctly divided among all slaves. */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        slaves_this_factor = nodes_in_slaves[n_id];
        n_slaves_this_factor = slaves_this_factor.size();

        /* Store total node energy here. */
        ValType * total_e = (ValType *)calloc(n_labels[n_id], sizeof(ValType));

        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* node_id for this node in this slave. */
            int _n_id_in_slave = slave_list[slaves_this_factor[s]]->node_map(n_id);
            for(int k = 0; k < n_labels[n_id]; k ++)
            {
                /* Accumulate in total_e[k]. */
                total_e[k] += slave_list[slaves_this_factor[s]]->node_energies[_n_id_in_slave][k];
            }
        }
        /* Check if the total is within EPS of the node energies specified by the Graph. */
        for(int k = 0; k < n_labels[n_id]; k ++)
        {
            if(total_e[k] - node_energies[n_id][k] > EPS*1e-2 || 
                    node_energies[n_id][k] - total_e[k] > EPS*1e-2)
            {
                /* total_e[k] and node_energies[n_id][k] are more than EPS apart. */
                printf("Energies do not match for node %d, label %d: Graph: %.10lf, Slaves: %.10lf\n", n_id, k, node_energies[n_id][k], total_e[k]);
                return false;
            }
        }
        /* Delete total_e, we do not need it anymore. */
        free(total_e);
    }               /* Finished checking nodes. */

    /* Check edges now. */
    for(int e_id = 0; e_id < n_edges; e_id ++)
    {
        int _from, _to;
        /* Retrieve ends of this edge. */
        _from = node_ids_from_edge_id[e_id].first;
        _to   = node_ids_from_edge_id[e_id].second;

        /* Slaves containing this edge. */
        slaves_this_factor   = edges_in_slaves[e_id];
        n_slaves_this_factor = slaves_this_factor.size();

        /* Store the total energy. */
        ValType * total_e = (ValType *)calloc(e_labels[e_id], sizeof(ValType));

        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* edge id for this edge in this slave. */
            int _e_id_in_slave = slave_list[slaves_this_factor[s]]->edge_map(e_id);
            /* Node IDs for _from and _to in this slave. */
            int _sfrom, _sto;
            _sfrom = slave_list[slaves_this_factor[s]]->node_map(_from);
            _sto   = slave_list[slaves_this_factor[s]]->node_map(_to);
            /* Whether to transpose the energies. */
            bool _transposed = (slave_list[slaves_this_factor[s]]->type() == 'c' && _from < _to) ||
                               (slave_list[slaves_this_factor[s]]->type() == 't' && _sfrom > _sto);
            /* Iterate over all labels to compute the total. */
            for(int j = 0; j < n_labels[_from]; j ++)
            {
                for(int k = 0; k < n_labels[_to]; k ++)
                {
                    if(_transposed)
                        total_e[j*n_labels[_to] + k] += slave_list[slaves_this_factor[s]]->edge_energies[_e_id_in_slave][k*n_labels[_from] + j];
                    else
                        total_e[j*n_labels[_to] + k] += slave_list[slaves_this_factor[s]]->edge_energies[_e_id_in_slave][j*n_labels[_to] + k];
                }
            }
        }               /* Finished summing over all slaves. */

        /* Check if the total is within EPS of the node energies specified by the Graph. */
        for(int k = 0; k < e_labels[e_id]; k ++)
        {
            if(total_e[k] - edge_energies[e_id][k] > EPS*1e-2 || 
                    edge_energies[e_id][k] - total_e[k] > EPS*1e-2)
            {
                /* total_e[k] and node_energies[n_id][k] are more than EPS apart. */
                printf("Energies do not match for edge %d, label %d: Graph: %.10lf, Slaves: %.10lf\n", e_id, k, edge_energies[e_id][k], total_e[k]);
                return false;
            }
        }               /* Finished checking for consistency. */
        /* Delete total_e, we do not need it anymore. */
        free(total_e);
    }

    /* No problems found. The decomposition is correct. */
    return true; 
}                       /* End of Graph::check_decomposition */

void Graph::optimise_slaves(void)
{
    /* Optimise all slaves in this Graph. */
    /* #pragma omp parallel for */
    for(int i = 0; i < n_slaves; i ++)
    {
        /* Check whether this slave is to be solved. */
        if(check_slaves[i])
        {
            slave_list[i]->optimise();
            slave_list[i]->compute_energy();
        }
    }
}                       /* End of Graph::optimise_slaves */

void Graph::estimate_primal(void)
{
    /* Very crude strategy for now: just assign the label 
       assigned by the first slave in nodes_in_slaves. 
       TODO: Use a better strategy. Min-marginals
       obtained from slaves? */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        std::vector<int> slist = nodes_in_slaves[n_id];
        /* Calculate how many votes each label gets. */
        int *votes = (int *)calloc(n_labels[n_id], sizeof(int));
        for(int i = 0; i < n_slaves_node[n_id]; i ++)
            ++ votes[slave_list[slist[i]]->get_node_label(n_id)];
        /* Choose the most-voted label. */
        int _argmax = 0;
        for(int j = 1; j < n_labels[n_id]; j ++)
        {
            if(votes[j] > votes[_argmax])
            {
                _argmax = j;
            }
        }
        /* Choose _argmax as the label for this node. */
        labels[n_id] = _argmax; 
        /* Free the memory allocated to votes. */
        free(votes);
    }

    /* Compute the primal cost now. */
    compute_primal_cost();
    return ;
}                       /* End of Graph::estiamte_primal */

void Graph::compute_primal_cost(void)
{
    /* Compute primal cost now. */
    primal_cost = 0;
    int _from, _to;
    int _lf, _lt;
    std::pair< int, int > _node_ids;

    /* Add energies due to nodes. */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        /* Get labelling and contribution from this node. */
        _lf = labels[n_id];
        primal_cost += node_energies[n_id][_lf];
    }

    /* Add energies due to edges. */
    for(int e_id = 0; e_id < n_edges; e_id ++)
    {
        /* Get node IDs for this edge. */
        _node_ids = node_ids_from_edge_id[e_id];
        _from     = _node_ids.first;
        _to       = _node_ids.second;
        /* Get labelling for this edge. */
        _lf       = labels[_from];
        _lt       = labels[_to];

        primal_cost += edge_energies[e_id][_lf*n_labels[_to] + _lt];
    }

    /* Update the best primal cost. */
    if(optim_it == 0 || best_primal_cost > primal_cost)
    {
        best_primal_cost     = primal_cost;
        /* Record the best primal solution. */
        for(int i = 0; i < n_nodes; i ++)
            best_primal_solution[i] = labels[i];
    }

    /* Push this cost into primal history. */
    primal_hist.push_back(primal_cost);
}                       /* End of Graph::compute_primal_cost */

void Graph::compute_dual(void)
{
    dual_cost = 0;
    /* The dual cost is just the sum of all slave energies. */
    for(int i = 0; i < n_slaves; i ++)
        dual_cost += slave_list[i]->energy;

    /* Update the best dual cost. */
    if(optim_it == 0 || best_dual_cost < dual_cost)
    {
        best_dual_cost = dual_cost;
    }

    /* Update dual cost history. */
    dual_hist.push_back(dual_cost);
}                       /* End of Graph::compute_dual */

void Graph::find_conflicts(void)
{
    std::vector<int> slaves_this_node;

    /* Reset n_miss to zero. */
    n_miss = 0;

    /* Reset check flags. */
    memset(check_nodes, 0, sizeof(bool)*n_nodes);
    memset(check_edges, 0, sizeof(bool)*n_edges);

    /* Iterate over every node. */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {

        /* Retrieve the slave list for this node. */
        slaves_this_node = nodes_in_slaves[n_id];

        int s;

        /* Iterate over slaves. We will check if they have 
           all assigned this node the same label. If not, 
           we have conflicts. */
        int l_s  = slave_list[slaves_this_node[0]]->get_node_label(n_id);
        int l_sn = l_s;
        for(s = 1; s < n_slaves_node[n_id]; s ++)
        {
            /* Get the label assigned by this slave and the next one. */
            l_s  = l_sn;
            l_sn = slave_list[slaves_this_node[s]]->get_node_label(n_id);

            /* If they don't match, break, because we found a conflict. */
            if(l_s != l_sn)
                break;
        }
        if(s < n_slaves_node[n_id])
        {
            /* We found a conflict mid-way. */
            check_nodes[n_id] = true;

            /* Increment n_miss. */
            ++ n_miss;
        }
    }

    /* Now mark check_edges. For this, we simply take every conflicted node, 
       and mark updates for all edges incident there. */
    for(int i = 0; i < n_nodes; i ++)
    {
        if(!check_nodes[i])
        {
            /* No need to check for updates here. */
            continue;
        }
        for(int j = 0; j < n_nodes; j ++)
        {
            /* TODO: Design choice - full adjacency matrix or sparse neighbours-only matrix? */
            if(adj_mat[i][j])
            {
                /* There is an edge between i and j. */
                int e_id = edge_id_from_node_ids[std::make_pair(i,j)];
                check_edges[e_id] = true;
            }
        }           /* Finished iterating over neighbours. */
    }               /* Finished iterating over nodes. */
}                   /* End of Graph::find_conflicts */


/*
 * Graph::_reset_updates -- Reset the updates variable to zero. 
 */
void Graph::_reset_updates(void)
{
    for(int s = 0; s < n_slaves; s ++)
    {
        int n_nodes_in_slave = slave_list[s]->n_nodes;
        int n_edges_in_slave = slave_list[s]->n_edges;
        /* Reset node updates to zero. */
        for(int i = 0; i < n_nodes_in_slave; i ++)
        {
            int n_labels_in_node = slave_list[s]->n_labels[i];
            for(int l = 0; l < n_labels_in_node; l ++)
            {
                node_updates[s][i][l] = 0.0;
            }
        }
        /* Reset edge updates to zero. */
        for(int e = 0; e < n_edges_in_slave; e ++)
        {
            int n_labels_in_edge = slave_list[s]->e_labels[e];
            for(int l = 0; l < n_labels_in_edge; l ++)
            {
                edge_updates[s][e][l] = 0.0;
            }
        }
    }

    /* Also reset flags which mark updates. */
    memset(_mark_slave_updates[0], 0, sizeof(bool)*n_slaves);
    memset(_mark_slave_updates[1], 0, sizeof(bool)*n_slaves);

    return;
}                   /* End of Graph::_reset_updates */


/*
 * Graph::compute_param_updates -- Compute parameter upates for an iteration of optimisation. 
 */
void Graph::compute_param_updates()
{
    std::vector< int > slaves_this_factor;
    int n_slaves_this_factor;

    /* Reset the vector check slaves. */
    memset(check_slaves, 0, sizeof(bool)*n_slaves);

    /* Variables for updates. */
    ValType ** _n_updates;
    ValType * _mean_labels = new ValType [MAX(max_n_label, max_e_label)];

    ValType _this_update;

    /* Reset the norm of the subgradient to zero. */
    cur_subgradient = 0.0;

    /* Reset the updates variable to zero. */
    _reset_updates();

//    ValType alpha = 0.01/sqrt(optim_it + 1);

    /* #pragma openmp parallel for */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        if(!check_nodes[n_id])
        {
            /* No need to make updates for this node. All labellings agree! */
            continue;
        }

        /* Reset _mean_labels to 0. */
        for(int j = 0; j < n_labels[n_id]; j ++)
        {
            _mean_labels[j] = 0;
        }

        /* Vectors to store labels assigned to a factor by contributing slaves, 
           and the ID of that factor for each of these Slaves. */
        std::vector<unsigned short> assigned_labels; 
        std::vector<unsigned short> f_id_in_slaves;

        /* Get the slave list for this node. */
        slaves_this_factor    = nodes_in_slaves[n_id];
        n_slaves_this_factor  = n_slaves_node[n_id];

        /* Get the labelling for this node from each slave. */
        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* Get the node ID for n_id in the Slave slave_list[slaves_this_factor[s]]. */
            int _nid_in_s = slave_list[slaves_this_factor[s]]->node_map(n_id);
            /* Add to the list of IDs for this node in the Slaves in its slave list. */
            f_id_in_slaves.push_back(_nid_in_s);
            /* Push the assigned label for this node by this slave into assigned_labels. */
            assigned_labels.push_back(slave_list[slaves_this_factor[s]]->labels[_nid_in_s]);
        }

        /* To compute updates, we need to convert the assigned labels into one-hot vectors. */
        _n_updates  = new ValType * [n_slaves_this_factor];
        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* We use calloc because we want the memory to be set to zero initially. */
            _n_updates[s] = (ValType *)calloc(n_labels[n_id], sizeof(ValType));
            /* One-hot the appropriate place. */
            _n_updates[s][assigned_labels[s]]  = 1;
            /* Add to the mean labels. */
            _mean_labels[assigned_labels[s]]  += 1;
        }

        /* Divide by the number of slaves in this slave list to finally get the mean. 
           This step is equivalent to the projection of the vector \lambda onto a 
           space where \sum_i \lambda_i = 0. */
        for(int j = 0; j < n_labels[n_id]; j ++)
        {
            _mean_labels[j] /= n_slaves_this_factor;
        }

        /* Subtract the vector _mean_labels from all assigned labellings, i.e., 
           one-hot labellings retrieved from all slaves in this node's slave list.
           This is finally the update that is to be made to the node energies 
           of the slave. We will directly make this update instead of unnecessarily
           iterating once more to subtract. */
        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            for(int j = 0; j < n_labels[n_id]; j ++)
            {
                /* For the slave
                   slaves_this_factor[i],
                   for the node 
                   f_id_in_slaves[i],
                   in this slave, and for the label 
                   j
                   for this node, make the update
                   _n_updates[i][j] - _mean_labels[j]. */

                /* The current update. */
                _this_update = _n_updates[s][j] - _mean_labels[j];
                /* Mark the update. */
                node_updates[slaves_this_factor[s]][f_id_in_slaves[s]][j] = _n_updates[s][j] - _mean_labels[j];
//                slave_list[slaves_this_factor[s]]->node_energies[f_id_in_slaves[s]][j] += alpha*(_n_updates[s][j] - _mean_labels[j]);
                /* Update the norm of the current subgradient. */
                cur_subgradient += _this_update * _this_update;
            }       /* Finished iterating over node labels for this node in this slave. */
            /* There were updates for this slave. So we solve it again in the next iteration. */
            check_slaves[slaves_this_factor[s]] = true;
            /* Mark this update. */
            _mark_slave_updates[0][slaves_this_factor[s]] = true;

            /* Delete _n_updates[i], we no longer need it. */
            free(_n_updates[s]);
        }           /* Finished iterating over slaves to make updates. */

        /* Free memory assigned here. We no longer need it. */
        delete [] _n_updates;

        /* Perform param updates for this node. */
    }               /* Finished iterating over nodes for param updates. */

    /* Now make updates for edges. */
    /* #pragma openmp parallel for */
    for(int e_id = 0; e_id < 0; e_id ++)
    {
        if(!check_edges[e_id])
        {
            /* No need to check this edge. */
            continue;
        }

        /* Reset _mean_labels to 0. */
        for(int j = 0; j < e_labels[e_id]; j ++)
        {
            _mean_labels[j] = 0;
        }

        /* Get the edge ends for this edge. */
        int _gfrom, _gto;
        _gfrom = node_ids_from_edge_id[e_id].first;
        _gto   = node_ids_from_edge_id[e_id].second;
        /* A slave's labelling for this edge. */
        int _sl_gfrom, _sl_gto;

        /* Create vectors to store labels assigned by slaves and IDs of this 
           edge in containing slaves. */
        std::vector< unsigned short > assigned_labels;
        std::vector< unsigned short > f_id_in_slaves;

        /* Get the slave list for this edge. */
        slaves_this_factor    = edges_in_slaves[e_id];
        n_slaves_this_factor  = n_slaves_edge[e_id];

        /* Get the ID of this edge for each slave, and also the labelling of this 
           edge by each slave. */
        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* ID of this edge in this slave. */
            int _eid_in_s = slave_list[slaves_this_factor[s]]->edge_map(e_id);
            /* Push this into f_id_in_slaves. */
            f_id_in_slaves.push_back(_eid_in_s);
            /* Push also the labelling of this edge by this slave into 
               assigned_labels. */
            _sl_gfrom  = slave_list[slaves_this_factor[s]]->get_node_label(_gfrom);
            _sl_gto    = slave_list[slaves_this_factor[s]]->get_node_label(_gto);
            assigned_labels.push_back(_sl_gfrom*n_labels[_gto] + _sl_gto);
        }

        /* To compute updates, we need to convert the assigned labels into one-hot vectors. */
        _n_updates  = new ValType * [n_slaves_this_factor];
        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* Allocate space for the one-hot vector for this slave. */
            _n_updates[s] = (ValType *)calloc(e_labels[e_id], sizeof(ValType));
            /* Make one-hot. */
            _n_updates[s][assigned_labels[s]] =  1;
            /* Add to the 'mean' vector. */
            _mean_labels[assigned_labels[s]]  += 1;
        }

        /* Divide by the number of slaves to finally calculate the mean. */
        for(int j = 0; j < e_labels[e_id]; j ++)
        {
            _mean_labels[j] /= n_slaves_this_factor;
        }

        /* Subtract the vector _mean_labels from all assigned labellings, i.e., 
           one-hot labellings retrieved from all slaves in this edge's slave list.
           This is finally the update that is to be made to the edge energies 
           of the slaves. We will directly make this update instead of unnecessarily
           iterating once more. */
        for(int s = 0; s < n_slaves_this_factor; s ++)
        {
            /* A flag that is raised if we must tranpose the edge energies, and hence, 
               the energy update. The energies are transposed if we have a cycle slave.
               Further, they are also tranposed if the node-ordering in the cycle slave
               is not the same as the Graph. Hence, in effect, this flag is raised
               only if the slave is a cycle slave and the node-ordering in the slave
               is the same as in the graph. */
            bool _transposed = (slave_list[slaves_this_factor[s]]->type() == 'c' && 
                    slave_list[slaves_this_factor[s]]->node_map(_gfrom) < slave_list[slaves_this_factor[s]]->node_map(_gto));
            for(int j = 0; j < n_labels[_gfrom]; j ++)
            {
                for(int k = 0; k < n_labels[_gto]; k ++)
                {
                    /* For the slave
                       slaves_this_factor[s]
                       for the edge
                       f_id_in_slaves[s],
                       in this slave, and for the label 
                       (j,k)
                       for this node, make the update
                       _n_updates[s][jk] - _mean_labels[jk]. */
                    int _label_in_slave;
                    int _label_in_update = j*n_labels[_gto] + k;
                    /* Choose the right label to update in the Slave. */
                    if(_transposed)
                    {
                        _label_in_slave = k*n_labels[_gfrom] + j;
                    }
                    else
                    {
                        _label_in_slave = _label_in_update;
                    }

                    /* The current update. */
                    _this_update = _n_updates[s][_label_in_update] - _mean_labels[_label_in_update];
                    /* Mark this update. */
                    edge_updates[slaves_this_factor[s]][f_id_in_slaves[s]][_label_in_slave] = _this_update;
//                    slave_list[slaves_this_factor[s]]->edge_energies[f_id_in_slaves[s]][_label_in_slave] += 
//                       alpha*(_n_updates[s][_label_in_update] - _mean_labels[_label_in_update]);
                    /* Update the norm of the current subgradient. */
                    cur_subgradient += _this_update * _this_update;     
                }       /* End of for over n_labels[_gto]. */
            }           /* End of for over n_labels[_gfrom]. */
            /* There were updates for this slave. So we solve it again in the next iteration. */
            check_slaves[slaves_this_factor[s]] = true;
            /* Mark this update. */
            _mark_slave_updates[1][slaves_this_factor[s]] = true;

            /* Delete this row of the update - we no longer need it. */
            free(_n_updates[s]);
        }               /* End of for over slaves_this_factor. */

        delete [] _n_updates;
    }                   /* Finished making updates for this slave. */
    delete [] _mean_labels;

    /* Reset the number of slaves to be solved in the next iteration. */
    n_slaves_to_solve = 0;
    for(int s = 0; s < n_slaves; s ++)
    {
        n_slaves_to_solve += check_slaves[s];
    }

    /* Push the norm of the subgradient into history. */
    subgrad_hist.push_back(cur_subgradient);
}                       /* End of Graph::compute_param_updates */

/*
 * Graph::apply_param_updates -- Apply previously computed parameter updates (computed
 *   by Graph::compute_param_updates).
 */
void Graph::apply_param_updates(ValType a_start, OptimStrategy strategy)
{
    /* The step size for alpha. This is derived from a_start depending on what
       strategy is chosen. */
    switch(strategy)
    {
        case OptimStrategy::STEP:            alpha = a_start/sqrt(optim_it + 1); //*(best_primal_cost - best_dual_cost)/10000;
                                             break;

        case OptimStrategy::STEP_SUBGRAD:    alpha = a_start/(sqrt(optim_it + 1) * cur_subgradient);
                                             break;

        case OptimStrategy::STEP_SQSUM:      alpha = a_start/(2.0 + optim_it);
                                             break;

        case OptimStrategy::STEP_K:          alpha = a_start/pow(2 + optim_it, 1.0/_decayk);
                                             break;

        case OptimStrategy::ADAPTIVE:        alpha = a_start*(best_primal_cost - dual_cost)/cur_subgradient;
                                             break;

        case OptimStrategy::ADAPTIVE_SDECAY: alpha = a_start*(best_primal_cost - dual_cost)/(sqrt(optim_it+1) * cur_subgradient);
                                             break;

        case OptimStrategy::ADAPTIVE_LDECAY: alpha = a_start*(best_primal_cost - dual_cost)/((2+optim_it)*cur_subgradient);
                                             break;

        case OptimStrategy::ADAPTIVE_KDECAY: alpha = a_start*(best_primal_cost - dual_cost)/(pow(2 + optim_it, 1.0/_decayk)*cur_subgradient);
                                             break;
    }

    /* Push alpha into alpha_hist. */
    alpha_hist.push_back(alpha);

    /* Make marked updates now. */
    for(int s = 0; s < n_slaves; s ++)
    {
        /* First, ensure that there are updates to be made for this slave. */
        if(!check_slaves[s])
            continue;

        /* Check if there are node updates to be made for this slave. */
        if(_mark_slave_updates[0][s])
        {
            int n_nodes_in_slave = slave_list[s]->n_nodes;
            /* Node updates. */
            for(int i = 0; i < n_nodes_in_slave; i ++)
            {
                int n_labels_in_node = slave_list[s]->n_labels[i];
                for(int l = 0; l < n_labels_in_node; l ++)
                {
                    slave_list[s]->node_energies[i][l] += alpha*(node_updates[s][i][l]);
                }
            }
        }

        /* Check if there are edge updates to be made for this slave. */
        if(_mark_slave_updates[1][s])
        {
            int n_edges_in_slave = slave_list[s]->n_edges;
            /* Edge updates. */
            for(int e = 0; e < n_edges_in_slave; e ++)
            {
                int n_labels_in_edge = slave_list[s]->e_labels[e];
                for(int l = 0; l < n_labels_in_edge; l ++)
                {
                    slave_list[s]->edge_energies[e][l] += alpha*(edge_updates[s][e][l]);
                }
            }
        }
    }

    /* Make the update. */
   // slave_list[slaves_this_factor[s]]->node_energies[f_id_in_slaves[s]][j] += alpha*(_n_updates[s][j] - _mean_labels[j]);

    /* Make the update. */
    //slave_list[slaves_this_factor[s]]->edge_energies[f_id_in_slaves[s]][_label_in_slave] += 
     //   alpha*(_n_updates[s][_label_in_update] - _mean_labels[_label_in_update]);
    return ;
}                       /* End of Graph::apply_param_updates */

/*
 * Graph::set_node_energies - Add node energies for a node in the Graph. 
 */
int Graph::set_node_energies(int n_id, ValType *e_vec)
{
    /* Set node energies specified by e_vec for node id n_id. */

    /* Check that it is a valid Id. */
    assert(n_id < n_nodes);

    /* We have assumed that e_vec will have the appropriate number
       of elements, specified by n_labels[n_id]; */
    for(int i = 0; i < n_labels[n_id]; i ++)
        node_energies[n_id][i] = e_vec[i];

    return n_id;
}                       /* End of set_node_energies(). */

/* 
 * Graph::set_edge_energies -- Specify edge energies for a edge between two nodes. 
 */
int Graph::set_edge_energies(int from, int to, ValType *e_vec)
{
    /* Set edge energies specified by e_vec for the edge
       between from and to. */

    /* Make sure we can still add another edge. */
    assert(current_edge_count < n_edges);

    /* The edge id is the current edge count. */
    int e_id = current_edge_count;

    /* Edges are always assumed to be from a lower ID to a higher ID. */
    int _from = MIN(from, to);
    int _to   = from + to - _from;

    /* Set e_labels for this edge. */
    e_labels[e_id] = n_labels[_from]*n_labels[_to];
    /* Update max_e_label. */
    if(e_labels[e_id] > max_e_label)
        max_e_label = e_labels[e_id];

    /* First allocate memory to store these energies. */
    edge_energies[e_id] = new ValType [e_labels[e_id]];

    /* The energies are stored in a row-first manner. That means
       for the labelling (i, j) for this edge, the energy
       can be retrieved by accessing the element i*n_labels[_to] + j. */
    for(int i = 0; i < n_labels[from]*n_labels[to]; i ++)
        edge_energies[e_id][i] = e_vec[i];

    /* Add this edge to the adjacency matrix. */
    adj_mat[from][to] = true;
    adj_mat[to][from] = true;

    /* Update the E -> V x V map for this edge. */
    node_ids_from_edge_id[e_id] = std::make_pair(_from, _to);
    /* Update the V x V -> E map for this edge. */
    edge_id_from_node_ids[std::make_pair(_from, _to)] = e_id;
    edge_id_from_node_ids[std::make_pair(_to, _from)] = e_id;

    /* Increment the current edge count. */
    ++ current_edge_count;

    /* Return the edge id. */
    return e_id;
}                       /* End of set_edge_energies(). */

/* 
 * Graph::optimise - Optimise the specified energy over this graph. 
 */
void Graph::optimise(ValType a_start, int n_iter, OptimStrategy strategy)
{
    if(!slaves_created)
    {
        printf("Graph decomposition has not yet been finalised. Please add some slaves by calling Graph::add_*_slaves().\n");
        return ;
    }

    if(!check_decomposition())
    {
        printf("The added slaves are not enough to decompose the Graph entirely.\n");
        printf("Please add more slaves to complete the decomposition before calling Graph::optimise().\n");
        return ;
    }

    optim_it = 0;
    bool converged = false;

    // check_decomposition();

    while(optim_it < n_iter && !converged)
    {
        if(_verbose && (optim_it == 0 || (optim_it+1)%_print_every == 0))
        {
            printf("Iteration %4d. Solving %5d subproblems. ", optim_it + 1, n_slaves_to_solve);
        }

        /* Optimise all slaves. */
        optimise_slaves();
        if(debug)
        {
            _print_state(0);
        }

        /* Find conflicts. */
        find_conflicts();

        /* Compute costs. */
        estimate_primal();
        compute_dual();

        /* Push the number of disagreeing nodes into n_miss history. */
        n_miss_hist.push_back(n_miss);

        /* Compute parameter updates. */
        compute_param_updates();

        /* If there are no conflits, we have converged! */
        if(n_miss == 0)
        {
            converged = true;
            break;
        }

        /* Make parameter updates according to chosen strategy. */
        apply_param_updates(a_start, strategy);

        if(_verbose && (optim_it == 0 || (optim_it+1)%_print_every == 0))
        {
            /* Print the costs. */
            printf("alpha = %.6f | n_miss = %d | ||dg||**2 = %.2lf | PRIMAL = %lf | DUAL = %lf | GAP = %lf | min(GAP) = %lf\n", alpha, n_miss, cur_subgradient, primal_cost, dual_cost, primal_cost - dual_cost, best_primal_cost - best_dual_cost);
        }

        /* Check decomposition here. */
        //check_decomposition();

        /* Increment iteration. */
        ++ optim_it;
    }               /* Finished optimise loop over n_iter iterations. */

    if(converged)
    {
        printf("Converged after %d iterations!\n", optim_it+1);
    }
    printf("Best PRIMAL = %.6lf; Best DUAL = %.6lf; Gap = %lf\n", best_primal_cost, best_dual_cost, best_primal_cost - best_dual_cost);

    /*
       printf("Labels: ");
       for(int i = 0; i < n_nodes; i ++)
       printf("%d ", labels[i]);
       printf("\n");
     */

    FILE *f_cost = fopen("costs.txt", "w");
    for(unsigned int i = 0; i < primal_hist.size(); i ++)
    {
        fprintf(f_cost, "%lf %lf %d %lf %lf\n", primal_hist[i], dual_hist[i], n_miss_hist[i], subgrad_hist[i], alpha_hist[i]);
    }
    fclose(f_cost);

}                       /* End of Graph::optimise */

/* 
 * Graph::get_edge_id -- Get edge ID for two given edge ends. 
 */
int Graph::get_edge_id(int x, int y)
{
    return edge_id_from_node_ids[std::make_pair(x,y)];
}

/*
 * Graph::get_node_ids -- Get node IDs for a given edge ID. 
 */
std::pair<int, int> Graph::get_node_id(int x)
{
    return node_ids_from_edge_id[x];
}

/* 
 * Graph::_print_state -- Print the state of the Graph during an optimisation. 
 * Also pauses for keypress after printing the state. 
 */
void Graph::_print_state(int state)
{
    printf("\n");
    /* Just after slaves have been optimised. */
    if(state == 0)
    {
        for(int s = 0; s < n_slaves; s ++)
        {
            printf(" --- Slave %d ----------------------------\n", s);
            printf(" Node list:       ");
            for(int i = 0; i < slave_list[s]->n_nodes; i ++)
            {
                printf("%3d ", slave_list[s]->node_list[i]);
            }
            printf("\n");
            printf(" Found optimum:   ");
            for(int i = 0; i < slave_list[s]->n_nodes; i ++)
            {
                printf("%3d ", slave_list[s]->labels[i]);
            }
            printf(" %lf\n", slave_list[s]->energy);
            printf(" Brute force:     ");
            std::vector<int> _best_labels = slave_list[s]->brute_force_solver();
            for(int i = 0; i < slave_list[s]->n_nodes; i ++)
            {
                printf("%3d ", _best_labels[i]);
            }
            ValType brute_force_energy = slave_list[s]->compute_energy_from_labels(_best_labels);
            printf(" %lf", brute_force_energy);
            if(brute_force_energy != slave_list[s]->energy)
            {
                printf("\t\t**********");
                getchar();
            }
            printf("\n\n");
        }
    }
    getchar();

    return ;
}

/*
 * Graph::verbose -- set or unset verbosity 
 */
void Graph::verbose(bool v)
{
    _verbose = v;
    return ;
}

/*
 * Graph::print_every -- specify how frequently to print optimisation status. 
 */
void Graph::print_every(int n)
{
    _print_every = n;
    return;
}

/* 
 * Graph::set_k_decayk -- specify K in OptimStrategy::STEP_K, and OptimStrategy::ADAPTIVE_KDECAY
 */
void Graph::set_k_decayk(float k)
{
    _decayk = k;
    return;
}
