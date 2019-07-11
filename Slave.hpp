#define __SLAVE_H

#include "utils.h"
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <assert.h>
#include <map>
#include <stack>
#include "thirdparty.h"
#include "cycle.h" 

typedef CycleSolver::ValType ValType;
typedef CycleSolver::Cycle Cycle;

ValType randn(double mean, double std)
{
    return CycleSolver::randn(mean, std);
}

class Slave
{
    private:
        /* Essential variables for the slave. */
        /* Self-descriptive. */
        int      slave_id;
        char     _type;
        bool **  adj_mat;

        /* Maximum possible labels for a node and edge. */
        int max_n_labels;
        int max_e_labels;

        /* Map to go from node ID in Graph to node ID in Slave. */
        /* The inverse node map already exists in node_list. */
        std::map< int, int > _node_map;
        /* Map to go from edge ID in Graph to edge ID in Slave. */
        /* The inverse edge map already exists in edge_list. */
        std::map< int, int > _edge_map;
        /* Node degrees. */
        int *     node_degrees;

        /* Maps to go from edge ID to edge ends, and vice versa.
           Here, they are pointers because these maps are actually
           made during the creation of slaves. */
        std::map< int, std::pair<int, int> > * node_ids_from_edge_id;
        std::map< std::pair<int, int>, int > * edge_id_from_node_ids;
    
        /* Max value of any energy in this Slave. */
        double _max_energy_this_slave;
        /* Adjusted node and edge energies, or rather potentials. */
        ValType ** adj_node_probs;
        ValType ** adj_edge_probs;
        /* Messages to be computed for max-product belief propagation. */
        ValType ** messages;
        /* Messages coming into nodes. */
        ValType ** messages_in;
        /* To temporarily store the node degrees. Requires by every
           execution of Slave::_max_prod_bp */
        int * _degrees;
        /* Copy of adjacency matrix. Also required by Slave::_max_prod_bp */
        bool ** _adj;

        /* ---- Optimisers ---- */
        /* Optimise a tree-structured problem using Max-Product Belief Propagation. */
        void _max_prod_bp(void);
        /* Optimise a cycle-structured problem using the Fast Cycle Solver (Wang and Koller). */
        void _fast_cycle_solver(void);
        /* Brute force optimisation. Not to be used other than for debugging. */
        std::vector<int> _brute_force_solver(void);

    public:
        /* Need the number of nodes to be public. */
        int      n_nodes;
        int      n_edges;

        /* labels is public because this labelling is accessed to update parameters of
           the Graph. */
        int *    labels;

        /* For debugging. */
        int *    node_list;
        int *    edge_list;

        /* Number of labels per node and edge. Needed by Graph containg this Slave. */
        int *    n_labels;
        int *    e_labels;

        /* Node and edge energies are public because they need to be modified by 
           the master problem. */
        ValType ** node_energies;
        ValType ** edge_energies;

        /* This slave's energy for the current configuration. */
        ValType energy;

        /* Solve this slave using brute force. */
        std::vector<int> brute_force_solver(void);

        /* 
         * Constructor specifying slave id, number 
         * of nodes, and number of edges 
         */
        Slave(int s_id, int nn, int ne)
        {
            slave_id  = s_id;
            n_nodes   = n_nodes;
            n_edges   = n_edges;
            node_list = 0;
            edge_list = 0;
            n_labels  = 0;
            e_labels  = 0;
            _type     = 0;
            adj_mat   = 0;

            max_n_labels = 0;
            max_e_labels = 0;

            /* Reset map pointers to zero. */
            edge_id_from_node_ids = 0;
            node_ids_from_edge_id = 0;
        }

        /*
         * Constructor specifying everything.
         */
        Slave(int s_id, int nn, int ne, int * nl, int *el, 
                int * nlab, ValType **node_erg, ValType **edge_erg,
                char tp, bool **adj,
                std::map< int, std::pair<int, int> > * n_from_e,
                std::map< std::pair<int, int>, int > * e_from_n)
        {
            slave_id   = s_id;
            n_nodes    = nn;
            n_edges    = ne;

            /* Save node list, edge list, n_labels, and adjacency matrix. */
            node_list = nl;
            edge_list = el;
            n_labels  = nlab;
            adj_mat   = adj;
            /* Save node and edge energies. */
            node_energies = node_erg;
            edge_energies = edge_erg;

            _type     = tp;
            labels    = new int [n_nodes];

            /* Update maps. */
            for(int i = 0; i < n_nodes; i ++)
            {
                _node_map[node_list[i]] = i;
            }
            for(int i = 0; i < n_edges; i ++)
            {
                _edge_map[edge_list[i]] = i;
            }

            /* Calculate node degrees if the subproblem is tree-structured.
               We will reuse this array later if we are optimise a tree slave. */
            if(_type == 't')
            {
                node_degrees = new int [n_nodes];
                for(int i = 0; i < n_nodes; i ++)
                {
                    int degree = 0;
                    for(int j = 0; j < n_nodes; j ++)
                    {
                        if(j != i && adj_mat[i][j])
                            ++ degree;
                    }
                    node_degrees[i] = degree;
                }
            }    

            /* Maps. */
            edge_id_from_node_ids = e_from_n;
            node_ids_from_edge_id = n_from_e;

            /* Get max n and e labels. */
            max_n_labels = n_labels[0];
            for(int i = 1; i < n_nodes; i ++)
            {
                if(max_n_labels < n_labels[i])
                {
                    max_n_labels = n_labels[i];
                }
            }

            /* Create e_labels. */
            e_labels     = new int [n_edges];
            if(_type == 't')
            {
                int _from    = (*node_ids_from_edge_id)[0].first;
                int _to      = (*node_ids_from_edge_id)[0].second;
                e_labels[0]  = n_labels[_from]*n_labels[_to];
                max_e_labels = e_labels[0];
                
                for(int e = 0; e < n_edges; e ++)
                {
                    int _from    = (*node_ids_from_edge_id)[e].first;
                    int _to      = (*node_ids_from_edge_id)[e].second;
                    e_labels[e]  = n_labels[_from]*n_labels[_to];
                    if(max_e_labels < e_labels[e])
                    {
                        max_e_labels = e_labels[e];
                    }
                }
            }
            else if (_type == 'c')
            {
                e_labels[0]  = n_labels[0]*n_labels[1];
                max_e_labels = e_labels[0];
                for(int e = 1; e < n_edges; e ++)
                {
                    e_labels[e] = n_labels[e]*n_labels[(e+1)%n_nodes];
                    if(max_e_labels < e_labels[e])
                        max_e_labels = e_labels[e];
                }
            }

            /* If the Slave is a tree, allocate space for adj_node and adj_edge probs, 
               and other variables required by Slave::_max_prod_bp. This is because 
               we can save time and energy by not reallocating these each time. */
            if(_type == 't')
            {
                adj_node_probs = new ValType * [n_nodes];
                adj_edge_probs = new ValType * [n_edges];
                for(int i = 0; i < n_nodes; i ++)
                {
                    adj_node_probs[i] = new ValType [n_labels[i]];
                }
                for(int e = 0; e < n_edges; e ++)
                {
                    adj_edge_probs[e] = new ValType [e_labels[e]];
                }

                _adj = new bool * [n_nodes];
                for(int i = 0; i < n_nodes; i ++)
                    _adj[i] = new bool [n_nodes];

                messages    = new ValType * [2*n_edges];
                messages_in = new ValType * [n_nodes];
                /* Messages sent between nodes. */
                /* One for each direction on each edge. */
                for(int e = 0; e < n_edges; e ++)
                {
                    int _efrom = (*node_ids_from_edge_id)[e].first;
                    int _eto   = (*node_ids_from_edge_id)[e].second;
                    /* Allocate space for all messages. */
                    messages[e]           = new ValType [n_labels[_eto]];
                    messages[e + n_edges] = new ValType [n_labels[_efrom]];
                }
                /* Messages coming into nodes. 
                   Saves computation time at the end 
                   when we need to calculate beliefs. */
                /* Set messages_in to 1 initially. */
                for(int i = 0; i < n_nodes; i ++)
                {
                    messages_in[i] = new ValType [n_labels[i]];
                }

                _degrees = new int [n_nodes];
            }
        }

        /*
         * Destructor
         */
        ~Slave()
        {
            /* Free all memory. The only references to this memory are in this slave. */
            /* So these values will not be used again. */
            delete [] node_list;
            delete [] edge_list;
            if(adj_mat)
            {
                for(int i = 0; i < n_nodes; i ++)
                {
                    free(adj_mat[i]);
                }
                delete [] adj_mat;
            }

            /* Delete n_labels in any case. It is not shared with any other
               object. */
            delete [] n_labels;
            delete [] e_labels;

            delete [] labels;

            /* Node degrees was assigned memory only if the Slave is of type tree. */
            if(_type == 't')
            {
                delete [] node_degrees;
                delete edge_id_from_node_ids;
                delete node_ids_from_edge_id;
                for(int i = 0; i < n_nodes; i ++)
                    delete [] adj_node_probs[i];
                for(int e = 0; e < n_edges; e ++)
                    delete [] adj_edge_probs[e];
                delete [] adj_node_probs;
                delete [] adj_edge_probs;

                for(int e = 0; e < 2*n_edges; e ++)
                    delete [] messages[e];
                delete [] messages;

                for(int i = 0; i < n_nodes; i ++)
                {
                    delete [] messages_in[i];
                    delete [] _adj[i];
                }
                delete [] messages_in;
                delete [] _adj;

                delete [] _degrees;
            }

            for(int i = 0; i < n_nodes; i ++)
                delete [] node_energies[i];
            for(int e = 0; e < n_edges; e ++)
                delete [] edge_energies[e];
            delete [] node_energies;
            delete [] edge_energies;
        }

        /* Member functions */
        
        /* A function to compute the maximum energy's value. This is 
           used to convert energies into potentials for max-product BP. */
        void _compute_max_energy();

        /* Optimise this slave. */
        void optimise();

        /* Compute this slave's energy for the current labelling. */
        void compute_energy();

        /* Compute this slave's energy given a labelling. */
        ValType compute_energy_from_labels(std::vector<int>);

        /* Set labels for this slave. */
        bool set_labels(int *);

        /* node_map: retrieve the node ID in Slave for a given ID in Graph. */
        int node_map(int n_id);
        /* edge_map: retrieve the edge ID in Slave for a given ID in Graph. */
        int edge_map(int e_id);

        /* Return the label assigned to a node by this slave. The ID of the node
           obeys the Graph, NOT the Slave. */
        int get_node_label(int n_id);
        /* Return the labelling of an edge by this Slave. The input is the ID
           of the edge in the Graph, NOT in the Slave. */
        std::pair<int, int> get_edge_labels(int e_id);

        /* Return the type of slave. */
        char type(void);
};

/* 
 * Slave::_compute_max_energy -- compute the maximum value of any energy
 * within this Slave.
 */
void Slave::_compute_max_energy(void)
{
    _max_energy_this_slave = node_energies[0][0];

    /* Iterate over nodes first. */
    for(int i = 0; i < n_nodes; i ++)
    {
        for(int j = 0; j < n_labels[i]; j ++)
        {
            if(node_energies[i][j] > _max_energy_this_slave)
            {
                _max_energy_this_slave = node_energies[i][j];
            }
        }
    }

    /* Now iterate over edges. */
    for(int i = 0; i < n_edges; i ++)
    {
        int _from = (*node_ids_from_edge_id)[i].first;
        int _to   = (*node_ids_from_edge_id)[i].second;
        for(int j = 0; j < n_labels[_from]*n_labels[_to]; j ++)
        {
            if(edge_energies[i][j] > _max_energy_this_slave)
            {
                _max_energy_this_slave = edge_energies[i][j];
            }
        }
    }
    
    /* Just to be safe. */
    _max_energy_this_slave += 1;
    /* The max is stored in _max_energy_this_slave. */

    return;
}

/* 
 * Slave::brute_force_solver -- Solve this slave using brute force. 
 */
std::vector<int> Slave::brute_force_solver(void)
{
    return _brute_force_solver();
}

/*
 * Slave::optimise -- Optimise this slave using the right solver, depending on 
 *    the type of subproblem. 
 */
void Slave::optimise(void)
{
    switch(_type)
    {
        case 'c': _fast_cycle_solver();
                  break;

        case 't': _max_prod_bp();
                  break;
    }
    return ;
}

/*
 * Slave::_fast_cycle_solver -- Optimise this cycle-structured Slave
 *    using the fast cycle solver from Wang and Koller.
 */
void Slave::_fast_cycle_solver(void)
{
    if(_type != 'c')
    {
        printf("Cannot call fast cycle solver on non-cycle slaves.\n");
        exit(1);
    }

    /* Create a cycle to solve this slave. */
    Cycle c;
    /* Initialise a cycle with these node and edge energies. */
    c.initialiseCycle(node_energies, edge_energies, n_labels, n_nodes);
    /* Run the fast cycle solver. */
    c.runFastSolver(0);
    /* 0 here says that the edge marginals for only the 0-th edge are calculated.
       These are stored in c.mmarg[n_nodes+0], which is of type
       std::vector< ValType >. */

    for(int i = 0; i < n_nodes; i ++)
    {
        labels[i] = c.assignment[i];
    }
    c.freeMemory();

    return ;
}

/*
 * Slave::__max_prod_bp -- Optimise this tree-structured Slave using
 *    max-product belief propagation. 
 */
void Slave::_max_prod_bp(void)
{
    /* Tree-structured slave. We solve this with max product BP. */
    if(_type != 't')
    {
        printf("Cannot call max-product belief propagation on non-tree slaves.\n");
        exit(1);
    }

    /* Convert energies to probabilities: Apply the exponential, and divide by the total
       for each factor. This converts them into probabilities, signifying that lower energy
       states are more probable. */
    ValType scaler = 1.0;
    ValType total_adj_pot;
    for(int i = 0; i < n_nodes; i ++)
    {
        total_adj_pot = 0.0;
        /* Apply exp. */
        for(int j = 0; j < n_labels[i]; j ++)
        {
            adj_node_probs[i][j] = exp(-1*scaler*node_energies[i][j]);
            total_adj_pot += adj_node_probs[i][j];
        }
        /* Turn them into probabilities. */
        for(int j = 0; j < n_labels[i]; j ++)
        {
            adj_node_probs[i][j] /= total_adj_pot;
        }
    }
    for(int e = 0; e < n_edges; e ++)
    {
        int _from = (*node_ids_from_edge_id)[e].first;
        int _to   = (*node_ids_from_edge_id)[e].second;
        total_adj_pot = 0.0;
        /* Apply exp. */
        for(int j = 0; j < n_labels[_from]*n_labels[_to]; j ++)
        {
            adj_edge_probs[e][j] = exp(-1*scaler*edge_energies[e][j]);
            total_adj_pot += adj_edge_probs[e][j];
        }
        /* Turn them into probabilities. */
        for(int j = 0; j < n_labels[_from]*n_labels[_to]; j ++)
        {
            adj_edge_probs[e][j] /= total_adj_pot;
        }
    }

    /* Initialise messages to zero. */
    for(int e = 0; e < n_edges; e ++)
    {
        int _efrom = (*node_ids_from_edge_id)[e].first;
        int _eto   = (*node_ids_from_edge_id)[e].second;
        /* Initialise messages to zero. */
        for(int j = 0; j < n_labels[_eto]; j ++)
        {
            messages[e][j]           = 0;
        }
        for(int j = 0; j < n_labels[_efrom]; j ++)
        {
            messages[e + n_edges][j] = 0;
        }
    }

    /* Initialise messages_in to 1. */
    for(int i = 0; i < n_nodes; i ++)
    {
        for(int j = 0; j < n_labels[i]; j ++)
            messages_in[i][j] = 1.0;
    }

    /* Store the path we took. */
    std::stack< std::pair<int, int> > path;

    /* Store the queued node. */
    std::queue<int> next_nodes;

    /* Copy node degrees into another array, because we will be modifying it. */
    for(int i = 0; i < n_nodes; i ++)
    {
        _degrees[i] = node_degrees[i];
    }

    /* Copy adj_mat into another array because we are going to modify it. */
    for(int i = 0; i < n_nodes; i ++)
    {
        for(int j = 0; j < n_nodes; j ++)
            _adj[i][j] = adj_mat[i][j];
    }

    /* Insert all leaf nodes in the queue. */
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        if(_degrees[n_id] == 1)
        {
            next_nodes.push(n_id);
        }
    }

    /* The from and to nodes for a message. */
    int _from;
    int _to;
    
    /* The node and edge energies. New memory must be allocated 
       because we convert energies to potentials. */
    ValType *node_erg = new ValType [max_n_labels];
    ValType *edge_erg = new ValType [max_e_labels];

    /* Iterate till we have sent all messages. */
    while(!next_nodes.empty())
    {
        /* Get the top node in the queue. */
        _from = next_nodes.front();
        /* Pop this from the queue. */
        next_nodes.pop();

        /* Find the next node to send messages to.
           Since the fact that a node is in the queue implies that 
           it has degree 1, the next node is the only remaining
           neighbour of _from. */
        for(_to = 0; !_adj[_from][_to]; ++ _to);

        /* Subtract 1 from the node degrees of _from and _to. */
        _degrees[_from]  -= 1;
        _degrees[_to]    -= 1;
        /* Remove this edge from the adjacency matrix's copy. */
        _adj[_from][_to]  = 0;
        _adj[_to][_from]  = 0;

        /* Push this edge into the path. */
        path.push(std::make_pair(_from, _to));

        /* Number of labels for _from and _to. */
        int _nl_from = n_labels[_from];
        int _nl_to   = n_labels[_to];

        /* Message id for this edge. For now, it is the same as
           the edge ID. */
        int e_id = (*edge_id_from_node_ids)[std::make_pair(_from,_to)];
        /* However, if _from > _to, we add n_edges to indicate that this 
           message is being sent in the other direction. */
        bool _transposed = false;
        int m_id         = e_id;
        if(_from > _to)
        {
            m_id        += n_edges;
            _transposed  = true;
        }

        /* Subtract the energies from the max to convert them into potentials. */
        for(int i = 0; i < _nl_from; i ++)
        {
            //node_erg[i] = _max_energy_this_slave - node_energies[_from][i];
            node_erg[i] = adj_node_probs[_from][i];
            assert(node_erg[i] > 0);
        }
        for(int j = 0; j < _nl_from*_nl_to; j ++)
        {
            //edge_erg[j] = _max_energy_this_slave - edge_energies[e_id][j];
            edge_erg[j] = adj_edge_probs[e_id][j];
            assert(edge_erg[j] > 0);
        }

        ValType _total_message = 0;
        /* Do the max-product update now. */
        for(int j = 0; j < _nl_to; j ++)
        {
            /* We must adjust the label ID according to whether we need
               to transpose the energies or not. */
            int _l_id = _transposed ? (j*_nl_from) : j;
            ValType _max = messages_in[_from][0]*node_erg[0]*edge_erg[_l_id];
            ValType _this;
            for(int i = 1; i < _nl_from; i ++)
            {
                /* Max-product. */
                _l_id = _transposed ? (j*_nl_from + i) : (i*_nl_to + j);
                _this = messages_in[_from][i]*node_erg[i]*edge_erg[_l_id];
                if(_this > _max)
                    _max = _this;
            }
            messages[m_id][j]  = _max;
            /* Add to the total messages. */
            _total_message    += _max;
        }

        /* Normalise messages to avoid overflow.
           Also, update the messages_in into _to. */
        for(int j = 0; j < _nl_to; j ++)
        {
            messages[m_id][j]   /= _total_message;
            messages_in[_to][j] *= messages[m_id][j];
        }

        /* If the remaining degree for the node _to is 0, 
           it is the root. We need not go any further - we can break. */
        if(_degrees[_to] == 0)
            break;

        /* However, if the remaining degree of _to is 1, we
           must add it to the queue. */
        if(_degrees[_to] == 1)
        {
            next_nodes.push(_to);
        }
    }           /* End of 1st pass of messages - from the leaves to the root. */

    /* Now we go back. */
    std::pair<int, int> _next_edge;

    while(!path.empty())
    {
        /* Retrieve the next edge. */
        _next_edge = path.top();
        /* Remove this from the stack. */
        path.pop();

        /* Get the _from and _to nodes. However, we reverse the order now, as
           we are going in the other direction. */
        _from = _next_edge.second;          // instead of .first
        _to   = _next_edge.first;           // instead of .second

        /* Number of labels for _from and _to. */
        int _nl_from = n_labels[_from];
        int _nl_to   = n_labels[_to];

        /* Message id for this edge. For now, it is the same as
           the edge ID. */
        int e_id = (*edge_id_from_node_ids)[std::make_pair(_from,_to)];
        /* We need messages into the node _to from all nodes but _from. 
           We use messages_in to accomplish this. messages_in[_to] now contains
           the product of all messages into _to. So, we 
           just divide the this message by the one from _from. */
        int tm_id = (_to < _from) ? e_id : (e_id + n_edges); 
        
        /* However, if _from > _to, we add n_edges to indicate that this 
           message is being sent in the other direction. */
        bool _transposed = false;
        int m_id         = e_id;
        if(_from > _to)
        {
            m_id        += n_edges;
            _transposed  = true;
        }

        /* Subtract the energies from the max to convert them into potentials. */
        for(int i = 0; i < _nl_from; i ++)
        {
            //node_erg[i] = _max_energy_this_slave - node_energies[_from][i];
            node_erg[i] = adj_node_probs[_from][i];
            assert(node_erg[i] > 0);
        }
        for(int j = 0; j < _nl_from*_nl_to; j ++)
        {
            //edge_erg[j] = _max_energy_this_slave - edge_energies[e_id][j];
            edge_erg[j] = adj_edge_probs[e_id][j];
            assert(edge_erg[j] > 0);
        }

        ValType _total_message = 0;
        /* Do the max-product update now. */
        for(int j = 0; j < _nl_to; j ++)
        {
            /* We must adjust the label ID according to whether we need
               to transpose the energies or not. */
            int _l_id = _transposed ? (j*_nl_from) : j;
            ValType _max = (messages_in[_from][0]/messages[tm_id][0])*node_erg[0]*edge_erg[_l_id];
            ValType _this;
            for(int i = 1; i < _nl_from; i ++)
            {
                /* Max-product. */
                _l_id = _transposed ? (j*_nl_from + i) : (i*_nl_to + j);
                _this = (messages_in[_from][i]/messages[tm_id][i])*node_erg[i]*edge_erg[_l_id];
                if(_this > _max)
                    _max = _this;
            }
            messages[m_id][j]  = _max;
            /* Add to the total messages. */
            _total_message    += _max;
        }

        /* Normalise messages to avoid overflow.
           Also, update the messages_in into _to. */
        for(int j = 0; j < _nl_to; j ++)
        {
            messages[m_id][j]   /= _total_message;
            messages_in[_to][j] *= messages[m_id][j];
        }
    }                   /* End of traversal from root to leaves. All messages computed now. */

    /* Messages have been computed. Assign labels. */
    int nl, _argmax;
    ValType _max, _this;
    for(int n_id = 0; n_id < n_nodes; n_id ++)
    {
        nl       = n_labels[n_id];
        //_max     = messages_in[n_id][0]*(_max_energy_this_slave - node_energies[n_id][0]);
        _max     = messages_in[n_id][0]*adj_node_probs[n_id][0];
        _argmax  = 0;

        for(int j = 1; j < nl; j ++)
        {
            //_this = messages_in[n_id][j]*(_max_energy_this_slave - node_energies[n_id][j]);
            _this = messages_in[n_id][j]*adj_node_probs[n_id][j];

            /* Find the label with maximum belief. */
            if(_this > _max)
            {
                _max    = _this;
                _argmax = j;
            }
        }
        /* Assign this node the "max" label. */
        labels[n_id] = _argmax;
    }

    /* Clean up */
        
    delete [] node_erg;
    delete [] edge_erg;

    return;
}

/* 
 * Slave::_brute_force_solver -- brute force solver for this slave. 
 * Simply tries all labellings and chooses the best one. 
 * Not be used other than for debugging. 
 */
std::vector<int> Slave::_brute_force_solver(void)
{
    std::vector< std::vector<int> > labellings = generate_label_permutations(n_nodes, n_labels);

    int     _best_labels = 0;
    ValType _best_energy = compute_energy_from_labels(labellings[_best_labels]);

    ValType _energy;

    for(unsigned int i = 1; i < labellings.size(); i ++)
    {
        _energy = compute_energy_from_labels(labellings[i]);
        if(_energy < _best_energy)
        {
            _best_energy = _energy;
            _best_labels = i;
        }
    }
    return labellings[_best_labels];
}

/*
 * Slave::compute_energy -- compute the Slave's energy for the current labelling. 
 */
void Slave::compute_energy(void)
{
    /* Compute the slave's energy for this labelling. */
    int _from, _to;
    int _lf, _lt;
    
    /* Reset energy to zero. */
    energy = 0.0;

    switch(_type)
    {
        case 'c': /* Iterate over all nodes. Since this is a cycle slave, we
                     can associate a cycle with a node, and hence compute 
                     the energy. */
                  for(int i = 0; i < n_nodes; i ++)
                  {
                      _from = i;
                      _to   = (i+1)%n_nodes;
                      _lf   = labels[_from];
                      _lt   = labels[_to];

                      /* Add the node and edge energies. */
                      energy += node_energies[_from][labels[i]];
                      energy += edge_energies[_from][_lt*n_labels[_from] + _lf];
                  }
                  break;

        case 't':  /* Iterate over all nodes and edges to compute the total
                      slave energy for this labelling. */
                  for(int i = 0; i < n_nodes; i ++)
                  {
                      /* Add node energies. */
                      energy += node_energies[i][labels[i]];
                  }
                  for(int e = 0; e < n_edges; e ++)
                  {
                      _from   = (*node_ids_from_edge_id)[e].first;
                      _to     = (*node_ids_from_edge_id)[e].second;
                      _lf     = labels[_from];
                      _lt     = labels[_to];
                      
                      /* Add edge energies. */
                      energy += edge_energies[e][_lf*n_labels[_to] + _lt];
                  }
    }
}

/*
 * Slave::compute_energy_from_labels -- compute the Slave's energy for a given labelling.
 */
ValType Slave::compute_energy_from_labels(std::vector<int> _labels)
{
    /* Compute the slave's energy for this labelling. */
    int _from, _to;
    int _lf, _lt;
    
    /* Reset energy to zero. */
    ValType _energy = 0.0;

    switch(_type)
    {
        case 'c': /* Iterate over all nodes. Since this is a cycle slave, we
                     can associate a cycle with a node, and hence compute 
                     the energy. */
                  for(int i = 0; i < n_nodes; i ++)
                  {
                      _from = i;
                      _to   = (i+1)%n_nodes;
                      _lf   = _labels[_from];
                      _lt   = _labels[_to];

                      /* Add the node and edge energies. */
                      _energy += node_energies[_from][_labels[i]];
                      _energy += edge_energies[_from][_lt*n_labels[_from] + _lf];
                  }
                  break;

        case 't':  /* Iterate over all nodes and edges to compute the total
                      slave energy for this labelling. */
                  for(int i = 0; i < n_nodes; i ++)
                  {
                      /* Add node energies. */
                      _energy += node_energies[i][_labels[i]];
                  }
                  for(int e = 0; e < n_edges; e ++)
                  {
                      _from   = (*node_ids_from_edge_id)[e].first;
                      _to     = (*node_ids_from_edge_id)[e].second;
                      _lf     = _labels[_from];
                      _lt     = _labels[_to];
                      
                      /* Add edge energies. */
                      _energy += edge_energies[e][_lf*n_labels[_to] + _lt];
                  }
    }

    return _energy;
}


/* 
 * Slave::set_labels -- set the Slave's labels to the specified ones. 
 */
bool Slave::set_labels(int *)
{
    return true;
}

/* 
 * Slave::node_map: Retrieve the node ID in Slave for a given ID in Graph.
 */
int Slave::node_map(int n_id)
{
    return _node_map[n_id];
}

/* 
 * Slave::edge_map: Retrieve the edge ID in Slave for a given ID in Graph.
 */
int Slave::edge_map(int e_id)
{
    return _edge_map[e_id];
}

/* 
 * Slave::get_node_label: Return the label assigned to a node by the Slave.
 */
int Slave::get_node_label(int n_id)
{
    return labels[_node_map[n_id]];
}

/*
 * Slave::get_edge_labels: Return the labelling of an edge by this Slave. 
 */
std::pair<int, int> Slave::get_edge_labels(int e_id)
{
    int _eid_in_s = _edge_map[e_id];
    std::pair<int, int> edge_labels; 

    /* If the type of slave is a cycle, we do not need to look up the
       edge ends in a map, but can simply retrieve them from node_list
       as we know the ID of the edge in the slave. */
    if(_type == 'c')
    {
        int ee0, ee1;
        ee0   = _eid_in_s;
        ee1   = (_eid_in_s+1)%n_nodes;
        return std::make_pair(ee0, ee1);
    }
    return edge_labels; 
}

/*
 * Slave::type: Return the type of slave.
 */
char Slave::type(void)
{
    return _type;
}
