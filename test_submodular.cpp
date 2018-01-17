#include "Graph.hpp"
#include <ctime>

/* A simple program to minimise
   a 2-D graph with unary and 
   pairwise energies using a
   row-column decomposition. */

int main(void)
{
    srand(time(NULL));

    /* Grpah size. */
    int rows = 20;
    int cols = 20;
    int n_edges = 2*rows*cols - rows - cols;

    /* Number of labels per node. */
    int nl = 2;
    int *n_labels = new int [rows*cols];
    for(int i = 0; i < rows*cols; i ++)
    {
        n_labels[i] = nl;
    }

    /* Create a new Graph. */
    Graph *G = new Graph(rows*cols, n_edges, n_labels);

    /* Create space for unaries and pairwise energies. */
    ValType * unaries  = new ValType [nl];
    ValType * pairwise = (ValType *)calloc(nl*nl, sizeof(ValType));

    /* Initialise random unary energies. */
    for(int n_id = 0; n_id < rows*cols; n_id ++)
    {
        for(int i = 0; i < nl; i ++)
        {
            unaries[i] = randn(0.0, 1.0);
        }
        G->set_node_energies(n_id, unaries);
    }
   
    /* Initialise random submodular pairwise energies. */
    for(int r = 0; r < rows; r ++)
    {
        for(int c = 0; c < cols; c ++)
        {
            int n_id = r*cols + c;
            if(r < rows - 1)
            {
                // Add down edge. 
                double lpq = (double)rand()/(double)RAND_MAX;
                for(int i = 0; i < nl; i ++)
                {
                    pairwise[i*nl + i] = lpq;
                }
                G->set_edge_energies(n_id, n_id + cols, pairwise);
            }
            if(c < cols - 1)
            {
                // Add right edge. 
                double lpq = (double)rand()/(double)RAND_MAX;
                for(int i = 0; i < nl; i ++)
                {
                    pairwise[i*nl + i] = lpq;
                }
                G->set_edge_energies(n_id, n_id + 1, pairwise);
            }

        }
    }
  
    /* Add slaves. */
    for(int r = 0; r < rows; r ++)
    {
        /* Add row slaves. */
        std::vector<int> nodelist;
        std::vector<int> edgelist;

        for(int c = 0; c < cols - 1; c ++)
        {
            int n_id = r*cols + c;
            nodelist.push_back(n_id);
            edgelist.push_back(G->get_edge_id(n_id, n_id+1));
        }
        nodelist.push_back(r*cols + cols - 1);
        G->add_tree_slave(nodelist, edgelist);
    } 

    for(int c = 0; c < cols; c ++)
    {
        /* Add col slaves. */
        std::vector<int> nodelist;
        std::vector<int> edgelist;

        for(int r = 0; r < rows - 1; r ++)
        {
            int n_id = r*cols + c;
            nodelist.push_back(n_id);
            edgelist.push_back(G->get_edge_id(n_id, n_id+cols));
        }
        nodelist.push_back((rows-1)*cols + c);
        G->add_tree_slave(nodelist, edgelist);
    }

    /* Finalise graph decomposition. */
    G->finalise_decomposition();

    /* Print optimisation status every 100 iterations. */
    G->print_every(100);

    /* Optimise using the subgradient method. */
    G->optimise(0.8, 5000, OptimStrategy::STEP);

    delete [] unaries;
    free(pairwise);
    delete [] n_labels;

    delete G;

    return 0;
}

