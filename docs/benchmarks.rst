Benchmarks
==========


.. |pyg| raw:: html

   <a href="https://github.com/pyg-team/pytorch_geometric" target="_blank">PyTorch Geometric (PyG)</a>

.. |dgl| raw:: html

   <a href="https://github.com/dmlc/dgl" target="_blank">Deep Graph Library (DGL)</a>

.. |spektral| raw:: html

   <a href="https://github.com/danielegrattarola/spektral" target="_blank">Spektral</a>

.. |pygloader| raw:: html

   <a href="https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html?highlight=TUDataset#torch_geometric.datasets.TUDataset" target="_blank">Dataset loader</a>

.. |spektralloader| raw:: html

   <a href="https://graphneural.network/datasets/#tudataset" target="_blank">Dataset loader</a>

.. |tudataset| raw:: html

   <a href="https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets" target="_blank">TUDataset Benchmark Data Sets</a>



Here we compare the speed of some popular GNN models encoded in PyNeuraLogic against some of the most popular GNN frameworks in their latest versions, namely |pyg| (2.0.2), |dgl| (0.6.1), and |spektral| (1.0.6).

The benchmarks report comparison of the average training time per epoch of three different architectures
- GCN (two GCNConv layers), GraphSAGE (two GraphSAGEConv layers), and GIN (five GINConv layers).

Datasets are picked from the common |tudataset| and are loaded into PyNeuraLogic, DGL, and PyG via PyG's |pygloader|.
Spektral benchmark uses Spektral's |spektralloader|.

We compare the frameworks in a binary graph classification task with only node's features. This is merely for the sake of 
simple reusability of the introduced architectures over the frameworks. Statistics of each dataset can be seen down below.

Due to its declarative nature, PyNeuraLogic has to transform each dataset into a logic form and then into a computation graph.
The time spent on this preprocessing task is labeled as "Dataset Build Time". Note that this transformation happens only once before
the training.

.. tabs::

    .. tab:: MUTAG

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/MUTAG.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.1238s     |0.1547s     |0.2491s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.1287s     |0.1795s     |0.5214s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.0897s     |0.1099s     |0.3399s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0083s** |**0.0119s** |**0.0393s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |1.4265s     |1.9372s     |2.3662s     |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 188                | ~17.9              | ~19.7              | 7                    |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: NCI1

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/NCI1.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |3.0152s     |3.1773s     |5.1924s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |3.1044s     |4.3426s     |11.3512s    |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |1.9226s     |2.6211s     |7.0598s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.2396s** |**0.3461s** |**1.5037s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |24.8405s    |25.2125s    |57.4115s    |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 4110               | ~29.8              | ~32.3              | 37                   |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: PROTEINS

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/PROTEINS.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.7221s     |1.0153s     |1.4591s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.7859s     |1.1963s     |3.1576s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.5047s     |0.6455s     |1.9786s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0741s** |**0.1111s** |**0.5524s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |9.9873s     |10.0125s    |24.2591s    |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 1113               | ~39.0              | ~72.8              | 3                    |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: BZR

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/BZR.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.2730s     |0.3238s     |0.5144s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.3035s     |0.4288s     |1.1171s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.1847s     |0.2464s     |0.7232s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0293s** |**0.0469s** |**0.1552s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |3.8219s     |3.9852s     |7.0831s     |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 405                | ~35.7              | ~38.3              | 53                   |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: COX2

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/COX2.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.3411s     |0.3705s     |0.5975s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.3513s     |0.5124s     |1.2988s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.2082s     |0.2857s     |0.8086s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0321s** |**0.0505s** |**0.1754s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |4.2805s     |4.5738s     |8.6356s     |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 467                | ~41.2              | ~43.4              | 35                   |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: DHFR

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/DHFR.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.5578s     |0.6058s     |0.9708s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.6063s     |0.8010s     |2.1136s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.3388s     |0.4588s     |1.3178s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0572s** |**0.0879s** |**0.3168s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |7.3361s     |7.3635s     |15.0887s    |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 467                | ~42.4              | ~44.5              | 53                   |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: KKI

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/KKI.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.0565s     |0.0797s     |0.1200s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.0611s     |0.0887s     |0.2292s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.0370s     |0.0535s     |0.1480s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0262s** |**0.0321s** |**0.0529s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |1.7563s     |2.0459s     |2.6008s     |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 83                 | ~26.9              | ~48.4              | 190                  |
        +--------------------+--------------------+--------------------+----------------------+


    .. tab:: Peking_1

        .. raw:: html

            <h4>Average Time Per Epoch</h4>

        .. image:: _static/benchmarks/Peking_1.svg
            :alt: Average Time Per Epoch
            :align: center

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |Spektral            |0.0597s     |0.0851s     |0.1244s     |
        +--------------------+------------+------------+------------+
        |Deep Graph Library  |0.0654s     |0.0923s     |0.2335s     |
        +--------------------+------------+------------+------------+
        |PyTorch Geometric   |0.0404s     |0.0608s     |0.1547s     |
        +--------------------+------------+------------+------------+
        |**PyNeuraLogic**    |**0.0371s** |**0.0469s** |**0.0778s** |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Build Time</h4>

        +--------------------+------------+------------+------------+
        |                    |GCN         |GraphSAGE   |GIN         |
        +====================+============+============+============+
        |PyNeuraLogic        |2.3414s     |2.2352s     |3.3951s     |
        +--------------------+------------+------------+------------+

        .. raw:: html

            <h4>Dataset Statistics</h4>

        +--------------------+--------------------+--------------------+----------------------+
        | Num. of Graphs     | Avg. num. of nodes | Avg. num. of edges | Num. node of features|
        +====================+====================+====================+======================+
        | 85                 | ~39.3              | ~77.3              | 190                  |
        +--------------------+--------------------+--------------------+----------------------+
