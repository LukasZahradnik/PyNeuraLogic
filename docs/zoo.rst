🦓 Module Zoo
==============

Welcome to our module zoo, the place where we discuss all pre-defined modules and outline how they are mapped to logic programs.

All modules listed here are defined in the :code:`neuralogic.nn.module` package, and their usage is quite similar to the usage
of regular rules. You can add them to your template via the :code:`+=` operator or :code:`add_module` method, e.g.:

.. code-block:: Python

    from neuralogic.nn.module import GCNConv

    template += GCNConv(...)
    # or
    template.add_module(GCNConv(...))

Right after adding a module into a template, it is expanded into logic form - rules. This allows you to build upon
pre-defined modules and create new variations by adding your own custom rules or just mixing modules together.


Pre-defined Modules
*******************

.. tabs::

    .. tab:: GNN

        .. list-table::
            :widths: 10 90
            :header-rows: 1

            * - Module
              - Edge formats

            * - :class:`~neuralogic.nn.module.gnn.gcn.GCNConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.gsage.SAGEConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.gin.GINConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.rgcn.RGCNConv`
              - :code:`R.<edge_name>(<source>, <relation>, <target>)` or :code:`R.<relation>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.tag.TAGConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.gatv2.GATv2Conv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.sg.SGConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.appnp.APPNPConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.res_gated.ResGatedGraphConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.gine.GINEConv`
              - :code:`R.<edge_name>(<source>, <target>)`
            * - :class:`~neuralogic.nn.module.gnn.gen.GENConv`
              - :code:`R.<edge_name>(<source>, <target>)`

    .. tab:: General Blocks

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Module
              -

            * - :class:`~neuralogic.nn.module.general.linear.Linear`
              -
            * - :class:`~neuralogic.nn.module.general.mlp.MLP`
              -

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Transformer module
              -

            * - :class:`~neuralogic.nn.module.general.transformer.Transformer`
              -
            * - :class:`~neuralogic.nn.module.general.transformer.TransformerEncoder`
              -
            * - :class:`~neuralogic.nn.module.general.transformer.TransformerDecoder`
              -

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Recurrent/Recursive module
              -

            * - :class:`~neuralogic.nn.module.general.rvnn.RvNN`
              -
            * - :class:`~neuralogic.nn.module.general.rnn.RNN`
              -
            * - :class:`~neuralogic.nn.module.general.gru.GRU`
              -
            * - :class:`~neuralogic.nn.module.general.lstm.LSTM`
              -

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Attention module
              -

            * - :class:`~neuralogic.nn.module.general.attention.Attention`
              -
            * - :class:`~neuralogic.nn.module.general.attention.MultiheadAttention`
              -

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Pooling module
              -

            * - :class:`~neuralogic.nn.module.general.pooling.Pooling`
              -
            * - :class:`~neuralogic.nn.module.general.pooling.SumPooling`
              -
            * - :class:`~neuralogic.nn.module.general.pooling.AvgPooling`
              -
            * - :class:`~neuralogic.nn.module.general.pooling.MaxPooling`
              -

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Encoding module
              -

            * - :class:`~neuralogic.nn.module.general.positional_encoding.PositionalEncoding`
              -

    .. tab:: Meta

        .. list-table::
            :widths: 99 1
            :header-rows: 1

            * - Module
              -

            * - :class:`~neuralogic.nn.module.meta.meta.MetaConv`
              -
            * - :class:`~neuralogic.nn.module.meta.magnn.MAGNNMean`
              -
            * - :class:`~neuralogic.nn.module.meta.magnn.MAGNNLinear`
              -

----

GNN Modules
***********

.. autoclass:: neuralogic.nn.module.gnn.gcn::GCNConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.gsage::SAGEConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.gin::GINConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.rgcn::RGCNConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.tag::TAGConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.gatv2::GATv2Conv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.sg::SGConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.appnp::APPNPConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.res_gated::ResGatedGraphConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.gine::GINEConv
   :members:

.. autoclass:: neuralogic.nn.module.gnn.gen::GENConv
   :members:

-----

General Block Modules
*********************

.. autoclass:: neuralogic.nn.module.general.linear::Linear
   :members:

.. autoclass:: neuralogic.nn.module.general.mlp::MLP
   :members:

.. autoclass:: neuralogic.nn.module.general.transformer::Transformer
   :members:

.. autoclass:: neuralogic.nn.module.general.transformer::TransformerEncoder
   :members:

.. autoclass:: neuralogic.nn.module.general.transformer::TransformerDecoder
   :members:

.. autoclass:: neuralogic.nn.module.general.rvnn::RvNN
   :members:

.. autoclass:: neuralogic.nn.module.general.rnn::RNN
   :members:

.. autoclass:: neuralogic.nn.module.general.gru::GRU
   :members:

.. autoclass:: neuralogic.nn.module.general.lstm::LSTM
   :members:

.. autoclass:: neuralogic.nn.module.general.attention::Attention
   :members:

.. autoclass:: neuralogic.nn.module.general.attention::MultiheadAttention
   :members:

.. autoclass:: neuralogic.nn.module.general.pooling::Pooling
   :members:

.. autoclass:: neuralogic.nn.module.general.pooling::SumPooling
   :members:

.. autoclass:: neuralogic.nn.module.general.pooling::AvgPooling
   :members:

.. autoclass:: neuralogic.nn.module.general.pooling::MaxPooling
   :members:

.. autoclass:: neuralogic.nn.module.general.positional_encoding::PositionalEncoding
   :members:

-----

Meta Modules
*********************

.. autoclass:: neuralogic.nn.module.meta.meta::MetaConv
   :members:

.. autoclass:: neuralogic.nn.module.meta.magnn::MAGNNMean
   :members:

.. autoclass:: neuralogic.nn.module.meta.magnn::MAGNNLinear
   :members:
