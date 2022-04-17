🦓 Model Zoo
==============

Welcome to our model zoo, the place where we discuss all pre-defined modules and outline how they are mapped to logic programs.

All modules listed here are defined in the :code:`neuralogic.nn.module` package, and their usage is quite similar to the usage
of regular rules. You can add them to your template via the :code:`+=` operator or :code:`add_module` method, e.g.:

.. code-block:: Python

    from neuralogic.nn.module import GCNConv

    template += GCNConv(...)
    # or
    template.add_module(GCNConv(...))

Right after adding a module into a template, it is expanded into logic form - rules. This allows you to build upon
pre-defined modules and create new variations by adding your own custom rules or just mixing modules together.


Pre-defined GNN Modules
***********************

.. list-table::
    :widths: 10 90
    :header-rows: 1

    * - Name
      - Edge formats

    * - :class:`~neuralogic.nn.module.gcn.GCNConv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.gsage.SAGEConv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.gin.GINConv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.rgcn.RGCNConv`
      - :code:`R.<edge_name>(<source>, <relation>, <target>)` or :code:`R.<relation>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.tag.TAGConv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.gatv2.GATv2Conv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.sg.SGConv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.appnp.APPNPConv`
      - :code:`R.<edge_name>(<source>, <target>)`
    * - :class:`~neuralogic.nn.module.res_gated.ResGatedGraphConv`
      - :code:`R.<edge_name>(<source>, <target>)`

.. autoclass:: neuralogic.nn.module.gcn::GCNConv
   :members:

.. autoclass:: neuralogic.nn.module.gsage::SAGEConv
   :members:

.. autoclass:: neuralogic.nn.module.gin::GINConv
   :members:

.. autoclass:: neuralogic.nn.module.rgcn::RGCNConv
   :members:

.. autoclass:: neuralogic.nn.module.tag::TAGConv
   :members:

.. autoclass:: neuralogic.nn.module.gatv2::GATv2Conv
   :members:

.. autoclass:: neuralogic.nn.module.sg::SGConv
   :members:

.. autoclass:: neuralogic.nn.module.appnp::APPNPConv
   :members:

.. autoclass:: neuralogic.nn.module.res_gated::ResGatedGraphConv
   :members: