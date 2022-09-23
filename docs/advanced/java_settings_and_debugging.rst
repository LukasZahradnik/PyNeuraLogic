Java Settings, Logging and Debugging
====================================

PyNeuraLogic, at its core, utilizes procedures (such as grounding) running on a Java Virtual Machine (JVM). JVM itself
offers plentiful options to set, such as memory limitations, garbage collectors settings, and more.

This section will go through interfaces that allow you to pass your own JVM settings.
We will also look into JVM logging and JVM debugging.

JVM Settings
************

.. important::

    Customizing JVM settings and JVM path is applicable only before a JVM is started. If you want to do some
    customizations, do them before working with PyNeuraLogic (building model/building samples, etc.)

By default, PyNeuraLogic uses JVM found on your ``PATH``. If you want to use a different JVM, you can do that by calling
the ``neuralogic.set_jvm_path`` function, such as:

.. code:: python

    import neuralogic

    neuralogic.set_jvm_path("/some/path/my_jvm/")

You can also make some adjustments to JVM settings via the ``neuralogic.set_jvm_options`` function.
By default, one option is passed into the JVM - ``"-Xms1g"``, which sets the minimum amount of heap memory size
to 1 GB. The maximum amount of the heap memory size can be set via the ``neuralogic.set_max_memory_size`` function.

This function overrides already set options, so if you want to keep defaults or previously set options,you will have
to specify them again. For example, you can inspect the garbage collector with customizing settings such as:

.. code:: python

    import neuralogic

    neuralogic.set_jvm_options(["-Xms1g", "-Xmx64g", "-XX:+PrintGCDetails"])


Java Logging
************

Looking into the Java logs can be valuable practice to get better insight into what is going on in the background.
It offers a lot of information about all steps, such as the grounding process. This info is also practical
when asking for help in discussion/issues.

You can add a logging handler anytime you want with any level by calling the ``add_handler`` function.
The first argument can be any object that implements a ``write(message: str)``
method (e.g., file handlers, ``sys.stdout``, etc.).

.. code:: python

    import sys
    from neuralogic.logging import add_handler, Formatter, Level

    add_handler(sys.stdout, Level.FINE, Formatter.COLOR)

If you decide you no longer want to subscribe to loggers, you can remove all logging handlers by calling
the ``clear_handlers`` function.

.. code:: python

    from neuralogic.logging import clear_handlers

    clear_handlers()


Java Debugging
**************

.. important::

    To run PyNeuraLogic in debug mode, you have to run the debug mode before a JVM is started -
    therefore, run the debug mode before working with PyNeuraLogic (building model/building samples, etc.)

There is a prepared interface to run the JVM in the debug mode, which allows to attach a remote debugger on the JVM and
then use breakpoints on the `NeuraLogic <https://github.com/GustikS/NeuraLogic>`_ project, as usual. You can enable
the debug mode by calling the ``neuralogic.initialize`` function with the argument ``debug_mode=True``.

.. code:: python

    import neuralogic

    neuralogic.initialize(debug_mode=True)

.. code:: console

    >>> Listening for transport dt_socket at address: 12999

Once you get the message above, the execution of the python program will wait (by default) for you to connect your
remote debugger to the port (by default, *12999*). Via other arguments of the initialize function,
it is possible to specify further things like debugging port, etc.

Once the remote debugger is attached, the execution of the Python program will continue until the execution hits a breakpoint.
