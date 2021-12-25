import os

import jpype


os.environ["CLASSPATH"] = os.path.join(os.path.abspath(os.path.dirname(__file__)), "jar", "NeuraLogic.jar")
jpype.startJVM(classpath=[os.environ["CLASSPATH"]])

jpype.java.lang.System.setOut(jpype.java.io.PrintStream(jpype.java.io.File("/dev/null")))
jpype.java.lang.System.setErr(jpype.java.io.PrintStream(jpype.java.io.File("/dev/null")))
