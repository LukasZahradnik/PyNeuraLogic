import jpype


def to_java_list(in_list):
    out_list = jpype.java.util.ArrayList()
    for item in in_list:
        out_list.add(item)
    return out_list
