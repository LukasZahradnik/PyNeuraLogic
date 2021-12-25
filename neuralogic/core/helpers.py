import jpype


def to_java_list(in_list, java_type=jpype.java.util.ArrayList):
    out_list = java_type()
    for item in in_list:
        out_list.add(item)
    return out_list
