
import tensorflow as tf

def load_graph(graph_filename, input_map=None, prev_graph=None):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the 
    # current default Graph
    if prev_graph is None:
        g = tf.Graph()
    else:
        g = prev_graph
    #TODO removing graph shape info
    with g.as_default() as graph:
        tf.import_graph_def(
            graph_def, 
            input_map=input_map, 
            return_elements=None, 
            name=''
        )
    return graph