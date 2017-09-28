import numpy as np
import tensorflow as tf
import pickle
from tensorflow.models.rnn.ptb import reader

chars_size = None

class GraphInput(object):
    cell_size = None 
    num_classes = None
    batch_size = None
    num_steps = None
    num_layers = None
    learning_rate = None
    embed_size = None
    
    def __init__(self, cell_size, num_classes, batch_size, num_steps, num_layers, learning_rate, embed_size):
        self.cell_size = cell_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.embed_size = embed_size

def create_graph_input(cell_size, num_classes, batch_size, num_steps, num_layers = 3, learning_rate = 1e-4, embed_size = 50):
    graph_input = GraphInput(cell_size, num_classes, batch_size, num_steps, num_layers, learning_rate, embed_size)
    return graph_input

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def adjust_data(file_name):
    
    global chars_size

    # read in data
    with open(file_name,'r') as f:
        raw_data = f.read()

    chars = set(raw_data) 
    chars_size = len(chars)

    # use dictionary to convert characters into integer, and vice versa
    int_to_chars = dict(enumerate(chars))
    chars_to_int = dict(zip(int_to_chars.values(), int_to_chars.keys()))

    # save dictionary for later reference
    save_obj(int_to_chars, "int_to_chars")
    save_obj(chars_to_int, "chars_to_int")

    data = [chars_to_int[x] for x in raw_data] 

    return data

def build_multiple_graphs(graph_object_array):
    
    graphs = []

    for graph_object in graph_object_array:
        if 'sess' in globals() and sess:
            sess.close()
        tf.reset_default_graph()

        # extract variables
        cell_size = graph_object.cell_size
        num_classes = graph_object.num_classes
        batch_size = graph_object.batch_size
        num_steps = graph_object.num_steps
        num_layers = graph_object.num_layers
        learning_rate = graph_object.learning_rate
        embed_size = graph_object.embed_size

        # tensor dimension batch_size X num_steps ==> batch_size # of rows, num_steps # cols
        inpt = tf.placeholder(tf.int32, [batch_size, num_steps])
        output = tf.placeholder(tf.int32, [batch_size, num_steps])

        # embedding dimension num_classes X cell_size
        #E = tf.Variable(tf.random_uniform([num_classes,embed_size]
        E = tf.get_variable('embedding', [num_classes, embed_size])
        embed_inputs = tf.nn.embedding_lookup(E, inpt)

        # create cell
        gru_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(cell_size)] * num_layers)

        init_state = gru_cell.zero_state(batch_size, tf.float32)
        h, final_state = tf.nn.dynamic_rnn(gru_cell, embed_inputs, initial_state=init_state)
        
        # softmax weights 
        with tf.variable_scope('softmax'):
            # W = tf.Variable(tf.truncated_normal([cell_size, num_classes], stddev=0.1))
            # b = tf.Variable(tf.constant(0.1, shape=[num_classes]))
            W = tf.get_variable('W', [cell_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))


        #reshape rnn_outputs and y
        h = tf.reshape(h, [-1, cell_size])
        output_hot_vector = tf.reshape(output, [-1])

        # logits and softmax
        logits = tf.matmul(h, W) + b
        prob_distribution = tf.nn.softmax(logits)

        # train
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, output_hot_vector))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # use dictionary to access variables easily
        graph_dictionary = {
            "input": inpt, 
            "output": output, 
            "init_state": init_state, 
            "final_state": final_state, 
            "loss": loss,
            "train_step": train_step, 
            "distribution": prob_distribution, 
            "saver": tf.train.Saver(),
            "sess": tf.Session()
        }

        graphs.append(graph_dictionary)

    return graphs


def train(train_graph, epochs, save_path):
    with train_graph["sess"] as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in epochs:
            first_run = True
            training_state = None

            for INPUT, OUTPUT in epoch:
                feed_dict = {train_graph['input']: INPUT, train_graph['output']: OUTPUT}

                if not first_run:
                    feed_dict[train_graph['init_state']] = training_state
                    first_run = False

                run_array = [train_graph['loss'], train_graph['final_state'], train_graph['train_step']]
                training_loss, training_state, train_step = sess.run(run_array, feed_dict)

            print epoch, training_loss    

        # save state
        train_graph["saver"].save(sess, save_path)

def generate(chargen_graph, checkpoint, num_chars):
    # arbitrarily pick first character
    first_char = "J"
    top_values = 5
    first_run = True

    with chargen_graph["sess"] as sess:
        sess.run(tf.initialize_all_variables())
        # bring back dictionary and weights
        chars_to_int, int_to_chars = restore_state(sess, checkpoint)

        current_char = chars_to_int[first_char]
        chars = [current_char]

        pred_state_array = [chargen_graph['distribution'], chargen_graph['final_state']]

        for i in range(num_chars):
            feed_dict = {chargen_graph['input']: [[current_char]]} if first_run else {chargen_graph['input']: [[current_char]], chargen_graph['init_state']: final_state}
            first_run = False
            distribution, final_state = sess.run(pred_state_array, feed_dict)
            
            # Choose top chars and append
            new_char = top_chars(top_values, distribution)
            chars.append(new_char)

    text = map(lambda x: int_to_chars[x], chars)
    return("".join(text))

def restore_state(sess, checkpoint):
    tf.train.Saver().restore(sess, checkpoint)

    # load dictionaries back
    chars_to_int = load_obj("chars_to_int")
    int_to_chars = load_obj("int_to_chars")

    return chars_to_int, int_to_chars

def top_chars(top_values, distribution):
    
    p = np.reshape(distribution,(-1))
    p[np.argsort(p)[0:len(p)-top_values]] = 0
    norm_p = p / np.sum(p)
    new_char = np.random.choice(chars_size, 1, p=norm_p)[0]

    p = np.squeeze(distribution)
    p[np.argsort(p)[:-top_values]] = 0
    norm_p = p / np.sum(p)
    new_char = np.random.choice(chars_size, 1, p=norm_p)[0]

    return new_char

if __name__ == "__main__":
  
    file_name = 'tinyshakespeare.txt'
    save_path = "saves/Shakespeare_epoch10_state100.ckpt"

    data = adjust_data(file_name)
    num_classes = chars_size

    # generate epochs
    epochs = []
    num_epoch = 5
    for i in range(num_epoch):
        epochs.append(reader.ptb_iterator(data, batch_size = 50, num_steps = 80)) 

    # create graphs
    train_graph_object = create_graph_input(num_steps = 80, cell_size = 256, batch_size = 50, num_classes = num_classes)
    chargen_graph_object = create_graph_input(num_steps = 1, cell_size = 256, batch_size = 1, num_classes = num_classes)
    train_graph, chargen_graph = build_multiple_graphs([train_graph_object, chargen_graph_object])
    #chargen_graph = build_multiple_graphs([chargen_graph_object])[0]

    train(train_graph, epochs, save_path)
    returned_text = generate(chargen_graph, save_path, 2000)

    print(returned_text)











