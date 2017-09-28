''' 
Sang Ha Park, Junsu Choi, Jongje Kim
CS2951K Final Project
Text Generation using mulit-RNN
'''

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

def create_graph_input(cell_size, num_classes, batch_size, num_steps, num_layers = 3, learning_rate = 5e-4, embed_size = 50):
    graph_input = GraphInput(cell_size, num_classes, batch_size, num_steps, num_layers, learning_rate, embed_size)
    return graph_input

def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_map_dicts(chars):
    int_to_chars = {}
    counter = 0

    for char in chars:
        int_to_chars[counter] = char
        counter += 1

    chars_to_int = {v: k for k, v in int_to_chars.items()}

    # save the dictionaries for future reference
    save_obj(int_to_chars, "int_to_chars")
    save_obj(chars_to_int, "chars_to_int")

    return int_to_chars, chars_to_int

def adjust_data(file_name):
    global chars_size

    with open(file_name,'r') as f:
        raw_text = f.read()

    chars = set(raw_text) 
    chars_size = len(chars)

    int_to_chars, chars_to_int = create_map_dicts(chars)
    inpt = [chars_to_int[char] for char in raw_text]

    return inpt


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

        # Placeholders
        inpt = tf.placeholder(tf.int32, [batch_size, num_steps])
        output = tf.placeholder(tf.int32, [batch_size, num_steps])

        # Embedding
        E = tf.get_variable('embedding', [num_classes, embed_size])
        embed_inputs = tf.nn.embedding_lookup(E, inpt)

        # multi-GRU cell
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(cell_size)] * num_layers)

        init_state = cell.zero_state(batch_size, tf.float32)
        rnn_outputs, final_state = tf.nn.dynamic_rnn(cell, embed_inputs, initial_state=init_state)

        with tf.variable_scope('softmax'):
            W = tf.get_variable('W', [cell_size, num_classes])
            b = tf.get_variable('b', [num_classes], initializer=tf.constant_initializer(0.0))

        
        rnn_outputs = tf.reshape(rnn_outputs, [-1, cell_size])
        output_reshaped = tf.reshape(output, [-1])

        # Run softmax, using logits
        logits = tf.matmul(rnn_outputs, W) + b
        distribution = tf.nn.softmax(logits)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, output_reshaped))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        graph_dictionary = {
            "input": inpt, 
            "output": output, 
            "init_state": init_state, 
            "final_state": final_state, 
            "loss": loss,
            "train_step": train_step, 
            "preds": distribution, 
            "saver": tf.train.Saver(),
            "sess": tf.Session()
        }

        graphs.append(graph_dictionary)

    return graphs

def train(train_graph, epochs, save):
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
        train_graph['saver'].save(sess, save)

def generate(chargen_graph, checkpoint, num_chars):
    first_char = "A"
    first_run = True
    top_values = 5

    with chargen_graph["sess"] as sess:
        sess.run(tf.initialize_all_variables())
        chars_to_int, int_to_chars = restore_state(sess, checkpoint)

        current_char = chars_to_int[first_char]
        chars = [current_char]

        pred_state_array = [chargen_graph['preds'], chargen_graph['final_state']]

        for i in range(num_chars):
            feed_dict = {chargen_graph['input']: [[current_char]]} if first_run else {chargen_graph['input']: [[current_char]], chargen_graph['init_state']: final_state}
            first_run = False
            distribution, final_state = sess.run(pred_state_array, feed_dict)
            
            # Choose top chars and append
            current_char = top_chars(top_values, distribution)
            chars.append(current_char)

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
    return new_char

if __name__ == "__main__":
    # process data
    file_name = 'tinyshakespeare.txt'
    save_path = "saves/Shakespeare_epoch10_state100.ckpt"

    data = adjust_data(file_name)
    num_classes = chars_size
    
    # generate epochs
    epochs = []
    num_epoch = 20
    for i in range(num_epoch):
        epochs.append(reader.ptb_iterator(data, batch_size = 50, num_steps = 80)) 

    # create graphs
    train_graph_object = create_graph_input(num_steps = 80, cell_size = 512, batch_size = 50, num_classes = num_classes)
    chargen_graph_object = create_graph_input(num_steps = 1, cell_size = 512, batch_size = 1, num_classes = num_classes)
    train_graph, chargen_graph = build_multiple_graphs([train_graph_object, chargen_graph_object])

    train(train_graph, epochs, save_path)
    returned_text = generate(chargen_graph, save_path, 2000)
    print (returned_text)











