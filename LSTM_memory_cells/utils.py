import keras.backend as K
import matplotlib.pyplot as plt


def run_lstm_one_step(cell, inputs, state):
    # refer to Keras's "recurrent.py -> LSTMCell.call"
    h_tm1 = state[0]  
    c_tm1 = state[1]  

    inputs_i = inputs
    inputs_f = inputs
    inputs_c = inputs
    inputs_o = inputs

    x_i = K.dot(inputs_i, cell.kernel_i)
    x_f = K.dot(inputs_f, cell.kernel_f)
    x_c = K.dot(inputs_c, cell.kernel_c)
    x_o = K.dot(inputs_o, cell.kernel_o)

    x_i = K.bias_add(x_i, cell.bias_i)
    x_f = K.bias_add(x_f, cell.bias_f)
    x_c = K.bias_add(x_c, cell.bias_c)
    x_o = K.bias_add(x_o, cell.bias_o)

    h_tm1_i = h_tm1
    h_tm1_f = h_tm1
    h_tm1_c = h_tm1
    h_tm1_o = h_tm1

    i = cell.recurrent_activation(x_i + K.dot(h_tm1_i,
                                              cell.recurrent_kernel_i))
    f = cell.recurrent_activation(x_f + K.dot(h_tm1_f,
                                              cell.recurrent_kernel_f))
    c = f * c_tm1 + i * cell.activation(x_c + K.dot(h_tm1_c,
                                                    cell.recurrent_kernel_c))
    o = cell.recurrent_activation(x_o + K.dot(h_tm1_o,
                                              cell.recurrent_kernel_o))

    h = o * cell.activation(c)

    return i, f, c, o, h


def run_lstm(cell, inputs, init_state, return_all=False):
    # inputs: (time, input_dim). One sample only.
    n_steps, input_dim = inputs.shape    
    state = init_state
    step_results = []
    for t in range(n_steps):  
        current_input = K.reshape(inputs[t, :], (1, input_dim))        
        ig, fg, cc, og, ho = run_lstm_one_step(cell, current_input, state) 
        if return_all:
            step_results.append([ig, fg, cc, og, ho])
        state = [ho, cc]
    if return_all:
        return step_results
    else:
        return state


def plot_text_with_color(strength, text, n_col = 30):
    n_row = len(text) // n_col
    strength = strength.reshape((n_row, n_col))
    _, ax = plt.subplots(figsize=(n_row, n_col))
    ax.imshow(strength)
    for i in range(n_row):
        for j in range(n_col):
            _ = ax.text(j, i, text[i*n_col + j], ha="center", va="center", color="w")
    plt.axis('off')
    plt.show()
    

def save_plot_text_with_color(strength, text, n_col = 30, file=''):
    n_row = len(text) // n_col
    strength = strength.reshape((n_row, n_col))
    fig, ax = plt.subplots(figsize=(n_row, n_col))
    ax.imshow(strength)
    for i in range(n_row):
        for j in range(n_col):
            _ = ax.text(j, i, text[i*n_col + j], ha="center", va="center", color="w")
    plt.axis('off')
    plt.savefig(file)
    plt.cla()
    plt.close(fig)
