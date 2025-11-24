import numpy as np
import pickle
import matplotlib.pyplot as plt
import time
import copy

import activation_functions as func
import optimizers

import layers


EPSILON = 1e-15

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

def decay_learning_rate(lr, final_lr = .001):
    decay = .75
    return max(lr * decay, final_lr)


def update_mean_std(mean1, std1, count1, mean2, std2, count2):
    '''**********ARGUMENTS**********
    :param mean1: self.data_type_means
    :param std1: self.data_type_stds
    :param count1: num_before
    :param mean2: this_data_mean
    :param std2: this_data_std
    :param count2: num_new
    **********RETURNS**********
    :return: new mean value, new std value
    '''
    # from www.burtonsys.com/climate/composite_sd.php#python
    countBoth = count1 + count2
    meanBoth = (count1 * mean1 + count2 * mean2) / countBoth
    var1 = std1 ** 2
    var2 = std2 ** 2
    # error sum of squares
    ESS = var1 * (count1 - 1) + var2 * (count2 - 1)
    # total group sum of squares
    TGSS = (mean1 - meanBoth) ** 2 * count1 + (mean2 - meanBoth) ** 2 * count2
    varBoth = (ESS + TGSS) / (countBoth - 1)
    stdBoth = np.sqrt(varBoth)

    return meanBoth, stdBoth


def standardize(X, means, stds):
    '''**********ARGUMENTS**********
    :param X: data to be standardized
    :param means: mean of sample and previous samples gotten from update_mean_std
    :param stds: std of sample and previous samples gotten from update_mean_std
    '**********RETURNS**********
    :return: standardized data
    '''
    return (X - means) / (stds + EPSILON)


def unstandardize(X, means, stds):
    '''**********ARGUMENTS**********
    :param X: data to be returned to normal space
    :param means: mean of sample and previous samples gotten from update_mean_std
    :param stds: std of sample and previous samples gotten from update_mean_std
    '**********RETURNS**********
    :return: unstandardized data
    '''
    return (stds - EPSILON) * X + means


def iterable(item):
    '''
    checks if item is iterable - if not makes it a list of one item.
    used for creating the hidden layer system of the neural network obj
    **********ARGUMENTS**********
    :param item: obj to test for iterability
    **********RETURNS**********
    :return: returns an iterable version of the item object
    '''
    try:
        iter(item)
    except TypeError:
        item = [item]
    return item


def load_network(filename):
    with open(filename, 'rb') as file:
        p_d, n_d = pickle.load(file)
    print(f'loaded network object from {filename}')
    model = p_d['network_class'](p_d['standardization'], p_d['problem_type'])
    model.cascade_error_trace = p_d['c_errors']
    model.network_error = p_d['n_errors']
    model.T_means = p_d['t_means']
    model.T_stds = p_d['t_stds']
    model.X_means = p_d['x_means']
    model.X_stds = p_d['x_stds']

    for c_i,  c_d in enumerate(n_d):
        model.assembly_loader(c_d)

        #cascade created, now set layer weights;
        #for w_i, w in enumerate(model.network[c_i].weights):
        for l_i, l in enumerate(model.network[c_i].layers):
            l.weights = c_d['layer_weights'][l_i]

    return model


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------

class CascadeAssembly:
    def __init__(self, ni, no, nhs, hidden_activation, output_activation, position):
        '''**********ARGUMENTS**********
        :param ni: number of inputs
        :param no: number of outputs
        :param nhs: structure of hidden units
        :param hidden_activation: string, passed in as 'linear', 'relu', 'sigmoid', 'tanh'
        :param output_activation: string, passed in as 'linear',
        :param position: position of cascade assembly - simple int
        '''
        self.ni = ni
        self.nhs = iterable(nhs)
        self.no = no


        self.assembly_id = position
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.layers = []

        self.value_output = [0]

        self.momentum = [0]
        self.mt = [0]
        self.vt = [0]

    def compute_gradients(self, delta):
        '''**********ARGUMENTS**********
        :param delta: delta computed at previous output (delta = - error / (n_samples * n_outputs)
        **********RETURNS**********
        :return: returns list of outputs of cascade assembly, list of gradients (one per layer), and new delta val
        '''
        for li in reversed(range(len(self.layers))):
            _grad = self.layers[li].build_gradient(delta)
            delta = self.layers[li].back_pass_gradient(delta)
        return self.outputs, self.gradients, delta


    def add_layer(self, ni, no, activation, is_output):
        '''**********ARGUMENTS**********
        :param ni: int, inputs shape
        :param no: int, number of outputs
        :param activation: string activation type
        :param is_output: boolean flag for output designation
        '''
        self.layers.append(layers.Layer(ni, no, activation, is_output))


    def zero_momentum(self):
        '''Resets momentum, MT and VT to zeros for new training data.
        '''
        self.momentum = np.zeros_like(self.weights)
        self.mt = np.zeros_like(self.weights)
        self.vt = np.zeros_like(self.weights)


    def forward(self, incoming_x):
        '''**********ARGUMENTS**********
        :param incoming_x: input values
        **********RETURNS**********
        :return: returns output of forward pass of all layers in this cascade assembly
        '''
        outs = incoming_x
        for l in self.layers:
            outs = l.forward(outs)
        self.value_output = outs
        return outs


    def update_weights(self, values):
        '''**********ARGUMENTS**********
        :param values: values created by optimizer using gradients, amount by which to change layer weight values
        '''
        for layer, gradient in zip(self.layers, values):
            #print(f'layer is {layer.shape}, grad is {gradient.shape}')
            try:  #try inserting a new axis if shapes don't match
                layer.update_weight(gradient[..., np.newaxis])
            except ValueError:
                layer.update_weight(gradient)
            except TypeError:
                layer.update_weight(gradient)


    # -------------------------------------------------------------------------------
    # ------------------------------UTILITY------------------------------------------
    def purge(self):
        '''
        Purges each layer of values, to be called when saving
        '''
        for l in self.layers:
            l.purge()


    # -------------------------------------------------------------------------------
    # ------------------------------PROPERTIES---------------------------------------
    @property
    def gradients(self):
        g = []
        for l in self.layers:
            g.append(l.gradient)
        try:
            g = np.squeeze(g)
        except ValueError:
            pass
        return g


    @property
    def outputs(self):
        o = []
        for l in self.layers:
            o.append(l.output)
        try:
            o = np.squeeze(o)
        except ValueError:
            try:
                o = np.array(o)
            except ValueError:
                pass
        return o


    @property
    def value_outputs(self):
        '''
        Value Outputs is a straight linear output from each cascade, rather than a softmax or sigmoid or other
        funciton when doing classification - these value outputs should be summed from all cascades, then have the
        activation applied to get the full network's classification.
        :param:
        ---
        :return:
        '''
        return self.value_output


    @property
    def inputs(self):
        i = []
        for l in self.layers:
            i.append(l.input)
        try:
            i = np.squeeze(i)
        except ValueError:
            pass
        return i


    @property
    def shapes(self):
        s = []
        for l in self.layers:
            s.append(l.shape)
        try:
            s = np.squeeze(s)
        except ValueError:
            pass
        return s


    @property
    def weights(self):
        w = []
        for l in self.layers:
            w.append(l.weights)
        try:
            w = np.squeeze(w)
        except ValueError:
            pass
        return w


    def __str__(self):
        return f'Cascade Assembly {self.assembly_id} of {len(self.layers)} {self.hidden_activation}, shaped {self.layers}'


class RegressionAssembly(CascadeAssembly):
    def __init__(self, ni, no, nhs, hidden_activation, output_activation, position):
        '''**********ARGUMENTS**********
        :param ni: number of inputs
        :param no: number of outputs
        :param nhs: structure of hidden units
        :param hidden_activation: string, passed in as 'linear', 'relu', 'sigmoid', 'tanh'
        :param output_activation: string, passed in as 'linear',
        :param position: position of cascade assembly - simple int
        '''

        super().__init__(ni, no, nhs, hidden_activation, output_activation, position)


        self.loss_func = 'mse'


        #Build Neural Network for this Assembly----
        if hidden_activation == 'linear':
            self.add_layer(ni, 1, hidden_activation, is_output=False)
            ni = 1
            self.add_layer(ni, no, output_activation, is_output=True)
        else:
            for l_i in np.arange(0, len(self.nhs)):
                self.add_layer(ni, self.nhs[l_i], hidden_activation, is_output=False)
                ni = self.nhs[l_i]
            self.add_layer(ni, no, output_activation, is_output=True)

        self.zero_momentum()


# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------



class AdaptiveCascade():
    def __init__(self, standardize_type=('x', 't'), problem_type = 'regression', error_threshold=0.15,
                 max_length=8, max_depth=8):
        '''
        Creates a neural network using linear and sigmoid hidden layers configured into cascade assemblies -
        these will each output an individual function of hidden / output layers that are then passed to the next
        assembly, and the T values are adjusted from the outputs. Then a full pass stage is done as if it is a
        standard hidden layer neural nework once the designated length of layers is reached.
        **********ARGUMENTS**********
        :param standardize_type: ('x', 't'), ('x'), ('t') which data to standardize
        :param problem_type: 'regression', 'classification', or 'multiclassification' for multi-label.
        :param error_threshold: CURRENTLY NONFUNCTIONAL
        :param max_length: CURRENTLY NONFUNCTIONAL
        :param max_depth: CURRENTLY NONFUNCTIONAL
        :param output_type: strings of activation function for output layers
        '''
        self.verbose = True
        self.assembly_position = 0

        self.trained = False
        self.training_now = False
        self.samples_trained = 0

        self.standardize_type = standardize_type
        self.problem_type = problem_type

        self.error_threshold = error_threshold
        self.max_length = max_length
        self.max_depth = max_depth

        self.network = []

        # gradient and STD traces
        self.cascade_error_trace = [1.0]
        self.network_error_trace = [1.0, 1.0]
        self.T_means = 0
        self.T_stds = 0
        self.X_means = 0
        self.X_stds = 0

        self.momentum = [0]
        self.mt_vt = [[0], [0]]


    # -------------------------------------------------------------------------------
    # ---------------------Creation Functions----------------------------------------

    def add_fc_assembly(self, ni, no, hidden_units, activation_type):
        '''**********ARGUMENTS**********
        :param ni: int for shape of inputs
        :param no: int for number of output values
        :param hidden_units:
        :param assembly_position: int corresponding with the cascade's position
        :param activation_type: activation for hidden layer
        '''

        self.ni = ni
        self.no = no

        self.network.append(RegressionAssembly(ni,
                                               no,
                                               hidden_units,
                                               activation_type,
                                               'linear',
                                               self.assembly_position))

        self.assembly_position += 1

        if self.verbose:
            print(f'')

    # ----------------------------------------------------------------------------
    # ---------------------Network Functions----------------------------------------
    def train(self, incoming_x, incoming_t, learning_rate=0.02, momentum=0.5, batch_size=4, iterations=10,
              epochs=2):
        '''**********ARGUMENTS**********
        :param incoming_x: input values
        :param incoming_t: target values
        :param learning_rate: learning rate eta - value between 0 and 1
        :param momentum: momentum alpha rate - value between 0 and 1
        :param batch_size: number of segments to split the input data into
        :param iterations: number of forward / backward updates
        :param epochs: number of passes through the X data to train, one cascade and one network pass per epoch
        **********RETURNS**********
        :return: list of moving targets as T values are updated
        '''
        self.training_now = True
        start_time = time.time()

        this_x = copy.copy(incoming_x)
        this_t = copy.copy(incoming_t)

        num_new = this_x.shape[0]
        num_before = self.samples_trained

        # -------VARIABLE STANDARDIZATION-----All Calculations Take Place in Standardized Space-----------
        if 'x' in self.standardize_type:
            this_x = self.standardize_dataset(this_x, num_before, num_new, data_type='inputs')

        if 't' in self.standardize_type:
            this_t = self.standardize_dataset(this_t, num_before, num_new, data_type='targets')
            org_t = copy.copy(this_t)
        else:
            org_t = copy.copy(this_t)
        target_trace = [0] * len(self.network)

        ep = 0
        while ep <= epochs:
            for cascade in self.network:
                cascade.zero_momentum()

            # ----------Cascade Assembly training-------------------------------------------
            for c_i, cascade in enumerate(self.network):
                cascade_output = self.train_assembly(incoming_x=this_x,
                                                     incoming_t=this_t,
                                                     cascade=cascade,
                                                     learning_rate=learning_rate,
                                                     momentum = momentum,
                                                     batch_size=batch_size,
                                                     iterations=iterations)

                #print(cascade_output.shape, this_t.shape)
                this_t = this_t - cascade_output
                target_trace[c_i] = this_t
                # ---end vertical cascade assembly training

            # ----------Full Network training-------------------------------------------

            self.train_network(incoming_x = this_x,
                               incoming_t = org_t,
                               learning_rate=learning_rate,
                               momentum = momentum,
                               batch_size=batch_size,
                               iterations=iterations)


            # TODO: Decay learning_rate
            if self.network_error_trace[-1] > self.network_error_trace[-2]:
                learning_rate = decay_learning_rate(learning_rate, final_lr = 0.0001)
                print(f'network error increasing from {self.network_error_trace[-1]}, to '
                      f'{self.network_error_trace[-2]}, decaying lr to {learning_rate}')

            ep += 1

        self.samples_trained += num_new
        self.trained = True
        self.training_now = False

        print(f'trained on {num_new} samples for {iterations} iterations, which took {time.time() - start_time} '
              f'seconds')

        return target_trace


    def train_assembly(self, incoming_x, incoming_t, cascade, learning_rate, momentum, batch_size, iterations):
        '''**********ARGUMENTS**********
        :param incoming_x: input values
        :param incoming_t: target values
        :param cascade: passes in cascade object
        :param learning_rate: learning rate eta - value between 0 and 1
        :param momentum: momentum alpha rate - value between 0 and 1
        :param batch_size: number of segments to split the input data into
        :param iterations: number of forward / backward updates
        **********RETURNS**********
        :return: outputs from the cascade assembly after training
        '''
        # -------USE BATCH SGD TO TRAIN THE NETWORK --------------------------------------

        print(f'training {cascade} with SGD_moment for {iterations}')
        optimizers.cascade_sgd_momentum(incoming_x = incoming_x,
                                        incoming_t = incoming_t,
                                        iterations = iterations,
                                        this_cascade = cascade,
                                        error_trace = self.cascade_error_trace,
                                        learning_rate_eta = learning_rate,
                                        momentum_alpha = momentum,
                                        batch_size = batch_size)


        return cascade.forward(incoming_x)


    def train_network(self, incoming_x, incoming_t, learning_rate, momentum, batch_size, iterations):
        '''**********ARGUMENTS**********
        :param incoming_x: input values
        :param incoming_t: target values
        :param learning_rate: learning rate eta - value between 0 and 1
        :param momentum: momentum alpha rate - value between 0 and 1
        :param batch_size: number of segments to split the input data into
        :param iterations: number of forward / backward updates
        '''

        # ----------------------Network Training----------------------------------------------
        optimizers.network_sgd_momentum(incoming_x=incoming_x,
                                        incoming_t=incoming_t,
                                        iterations=iterations,
                                        network_object=self,
                                        error_trace=self.network_error_trace,
                                        learning_rate_eta=learning_rate,
                                        momentum_alpha=momentum,
                                        batch_size=batch_size)


    # ----------------------------------------------------------------------------
    # ---------------------Forward and Use----------------------------------------


    def forward(self, incoming_x):
        '''**********ARGUMENTS**********
        Forward pass through each cascade assembly - includes no standardization or unstandardization
        :param incoming_x: input values - should already be standardized
        **********RETURNS**********
        :return: output of the full network's outputs - sum of each cascade's output
        '''

        this_x = copy.copy(incoming_x)
        output_by_cascade = []
        for cascade in self.network:
            _o = cascade.forward(this_x)
            outs = cascade.value_outputs
            output_by_cascade.append(outs)
            o_activ = cascade.output_activation
            # end cascade loop-----------

        full_output = np.sum(output_by_cascade, axis=0)
        full_output = func.activation_dictionary[o_activ](full_output)
        return full_output


    def use(self, incoming_x):
        '''**********ARGUMENTS**********
        :param incoming_x: input values - raw
        **********RETURNS**********
        :return: full_output, [output per cascade], [unstandardized output]
        '''
        # TODO: ADD Lock network weights and standardization?
        this_x = copy.copy(incoming_x)
        # -----VARIABLE STANDARDIZATION-------
        if 'x' in self.standardize_type:
            this_x = standardize(this_x, self.X_means, self.X_stds)

        output_by_cascade = []
        value_outputs = []

        for cascade in self.network:
            outs = cascade.forward(this_x)
            output_by_cascade.append(outs)
            value_outputs.append(cascade.value_outputs)
            o_activ = cascade.output_activation
        full_output = np.sum(value_outputs, axis=0)
        full_output = func.activation_dictionary[o_activ](full_output)

        if 't' in self.standardize_type:
            full_output = unstandardize(full_output, self.T_means, self.T_stds)

        return full_output, output_by_cascade


    # ------------------------------------------------------------------------
    # ---------------------Standardize----------------------------------------

    def standardize_dataset(self, data, num_before, num_new, data_type):
        '''**********ARGUMENTS**********
        :param data: incoming data, could be inputs or targets
        :param num_before: number of samples already trained. Int, should be passed in as self.num_trained
        :param num_new: Number of samples in current training.  Int, should be X.shape[0]
        :param data_type: category of input data - string, either 'targets', or 'inputs'
        **********RETURNS**********
        :return: returns standardized version of data
        '''

        this_mean = np.array(data.mean(0))

        if data.shape[0] == 1:
            this_std = np.ones(data.shape)
        else:
            this_std = np.array(data.std(0))
            this_std[this_std == 0] = 1.0

        if data_type == 'targets':
            if num_before == 0:
                self.T_means = this_mean
                self.T_stds = this_std
            else:
                self.T_means, self.T_stds = update_mean_std(self.T_means,
                                                                 self.T_stds,
                                                                 num_before,
                                                                 this_mean,
                                                                 this_std,
                                                                 num_new)

            std_data = standardize(data, self.T_means, self.T_stds)

        elif data_type == 'inputs':
            if num_before == 0:
                self.X_means = this_mean
                self.X_stds = this_std
            else:
                self.X_means, self.X_stds = update_mean_std(self.X_means,
                                                                 self.X_stds,
                                                                 num_before,
                                                                 this_mean,
                                                                 this_std,
                                                                 num_new)

            std_data = standardize(data, self.X_means, self.X_stds)

        else:
            raise TypeError('Data_type must be either "inputs" or "targets"')

        return std_data


    # ------------------------------------------------------------------------
    # ---------------------Properties----------------------------------------
    @property
    def gradients(self):
        g = []
        for cas in self.network:
            g.append(cas.gradients)
        try:
            g = np.squeeze(g)
        except ValueError:
            g = np.array(g)
        return g


    @property
    def outputs(self):
        o = []
        for cas in self.network:
            o.append(cas.outputs)
        try:
            o = np.squeeze(o)
        except ValueError:
            o = np.array(o)
        return o


    @property
    def inputs(self):
        i = []
        for cas in self.network:
            i.append(cas.inputs)
        try:
            i = np.squeeze(i)
        except ValueError:
            i = np.array(i)
        return i


    @property
    def weights(self):
        w = []
        for cas in self.network:
            w.append(cas.weights)
        try:
            w = np.squeeze(w)
        except ValueError:
            w = np.array(w)
        return w

    def save(self, filename):
        f_map = {'AdaptiveCascade': AdaptiveCascade,
                 'RegressionAssembly': 'add_fc_assembly',
                 'ClassificationAssembly': 'add_fc_assembly',
                 'ConvolutionalRegressionAssembly': 'add_conv_assembly',
                 'ConvolutionalClassificationAssembly': 'self.add_conv_assembly'
                 }
        n_d = {'network_class': f_map[type(self).__name__],
               'standardization': self.standardize_type,
               'problem_type': self.problem_type,
               'c_errors': self.cascade_error_trace,
               'n_errors': self.network_error_trace,
               't_means': self.T_means,
               't_stds': self.T_stds,
               'x_means': self.X_means,
               'x_stds': self.X_stds}

        # each cascade is a dict, network is a list of dicts.
        c_d = []
        for c in self.network:
            # cascade called eg RegressionAssembly(ni, no, nhs, hidden_activation, output_activation, position)
            c_dict = {'cascade_class': f_map[type(c).__name__],
                      'ni': c.ni,
                      'no': c.no,
                      'nhs': c.nhs,
                      'output_activation': c.output_activation,
                      'hidden_activation': c.hidden_activation,
                      'position': c.assembly_id,
                      'layer_weights': c.weights}

            c_d.append(c_dict)

        packed_network_dict = (n_d, c_d)
        with open(filename, 'wb') as handle:
            pickle.dump(packed_network_dict, handle)
        handle.close()
        print(f'saved Adaptive Cascade Network of {len(self.network)} assemblies')

    def assembly_loader(self, assembly_dict):
        getattr(self, assembly_dict['cascade_class'])(assembly_dict['ni'],
                                                      assembly_dict['no'],
                                                      assembly_dict['nhs'],
                                                      assembly_dict['hidden_activation'])

        assert self.network[-1].output_activation == assembly_dict['output_activation']
        assert self.network[-1].assembly_id == assembly_dict['position']

# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

if __name__ == '__main__':
    '''
    create random data to test
    '''
    np.random.seed(20)
    X = np.arange(100).reshape((-1, 1))
    T = copy.copy(X)
    T[80:] = 80
    T[:20] = 20
    T = T * 2 * np.random.normal(1.0, .08, (T.shape))
    T1 = (np.tanh(X * 0.02) + 4) ** 3
    T2 = (np.sin(X * 0.025) + 4) ** 2
    T3 = X ** 1.33
    T = T1

    model = AdaptiveCascade(('x', 't'), problem_type = 'regression')

    model.add_fc_assembly(ni=X.shape[-1], no=T.shape[-1], hidden_units = [1], activation_type='linear')
    model.add_fc_assembly(ni=X.shape[-1], no=T.shape[-1], hidden_units = [3], activation_type='relu_leaky')

    target_trace = model.train(X, T, learning_rate=0.002, momentum=0.9, batch_size=50, iterations=20, epochs=0)
    target_trace = model.train(X, T, learning_rate=0.0002, momentum=0.9, batch_size=20, iterations=20, epochs=0)
    target_trace = model.train(X, T, learning_rate=0.00002, momentum=0.9, batch_size=2, iterations=20, epochs=0)
    #plotting.plot_training(model, X, T, target_trace)
    plt.plot(model.network_error_trace)
    plt.plot(model.cascade_error_trace)
    plt.show()
    plt.plot(model.use(X)[0], T)
    plt.show()