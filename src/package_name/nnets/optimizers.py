import numpy as np
import loss_functions as loss


# ------------------------------------------------------------------------------------------------------------
# ----------------------------------CASCADE OPTIMIZERS------------------------------------------------------
def cascade_sgd_momentum(incoming_x, incoming_t, iterations, this_cascade, error_trace, learning_rate_eta=0.02,
                         momentum_alpha=0.96, batch_size=4):
    '''**********ARGUMENTS**********
    :param incoming_x: input values - incoming data should be standardized
    :param incoming_t: target values - targets should be standardized
    :param iterations: number of updates
    :param this_cascade: passes in cascade assembly object
    :param error_trace: error trace object from network object
    :param learning_rate_eta: learning rate value for nnet weight updates
    :param momentum_alpha: Momentum to roll down gradient
    :param batch_size: split batch as 1/nth of the full dataset at a time
    '''
    n_samples = incoming_x.shape[0]
    # n_outputs = incoming_t.shape[-1]

    # make indexing array, shuffle it and use it to index X and T to create batches
    selector = np.array(np.arange(0, n_samples))
    #batch_size = int((1 / batch_size) * n_samples)

    #velocity = this_cascade.momentum

    for i in range(0, iterations):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]
            # print(f'training on {batch_i} : {batch_i + batch_size}, {this_batch.shape}')
            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            output = this_cascade.forward(_x)

            error = loss.loss_dictionary[this_cascade.loss_func](output, _t)

            d_error = loss.derivative_dictionary[this_cascade.loss_func + '_derivative'](output, _t)
            #TODO: add L2 / weight decay: seperate method / function here -
            #TODO: L2 = costF + gamma / n_samples * sum (abs(weights))
            #add gamma/n * w to der of all weights - leave biases intact.
            delta = d_error / batch_size

            output, gradients, delta = this_cascade.compute_gradients(delta)

            # update V factors in this batch
            this_cascade.momentum = (momentum_alpha * this_cascade.momentum) - (learning_rate_eta * gradients)

            #Update weights
            this_cascade.update_weights(this_cascade.momentum)

            batch_error.append(error)

        error_trace.append(np.mean(batch_error))

# ------------------------------------------------------------------------------------------------------------
# ----------------------------------NETWORK OPTIMIZERS------------------------------------------------------

def network_sgd_momentum(incoming_x, incoming_t,
                         network_object, error_trace, iterations,
                         learning_rate_eta=0.02, momentum_alpha=0.5, batch_size=4):
    '''**********ARGUMENTS**********
    :param incoming_x: input values - incoming data should be standardized
    :param incoming_t: target values - targets should be standardized
    :param iterations: number of updates
    :param network_object: passes in as self from Adaptive_Cascade Object
    :param error_trace: error trace object from network object
    :param learning_rate_eta: learning rate value for nnet weight updates
    :param momentum_alpha: Momentum to roll down gradient
    :param batch_size: split batch as 1/nth of the full dataset at a time
    '''
    n_samples = incoming_x.shape[0]

    selector = np.array(np.arange(0, n_samples))

    for i in range(0, iterations):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            full_output = network_object.forward(_x)
            error = loss.loss_dictionary[network_object.network[-1].loss_func](full_output, _t)
            d_error = loss.derivative_dictionary[network_object.network[-1].loss_func + '_derivative'](full_output, _t)
            delta = d_error / batch_size

            for c_i, this_cascade in enumerate(reversed(network_object.network)):

                _out, gradients, delta = this_cascade.compute_gradients(delta)

                this_cascade.momentum = -(momentum_alpha * this_cascade.momentum) - (learning_rate_eta * gradients)
                this_cascade.update_weights(this_cascade.momentum)
                delta = d_error / batch_size

            batch_error.append(error)

        error_trace.append(np.mean(batch_error))