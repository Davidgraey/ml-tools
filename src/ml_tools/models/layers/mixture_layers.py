from ml_tools.models.layers import Layer

class VotingWeight(Layer):
    def __init__(self, input_shape: int, hidden_size: int, num_experts: int):
        """
        Voting Weight (0-1)  values via linear activation

        our input is one time-domain, one frequency domain array (stacked)
        shape = (num_samples, 2, hidden_space)
        our intake layer will be (2*hidden_space, ...), and we'll have to reshape the array as it comes in.

        Parameters
        ----------
        input_shape : the shape of the embed stage output. (data that the gate is voting on)
        num_experts : the number of "experts" that we'll have as the output.
            This is softmaxed and those probs are used as a weight.
            SUM(softmax(vote_output) * experts_output) (one for each)

        """
        self.activation = "linear"
        self.input_shape = input_shape
        self.num_experts = num_experts

        # self.weights = (input_shape, hidden_size, num_experts)
        self.fc_1 = 1 / num_experts
        self.fc_2 = 0

        def forward(self, input_data):
            # input_data = (num_samples, 2, hidden_space)
            f_x = self.weights * input_data

            # output = self.num_experts
            # probs = softmax(output)

            # return the probs from forward pass -> perform the weighting and sum in the full network class


class VotingGate(Layer):
    #  determines the weighting of each "expert" in the final output
    #
    def __init__(self, input_shape: int, hidden_size: int, num_experts: int):
        """
        Voting Gate for Fourier Net --

        our input is one time-domain, one frequency domain array (stacked)
        shape = (num_samples, 2, hidden_space)
        our intake layer will be (2*hidden_space, ...), and we'll have to reshape the array as it comes in.

        Parameters
        ----------
        input_shape : the shape of the embed stage output. (data that the gate is voting on)
        num_experts : the number of "experts" that we'll have as the output.
            This is softmaxed and those probs are used as a weight.
            SUM(softmax(vote_output) * experts_output) (one for each)

        """
        self.activation = "softmax"
        self.input_shape = input_shape
        # number of fc layers?
        # any connections?
        self.num_experts = num_experts

        # self.input_layer = (2*input_shape, hidden_size)
        # hiddens

        def forward(self, input_data):
            # input_data = (num_samples, 2, hidden_space)
            # output = self.num_experts
            # probs = softmax(output)

            # return the probs from forward pass -> perform the weighting and sum in the full network class

            pass
