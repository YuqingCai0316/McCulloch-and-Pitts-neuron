class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        """
        Initialize the neuron with given weights and threshold.

        :param weights: List of weights for the inputs.
        :param threshold: The activation threshold for the neuron.
        """
        self.weights = weights
        self.threshold = threshold

    def activation_function(self, weighted_sum):
        """
        The activation function: Step function that mimics a binary threshold.

        :param weighted_sum: The sum of the weighted inputs.
        :return: 1 if the sum is greater than or equal to the threshold, otherwise 0.
        """
        return 1 if weighted_sum >= self.threshold else 0

    def compute_output(self, inputs):
        """
        Compute the output of the neuron for the given inputs.

        :param inputs: List of binary input signals (0 or 1).
        :return: Binary output of the neuron (0 or 1).
        """
        if len(inputs) != len(self.weights):
            raise ValueError("Number of inputs must match the number of weights.")

        # Calculate the weighted sum of inputs
        weighted_sum = sum(input_signal * weight for input_signal, weight in zip(inputs, self.weights))
        print('weighted_sum',weighted_sum)
        # Apply the activation function
        return self.activation_function(weighted_sum)


# Example Usage
if __name__ == "__main__":
    # Define weights and threshold
    weights = [1, 1, 1]  # Weights for each input
    threshold = 2  # Threshold for activation

    # Create a neuron
    neuron = McCullochPittsNeuron(weights, threshold)

    # Test with some inputs
    inputs_list = [
        [0, 0, 0],  # Example 1: All inputs 0
        [0, 1, 1],  # Example 2: Two inputs 1
        [1, 1, 1],  # Example 3: All inputs 1
        [1, 0, 1],  # Example 4: Mixed inputs
    ]

    # Simulate neuron output for each set of inputs
    for inputs in inputs_list:
        output = neuron.compute_output(inputs)
        print(f"Inputs: {inputs} -> Output: {output}")
