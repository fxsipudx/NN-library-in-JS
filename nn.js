function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
    return y * (1 - y);
}

class NeuralNetwork {
    constructor(input_nodes, hidden_nodes, output_nodes) {
        this.input_nodes = input_nodes;
        this.hidden_nodes = hidden_nodes;
        this.output_nodes = output_nodes;

        this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
        this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);

        this.weights_ih.randomise();
        this.weights_ho.randomise();

        this.bias_h = new Matrix(this.hidden_nodes, 1);
        this.bias_o = new Matrix(this.output_nodes, 1);

        this.bias_h.randomise();
        this.bias_o.randomise();

        this.learning_rate = 0.01;
    }

    feedforward(input_arr) {
        let inputs = Matrix.fromArray(input_arr);
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        hidden.map(sigmoid);

        let outputs = Matrix.multiply(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(sigmoid);

        return outputs.toArray();
    }

    train(input_arr, targets_arr) {
        // Feedforward
        let inputs = Matrix.fromArray(input_arr);
        let hidden = Matrix.multiply(this.weights_ih, inputs);
        hidden.add(this.bias_h);
        hidden.map(sigmoid);

        let outputs = Matrix.multiply(this.weights_ho, hidden);
        outputs.add(this.bias_o);
        outputs.map(sigmoid);

        // Convert targets to matrix
        let targets = Matrix.fromArray(targets_arr);

        // Calculate output errors (targets - outputs)
        let output_errors = Matrix.subtract(targets, outputs);

        // Calculate gradient
        let gradients = Matrix.map(outputs, dsigmoid);
        gradients.multiply(output_errors);
        gradients.multiply(this.learning_rate);

        // Calculate deltas for weights_ho
        let hidden_t = Matrix.transpose(hidden);
        let weights_ho_deltas = Matrix.multiply(gradients, hidden_t);

        // Update weights and biases for the hidden-to-output layer
        this.weights_ho.add(weights_ho_deltas);
        this.bias_o.add(gradients);

        // Calculate hidden layer errors
        let who_t = Matrix.transpose(this.weights_ho);
        let hidden_errors = Matrix.multiply(who_t, output_errors);

        // Calculate hidden gradients
        let hidden_gradient = Matrix.map(hidden, dsigmoid);
        hidden_gradient.multiply(hidden_errors);
        hidden_gradient.multiply(this.learning_rate);

        // Calculate deltas for weights_ih
        let inputs_t = Matrix.transpose(inputs);
        let weights_ih_deltas = Matrix.multiply(hidden_gradient, inputs_t);

        // Update weights and biases for the input-to-hidden layer
        this.weights_ih.add(weights_ih_deltas);
        this.bias_h.add(hidden_gradient);
    }
}