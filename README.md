# Neural Network Library in JavaScript

This project implements a simple neural network library in JavaScript, including matrix operations and feed-forward neural network functionality.

## Features

- Matrix operations (addition, multiplication, transpose, etc.)
- Feed-forward neural network with customizable input, hidden, and output layers
- Training functionality with backpropagation

## Installation

To use this library in your project, you can clone this repository:

```bash
git clone https://github.com/your-username/nn-library-in-js.git
```

Then, include the necessary files in your HTML:

```html
<script src="path/to/matrix.js"></script>
<script src="path/to/neuralnetwork.js"></script>
```

## Usage

Here's a basic example of how to use the neural network:

```javascript
// Create a neural network with 2 input nodes, 3 hidden nodes, and 1 output node
let nn = new NeuralNetwork(2, 3, 1);

// Train the network
let trainingData = [
  { inputs: [0, 0], targets: [0] },
  { inputs: [0, 1], targets: [1] },
  { inputs: [1, 0], targets: [1] },
  { inputs: [1, 1], targets: [0] }
];

for (let i = 0; i < 10000; i++) {
  let data = random(trainingData);
  nn.train(data.inputs, data.targets);
}

// Use the trained network
let output = nn.feedforward([1, 0]);
console.log(output);
```

## API Reference

### Matrix Class

- `constructor(rows, cols)`: Create a new matrix
- `static transpose(matrix)`: Transpose a matrix
- `static fromArray(arr)`: Create a matrix from an array
- `toArray()`: Convert matrix to array
- `static multiply(a, b)`: Multiply two matrices
- `multiply(n)`: Scalar multiplication or element-wise multiplication with another matrix
- `static subtract(a, b)`: Subtract two matrices
- `add(n)`: Add a scalar or another matrix
- `randomise()`: Fill the matrix with random values
- `map(func)`: Apply a function to every element of the matrix
- `static map(matrix, func)`: Apply a function to every element of a matrix
- `print()`: Print the matrix to the console

### NeuralNetwork Class

- `constructor(input_nodes, hidden_nodes, output_nodes)`: Create a new neural network
- `feedforward(input_arr)`: Feed an input through the network
- `train(input_arr, targets_arr)`: Train the network with a single input-target pair

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).

## Contact

If you have any questions or feedback, please open an issue on the GitHub repository.

---

Happy coding!
