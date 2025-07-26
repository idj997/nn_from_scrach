# Neural Network Fundamentals from Scratch

A comprehensive collection of Jupyter notebooks demonstrating the core concepts of neural networks built from scratch using Python and NumPy. This repository provides a step-by-step learning path from basic neuron operations to multi-layer networks.

## 📋 Table of Contents
- [Overview](#overview)
- [Learning Path](#learning-path)
- [Notebooks Description](#notebooks-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Key Concepts Covered](#key-concepts-covered)
- [Mathematical Foundation](#mathematical-foundation)
- [Next Steps](#next-steps)
- [Contributing](#contributing)

## 🚀 Overview

This repository contains educational materials for understanding neural networks from the ground up. Instead of using high-level frameworks like TensorFlow or PyTorch, these notebooks implement neural network components using basic Python and NumPy operations to provide deep insights into how neural networks actually work.

**Perfect for:**
- Students learning machine learning fundamentals
- Developers wanting to understand neural networks at a low level
- Anyone preparing for ML interviews or academic coursework
- Practitioners seeking to implement custom neural network components

## 📚 Learning Path

Follow the notebooks in this recommended order:

1. **intro_neuron_code.ipynb** - Start here to understand single neurons
2. **coding_a_layer.ipynb** - Learn how multiple neurons form layers
3. **calculating_losses.ipynb** - Understand error calculation and optimization
4. **multidimensional_inputs_and_multiple_layers.ipynb** - Build complete networks

## 📖 Notebooks Description

### 1. intro_neuron_code.ipynb
**🧠 Single Neuron Fundamentals**

Learn the basic building block of neural networks:
- How a single neuron processes inputs
- Weighted sum calculation: `output = (input₁ × weight₁) + (input₂ × weight₂) + ... + bias`
- Understanding weights and bias parameters
- Simple forward propagation

**Key Learning:**
```python
# Basic neuron operation
inputs = [1.3, 6.7, 2.9]
weights = [4.4, 5.9, 3.3]
bias = 3
output = sum(i * w for i, w in zip(inputs, weights)) + bias
```

### 2. coding_a_layer.ipynb
**🔗 Neural Network Layers**

Expand from single neurons to complete layers:
- Multiple neurons working together
- Layer-wise computation patterns
- Understanding how outputs from one layer become inputs to the next
- Implementing multiple neurons with different weights and biases

**Key Learning:**
- Each neuron in a layer has its own set of weights and bias
- All neurons in a layer process the same inputs simultaneously
- Layer output is a vector of individual neuron outputs

### 3. calculating_losses.ipynb
**📊 Loss Calculation and Optimization**

Introduction to training neural networks:
- **Manual Implementation**: Understanding loops and basic operations
- **NumPy Implementation**: Efficient vectorized operations using `np.dot()`
- **Loss Calculation**: Computing prediction errors
- **Comparison**: Manual loops vs. NumPy vectorization

**Key Learning:**
```python
# Efficient computation using NumPy
np_output = np.dot(weights, inputs) + bias
losses = actual_values - np_output
```

### 4. multidimensional_inputs_and_multiple_layers.ipynb
**🏗️ Complete Neural Networks**

Build full neural networks with multiple layers:
- **Batch Processing**: Handle multiple samples simultaneously
- **Matrix Operations**: Understanding shapes and dimensions
- **Multiple Hidden Layers**: Connecting layers in sequence
- **Object-Oriented Design**: Creating reusable layer classes
- **Weight Initialization**: Best practices for starting weights

**Key Features:**
- Custom `layer_creation` class for modular design
- Proper weight initialization using small random values
- Understanding input/output dimensions between layers
- Scalable architecture (easily add more layers/neurons)

## 🛠 Requirements

### Software Dependencies
- **Python**: 3.7+
- **NumPy**: 1.19+
- **Jupyter Notebook**: 6.0+ or Jupyter Lab

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd neural-network-fundamentals
```

2. **Install dependencies**
```bash
pip install numpy jupyter
```

3. **Launch Jupyter**
```bash
jupyter notebook
# or
jupyter lab
```

## 🚀 Usage

### Running the Notebooks

1. **Start with the basics**:
   ```bash
   jupyter notebook intro_neuron_code.ipynb
   ```

2. **Progress through each notebook**:
   - Run cells sequentially (Shift + Enter)
   - Experiment with different input values
   - Modify weights and biases to see effects

3. **Hands-on Learning**:
   - Change input values and observe outputs
   - Modify network architecture (neurons, layers)
   - Experiment with different weight initialization strategies

### Example Usage

```python
# Create a simple 2-layer network
import numpy as np

# Define input data (3 samples, 4 features each)
inputs = [[1.3,6.7,2.9,3.2],
          [6.2,9.7,1.9,4.2],
          [3.3,5.2,1.3,6.6]]

# Create layers
layer1 = layer_creation(4, 10)    # 4 inputs → 10 neurons
layer2 = layer_creation(10, 3)    # 10 inputs → 3 outputs

# Forward propagation
output1 = layer1.forward(inputs)
final_output = layer2.forward(output1)
```

## 🔑 Key Concepts Covered

### Fundamental Operations
- **Weighted Sum**: Core neuron computation
- **Bias Addition**: Shifting activation threshold
- **Matrix Multiplication**: Efficient batch processing
- **Forward Propagation**: Data flow through network

### Neural Network Architecture
- **Input Layer**: Data entry point
- **Hidden Layers**: Feature extraction and transformation
- **Output Layer**: Final predictions
- **Layer Connectivity**: How layers connect and communicate

### Implementation Details
- **Weight Initialization**: Starting with small random values
- **Batch Processing**: Handling multiple samples efficiently
- **Shape Management**: Understanding tensor dimensions
- **Object-Oriented Design**: Clean, reusable code structure

### Optimization Concepts
- **Loss Calculation**: Measuring prediction accuracy
- **Error Analysis**: Understanding prediction differences
- **Performance**: NumPy vectorization vs. manual loops

## 📐 Mathematical Foundation

### Single Neuron Output
```
output = Σ(inputᵢ × weightᵢ) + bias
```

### Layer Output (Matrix Form)
```
Output = Input × Weights + Bias
```

### Loss Calculation
```
Loss = Actual_Values - Predicted_Values
```

### Matrix Dimensions
- Input: `(batch_size, input_features)`
- Weights: `(input_features, output_neurons)`  
- Output: `(batch_size, output_neurons)`

## 🎯 Learning Outcomes

After completing these notebooks, you will understand:

1. **How neural networks process information** at the fundamental level
2. **Why matrix operations are essential** for efficient computation
3. **How to implement neural networks** without high-level frameworks
4. **The importance of proper weight initialization** and network architecture
5. **How to structure code** for scalable neural network implementations

## 🚀 Next Steps

### Immediate Extensions
1. **Add Activation Functions**: Implement ReLU, Sigmoid, Tanh
2. **Implement Backpropagation**: Add learning capabilities
3. **Add Regularization**: Prevent overfitting
4. **Create Training Loops**: Full learning implementation

### Advanced Topics
- Gradient descent optimization
- Different loss functions (MSE, Cross-entropy)
- Batch normalization
- Dropout for regularization
- Convolutional and recurrent layers

### Recommended Learning Path
1. Complete these fundamentals
2. Study activation functions and backpropagation
3. Implement a complete training algorithm
4. Explore specialized architectures (CNNs, RNNs)
5. Learn modern frameworks (TensorFlow, PyTorch)

## 🤝 Contributing

We welcome contributions to improve these educational materials!

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes (add examples, fix typos, improve explanations)
4. Commit changes (`git commit -m 'Add helpful example'`)
5. Push to branch (`git push origin feature/improvement`)
6. Create Pull Request

### Contribution Ideas
- Add more detailed explanations
- Include visualization code
- Create additional practice exercises
- Improve code comments and documentation
- Add error handling examples

## 📝 Educational Notes

### Important Concepts Highlighted

1. **Weight Initialization**: Small values (0.01-0.1) prevent exploding gradients
2. **Bias Importance**: Non-zero bias prevents "dead" networks
3. **Matrix Shapes**: Critical for successful layer connections
4. **Vectorization**: NumPy operations are much faster than Python loops

### Common Pitfalls Addressed

- **Shape Mismatches**: Understanding when to transpose matrices
- **Dead Networks**: Why proper initialization matters
- **Scaling**: How large weights can cause numerical instability

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create GitHub issues for bugs or questions
- **Discussions**: Use GitHub Discussions for general questions
- **Documentation**: All notebooks include detailed comments and explanations

---

**Happy Learning!** 🎓 These notebooks provide the foundation for understanding how modern deep learning frameworks work under the hood.
