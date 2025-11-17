# Capstoone Week 2
# XOR Neural Network – Multiple Implementations  
**Instruction:** *Just run all cells at once.*

This notebook explores four different implementations of a simple neural network for solving the  **XOR classification problem**, progressing from fully manual NumPy code to increasingly automated PyTorch workflows.

---

## Overview of the Four Implementations

### **1. NumPy Implementation (Fully Manual)**
A from-scratch neural network with manual:
- Forward pass  
- Backpropagation  
- Gradient descent parameter updates  

**Notes:**
- Initially encountered “dead ReLUs,” as predicted in *Deep Learning Basics with PyTorch*.  
- Improved stability using:
  - He/Xavier initialization  
  - `leaky_relu` activation  
- Training was slow without batching, so batching was added (with help from AI).  
- Using learning rate = **0.2**, batch averaging, He initialization, and leaky ReLU, the network learns XOR successfully.

---

### **2. PyTorch Manual Loop (Autograd + Manual Forward Pass)**
A semi-manual approach:
- Forward computation still written manually  
- PyTorch’s autograd handles gradient computation  
- Parameter updates performed explicitly  

**Results:**
- With 16 hidden units: 0.99 accuracy**  
- With only 2 hidden units: 0.82 accuracy**

---

### **3. PyTorch `nn.Module` (Custom Model Class)**
A more idiomatic PyTorch solution:
- Model defined using a subclass of `nn.Module`  
- Uses `nn.Linear` layers  
- Forward pass defined cleanly inside the model class  
- Training loop still manual, but using PyTorch optimizers  

This approach works well and is structurally closer to standard PyTorch neural network implementations.

---

### **4. PyTorch `nn.Sequential` (Fully Automated Model Definition)**
The highest-level PyTorch approach:
- Model constructed declaratively using `nn.Sequential`  
- Automatically handles forward computation based on layer order  
- Same training loop as Section 3  
 
`nn.Sequential` trains quickly and performs well, especially during the first 100 epochs.



---
