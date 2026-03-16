	- **Neurons** -A neuron stores an activation value (a real number), sometimes normalized between 0 and 1.
 - for example 
	 - the 28 X 28 pixel of the input image has 784 neurons holding a value that corresponds to the gray scale value of the pixel
	 - if its 0 - its black
	 - if its 1 - its white
	 - the number inside a neuron is called activation
	 - ![[Pasted image 20260213130248.png]]
- these 784 neurons make up our first layer of the network
- ![[Pasted image 20260213130325.png]]
- the last layer has 10 neurons - that can be from 0 to 1 representing how much the system thinks the given image corresponds with a given digit
- the hidden layers are the ones in between
- the way a network operates is - activation in one layer determines the activation in the next layer

- Lets imagine the way neural networks work is - each layer's job is to recognize patterns of the input
- ![[Pasted image 20260213131524.png]]
- each number is broken down to straights, edges, loops etc.
- each layer takes in bigger and bigger patterns and the last layer is able to recognize the number
- this is how we can "hope" the NN works

## But how does this actually work?
- we need to design how activations in one layer might determine the activations in the next
- lets take an example
- say, one particular neuron in the second layer to pick up if the image has an edge in the white region
- ![[Pasted image 20260213132828.png]]
- so what parameters should the network have so that it can capture that pattern or any other pixel pattern like several edges making a loop
- ![[Pasted image 20260213133725.png]]
- we assign a "weight" to each of the connections between our neuron and the neurons in the next layer

### what are weights?

- not all inputs matter equally for a decision
- for example
	- if we want to recognize a 9 
	- the pixels on top are more important (as the form a loopy pattern) than the ones in the middle or at the bottom
	- so we assign positive weights at the top part telling the neuron to "care" about that region more
- for a single neuron 
	- $\text{z} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n$
	- $w$ is weight and $x$ is the input value
	- in this case $x$ is pixel brightness value
	- if we have a large positive weight - the input should be ON
	- if we have a large negative weight - the input should be OFF

- the weighted sum or $z$ we got, can be any number on the number line 
- so we use the sigmoid function to convert the $z$ to a value between 0 and 1 and get $activation$
- ![[Pasted image 20260213135146.png]]
- so the $activation$ measures how positive the weighed sum is but we don't always want the neuron to light up as soon as its bigger than 0, we want it light up after a certain threshold
- that's where bias comes in
- so $\text{activation} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n - \theta$
- $\theta$ is the threshold
- $-\theta$ = $b$
- $\text{activation} = w_1 x_1 + w_2 x_2 + \cdots + w_n x_n + b$
- this is just for one neuron, it repeats for all the connections from the first layer to the next with its own weights and biases
- ![[Pasted image 20260213140424.png]]
- when we say "Learning" - it means finding the right weights and biases 
- ![[Pasted image 20260213140706.png]]
## Meaning of $w_{k,n}$
- $w_{k,n}$ - Weight from previous-layer neuron $n$ to current-layer neuron $k$.

It tells how much neuron $n$ influences neuron $k$.
$$
z_k = \sum_{n} w_{k,n}\, x_n + b_k
$$
$$
a_k = \sigma(z_k)
$$
Where:
- $x_n$ = activation of previous neuron $n$
- $w_{k,n}$ = weight from $n \rightarrow k$
- $b_k$ = bias of neuron $k$
- $a_k$ = output of neuron $k$
- Weight matrix $W$
$$
W =
\begin{bmatrix}
w_{0,0} & w_{0,1} & \dots & w_{0,n} \\
w_{1,0} & w_{1,1} & \dots & w_{1,n} \\
\vdots  & \vdots  & \ddots & \vdots \\
w_{k,0} & w_{k,1} & \dots & w_{k,n}
\end{bmatrix}
$$
- Row $k$ = all weights into neuron $k$
- Column $n$ = all weights from neuron $n$
 - Layer equation
	 - $a^{(1)} = \sigma\!\left(W\,a^{(0)} + b\right)$
- $a^{(0)}$ = activations of previous layer  
- $W\,a^{(0)}$ = all weighted sums  
- $b$ = biases of current neurons  
- $a^{(1)}$ = activations of current layer  

- modern neural networks use ReLU function rather than sigmoid function

### ReLU activation function
- stands for rectified linear unit activation function
- ![[Pasted image 20260213194154.png]]
$\mathrm{ReLU}(x) = \max(0, x)$
Meaning:
- if $x > 0$ → output $x$
- if $x \le 0$ → output $0$
Purpose:
- keeps positive signals
- removes negative signals
- introduces non-linearity
In neuron
- input: $z_k$ (weighted sum)
- output: $a_k = \mathrm{ReLU}(z_k)$


## Cost functions

- at the beginning of training the network - all of our weights and biases are totally random and our network performs horribly 
- ex: ![[Pasted image 20260213194838.png]]
- so we apply a cost function
- cost function is basically the sum of the squares of differences of the *trash output* to the desired output
- ![[Pasted image 20260213195007.png]]
- the sum is small if the network performs good, and its big if it performs bad
- now we consider the **average cost of all the training examples**
- the bigger the average cost function - the worse the neural network

##### Now we need to tell the network how to change its weights and biases to minimize the cost function
- a neural network has a lot of inputs (13000 in the case of mnist dataset), but lets **consider just one input $C(w)$ - one input and one output**
- we need to find and input that minimizes the value of the function $c(w)$
- ![[Pasted image 20260213201729.png]]
	- we start at any input and figure out in what direction if we move we get to the minimum
	- so we get the slope of the function
		- shift to left if the slope is positive and to the right if its negative
		- do this repeatedly until slope is 0 
		- this is local minimum
		- this is just gradient descent


	- now imagine a cost function with **two inputs** - x-y plane as the input space and the cost function is on it
	- ![[Pasted image 20260217174750.png]]
	- here we need to find the direction in this **input space** in which we get to decrease the output of the function most quickly 
	- the gradient of this functions gives you the direction of steepest ascent - so you take the negative of this gradient
	- we repeat this over and over till we get to the minimum cost 

#### We apply the same idea for our 13000 input neural network - instead of two

- the algorithm to compute the gradient efficiently is called **backpropagation**
- it can be represented as a vector
- ![[Pasted image 20260217175432.png]]
- if the gradient is big - it carries more weight (more bang for your buck) - or which change matters the most 


#### How back propagation works?

- lets take an example - an image of a "2"
- The network isn't trained yet - so the output looks random compared to the desired output
- ![[Pasted image 20260224174318.png]]
- we cant directly change the output layer's activations, but we have influences on the weights and biases
- ![[Pasted image 20260224174440.png]]
- we need the 3rd value to go up and the rest of the values to be nudged down
- the sizes of these nudges should be propertional to how far away the current value is to the target value
- example - the increase for 2 is more important than for 8 - which is already less
- to increase the activation for 2 - we have 3 ways
- ![[Pasted image 20260224175241.png]]
- but we cant directly change the activations of the previous layer - we only control the weights and biases
- this is only for the "digit 2" output neuron
- we need this similar thing to happen to all the digits - and each of these output neurons have its own effect on its previous layer
- ![[Pasted image 20260224175710.png]]
- the desire of the "digit 2" neuron is added together with the desires of all the other output neurons for what should happen to its previous layer in  proportion to its corresponding wieghts and how much each of the neurons need to change
- this is the "idea" for back propagation
- we add all the desired and obtain a list of nudges we need to make happen to the 2nd last layer and we recursively apply it to the previous layers moving backwards through the network
- this is all just for a single training example - if we only listened to that one training example - then the output will be the same no matter what
- so we take the average of corresponding list element over every other weights and biases over every other training example
- the *average list* that we obtain is somewhat loosely proportional to negetive gradient of the cost fxn
- ![[Pasted image 20260224180540.png]]
- but it takes an extremely long time for the computer to get the average of weights and biases - cuz a lot of math
- so we randommly shuffle the training set and we divide the training set to batches (for example 100 images a batch) and then we compute the back propagation for the batch
- each batch gives a good approximation of the desired weight and bias and a signifiacant computation speed up - this is **stochastic gradient descent**


