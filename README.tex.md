# Point Packing
## 📝 Description
In this project i use gradient descent to find the optimal 'packing' of points confined to a rectangular 2D container. I define this optimal packing as minimizing the distance to all other points and the borders. On the basis of this definition i formulate the following loss function: 

$$
\begin{aligned}
loss(p) &= \alpha \left( \sum_{i=1}^{n} \frac{1}{p_i.x}+\frac{1}{p_i.y}+\frac{1}{w-p_i.x}+\frac{1}{h-p_i.y}\right ) +\\ &\quad\sum_{i=1}^{n}\sum_{j=i+1}^{n}\frac{1}{\sqrt{\left (p_i.x-p_j.x\right )^2+\left (p_i.y-p_j.y\right )^2}}
\end{aligned}
$$


Here $p$ is some list of points of length $n$. Each point stores its $x$ and $y$ position. The width and height of the rectangle are denoted by $w$ and $h$. The first summation describes the loss attributed by the proximity of points to the borders. As a result, the constant $\alpha$ is a hyperparameter that controls the strength of the borders influence. The second term describes the loss attributed by the points lying close to each other (hence the summation over all possible pairs). This function is only well defined if none of the points overlap with each other or with the borders.


## 🔍 Example
Below you can see the (locally) optimal packing with $n=20$, $\alpha=1$ and $w=h=800$:

<center>
<img src="images/20points.png" alt="drawing" width="300"/>
</center>

Or a packing with $n=1000$, $\alpha=250$ and $w=h=800$:

<center>
<img src="images/1000points.png" alt="drawing" width="300"/>
</center>

It is visually intuitive how $\alpha$ corresponds to the amount 'repulsion' the borders enforce on the points. 

## 🏗 Implementation
In order to optimize the defined loss function, you simply update the point positions using the gradient. This computation is done in C with openmp for parallelization. The visualization is done in Python with PyGame.

## 🏄‍♂️ Usage
In order to use the project run `
python3 draw.py`. Try clicking and dragging the mouse for repulsion. Press **R** to reset the points. You can try different setups by changing the settings in **points.py**.

