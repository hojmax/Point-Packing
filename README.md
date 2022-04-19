# Point Packing
## 📝 Description
In this project i use gradient descent to find the optimal 'packing' of points confined to a rectangular 2D container. I define this optimal packing as minimizing the distance to all other points and the borders. On the basis of this definition i formulate the following loss function (Sorry to light theme users 😊 ): 

![eq](images/eq1.png)

Here **p** is some list of points of length **n**. Each point stores its **x** and **y** position. The width and height of the rectangle are denoted by **w** and **h**. The first summation describes the loss attributed by the proximity of points to the borders. As a result, the constant **α** is a hyperparameter that controls the strength of the borders influence. The second term describes the loss attributed by the points lying close to each other (hence the summation over all possible pairs). This function is only well defined if none of the points overlap with each other or with the borders.


## 🔍 Example
Below you can see the (locally) optimal packing with **n=20**, **α=1** and **w=h=800**:

![eq](images/20points.png)

Or a packing with **n=1000**, **α=250** and **w=h=800**:

![eq](images/1000points.png)

It is visually intuitive how **α** corresponds to the amount 'repulsion' the borders enforce on the points. 

## 🏗 Implementation
In order to optimize the defined loss function, you simply update the point positions using the gradient. This computation is done in C with openmp for parallelization. The visualization is done in Python with PyGame.

## 🏄‍♂️ Usage
To use the project run `
python3 draw.py`. Try clicking and dragging the mouse for repulsion. Press **R** to reset the points. You can try different setups by changing the settings in **points.py**.

