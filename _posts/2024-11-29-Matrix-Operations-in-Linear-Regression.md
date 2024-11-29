---
layout: post
title: "Matrix Operations in Linear Regression"
date: 2024-11-29
categories: [Machine Learning]
tags: [Linear Regression, Linear Algebra, Matrix Operations, Gradient Descent]
---

# Matrix Operations in Linear Regression

# 01 Overview
### Introduction

This article aims to explain the Matrix operations involved in different stages of a Linear Regression Model, whether it’s making model predictions or training the model. Having this additional perspective, alongside the Mathematical details provides a deeper understanding of how Machine Learning Models work. In general, this knowledge equips you to implement the models from a First Principles approach providing a comprehensive view of nuances involved thus paving the way for more advanced learning in the field.

This article is structured in two main parts: First we will assume that we have a pre-built Linear Regression model. This will help us to focus on learning operations involved in making model predictions. In the later part of this article, we’ll shift our focus to the operations under “Model Training” which is where the Model weights are finalized

### Topics covered in this Article:

* Input data representation and Model equation
* Breakdown of Model equation as Matrix Operations
* Model Training and Loss Computation
* Matrix operations behind Gradient Descent and Weight update

### Topics not covered in this article:

* Introduction to Linear Regression - it's assumed that you already have a basic familiarity about this ML technique
* Closed-form solution of Linear Regression
* Introduction to Linear Algebra and Matrix Operations - You'll need to have a basic familiarity with concepts like Vectors, Matrices, Dot product, etc.

# 02 Input Data representation and Model Equation
To get a clear picture of how a Linear Regression model works behind the scenes, let’s begin by looking at how its underlying input data is organized. In Linear Regression, the input data is arranged into features (the variables used for predictions) and observations (individual data points). The target values, which the model tries to predict, are also included in the dataset. The table below gives a simple reference of how the input data is typically organized

<p align="center">
  <img src="https://github.com/user-attachments/assets/af209621-851e-4f6b-9bf1-3dcd670cb39e" 
       alt="Input Data Representation in Linear Regression" 
       title="Input Data Representation in Linear Regression" 
       width="800">
</p>

Here:

* Each row represents an observation (a single data point). The dataset shown has a total of ‘n’ observations
* Each column represents a feature used for prediction. Here we are showing in total ‘m’ features. In terms of notation, capital X represents the entire dataset excluding the target variable, and X with subscript indicate the individual features
* The final column (Y) contains the target values, which the model tries to predict.

**Example :** A dataset with two features and three observations will look like this:

<p align="center">
  <img src="https://github.com/user-attachments/assets/ff02f060-f45e-4726-ab57-0ddf101b8fb9" 
       alt="Example of a very simple Input Dataset for Linear Regression" 
       title="Example of a very simple Input Dataset for Linear Regression" 
       width="300">
</p>



Now, let’s look at the Linear Regression model equation for predicting the target (𝑌)

<p align="center">
  <img src="https://github.com/user-attachments/assets/4acce624-be0a-4ba7-9b1e-3ba5592a1e82" 
       alt="Linear Regression model equation" 
       title="Linear Regression model equation" 
       width="600">
</p>




**Key takeaway from this equation** - Once we know the values of feature weights and the model intercept, we can plug in the feature values for any observation in the RHS (right hand side) of the above expression and compute the corresponding predicted target value.

But a more pertinent question here is, “how do we get the model predictions for multiple data points” – one by one or is there a more efficient way? The answer to this question also provides insights into the common steps performed while training a Linear Regression model.

Let’s address this question in the next section.


# 03 Breakdown of Model equation as Matrix Operations
The previously shown model equation can be written as an Algebraic expression that uses Three entities:

* Feature Vector
* Weights Vector
* Intercept

This is how these entities look.

<p align="center">
  <img src="https://github.com/user-attachments/assets/455d94c7-9821-4f66-a97d-194a4dcc94a1" 
       alt="Vector representation of the components required to make a Prediction on a single data point" 
       title="Vector representation of the components required to make a Prediction on a single data point" 
       width="600">
</p>


Here the Feature and Weight Vectors are Two-Dimensional vectors or Matrices, whereas the Intercept is a scalar (single number with no dimensions). The dimensions of the Feature vector shown indicate that it holds information for a single data point. As a result, as model prediction for this single data point should also be a single number (a scalar) or a 1x1 Matrix

To get this 1x1output, we’ll perform the below Matrix operations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/c30d1112-900c-4fa3-a329-f19d9164998b" 
       alt="Matrix Operations to make a Prediction on a single data point in Linear Regression" 
       title="Matrix Operations to make a Prediction on a single data point in Linear Regression" 
       width="600">
</p>

Note that the Weight Vector is first transposed before being multiplied with the Feature Vector. For a single data point, we can transpose the Feature Vector too and then multiply it by the Weight vector to get the same 1x1 result. So, which of these options should we choose? – The answer lies in knowing “what works for both single and multiple data points”, so let’s wait to see what works for multiple data points (discussed later), and decide this then

The matrix multiplication referred here is actually the dot product operation where we multiply the elements of these Two vectors at the same positions (. i.e., X1 with w1 ,  X2 with w2 and so on) and add the resulting values

The output of this dot product is a 1x1 Matrix. To this we add the intercept, a scalar value. Thus, the resulting output is of shape 1x1, which is the predicted value for the single data point

 For the Two vectors shown, if we simply take the product of the corresponding elements, add all of them along with the intercept, we get the RHS of the Linear Regression equation:

<p align="center">
  <img src="https://github.com/user-attachments/assets/51c5c4e1-a219-462a-a0fd-4e63d31f6b63" 
       alt="Linear Regression model equation" 
       title="Linear Regression model equation" 
       width="600">
</p>


The above shown Matrix computations make sense for a single data point. Next, we’ll look at how predictions are made for Multiple data points. First let’s observe how the Feature and Weight matrices look for multiple data points (let’s say n data points):

<p align="center">
  <img src="https://github.com/user-attachments/assets/b0f14ff2-aad9-4bba-9f39-3ba2bf40a870" 
       alt="Vector representation of the components required to make a Prediction on multiple data points" 
       title="Vector representation of the components required to make a Prediction on multiple data points" 
       width="600">
</p>

 
The notation used in the above shown Feature Matrix is explained using this below visual :

<p align="center">
  <img src="https://github.com/user-attachments/assets/b5c006c3-60fb-4241-98ed-7895450c622c" 
       alt="Explanation of Notation used in Feature Matrix" 
       title="Explanation of Notation used in Feature Matrix" 
       width="600">
</p>


* Subscript of X – indicates which Feature we are talking about
* Superscript of X – indicates the data point or the observation in the dataset


It’s evident that the representation of Feature Matrix has changed significantly but for both the Weight Vector and the Intercept it remains the same. This is because the Weight Vector is not dependent on the number of data observations, rather it is dependent on the number of features.

The predicted output for these multiple data points will be a nx1 Vector (one prediction for each of the n data points). Below are the Matrix operations to compute this prediction vector:

<p align="center">
  <img src="https://github.com/user-attachments/assets/c5e256c9-d0da-4bc8-bd2c-bba8d57845a1" 
       alt="Matrix Operations to make a Prediction on multiple data points in Linear Regression" 
       title="Matrix Operations to make a Prediction on multiple data points in Linear Regression" 
       width="700">
</p>


Here, first the Feature Matrix and the Weight Vector are multiplied to produce a nx1 vector (denoted in the graphic as the intermediate vector). Next, the intercept value is simply added to this vector. This addition is performed element-wise, where the scalar is added to each of the ‘n’ elements to produce the nx1 Predictions vector

Did you notice that regardless of the number of data points (single or multiple), the operations are same, i.e., Feature matrix (X) multiplied by the Transposed Weight vector (w) and the result added to the Intercept value

Let’s formalize this general expression so it captures these Matrix operations to get the Model predictions

<p align="center">
  <img src="https://github.com/user-attachments/assets/f108d4c9-517a-44b9-a20f-7852f9287686" 
       alt="Linear Regression Model Equation represented as an Algebraic Expression" 
       title="Linear Regression Model Equation represented as an Algebraic Expression" 
       width="600">
</p>


This expression is ‘general’ since:

* It can represent computation of the predicted value for both single and multiple data points (. i.e. n being 1 or a higher value)
* It works for any number of Features.
* Also, this concise expression is more convenient to write compared to the Linear Regression equation, where we need to write a longer expression indicating the ‘m’ features

Up until now, we have explored the Matrix operations involved to make predictions in a Linear Regression Model. With the Model Weights in hand, we can get the Model predictions for both single and multiple data points.

But as you might already know, the model weights aren’t available to start with (unlike the Features and Target Values), rather they need to be learned through model training. This makes it essential to understand the steps involved in training the Model and the associated Matrix Operations. While it might sound like a humongous task, but don’t worry, we have already taken care of the foundational concepts of Model Training and going ahead we just need to build upon them.

Before moving to the specific steps of model training, let’s first briefly look at the overall process.


# 04 Model training and Loss Computation
When we are building a Linear Regression model from scratch, all we have is the input data, i.e. the Features and the Target. The Model weights are not known at this point. But why do we need Model weights - to make the model predictions, as seen in the previous sections.

To begin the Model training process, we assume some ‘initial values’ of the Feature Weights and the Intercept. These ‘initial weight’ values are randomly chosen; thus, they are expected to be off from the “final optimal values” we aim to achieve (unless we get lucky with the initial random choices). As a result, the model’s initial predictions will also be off or poor copies of the actual Target values.

This quality of prediction (i.e., how close or far the model predictions are from the actual target values) is quantified or measured through a Loss function. It’s called a ‘function’, because different values of the model weights will produce different loss values – thus it is a function between the model weights and the loss.

In summary the Model Training exercise is an activity of finding the model weights for which we observe the Minimum Loss; why minimum – since then, the model prediction will be closest to the actual target values

The most common loss function used in Linear Regression is the Mean Squared Error (MSE), calculated as:

<p align="center">
  <img src="https://github.com/user-attachments/assets/26ce0185-13b7-4b2b-a2ad-491ebc6fe252" 
       alt="MSE (Mean Squared Error) Loss expression in Linear Regression" 
       title="MSE (Mean Squared Error) Loss expression in Linear Regression" 
       width="400">
</p>


This expression can be broken down into graduals steps for the purpose of understanding the underlying operations:

* For each of the ‘n’ observations in the dataset, we compute the difference between the Actual Target value and the Predicted value. These differences are called as Errors
* Next, we square these Error values to ensure that the positive and negative errors do not cancel out, when looking at the overall Model Error or Loss
* Next, we compute the Mean of these squared errors – thus the name “Mean Squared Error”. Essentially, we add all the squared values and divide by their count
* Finally, the computed Mean is divided by the constant Two. The role of this constant will become clearer in the next section (when we compute the derivative of the Loss Function). To clarify at this point, dividing the loss by Two is done uniformly and doesn't change the model training process. Our main goal is to see if the loss value decreases eventually, so we’re focused on relative comparisons rather than the exact absolute value of the loss.

Now that we understand the computations under the Loss Function, let’s explore the corresponding Matrix operations:

<p align="center">
  <img src="https://github.com/user-attachments/assets/dca4a36b-a705-4b14-912a-26850f03d150" 
       alt="Matrix Operations to compute the Loss Value" 
       title="Matrix Operations to compute the Loss Value" 
       width="1000">
</p>


Here’s a summary of the model training steps covered so far:
* Choose a dataset with ‘n’ observations, ‘m’ Features and one Target variable
* Initialize the Model Weights, i.e. both ‘m’ Feature weights and one intercept
* Compute the Loss using these initial Model Weight values

 As discussed earlier, model predictions with the initial values of model weights will be off, which will also reflect in the Loss value we just learned to compute. Next, we will take steps to improve the model so that its predictions get closer to the Actual Target values thereby reducing the Loss value.


# 05 Matrix operations behind Gradient Descent and Weight update
 
**Gradient Descent preview**

Starting with the randomly chosen ‘initial values’, we update the Model weights by giving them a small nudge in the right direction (i.e., by either increasing or decreasing their values). If we rather update the weights randomly, then it can potentially cause the Loss to go up instead of lowering it. Thus, to update the weights so that it only lowers the loss, we take into account how these weights impact the Loss. This calls for computing the derivative of the Loss function with these weights.

We know from Calculus that the derivative of a function w.r.t a variable, gives a directional sense of where the function increases. However, since our goal is to find the direction in which the Loss function decreases, we simply take the negative of the derivative.

The next set of Model weights values are computed from the previous set by adding to them a small number, which is computed using the Loss Derivative and a term called as ‘learning rate’. The expression for this update looks as follows:

<p align="center">
  <img src="https://github.com/user-attachments/assets/a795a270-3753-4e68-b820-de6e30523737" 
       alt="Expressions to update Model Weights" 
       title="Expressions to update Model Weights" 
       width="600">
</p>


Here we first compute the RHS of both the expressions and then update the weight values against the variable on the LHS. Learning rate (the symbol eta) in the expression helps us control how big or small of an update we want to make to the previous weight value.

From a terminology point of view, we’ve just covered the core idea behind the **Gradient Descent Algorithm** for model training.

**The most important (and often overlooked) aspect of these expressions is the shape of the derivatives**. The derivative of Loss function with respect to the:

* The Weights vector comes out as a vector of dimension 1xm. This makes sense as we need to update all the ‘m’ Features Weights in the weights vector
* The Intercept comes as a scalar or a matrix of dimension 1x1

Next, we will aim to get to the final expressions of these weight updates by computing the derivative of the Loss function. We’ll also understand side-by-side the Matrix operations required to implement these weight updates.

 Since we need to compute the derivative of the Loss function, let’s take a second look at its expression by plugging in the Model Weights

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7da2020-80f6-4521-95ae-3a9cd6ff30dc" 
       alt="Loss Function expression with Model Weights" 
       title="Loss Function expression with Model Weights" 
       width="600">
</p>

Next, let’s compute the derivative of the Loss function with the Weight Vector and the Intercept

**Derivative of the Loss function w.r.t the Weight vector**

<p align="center">
  <img src="https://github.com/user-attachments/assets/811d07f1-67bb-4314-8a07-7c0118a11324" 
       alt="Derivative of Loss function with the Weight vector" 
       title="Derivative of Loss function with the Weight vector" 
       width="600">
</p>

The matrix operations for the last shown expression will be performed as follows.

<p align="center">
  <img src="https://github.com/user-attachments/assets/7ebe007d-bbbc-448e-89d9-46f26d1b9b16" 
       alt="Matrix Operations to compute the derivative of Loss function with the Weight vector" 
       title="Matrix Operations to compute the derivative of Loss function with the Weight vector" 
       width="800">
</p>


Below is a breakdown of what’s happening here:

* First the difference between the Actual Target values and the Predicted Target values are computed. This difference is the earlier seen Error vector which has the dimension nx1
* Next, we multiply the Transpose of this Error vector (1xn) with the Feature Matrix (nxm)
* This matrix multiplication inherently accounts for the summation in the expression. Here is how: The Error Vector gets multiplied by each of the ‘m’ columns in the Feature matrix, effectively performing a dot product. Thus, for any one column, ‘n’ products are computed and summed, producing a single value. Repeating this for all the ‘m’ columns results in a 1xm vector
* Lastly all the values in this resulting 1xm vector are multiplied by a factor of 1/n. This is like applying scaling to each of these ‘m’ terms, as they have been derived from ‘n’ observations under each feature

**Derivative of the Loss function w.r.t the Intercept**

<p align="center">
  <img src="https://github.com/user-attachments/assets/0eaed6d9-d137-46ba-bb0b-e46ce535b948" 
       alt="Derivative of Loss function with the Intercept" 
       title="Derivative of Loss function with the Intercept" 
       width="600">
</p>


The matrix operations required for computing the derivative w.r.t the Intercept are relatively simpler and requires just the computation of the Error vector. This is followed by taking the average of all Error values in the Vector to get a scalar number (or a matrix with dimension 1x1) – why average? since we need to apply the Summation from the expression and then divide by the number of observations.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b4852fc6-f502-4125-8929-02fc14805786" 
       alt="Matrix Operations to compute the derivative of Loss function with the Intercept" 
       title="Matrix Operations to compute the derivative of Loss function with the Intercept" 
       width="400">
</p>

Since both the derivatives have been computed, we can write the final weight update expressions:

<p align="center">
  <img src="https://github.com/user-attachments/assets/92a6aa11-ad1c-4124-8d23-f49685c4556e" 
       alt="Weight update expressions for Weight vector and Intercept" 
       title="Weight update expressions for Weight vector and Intercept" 
       width="800">
</p>


The above expressions may seem intimidating at first and may not fully explain the process on their own. Fortunately, we’ve already explored the details of how these operations work through Matrices, making them much easier to understand.

 Here is a view of how these weight updates look in the Matrix form

**Weight vector**

<p align="center">
  <img src="https://github.com/user-attachments/assets/af0eb4a4-460b-49d7-82a1-f137365ffc12" 
       alt="Matrix operation to update the Weight vector" 
       title="Matrix operation to update the Weight vector" 
       width="700">
</p>


**Intercept**

<p align="center">
  <img src="https://github.com/user-attachments/assets/e49e99a5-ef42-4aed-8315-438997cfce65" 
       alt="Matrix operation to update the Intercept" 
       title="Matrix operation to update the Intercept" 
       width="500">
</p>

These weight updates happen iteratively, and we keep a track of Loss values under each iteration. A good point to stop these iterations is when the drop in loss in consecutive iterations becomes negligible. This concludes the process of Model training providing us with the Final Model weights, that minimize the Loss and give us the most accurate predictions based on the data.

The expressions covered in this section not only take us closer to the Mathematical and Operational details of Gradient Descent but also help us see that each weight update requires computations on all ‘n’ observations of the Feature Matrix. Also, to get to the Final Model Weights, we need to perform multiple such iterations. Thus, for large input datasets the Model training process could be quite slow. To overcome this challenge, different variations of the Gradient Descent Algorithm such as Mini Batch Gradient Descent and Stochastic Gradient Descent are used. We’ll not go into these details in this article.


# 06 Conclusion
In this article we explored the kind of Matrix Operations that are performed in both Training a Linear Regression Model as well as in using it to make predictions. The approach to Model training covered here is a basic or ‘vanilla’ version to get the core idea behind Gradient Descent Algorithm. The more advanced approaches to Model training involve: tweaking the learning rate as training happens, incorporating the Regularization terms to avoid overfitting and also using different and more advanced variants of the Gradient Descent Algorithm itself.

The covered topics and insights go well beyond Linear Regression and serve as a foundation for understanding more complex Machine Learning algorithms. With this knowledge, you can explore advanced topics like Mini-batch Gradient Descent, Stochastic Gradient Descent, and optimization techniques used in Neural Networks. This step-by-step approach also highlights the importance of starting with the basics, helping you build a solid understanding of how models and algorithms truly work.
