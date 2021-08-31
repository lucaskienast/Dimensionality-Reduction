# Dimensionality Reduction
The number of input variables or features for a dataset is referred to as its dimensionality. Large numbers of input features can cause poor performance for machine learning algorithms. Having a large number of dimensions in the feature space can mean that the volume of that space is very large, and in turn, the points that we have in that space (rows of data) often represent a small and non-representative sample. Dimensionality reduction refers to techniques that reduce the number of input variables in a dataset. These techniques can be used in applied machine learning to simplify a classification or regression dataset in order to better fit a predictive model. It might be performed after data cleaning and data scaling and before training a predictive model. As such, any dimensionality reduction performed on training data must also be performed on new data.

Some of the main techniques for dimensionality reduction are:

- Feature selection methods
- Matrix factorization/decomposition
- Manifold learning
- Autoencoders

## Feature Selection 
Perhaps the most common methods are so-called feature selection techniques that use scoring or statistical methods to select which features to keep and which features to delete. Two main classes of feature selection techniques include wrapper methods and filter methods. Wrapper methods, as the name suggests, wrap a machine learning model, fitting and evaluating the model with different subsets of input features and selecting the subset the results in the best model performance. RFE is an example of a wrapper feature selection method. Filter methods use scoring methods, like correlation between the feature and the target variable, to select a subset of input features that are most predictive. 

Examples include:

- Missing value ratio
- Low variance filter
- High correlation filter
- Random forest
- Backward/ Forward feature elimination 
- Factor analysis

See: https://github.com/lucaskienast/Feature-Selection

## Matrix Factorization
Techniques from linear algebra can be used for dimensionality reduction. Specifically, matrix factorization methods can be used to reduce a dataset matrix into its constituent parts. The parts can then be ranked and a subset of those parts can be selected that best captures the salient structure of the matrix that can be used to represent the dataset. Typically, linear algebra methods assume that all input features have the same scale or distribution. This suggests that it is good practice to either normalize or standardize data prior to using these methods if the input variables have differing scales or units. 

Examples include:

- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA)

## Manifold Learning
Techniques from high-dimensionality statistics can also be used for dimensionality reduction. These techniques are sometimes referred to as “manifold learning” and are used to create a low-dimensional projection of high-dimensional data, often for the purposes of data visualization. The projection is designed to both create a low-dimensional representation of the dataset whilst best preserving the salient structure or relationships in the data. The features in the projection often have little relationship with the original columns, e.g. they do not have column names, which can be confusing to beginners. Typically, manifold learning methods assume that all input features have the same scale or distribution. This suggests that it is good practice to either normalize or standardize data prior to using these methods if the input variables have differing scales or units. 

Examples include:

- Isomap Embedding
- Locally Linear Embedding (LLE)
- Spectral Embedding
- Kohonen Self-Organizing Map (SOM)
- Sammons Mapping
- Multidimensional Scaling (MDS)
- t-distributed Stochastic Neighbor Embedding (t-SNE)

## Autoencoders
Deep learning neural networks can be constructed to perform dimensionality reduction. A popular approach is called autoencoders. This involves framing a self-supervised learning problem where a model must reproduce the input correctly. More precisely, an auto-encoder is a feedforward neural network that is trained to predict the input itself. A network model is used that seeks to compress the data flow to a bottleneck layer with far fewer dimensions than the original input data. The part of the model prior to and including the bottleneck is referred to as the encoder, and the part of the model that reads the bottleneck output and reconstructs the input is called the decoder. After training, the decoder is discarded and the output from the bottleneck is used directly as the reduced dimensionality of the input. Inputs transformed by this encoder can then be fed into another model, not necessarily a neural network model. The output of the encoder is a type of projection, and like other projection methods, there is no direct relationship to the bottleneck output back to the original input variables, making them challenging to interpret. 

Examples include:

- LSTM autoencoders
- Denoising autoencoders
- Sparse autoencoders
- Deep autoencoders
- Contractive autoencoders
- Undercomplete autoencoders
- Convolutional autoencoders
- Variational autoencoders

## References

Brownlee, J. (2020) A Gentle Introduction to LSTM Autoencoders. Available at: https://machinelearningmastery.com/lstm-autoencoders/ (Accessed: 31 August 2021)

Brownlee, J. (2019) A Gentle Introduction to Matrix Factorization for Machine Learning. Available at: https://machinelearningmastery.com/introduction-to-matrix-decompositions-for-machine-learning/ (Accessed: 31 August 2021)

Brownlee, J. (2019) Gentle Introduction to Eigenvalues and Eigenvectors for Machine Learning. Available at: https://machinelearningmastery.com/introduction-to-eigendecomposition-eigenvalues-and-eigenvectors/ (Accessed: 31 August 2021)

Brownlee, J. (2019) How to Calculate Principal Component Analysis (PCA) from Scratch in Python. Available at: https://machinelearningmastery.com/calculate-principal-component-analysis-scratch-python/ (Accessed: 31 August 2021)

Brownlee, J. (2019) How to Calculate the SVD from Scratch with Python. Available at: https://machinelearningmastery.com/singular-value-decomposition-for-machine-learning/ (Accessed: 31 August 2021)

Brownlee, J. (2020) Introduction to Dimensionality Reduction for Machine Learning. Available at: https://machinelearningmastery.com/dimensionality-reduction-for-machine-learning/ (Accessed: 31 August 2021)

Brownlee, J. (2020) Linear Discriminant Analysis for Dimensionality Reduction in Python. Available at: https://machinelearningmastery.com/linear-discriminant-analysis-for-dimensionality-reduction-in-python/ (Accessed: 31 August 2021)

Brownlee, J. (2020) Principal Component Analysis for Dimensionality Reduction in Python. Available at: https://machinelearningmastery.com/principal-components-analysis-for-dimensionality-reduction-in-python/ (Accessed: 31 August 2021)

Brownlee, J. (2020) Singular Value Decomposition for Dimensionality Reduction in Python. Available at: https://machinelearningmastery.com/singular-value-decomposition-for-dimensionality-reduction-in-python/ (Accessed: 31 August 2021)

Brownlee, J. (2020) 6 Dimensionality Reduction Algorithms With Python. Available at: https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/ (Accessed: 31 August 2021)

Kapri, A. (2020) PCA vs LDA vs T-SNE — Let’s Understand the difference between them. Available at: https://medium.com/analytics-vidhya/pca-vs-lda-vs-t-sne-lets-understand-the-difference-between-them-22fa6b9be9d0 (Accessed: 31 August 2021)

Krohn, J. (2021) Mathematical Foundations of Machine Learning. Available at: https://github.com/jonkrohn/ML-foundations (Accessed: 31 August 2021)

Mungoli, A. (2020) Dimensionality Reduction: PCA versus Autoencoders. Available at: https://towardsdatascience.com/dimensionality-reduction-pca-versus-autoencoders-338fcaf3297d (Accessed: 31 August 2021)

Prakash, A. (2020) Different Types of Autoencoders. Available at: https://iq.opengenus.org/types-of-autoencoder/ (Accessed: 31 August 2021)

Sharma, P. (2018) The Ultimate Guide to 12 Dimensionality Reduction Techniques (with Python codes). Available at: https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/ (Accessed: 31 August 2021)

Silipo, R. (2015) Seven Techniques for Data Dimensionality Reduction. Available at: https://www.kdnuggets.com/2015/05/7-methods-data-dimensionality-reduction.html (Accessed: 31 August 2021)

Thaenraj, P. (2021) PCA, LDA, and SVD: Model Tuning Through Feature Reduction for Transportation POI Classification. Available at: https://towardsdatascience.com/pca-lda-and-svd-model-tuning-through-feature-reduction-for-transportation-poi-classification-8d20501ee255 (Accessed: 31 August 2021)

Udacity (2021) Artificial Intelligence for Trading. Available at: https://www.udacity.com/course/ai-for-trading--nd880 (Accessed: 31 August 2021)
