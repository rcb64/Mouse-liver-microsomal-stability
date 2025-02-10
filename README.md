Mouse Microsomal Stability Prediction with Machine Learning

Overview

I built a model which predicts mouse liver microsomal stability of chemical compounds using machine learning. I implemented 3 models: Naive Bayesian, SVM and Random Forest. The random forest model emerges to be the best in a 5-fold internal cross-validation, and is chosen for further refinement. The best model underwent hyperparameter optimization and calibration, which yielded a model with good predicting power (AUC = 0.82, Log Loss = 0.47) on a training set of 759 compounds. The model performs moderately on an independent unseen dataset of 571 compounds (AUC = 0.82, Log Loss = 0.47). 

Features

Data Processing: Extracts molecular descriptors and extended-connectivity fingerprints (Morgan fingerprints) from SMILES notation.

Feature Engineering: Standardizes features and reduces dimensionality using Principal Component Analysis (PCA).

Model Training & Evaluation: Compares multiple classifiers through cross-validation and selects the best-performing model.

Hyperparameter Tuning: Implements randomized search for optimizing Random Forest parameters.

Calibration: Uses isotonic calibration for improved probability estimates.

Applicability Domain Quantification: Assesses when the model is operating within its intended domain by leveraging uncertainty estimation techniques.

Performance Metrics: Evaluates models using accuracy, ROC AUC, log loss, and confusion matrices.

Requirements

Ensure you have the following dependencies installed:

pip install numpy pandas scikit-learn matplotlib seaborn
conda install -c conda-forge rdkit


Dataset

The model is trained on a dataset of 759 compounds stored in an SDF file. The dataset includes molecular structures and stability labels:

SMILES column: Contains molecular structures.

Stable column: Binary classification (1 for stable, 0 for unstable).

Stability is defined as:

0 = Half-life < 30 min

1 = Half-life > 59.5 min

Pipeline

1. Load Dataset

Reads the SDF file and converts it into a Pandas DataFrame.

2. Feature Extraction

Computes molecular descriptors (e.g., logP, molecular weight, number of rings, H-bond donors/acceptors).

Generates Morgan fingerprints (FCFP6) for molecular representation.

3. Feature Engineering

Standardizes numerical features using StandardScaler.

Applies Principal Component Analysis (PCA) for dimensionality reduction.

4. Model Training & Evaluation

Trains and evaluates Naive Bayes, SVM, and Random Forest classifiers.

Performs 5-fold cross-validation to compare models.

Tunes hyperparameters for Random Forest using RandomizedSearchCV.

Calibrates the best Random Forest model using isotonic regression.

5. Applicability Domain Quantification

Uses uncertainty estimation methods to determine when the model is making out-of-distribution predictions.

Implements probability-based confidence scores to flag unreliable predictions.

Proposes a strategy to refine the applicability domain by integrating external datasets.

6. Model Performance Analysis

Generates confusion matrices and classification reports.

Plots ROC curves for AUC evaluation.

Computes log loss to assess predictive uncertainty.


Interpreting Results

The script outputs performance metrics and visualizations.

Results include cross-validation scores, classification reports, and calibration curves.

Applicability domain quantification results indicate when predictions may be unreliable due to high uncertainty.

Data Limitations

The datasets (both training and validation) are heavily skewed toward unstable compounds. The training dataset has 759 compounds, out of which 262 are stable. The validation dataset has 571 compounds, out of which 109 are stable. This can bias the model towards predicting unstable compounds with a good accuracy and recall, at the expense of the performance on the positive set, which is exactly what I observe in my model. 

There is limited publicly available mouse microsomal stability data.

The datasets used for training and validation are sourced from Freundlich et al., Pharm Res. 2016 February; 33(2): 433–449.

Human microsomal stability data is much more widely available (e.g., Therapeutic Data Commons, AZ dataset: TDC ADME). 

Future Improvements

Create / curate other databases that are not skewed towards stable or unstable compounds. 

Try deep learning models (e.g., Graph Neural Networks).

Further refine applicability domain quantification using additional uncertainty metrics.

Use more complex fingerprints, such as Pharmacophore fingerprints or combinations.


Key findings / discussion: 
The random forest model emerges to be the best, especially after hyperparameter tuning and calibration. On the test set using the training data, it achieves an accuracy of 78%, with an AUC of 0.83 and a log loss of 0.47, indicating a strong ability to discriminate between classes but with room for improvement in positive class recall. This is not surprising given that the training database is severely skewed towards unstable compounds.  For unseen validation data, it predicts with 76% accuracy and an AUC of 0.63, reflecting a decline in discriminatory power potentially due to model overfitting to training data. Another explanations is that the molecular descriptors and the fcfp_6 fingerprints are simply not powerful enough to predict such a complex feature as microsomal stability, which is clearly a function of more than just molecule geometry and phys-chem properties. A potential solution is to use deep neural networks, using graph representations for the molecules. 




Contributions 

Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

Downloading the Repository

To get started, clone the repository from GitHub:

git clone https://github.com/rcb64/Mouse-liver-microsomal-stability.git

Navigate to the project directory:

cd Mouse-liver-microsomal-stability

Acknowledgments 

The datasets used in this project were created and curated by Freundlich et al. and are available as supplementary information in Pharm Res. 2016 February; 33(2): 433–449.
Code generation and debugging were performed with OpenAI (ChatGPT). 