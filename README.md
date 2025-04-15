Mouse Microsomal Stability Prediction with Machine Learning

Overview

I built a model which predicts mouse liver microsomal stability of chemical compounds using machine learning. I implemented 3 models: Naive Bayesian, SVM and Random Forest. The random forest model emerges to be the best in a 5-fold internal cross-validation, and is chosen for further refinement. The best model underwent hyperparameter optimization and calibration, which yielded a model with good predicting power (AUC = 0.87, Log Loss = 0.43) on a training set of 759 compounds. The model is powerful in performing on an independent unseen dataset of 571 compounds (AUC = 0.78, Log Loss = 0.53). 

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

Computes 3D molecular descriptors using Maestro(Schrodinger)'s QikProp (e.g., volume, CACO permeability, PSA).

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

Discussion:

Key findings:
The Random Forest model demonstrates effective prediction of mouse liver microsomal stability, achieving significant results with a Test AUC of 0.867 following hyperparameter optimization (n_estimators=50, min_samples_split=5, min_samples_leaf=4, max_features=None, and max_depth=30). The calibrated model further attains a Test AUC of 0.87 and Log Loss of 0.43, indicating robust performance, albeit with potential overfitting evidenced by a perfect internal Train AUC. In comparison, the Naive Bayes model recorded a lower Mean Test AUC of 0.621, while the SVM displayed competitive performance with a Mean Test AUC of 0.873. On an independent validation set, the calibrated Random Forest model maintains a commendable AUC of 0.78, despite decreased precision and recall for the positive class. These metrics highlight the model's substantial generalization capability and effective calibration, emphasizing improved probability estimations. This aligns with findings from the referenced study, which reported Bayesian models exhibiting notable external validation performance with a comparable AUC of 0.778. 

Database preparation: 
The two databases created and curated by Freundlich et al. were used as a starting point. Maestro's LigPrep was used to clean the structures, protonate at pH 7.4 and generate 3D conformers. MacroModel minimisation was next used to minimise the conformations of the molecules. This lead to some molecules having more than 1 conformer predicted, with their associated potential energies. This database was then used as an input for the QikProp calculation in Maestro, which created and predicted various 3D properties for the molecules, such as volume, PSA, ionisation potential, CACO permeability. I then created a Python script (available in the package) to filter out the duplicate molecules and create a novel database, where only the lowest energy conformer is kept for purposes of model training. 

RDKit derived properties vs. QikProp derived properties: 
In my comparative analysis of predictive models for mouse microsomal stability, the QikProp model distinctly outperformed my RDKit-derived properties model across several key performance metrics. The QikProp model achieved a test set accuracy of 79% and an impressive AUC of 0.87. On the validation dataset, it further demonstrated robustness with an 80% accuracy and a notable AUC of 0.78. In contrast, while the RDKit model achieved promising cross-validation scores with a mean test accuracy of 81.9% and mean AUC of 0.899, it fell short in validation performance, securing a lower test accuracy of 77% and AUC of 0.85. The RDKit model also struggled with class 1 predictions on the validation set, with a precision of 25% and recall of 16%. Conversely, the QikProp model balanced precision and recall more effectively, achieving a precision of 48% and recall of 60% for class 1. The QikProp model's superior handling of class imbalance and potential for further enhancement highlights it as my preferred model for reliable and insightful predictions in my dataset.

Exploring different fingerprint radii:
In an extensive analysis of the QikProp model across varying fingerprint radii, distinct performance trends emerged. The model with a radius of 3 demonstrated superior test set results, achieving an accuracy of 80% and an AUC of 0.88, outperforming the radius 6 and 10 configurations. Meanwhile, the model with a radius of 6 balanced strong test accuracy (79%) and AUC (0.87) with remarkable validation performance, securing an 80% accuracy and an AUC of 0.78. In contrast, the radius 10 model, while competitive on the test set with a 77% accuracy and an AUC of 0.87, underperformed in validation, reflecting a lower AUC of 0.71 and an accuracy of 75%. Class-specific analysis revealed that both radii 3 and 6 maintained high precision and recall for the negative class, while radius 10 showed challenges in accurately predicting the positive class, with a mere F1-score of 0.27. Radius 6 emerges as the optimal choice for robust generalization, whereas radius 3 excels in maximizing in-sample accuracy, offering a compelling trade-off depending on the desired balance between predictive prowess and generalization capacity.

Feature importance analysis:
The analysis reveals that Morgan fingerprints account for almost the entire importance score (0.999989), dwarfing the contributions from the selected molecular descriptors. This suggests that the structural patterns captured by the fingerprints are highly predictive of the target outcome in our dataset. While the traditional descriptors add complexity to the model, they appear less influential in this context. This is in line with running a multi model hyperparameter optimization using the Qptuna platform (AstraZeneca) which converged to random forest models using ecfp6 descriptors as the best models. 

Descriptors versus fingerprints:
When using only Morgan fingerprints as descriptors, model performance improved slightly compared to the full model. The Random Forest's mean test accuracy increased from 79.0% with the full model to 81.2% using only fingerprints. The AUC improved from 0.867 to 0.894. However, both models showed similar challenges with class 1 recall, highlighting areas for improvement. On the validation set, accuracy was slightly lower with fingerprints (77% vs. 80%), suggesting fingerprints alone provide strong predictive power but refinement is needed for better class balance. This is expected given the findings from feature importance analysis. 
When using only 3D descriptors, the model's performance was lower compared to using fingerprints. The Random Forest with descriptors achieved a test accuracy of 70% (AUC: 0.74), showcasing challenges in classifying the minority class with a recall of 0.41 for class 1. In contrast, the fingerprints-only model had a higher test accuracy of 81.2% (AUC: 0.894), indicating better predictive capability.
On the validation set, the descriptors-only model showed a reduced accuracy of 64% (AUC: 0.52) compared to 77% with fingerprints. This highlights that while 3D descriptors provide some predictive value, fingerprints capture more critical information leading to superior model performance overall. 


Information gain analysis and privileged substructures: 
In my analysis of mouse microsomal stability, I used BRICS decomposition to identify key molecular fragments that influence compound stability. By calculating mutual information scores and performing chi-square tests, I identified fragments with significant predictive value. The scatter plot illustrates a strong correlation between fragment frequency in unstable compounds and information gain, with certain outliers showing exceptional significance. Known chemotypes such as secondary amines, ethers, electron rich heterocycles, alkyl chain and ketones emerge as powerful discriminants and predictors for stable versus unstable compounds. There is a strong correlation between information gain and frequency, with a steep slope, indicating that molecular structures are important predictors through their fingerprints, as evidenced by the feature importance analysis. 

Uncertainty quantification:
The uncertainty quantification for the Random Forest model revealed insights into prediction confidence. On the test set, the average interval width was 0.784, and the standard deviation of predictions was 0.024, reflecting relatively narrow prediction variability. However, the coverage probability was 0.000, indicating that true labels often fell outside the predicted intervals—a potential area for model or method refinement. The entropy of 0.521 suggests moderate prediction uncertainty.
For the validation set, the average interval width was slightly narrower at 0.508, with a standard deviation of 0.014. The coverage probability was also 0.000, and the entropy increased to 0.632, suggesting increased prediction uncertainty compared to the test set. These findings highlight the need to enhance model calibration or uncertainty estimation techniques to improve coverage while maintaining prediction precision.


Applicability domain:
In my application of k-nearest neighbors to define the applicability domain (AD) of the QikProp model with a fingerprint radius of 6, I successfully implemented a robust mechanism that ensures prediction reliability. By establishing a clear threshold at the 95th percentile of nearest-neighbor distances within the training data, I accurately identified and categorized out-of-domain instances. This process flagged only 4 out of 150 test instances, indicative of a comprehensive training set that effectively captures the feature space necessary for my model's stable predictions. 

PCA analysis:
In my model development process, I employed Principal Component Analysis (PCA) to reduce the dimensionality of the dataset while retaining key information. Upon examining the cumulative explained variance against the number of PCA components, I observed that using nine components effectively captured about 90% of the variance in the data. This was crucial for striking a balance between simplification and the preservation of the original data structure. Interestingly, when I tried increasing the number of components to 13, the model's performance on the validation set actually deteriorated. This experience highlighted the importance of not only maintaining a sufficient level of variance but also ensuring that the model remains manageable and less prone to overfitting. 

Dataset exploration:
In my analysis of the chemical space represented by my data, I utilized t-SNE visualization and Tanimoto similarity heatmaps for both the training and validation sets. The t-SNE plot reveals that the training data consists of a diverse array of compounds clustered in distinct regions, indicating that I have effectively captured various chemical classes. However, I also noticed that the validation set contains several compounds positioned in areas that are either sparse or not represented in the training set. This suggests potential challenges for model generalization, as some validation samples may possess unique characteristics that the model has not encountered during training. The Tanimoto similarity heatmap for the training set shows prominent blocks of high similarity among many molecules, hinting at closely related compounds, which is essential for understanding structural relationships. Conversely, the validation heatmap indicates that while there are clusters of similar compounds, they do not fully overlap with the training data, emphasizing gaps in my coverage.




Contributions 

Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

Downloading the Repository

To get started, clone the repository from GitHub:

git clone https://github.com/rcb64/Mouse-liver-microsomal-stability.git

Navigate to the project directory:

cd Mouse-liver-microsomal-stability

Acknowledgments 

The datasets used in this project were created and curated by Freundlich et al. and are available as supplementary information in Pharm Res. 2016 February; 33(2): 433–449.
