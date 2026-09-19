<div align="center">

# 🤖 Machine Learning Masterclass: Foundations, Pipelines & Production
### Comprehensive Curriculum by STP (Steps Towards Progress)

**An end-to-end repository tracking machine learning algorithms: statistical foundations, EDA, optimization dynamics, tree ensembles, margin classifiers, and a fully deployed financial capstone application.**

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/Mohammed-Nasr-Aldin/ML-with-STP)
[![Live Portfolio](https://img.shields.io/badge/Live-ML%20Portfolio-4F46E5?logo=googlechrome&logoColor=white)](https://github.com/Mohammed-Nasr-Aldin/ML-Portfolio)
[![Capstone Dashboard](https://img.shields.io/badge/Capstone-Credit%20Score%20Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://creditscore-dashboard.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-success)

<br>

[**🌐 Open Live Portfolio**](https://github.com/Mohammed-Nasr-Aldin/ML-Portfolio) &nbsp;&nbsp;•&nbsp;&nbsp; [**🚀 Test Credit Score App**](https://creditscore-dashboard.streamlit.app/) &nbsp;&nbsp;•&nbsp;&nbsp; [**📑 Lecture Materials (Google Drive)**](https://drive.google.com/drive/folders/1N9PkLzjkwIPvVRdG7err-hu7tFiXfxEY)

</div>

---

## 📑 Table of Contents
- [Track Architecture](#-track-architecture)
- [Curriculum & Theoretical Foundations](#-curriculum--theoretical-foundations)
- [Practical Notebooks & Implementations](#-practical-notebooks--implementations)
- [Featured Capstone: Credit Score System](#-featured-capstone-credit-score-system)
- [Datasets Catalog](#-datasets-catalog)
- [Installation & Quickstart](#-installation--quickstart)
- [Portfolio Integration](#-portfolio-integration)
- [Author](#-author)

---

## 📐 Track Architecture

```text
                                    Machine Learning Lifecycle
┌───────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Business Diagnosis ➔ Exploratory Analytics ➔ Preprocessing ➔ Optimization ➔ Validation ➔ Release  │
└───────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                  │
          ┌───────────────────────────────────────┴───────────────────────────────────────┐
          ▼                                                                               ▼
     Supervised Learning                                                         Unsupervised Learning
     ├── Linear Regression (OLS, GD)                                             ├── K-Means Clustering
     ├── Shrinkage (Lasso L1, Ridge L2)                                          └── Dimensionality Reduction
     ├── Logistic Classification (MLE, Cross-Entropy)
     ├── Tree Ensembles (Gini Gain, Random Forests, Tuning)
     └── Support Vector Machines (Lagrangian Duality, RBF Kernel)
```

---

## 🧠 Curriculum & Theoretical Foundations

### 1. Data Manipulation & Exploratory Data Analysis (Session 1)
* **Data Taxonomy**: Tabular structures, images as matrix tensors, and video streams as temporal stacks.
* **Matrix Calculus & Vectorization**: Array broadcasting with `NumPy` and structural indexing (`.loc` vs `.iloc`) with `Pandas`.
* **Statistical Integrity (5-Stage Protocol)**:
  - Missing value imputation: Central tendency matching (Median for skewed numeric, Mode for categorical).
  - Outlier isolation: Tukey's Fences ($IQR = Q_3 - Q_1$, $[\text{Lower: } Q_1 - 1.5 \cdot IQR, \text{Upper: } Q_3 + 1.5 \cdot IQR]$).
  - Association measurements: Pearson ($r$) for linear dependencies vs. Spearman rank ($\rho$) for monotonic sequences.
* **Materials**: Available on [Google Drive Folder](https://drive.google.com/drive/folders/1OdKG4Dk-LDX6Q829oObhnNupnptL1h3Y).

### 2. Continuous Optimization & Regularization (Session 2)
* **Hypothesis Formulation**: Multivariate mapping $\hat{y} = \theta_0 + \sum_{j=1}^n \theta_j x_j$.
* **Analytical vs Iterative Convergence**: Closed-form Normal Equation ($\theta = (X^T X)^{-1} X^T Y$) vs Gradient Descent weight updates:
  $$w_{k+1} = w_k - \alpha \frac{\partial \mathcal{J}}{\partial w}$$
* **Bias-Variance Stabilization**:
  - **L1 (Lasso)**: $\lambda \sum \vert{}w_j\vert{}$ introduces geometric sparsity for native feature selection.
  - **L2 (Ridge)**: $\lambda \sum w_j^2$ smoothly contracts correlated weights against collinearity.
* **Data Scaling**: Z-Score Standardization ($z = \frac{x - \mu}{\sigma}$) vs Min-Max Normalization ($x' \in [0, 1]$).
* **Materials**: Available on [Google Drive Folder](https://drive.google.com/drive/folders/1MCyKLKkolHwJFjwg3azErUeTqoD3nabf).

### 3. Logistic Modeling & Validation Diagnostics (Session 3)
* **Logistic Mapping**: Mapping linear combiners to non-linear probabilities via Sigmoid:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
* **Loss Optimization**: Negative Log-Likelihood objective translated into Binary Cross-Entropy:

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \ln(\hat{y}_i) + (1 - y_i) \ln(1 - \hat{y}_i) \right]
$$
* **Evaluation Diagnostics**: Precision, Recall, $F_1\text{-score}$ (harmonic balance), Confusion Matrix, and ROC-AUC curve.
* **Resampling Strategy**: Stratified K-Fold to prevent target frequency distortion during validation.
* **Materials**: Available on [Google Drive Folder](https://drive.google.com/drive/folders/1zRXfJhD9WJzR3mEb2M4VqErxROB4-Zge).

### 4. Non-Parametric Trees & Bagging Ensembles (Session 4)
* **Impurity Partitioning**: Node splitting based on Gini Impurity:
  $$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$
* **Variance Reduction via Random Forests**: Bootstrap Aggregating (Bagging) de-correlating trees with random feature sub-sampling.
* **Hyperparameter Search**: Systematically tuning `max_depth`, `n_estimators`, and `min_samples_leaf` using `GridSearchCV`.
* **Materials**: Available on [Google Drive Folder](https://drive.google.com/drive/folders/1p7wvo5WwRkgHxmD2ovN-p_fgq-SJslMY).

### 5. Margin Maximization & Deep Representations (Session 5)
* **Convex Margin Geometry**: Maximizing functional separation margin $\frac{2}{\vert{}\vert{}w\vert{}\vert{}}$ subject to $y_i(w^T x_i + b) \ge 1$.
* **Kernel Projections**: Mapping complex boundaries to higher-dimensional spaces using Radial Basis Function (RBF):
  $$K(x, x') = \exp(-\gamma \vert{}\vert{}x - x'\vert{}\vert{}^2)$$
* **Deep Neural Architectures**: Artificial Perceptron mechanics, activation functions (ReLU, Sigmoid, Tanh, Softmax), and feedforward transformations.
* **Materials**: Available on [Google Drive Folder](https://drive.google.com/drive/folders/1NFjNchZiUpMDBo6afu0jYzOTKbzmEeHt).

---

## 💻 Practical Notebooks & Implementations

| # | Notebook & Task | Domain & Target | Techniques & Benchmarks |
|:---:|:---|:---|:---|
| **01** | [`01_EDA_and_Outliers_Insurance.ipynb`](notebooks/01_EDA_and_Outliers_Insurance.ipynb) | Healthcare Charges | Tukey's IQR boundaries, Boxplots, Kernel Density Estimation |
| **02** | [`02_Linear_Regression_House_Rent.ipynb`](notebooks/02_Linear_Regression_House_Rent.ipynb) | Housing Rental Price | Multiple Linear Regression, Categorical Mapping (City, Contact, Area), Z-score scaling |
| **03** | [`03_Logistic_Regression_Income_Prediction.ipynb`](notebooks/03_Logistic_Regression_Income_Prediction.ipynb) | Census Income (>50K) | Logistic Regression, Categorical Dummies, Stratified 5-Fold ($F_1 = 0.62$, Acc = 84%) |
| **04** | [`04_Decision_Trees_Random_Forest_GridSearch.ipynb`](notebooks/04_Decision_Trees_Random_Forest_GridSearch.ipynb) | Income Classification | Decision Tree ($86\%$) vs Default RF ($84\%$) vs **GridSearch RF ($87\%$)** |
| **05** | [`05_SVM_and_Model_Comparison_Amazon.ipynb`](notebooks/05_SVM_and_Model_Comparison_Amazon.ipynb) | E-commerce Sentiment | Non-linear boundary projection with RBF Kernel vs Linear Models |

---

## 💳 Featured Capstone: Credit Score System

An industrial-grade solution analyzing 100,000 banking customer profiles to evaluate credit default probability.

- **Live Streamlit App**: [creditscore-dashboard.streamlit.app](https://creditscore-dashboard.streamlit.app/)
- **Capstone Source Repository**: [github.com/Mohammed-Nasr-Aldin/Credit-Score-Dashboard](https://github.com/Mohammed-Nasr-Aldin/Credit-Score-Dashboard)
- **Detailed Slide Deck**: [`credit-score-slides.pdf`](capstone_project/credit-score-slides.pdf)

### Engineering Highlights & Performance:
* **Dataset Scale**: 100,000 labeled samples (Good: 17,828, Standard: 53,174, Poor: 28,998).
* **Engineered Ratios**: Debt-to-Income, EMI-to-Income, Invested-to-Income, Balance-to-Income.
* **Feature Selection**: Trimmed from 18 raw inputs to the top **9 high-impact drivers** (Threshold $\ge 0.04$).
* **Accuracy Benchmark**: Random Forest achieved **76% test accuracy** (Recall: Good: $0.85$, Poor: $0.81$, Standard: $0.61$), outperforming Logistic Regression ($66\%$).

---

## 📊 Datasets Catalog

Raw source datasets are located under [`datasets/`](datasets/):
* `income.csv`: US Census demographic records for salary threshold classification (32,561 rows).
* `insurance.csv`: Actuarial health metrics tracking medical expenses (1,338 rows).
* `House_Rent_Dataset.csv`: Indian metropolitan residential rental listings (4,746 rows).
* `amazon_sales_dataset.csv`: Retail transaction orders, discount margins, and customer ratings (50,000 rows).

---

## ⚙️ Installation & Quickstart

```bash
# Clone the repository
git clone [https://github.com/Mohammed-Nasr-Aldin/ML-with-STP.git](https://github.com/Mohammed-Nasr-Aldin/ML-with-STP.git)
cd ML-with-STP

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run notebooks
jupyter notebook
```

## 🌐 Portfolio Integration

This curriculum and its production outputs form a foundational pillar in my personal web portfolio:

👉 [**Inspect ML Portfolio Projects**](https://github.com/Mohammed-Nasr-Aldin/ML-Portfolio)

---

## 👤 Author

**Mohamed Nasr Eldin**  
*Electronics and Communications Engineering Student @ Ain Shams University*  
*Specialized in Machine Learning, Embedded Systems, and Analog IC Design*

[![GitHub](https://img.shields.io/badge/GitHub-Mohammed--Nasr--Aldin-181717?logo=github&logoColor=white)](https://github.com/Mohammed-Nasr-Aldin)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/)

---

## ⭐️ Acknowledgments

Special thanks to the **STP (Steps Towards Progress)** community, instructors, and mentors for delivering an impactful and comprehensive Machine Learning curriculum.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE) - feel free to use and adapt the code for learning purposes.

If you find this repository helpful, consider giving it a ⭐!
