# Heart Disease Prediction

This project explores multiple machine learning models to predict the presence of heart disease using the [Heart Disease dataset](heart.csv).  
The workflow covers feature engineering, model selection, hyperparameter tuning, and comparative evaluation of traditional ML models and a neural network.

---

## 📂 Project Structure

- **Data Preprocessing & Feature Engineering**
  - Load and inspect dataset (`pandas`)
  - One-hot encode categorical variables
  - Standardize numerical features
  - Split into training and test sets

- **Models Implemented**
  - Linear Models: Logistic Regression, SGD Classifier
  - Tree-Based Models: Decision Tree, Random Forest, AdaBoost, Gradient Boosting
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Neural Network (TensorFlow / Keras)

- **Evaluation Metrics**
  - Accuracy (for baseline comparison)
  - Precision (% of predicted positives that are true positives)
  - Recall (% of true positives identified)  
  *Due to the medical nature of the problem, recall was prioritized while maintaining acceptable precision.*

---

## ⚙️ Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place `heart.csv` in the working directory.  
   If using Google Colab, adjust the `%cd` path as needed.

---

## 📊 Results & Analysis

### Linear Models
- Logistic Regression and SGD performed moderately.  
- Showed that features explained variance, but accuracy was limited.

### Decision Tree Models
- **Random Forest** and **Gradient Boosting** significantly outperformed linear models.  
- Provided stronger recall/precision balance.

### KNN
- Competitive results at `k=9`, but limited tunability and sensitivity to data scaling.

### SVC
- Explored kernels and hyperparameters, but performance was not superior to ensemble trees.

### Neural Network
- Multi-layer architecture (128 → 64 → 32 → 16 → 1, ReLU + Sigmoid).  
- Achieved ~90% test accuracy.  
- However, the model tended to overfit without significantly outperforming simpler models.

### Best Models
- **Random Forest (n_estimators ≈ 325)** gave the best balance between precision and recall.  
- **Gradient Boosting** also performed well but with more variance.  
- Neural networks were powerful but unnecessarily complex for this dataset.

---

## 📈 Visualizations

- Training curves for precision/recall across epochs (Neural Network).
- Precision vs Recall curves for Random Forest and Gradient Boosting across `n_estimators`.

---

## 🧾 Key Findings

- **Precision > Recall across all models** — a trade-off not ideal in a medical detection context.  
- **Goal:** Recall ≥ 0.95 with Precision ≥ 0.90.  
- None of the tested models consistently met this threshold.  
- **Conclusion:** The limitation lies in the dataset itself — additional features or more examples are needed for clinically useful predictions.

---

## 🚀 Next Steps

- Acquire more diverse patient data (lab results, demographics, genetic factors).  
- Explore feature selection and dimensionality reduction.  
- Experiment with ensemble stacking.  
- Test advanced neural architectures with regularization techniques.

---

## 🛠️ Technologies Used

- **Python 3.9+**
- **pandas**, **numpy**, **scikit-learn**
- **TensorFlow / Keras**
- **Matplotlib**

---

## 📜 License

This project is released under the MIT License.  
See [LICENSE](LICENSE) for more details.

---
