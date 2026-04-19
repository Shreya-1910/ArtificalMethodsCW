# ArtificalMethodsCW
# Intrusion Detection System using Feature Selection (PSO, GA, Hybrid PSO-GA)

This project focuses on intrusion detection using the CICIDS2017 dataset, where feature selection is performed using metaheuristic optimization algorithms (PSO, GA, and Hybrid PSO-GA) to reduce dimensionality while maintaining high classification performance with a Random Forest model.

---

##  Folder Structure
project/
│
├── data/
│ ├── raw_dataset/ # Original CICIDS2017 dataset
│ ├── preprocessed_data/ # Cleaned and processed dataset
│ └── sample_data/ # 5% sampled dataset 
│
├── notebooks/
│ ├── random_forest.ipynb # Baseline Random Forest model
│ ├── ga_feature_selection.ipynb# Genetic Algorithm feature selection
│ ├── pso_feature_selection.ipynb# Particle Swarm Optimization feature selection
│ └── hybrid_pso_ga.ipynb # Hybrid PSO-GA feature selection
│
├── paper/ # Conference paper submission
│
├── video/ # Project explanation
│
└── README.md # Project documentation

---

##Important Notes

- The project includes a **5% sample dataset** for quick execution and testing.
- For full experiments, use the **complete CICIDS2017 dataset**.
- You MUST update the dataset file paths in the notebooks:
  - `sample_data/` path 
  - `raw_dataset/` path 
- Ensure all paths point to the correct local directory where the datasets are stored before running any notebook.

---

##  How to Use

1. Open the required notebook in Jupyter Notebook or Jupyter Lab.
2. Set the dataset path (sample or full dataset depending on usage).
3. Run cells sequentially to perform preprocessing, feature selection and model evaluation.
4. View results in the output cells.
---

##  Summary
This project demonstrates how metaheuristic algorithms (PSO, GA, and Hybrid PSO-GA) can effectively reduce feature selected while improving or maintaining intrusion detection performance using a Random Forest classifier.
