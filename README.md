# ArtificalMethodsCW
# Intrusion Detection System using Feature Selection (PSO, GA, Hybrid PSO-GA)

This project focuses on intrusion detection using the CICIDS2017 dataset, where feature selection is performed using metaheuristic optimization algorithms (PSO, GA, and Hybrid PSO-GA) to reduce dimensionality while maintaining high classification performance with a Random Forest model.

---

##  Folder Structure
 Directory | Description 
|-----------|-------------
| `CXICIDS2017` | Original CICIDS2017 dataset (unprocessed) 
| `preprocessed_data` | Cleaned data 
| `sample_data` | 5%  sample 
| `GA/PSO/GWO/GA+PSO notebooks` | Jupyter notebooks 
| `conference paper` | Conference paper
| `video` | Project explanation 


---

Important Notes

- The project includes a **5% sample dataset** for quick execution and testing.
- For full experiments, use the **complete CICIDS2017 dataset**.
- You MUST update the dataset file paths in the notebooks:
  - `sample_data/` path 
  - `full_dataset/` path 
- Ensure all paths point to the correct local directory where the datasets are stored before running any notebook.

---

##  How to Use

1. Open the required notebook in Jupyter Notebook.
2. Set the dataset path (sample and full).
3. Run cells sequentially.
4. View results in the output cells.
---

##  Summary
This project demonstrates how metaheuristic algorithms (PSO, GA, and Hybrid PSO-GA) can effectively reduce feature selected while improving or maintaining intrusion detection performance using a Random Forest classifier.
