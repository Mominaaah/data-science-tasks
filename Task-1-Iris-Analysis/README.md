# Iris Dataset Explorer
### Exploratory Data Analysis with Python · Pandas · Seaborn · Matplotlib

## Overview

A complete **Exploratory Data Analysis (EDA)** of the Iris dataset — one of the most iconic datasets in data science, first introduced by British statistician **R. A. Fisher in 1936**. This project applies the full EDA pipeline: data loading, statistical summarization, and multi-layered visualization to extract meaningful insights about three iris species.

The entire analysis is built inside a **Jupyter Notebook in VS Code**, with every section documented using markdown for clarity and reproducibility.

## The Dataset

| Property       | Detail                                  |
|----------------|-----------------------------------------|
| Source         | UCI Machine Learning Repository         |
| Introduced by  | R. A. Fisher (1936)                     |
| Total Samples  | 150 — perfectly balanced, 50 per class  |
| Features       | 4 continuous numeric measurements       |
| Target Classes | 3 species                               |
| Missing Values | None                                    |

### Measurements Collected

| Feature       | Unit | Description                  |
|---------------|------|------------------------------|
| Sepal Length  | cm   | Length of the outer leaf     |
| Sepal Width   | cm   | Width of the outer leaf      |
| Petal Length  | cm   | Length of the inner petal    |
| Petal Width   | cm   | Width of the inner petal     |

### Species

| Species | Color Code | Characteristic |
|---------|------------|----------------|
| *Iris setosa* | 🟢 Teal | Smallest petals — perfectly separable |
| *Iris versicolor* | 🟠 Amber | Medium — slight overlap with virginica |
| *Iris virginica* | 🟣 Purple | Largest petals — highest measurements |

## Stack

| Tool | Role |
|------|------|
| `pandas` | Data loading, grouping, and statistical summaries |
| `numpy` | Numerical computation |
| `matplotlib` | Base chart rendering and layout |
| `seaborn` | Statistical plots — box, scatter, pair, heatmap |
| `scipy.stats` | One-way ANOVA significance testing |
| `sklearn.datasets` | Built-in Iris dataset loader |

## Project Structure

```
iris-dataset-explorer/
│
├── iris_notebook.ipynb      ← Full analysis notebook with markdown + code
├── iris_explorer.py         ← Standalone Python script version
└── README.md                ← This file
```

## Setup

**Clone and install:**

```bash
git clone https://github.com/yourusername/iris-dataset-explorer.git
cd iris-dataset-explorer
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

**Run the notebook:**

Open `iris_notebook.ipynb` in VS Code, select your Python interpreter, and run cells with `Shift + Enter`.

> Requires the **Jupyter extension** in VS Code.

---

## Analysis Breakdown

### Data Profiling
The dataset is loaded directly from `sklearn` — no external CSV required. After converting to a Pandas DataFrame, the data is profiled using `df.info()` and `df.describe()` to inspect types, null values, and overall statistical distributions. A per-species groupby mean reveals the numerical distance between classes before any visualization.

### Scatter Plots — Separability Analysis
Two scatter plots are generated: one for **petal dimensions** and one for **sepal dimensions**. The contrast between them is a core finding of this analysis — petal measurements create clean visual boundaries between species while sepal measurements produce substantial overlap between Versicolor and Virginica.

### Pair Plot — All Feature Combinations
A `seaborn.pairplot()` with KDE diagonals maps every possible feature combination (6 pairs) in a single grid. This is the fastest way to identify which feature pairs carry the most class-discriminating information without building a model.

### Histograms — Distribution Shape
Overlapping histograms per species are generated for all four features in a 2×2 grid. Petal features show **near-zero distribution overlap** for Setosa — a strong visual indicator of separability.

### Box Plots — Spread and Outliers
Grouped box plots visualize the median, interquartile range, and outliers for each feature across species. Virginica exhibits the widest spread in petal measurements; Setosa is the most compact.

### Correlation Heatmap — Feature Relationships

Pearson correlation coefficients computed across all feature pairs:

| Feature Pair                   | r        |
|--------------------------------|----------|
| Petal Length ↔ Petal Width     | **0.96** |
| Sepal Length ↔ Petal Length    | 0.87     |
| Sepal Length ↔ Petal Width     | 0.82     |
| Sepal Width ↔ Petal Length     | −0.43    |

The near-perfect correlation between petal length and petal width indicates **multicollinearity** — a critical consideration for any downstream classification modeling.

### Bar Chart — Class Mean Comparison
A grouped bar chart compares mean values across all three species for each feature simultaneously. Provides a clean quantitative summary of class separation without looking at individual points.

### ANOVA — Statistical Significance Testing

One-way ANOVA confirms whether differences between species are statistically significant or due to chance:

| Feature        | F-Statistic  | p-value    | Result         |
|----------------|--------------|------------|----------------|
| Sepal Length   | 119.26       | 1.67e-31   |  Significant |
| Sepal Width    | 49.16        | 4.49e-17   |  Significant |
| Petal Length   | **1180.16**  | 2.86e-91   |  Significant |
| Petal Width    | **960.01**   | 4.17e-85   |  Significant |

All four features reject the null hypothesis (p < 0.05). The substantially higher F-statistics for petal features reinforce their superiority as discriminating variables.

---

## Key Findings

**1. Petal features are the stronger discriminators.**
Petal length and width separate species far more cleanly than sepal measurements. Sepal features introduce significant overlap between Versicolor and Virginica.

**2. Setosa is linearly separable.**
Across every visualization — scatter, histogram, box plot, and pair plot — Setosa forms an isolated cluster. It can be identified with near-perfect accuracy using petal measurements alone.

**3. Versicolor and Virginica partially overlap.**
These two species share measurement ranges in all four features, though they remain statistically distinct. Any classifier built on this data will need to handle this boundary carefully.

**4. Strong multicollinearity in petal features (r = 0.96).**
Petal length and petal width grow together almost perfectly. For downstream modeling, using both may be redundant — dimensionality reduction or feature selection is advisable.

**5. All features are statistically significant.**
ANOVA results confirm that none of the four features are noise. Each one carries genuine class-discriminating information, validated at p < 0.001.

## Statistical Summary

| Feature       | Mean  | Std Dev | Min  | Median | Max  |
|---------------|-------|---------|------|--------|------|
| Sepal Length  | 5.84  | 0.83    | 4.30 | 5.80   | 7.90 |
| Sepal Width   | 3.06  | 0.44    | 2.00 | 3.00   | 4.40 |
| Petal Length  | 3.76  | 1.77    | 1.00 | 4.35   | 6.90 |
| Petal Width   | 1.20  | 0.76    | 0.10 | 1.30   | 2.50 |

## Author

**Momina Ramzan**

## Acknowledgements

- R. A. Fisher — original dataset and statistical methodology (1936)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris) — public dataset hosting
- Scikit-learn contributors — built-in dataset access via `sklearn.datasets`
