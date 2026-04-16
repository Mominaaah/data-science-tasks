# Customer Segmentation Using Unsupervised Learning

Cluster mall customers by spending habits using K-Means and visualise segments with PCA and t-SNE. Each segment gets a tailored marketing strategy.


## Dataset

- **Source:** Mall Customers Dataset (200 records)
- **Features:** Customer ID, Gender, Age, Annual Income (k$), Spending Score (1–100)
- **No labels** — pure unsupervised learning task


## Project Structure

```
Task-2-Customer-Segmentation/
│
├── customer_segmentation.ipynb   # Main notebook
└── README.md                     # This file
```
## What This Notebook Covers

### 1. Exploratory Data Analysis
- Distribution plots for Age, Income, Spending Score split by gender
- Scatter plots: Age vs Spending, Income vs Spending
- Correlation heatmap
- Boxplots by gender

### 2. Preprocessing
- Feature selection (2-D and 3-D clustering versions)
- StandardScaler for 3-D feature space

### 3. Finding Optimal K
- **Elbow Method** — inertia (WCSS) vs number of clusters
- **Silhouette Score** — cluster cohesion and separation
- **Davies-Bouldin Score** — average cluster similarity

### 4. K-Means Clustering (K = 5)
- Fitted on Income × Spending Score (2-D, interpretable)
- Also fitted on Age + Income + Spending (3-D, richer)
- Cluster size and centroid analysis

### 5. Dimensionality Reduction
- **PCA** — linear projection with explained variance, scree plot, biplot
- **t-SNE** — non-linear projection for cluster validation
- Side-by-side PCA vs t-SNE comparison

### 6. Cluster Profiling
- Mean Age, Income, Spending per cluster
- Violin plots per feature per cluster
- Normalised heatmap of cluster profiles

### 7. Marketing Strategies
Tailored strategy for each of the 5 identified segments:

| Cluster | Segment | Key Strategy |
|---|---|---|
| 0 | High Income, Low Spending | Premium loyalty + trust building |
| 1 | Medium Income, Medium Spending | Rewards, bundles, social proof |
| 2 | High Income, High Spending | VIP retention + luxury positioning |
| 3 | Low Income, High Spending | BNPL, influencer marketing, flash sales |
| 4 | Low Income, Low Spending | Essential products, heavy discounts |

## Clustering Metrics

| Metric | Direction | Meaning |
|---|---|---|
| Silhouette Score | Higher = better | How well separated clusters are |
| Davies-Bouldin Score | Lower = better | Average similarity between clusters |
| Calinski-Harabasz Score | Higher = better | Ratio of between/within cluster variance |

## How to Run

**Step 1 — Install dependencies** (run Cell 0 or paste in terminal):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**Step 2 — Run all cells top to bottom**

The dataset downloads automatically — no manual download needed.

## Skills Demonstrated

- Unsupervised learning with K-Means
- Optimal cluster selection (Elbow, Silhouette, Davies-Bouldin)
- Dimensionality reduction with PCA and t-SNE
- Customer segmentation and profiling
- Data-driven marketing strategy development

