# 🧠 Smart Data Analyzer

A full-featured **Streamlit** web app that combines **Statistics**, **Probability**, and **Vector Similarity** to help you understand any CSV dataset — built as a Weekly Python Project.

---

## 🚀 Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | 📥 Upload Dataset | Upload any CSV file |
| 2 | 📐 Statistical Summary | Mean, Median, Mode, Std Dev, Variance, Min, Max |
| 3 | 🎲 Probability Insights | Range-based probability + normal distribution fit |
| 4 | 🔢 Vector Similarity | Dot product & cosine similarity between two rows |
| 5 | 📊 Visual Charts | Histogram, bar chart, radar/spider chart |
| 6 | 💡 Auto Insights | Smart observations about skew, correlation, outliers |

---

## 🧠 Concepts Covered

### 📐 Statistics
- **Mean** – Average of all values in a column
- **Median** – Middle value when data is sorted
- **Mode** – Most frequently occurring value
- **Standard Deviation** – How spread out data is from the mean
- **Variance** – Square of standard deviation

### 🎲 Probability
- **Basic Probability** – P(x) = count in range / total values
- **Normal Distribution** – Fitting a bell curve to actual data using mean & std dev
- **Range Probability** – Probability of a value falling within a custom range

### 🔢 Vector Similarity
- **Dot Product** – Sum of element-wise products of two row vectors
- **Cosine Similarity** – Angle-based similarity score between −1 and 1

---

## 🛠️ Tech Stack

- [Python 3](https://www.python.org/)
- [Streamlit](https://streamlit.io/) – Web UI framework
- [Pandas](https://pandas.pydata.org/) – Data loading & manipulation
- [NumPy](https://numpy.org/) – Numerical operations & vector math
- [Matplotlib](https://matplotlib.org/) – Charts and visualizations
- [SciPy](https://scipy.org/) – Statistics (mode, normal distribution)

---

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/Ulas33/smart-data-analyzer.git
cd smart-data-analyzer
```

**2. Install dependencies**
```bash
pip install streamlit pandas numpy matplotlib scipy
```

**3. Run the app**
```bash
streamlit run smart_data_analyzer.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📁 Project Structure

```
smart-data-analyzer/
│
├── .gitignore
├── requirements.txt
├── smart_data_analyzer.py   # Main Streamlit app
└── README.md                # Project documentation
```

---

## 📖 How to Use

1. **Upload** a `.csv` file using the sidebar
2. **View** the full statistical summary table
3. **Select a column** to see metric cards and histogram
4. **Set a range** using the slider to compute probability P(x)
5. **Pick two row indices** to compute dot product and cosine similarity
6. **Read the insights** auto-generated at the bottom

---

## 📊 App Sections

### Section 1 — Statistical Summary
- Summary table for all numeric columns
- Interactive column selector with 6 metric cards
- Histogram with Mean and Median reference lines

### Section 2 — Probability Insights
- Custom range slider to define P(x) window
- Highlighted histogram showing values in range
- Normal distribution curve fitted to actual data

### Section 3 — Row Similarity (Dot Product)
- Select any two rows by index number
- Computes dot product and cosine similarity
- Side-by-side comparison table with differences
- Grouped bar chart for visual comparison
- Radar / Spider chart (shown when ≤ 10 columns)

### Section 4 — Auto-Generated Insights
- Skewness detection per column
- Missing value report
- Probability range summary
- Similarity interpretation (duplicate / similar / different)
- High-variance column flags
- Strong correlation detection between column pairs

---

## 📌 Sample Datasets to Try

You can test this app with any of these free datasets:

- [Iris Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv)
- [Titanic Dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv)
- [Tips Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv)

Download any as a `.csv` and upload directly into the app.

---

## 📚 Assignment Info

- **Type:** Weekly Project
- **Subject:** Python / Data Science / Math
- **Concepts:** Statistics, Probability, Linear Algebra (Vectors)
- **Tools:** Streamlit, Pandas, NumPy, Matplotlib, SciPy

---

## 📄 License

This project is open source and free to use for educational purposes.
