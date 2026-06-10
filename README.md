# Demand Prediction — Inventory Management Optimization

A machine learning project that predicts product demand for a retail store chain to optimize inventory management, reduce holding costs, and prevent stockouts. The project follows a complete project lifecycle — planning, design, implementation, and reporting.

> Built as **Assignment 2** of the **Joint Tech Internship Community Program** (Generative AI Consortium – MSME, SystimaNX IT Solutions Pvt Ltd.) — AI/ML Internship, Deep Learning track.

## 👥 Team

- S.K. Surya Prasath
- B. Suriya Balaji
- Pagalavan K.S

*BE Computer Science, Kongu Engineering College*

## 📁 Repository Contents

| File | Description |
|------|-------------|
| `Demand_prediction.ipynb` | Main Jupyter notebook — full pipeline from data exploration to blended model predictions |
| `product_demand_prediction_dataset.csv` | Dataset of 5,000 daily product–store sales records |
| `Assignment_2_phase1.pdf` | Planning phase document — task breakdown, timelines, and team collaboration plan |
| `Assignment_2_designphase -.pdf` | Design phase document — train/test split rationale, target error rates, and design decisions |
| `Assignment_2_report.pdf` | Final report — model performance, feature importance insights, and inventory recommendations |

## 📊 Dataset

Each record contains:

- **ProductID, StoreID, Date** — identifiers and time reference
- **Sales, Price, CompetitorPrice** — sales and pricing data
- **Promotion, Season, Holiday, DayOfWeek, Weather** — categorical/contextual factors
- **EconomicIndicator, StockLevel** — macro and inventory signals
- **Demand** — target variable

## 🔍 Project Workflow

1. **Data Exploration** — summary statistics, data types, missing value checks
2. **Outlier Handling** — capping the `Demand` target at the 95th percentile
3. **Feature Engineering**
   - Lag features: `Lag_1`, `Lag_7`, `Lag_14`, `Lag_30`
   - Moving averages: 7-day, 14-day, and 30-day windows
   - Interaction term: `Price × Promotion`
   - Temporal features: day of year, and cyclical sine/cosine encoding of day of week
4. **Preprocessing** — `ColumnTransformer` pipeline with `StandardScaler` for numerical features and `OneHotEncoder` for categorical features
5. **Train-Test Split** — 80/20 split (per the design phase document)
6. **Model Training & Hyperparameter Tuning** — `RandomizedSearchCV` over three gradient-boosted/ensemble models:
   - LightGBM
   - XGBoost
   - Random Forest
7. **Model Evaluation** — MAE (target: < 50, per design doc), MSE, and R²
8. **Blended Predictions** — ensemble averaging across the best tuned models, which further improved MAE
9. **Feature Importance** — visualization of the most influential features (LightGBM)

## 💡 Key Insights

- Temporal features (day of week, holidays) play a crucial role in demand prediction
- Promotions are linked to demand spikes — important for marketing-aware inventory planning
- Clear seasonal patterns require proactive inventory adjustments
- Blending model predictions yields more reliable forecasts than any single model

## 📋 Recommendations (from the report)

- Retrain the model regularly with new data and use blended predictions for planning
- Adjust safety stock levels based on predicted demand variability
- Implement dynamic reordering that accounts for lead times and forecasted demand
- Align inventory and promotion planning using historical promotion effectiveness

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook / JupyterLab

### Installation

```bash
# Clone the repository
git clone https://github.com/surya2377/Joint-Tech-Internship-Community-Program-assignment.git
cd Joint-Tech-Internship-Community-Program-assignment

# Install dependencies
pip install pandas numpy matplotlib scikit-learn lightgbm xgboost jupyter
```

### Usage

1. Launch Jupyter:
   ```bash
   jupyter notebook
   ```
2. Open `Demand_prediction.ipynb`
3. Update the dataset path in the data loading cell to the local file:
   ```python
   df = pd.read_csv('product_demand_prediction_dataset.csv')
   ```
4. Run all cells in order. (Note: `RandomizedSearchCV` tuning may take several minutes.)

## 🛠️ Tech Stack

- **Python** — core language
- **pandas / NumPy** — data manipulation
- **scikit-learn** — preprocessing pipelines, tuning, and evaluation
- **LightGBM / XGBoost** — gradient boosting models
- **Matplotlib** — visualization

## 📄 License

This project was created for educational purposes as part of an internship assignment.
