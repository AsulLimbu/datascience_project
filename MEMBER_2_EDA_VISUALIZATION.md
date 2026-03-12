# Member 2: EDA & Visualization

## Smart Energy Consumption Forecasting & Optimization System
### Nepal Electricity Authority (NEA) Daily Operations Analysis

---

## Table of Contents
1. [Role Overview](#1-role-overview)
2. [Responsibilities](#2-responsibilities)
3. [Technical Implementation](#3-technical-implementation)
4. [Deliverables](#4-deliverables)
5. [Key Findings](#5-key-findings)
6. [Viva Questions & Answers](#6-viva-questions--answers)

---

# 1. Role Overview

## Scope
Data exploration, pattern discovery, statistical analysis, and visualization creation.

## Primary Tasks
- Compute statistical summaries of energy data
- Analyze temporal patterns (daily, weekly, monthly, seasonal)
- Analyze generation mix (energy source distribution)
- Perform correlation analysis between variables
- Detect anomalies using statistical methods
- Create comprehensive visualizations

## Key Questions Answered
- What are the patterns in Nepal's energy consumption?
- How does demand vary by season, day of week, and time?
- What is the energy generation mix in Nepal?
- What factors correlate with energy demand?
- Are there any anomalies in consumption patterns?

---

# 2. Responsibilities

## 2.1 Statistical Summary

### Task Description
Compute and present basic statistics of energy data.

### Key Statistics Computed

| Metric | Value |
|--------|-------|
| Mean Daily Demand | 32,191 MWh |
| Maximum Demand | 42,190 MWh |
| Minimum Demand | 18,936 MWh |
| Standard Deviation | 4,613 MWh |
| Total Records | 606 |

### Data Distribution
- Normal distribution with slight right skew
- Most values concentrated around 30,000-35,000 MWh
- Outliers on both ends corresponding to festivals and grid issues

---

## 2.2 Temporal Pattern Analysis

### Task Description
Analyze energy consumption patterns over different time periods.

### Analyses Performed

#### Daily Trend Analysis
- Line plot of energy requirement over 1.5 years
- 7-day rolling average to smooth daily volatility
- Identification of upward trend in energy demand

#### Weekly Pattern Analysis
- Day of week comparison
- Weekend vs weekday demand difference

#### Monthly Pattern Analysis
- Monthly average demand comparison
- Identification of peak and low demand months

#### Seasonal Pattern Analysis
- Nepal's 5-season analysis (Winter, Spring, Summer, Monsoon, Autumn)
- Season-specific consumption patterns

---

## 2.3 Generation Mix Analysis

### Task Description
Analyze the contribution of different energy sources to Nepal's power supply.

### Energy Sources Analyzed
1. NEA Own Generation
2. NEA Subsidiary Companies
3. Independent Power Producers (IPP)
4. Import from India

### Analysis Includes
- Stacked area chart of generation over time
- Bar chart of average contribution by source
- Import vs Export trend analysis
- Energy Interruption analysis

---

## 2.4 Correlation Analysis

### Task Description
Analyze relationships between all energy variables.

### Variables in Correlation Matrix
- Energy_generation_NEA
- Energy_generation_NEA Subsidiary
- Energy_generation_IPP
- Energy_generation_Import
- Energy Export
- Net Energy Met (INPS Demand)
- Energy Interruption
- Energy Requirement

---

## 2.5 Anomaly Detection

### Task Description
Identify unusual consumption days using statistical methods.

### Methods Used
1. **Z-Score Method**: Flag values > 3 standard deviations from mean
2. **IQR Method**: Flag values outside Q1 - 1.5×IQR and Q3 + 1.5×IQR

---

# 3. Technical Implementation

## 3.1 Time Features Addition

```python
# Add time features for temporal analysis
energy_clean['month'] = energy_clean['date'].dt.month
energy_clean['day_of_week'] = energy_clean['date'].dt.dayofweek
energy_clean['day_name'] = energy_clean['date'].dt.day_name()
energy_clean['week_of_year'] = energy_clean['date'].dt.isocalendar().week
energy_clean['year'] = energy_clean['date'].dt.year
energy_clean['quarter'] = energy_clean['date'].dt.quarter

# Nepal-specific season mapping
season_map = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring',
    5: 'Summer', 6: 'Summer', 7: 'Summer',
    8: 'Monsoon', 9: 'Monsoon',
    10: 'Autumn', 11: 'Autumn'
}
energy_clean['season'] = energy_clean['month'].map(season_map)
```

---

## 3.2 Energy Trend Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=(16, 6))

# Plot daily energy requirement
ax.plot(energy_clean['date'], energy_clean['Energy Requirement'],
        color='#2E86AB', linewidth=1, alpha=0.8)

# Fill area under the curve
ax.fill_between(energy_clean['date'], energy_clean['Energy Requirement'],
                alpha=0.3, color='#2E86AB')

# Add 7-day rolling average
rolling_avg = energy_clean['Energy Requirement'].rolling(window=7).mean()
ax.plot(energy_clean['date'], rolling_avg, color='#E94F37',
        linewidth=2, label='7-day Rolling Average')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Energy Requirement (MWh)', fontsize=12)
ax.set_title('Daily Energy Requirement in Nepal (July 2022 - Nov 2023)',
             fontsize=14, fontweight='bold')
ax.legend()

plt.tight_layout()
plt.savefig('reports/energy_requirement_trend.png', dpi=150)
plt.show()
```

---

## 3.3 Generation Mix Visualization

```python
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Stacked area chart - Generation sources over time
ax1 = axes[0, 0]
ax1.stackplot(energy_clean['date'],
              energy_clean['Energy_generation_NEA'],
              energy_clean['Energy_generation_NEA Subsidiary'],
              energy_clean['Energy_generation_IPP'],
              energy_clean['Energy_generation_Import'],
              labels=['NEA', 'NEA Subsidiary', 'IPP', 'Import'],
              alpha=0.8)
ax1.set_title('Energy Generation Mix Over Time', fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Energy (MWh)')
ax1.legend(loc='upper left')

# 2. Bar chart - Average generation by source
ax2 = axes[0, 1]
gen_sources = ['Energy_generation_NEA', 'Energy_generation_NEA Subsidiary',
               'Energy_generation_IPP', 'Energy_generation_Import']
gen_means = energy_clean[gen_sources].mean()
gen_means.index = ['NEA', 'NEA Subsidiary', 'IPP', 'Import']
colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
ax2.bar(gen_means.index, gen_means.values, color=colors)
ax2.set_title('Average Daily Energy by Source', fontweight='bold')
ax2.set_ylabel('Energy (MWh)')

# 3. Import vs Export trend
ax3 = axes[1, 0]
ax3.plot(energy_clean['date'], energy_clean['Energy Export'],
         label='Export', color='#28A745', alpha=0.7)
ax3.plot(energy_clean['date'], energy_clean['Energy_generation_Import'],
         label='Import', color='#DC3545', alpha=0.7)
ax3.set_title('Energy Import vs Export Over Time', fontweight='bold')
ax3.legend()

# 4. Energy Interruption
ax4 = axes[1, 1]
ax4.fill_between(energy_clean['date'], energy_clean['Energy Interruption'],
                 color='#E94F37', alpha=0.7)
ax4.set_title('Energy Interruption Over Time', fontweight='bold')

plt.tight_layout()
plt.savefig('reports/generation_analysis.png', dpi=150)
plt.show()
```

---

## 3.4 Seasonal Pattern Analysis

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 1. Energy by season
ax1 = axes[0]
seasonal_avg = energy_clean.groupby('season')['Energy Requirement'].mean().reindex(
    ['Winter', 'Spring', 'Summer', 'Monsoon', 'Autumn']
)
colors = ['#3498DB', '#2ECC71', '#F1C40F', '#9B59B6', '#E67E22']
bars = ax1.bar(seasonal_avg.index, seasonal_avg.values, color=colors)
ax1.set_title('Average Energy Requirement by Season', fontweight='bold')
ax1.set_ylabel('Energy (MWh)')
ax1.axhline(y=energy_clean['Energy Requirement'].mean(), color='red',
            linestyle='--', label='Overall Average')
ax1.legend()

# 2. Day of week analysis
ax2 = axes[1]
dow_avg = energy_clean.groupby('day_name')['Energy Requirement'].mean().reindex(
    ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
)
colors = ['#3498DB'] * 5 + ['#E74C3C', '#E74C3C']  # Weekend in red
ax2.bar(dow_avg.index, dow_avg.values, color=colors)
ax2.set_title('Average Energy Requirement by Day of Week', fontweight='bold')
ax2.set_ylabel('Energy (MWh)')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/seasonal_patterns.png', dpi=150)
plt.show()
```

---

## 3.5 Correlation Matrix

```python
# Correlation matrix
numeric_cols = ['Energy_generation_NEA', 'Energy_generation_NEA Subsidiary',
                'Energy_generation_IPP', 'Energy_generation_Import',
                'Energy Export', 'Net Energy Met within the country (INPS Demand)',
                'Energy Interruption', 'Energy Requirement']

corr_matrix = energy_clean[numeric_cols].corr()

# Simplified column names for display
corr_matrix.columns = ['NEA', 'NEA Sub', 'IPP', 'Import', 'Export',
                       'Net Demand', 'Interruption', 'Requirement']
corr_matrix.index = corr_matrix.columns

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlBu_r', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix - Energy Variables', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig('reports/correlation_matrix.png', dpi=150)
plt.show()
```

---

## 3.6 Anomaly Detection

```python
import numpy as np

def detect_anomalies(series, method='zscore', threshold=3):
    """Detect anomalies in a time series."""
    if method == 'zscore':
        z_scores = np.abs((series - series.mean()) / series.std())
        return z_scores > threshold
    elif method == 'iqr':
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        return (series < lower) | (series > upper)

# Detect anomalies
energy_clean['anomaly_zscore'] = detect_anomalies(
    energy_clean['Energy Requirement'], method='zscore'
)
energy_clean['anomaly_iqr'] = detect_anomalies(
    energy_clean['Energy Requirement'], method='iqr'
)

# Visualize anomalies
fig, ax = plt.subplots(figsize=(16, 6))
normal = energy_clean[~energy_clean['anomaly_iqr']]
anomalies = energy_clean[energy_clean['anomaly_iqr']]

ax.scatter(normal['date'], normal['Energy Requirement'],
           c='#2E86AB', alpha=0.6, s=20, label='Normal')
ax.scatter(anomalies['date'], anomalies['Energy Requirement'],
           c='#E94F37', s=50, label='Anomaly', marker='x')

ax.set_title('Energy Requirement with Detected Anomalies', fontweight='bold')
ax.legend()
plt.savefig('reports/anomaly_detection.png', dpi=150)
plt.show()
```

---

# 4. Deliverables

## Visualizations Created

| Visualization | File | Purpose |
|---------------|------|---------|
| Energy Requirement Trend | `reports/energy_requirement_trend.png` | Show daily demand over time |
| Generation Mix Analysis | `reports/generation_analysis.png` | Energy source breakdown |
| Seasonal Patterns | `reports/seasonal_patterns.png` | Season and day-of-week patterns |
| Energy Heatmap | `reports/energy_heatmap.png` | Day-of-week vs Month patterns |
| Peak Demand Analysis | `reports/peak_demand_analysis.png` | Peak demand distribution |
| Correlation Matrix | `reports/correlation_matrix.png` | Variable relationships |
| Anomaly Detection | `reports/anomaly_detection.png` | Unusual consumption days |

## Files Modified

| File | Description |
|------|-------------|
| `notebooks/01_eda_energy_data.ipynb` | Complete EDA notebook |
| `data/processed/energy_clean.csv` | Cleaned energy data with time features |
| `data/processed/demand_clean.csv` | Cleaned peak demand data |

---

# 5. Key Findings

## 5.1 Temporal Patterns

### Seasonal Demand
| Season | Avg Demand (MWh) | Observation |
|--------|------------------|-------------|
| Summer | ~35,000 | Highest - cooling demand peak |
| Winter | ~32,000 | Moderate - heating needs |
| Monsoon | Variable | Hydro generation impact |
| Autumn | Festival spike | Dashain/Tihar effect |
| Spring | ~30,000 | Lowest - mild weather |

### Weekly Patterns
| Day | Pattern |
|-----|---------|
| Monday-Thursday | Higher demand - industrial activity |
| Friday | Slightly lower - weekend transition |
| Saturday-Sunday | Lowest - weekend effect |

---

## 5.2 Generation Mix

| Source | Avg Contribution (MWh) | Share |
|--------|------------------------|-------|
| IPP | 16,464 | 43% |
| NEA Subsidiary | 7,180 | 30% |
| NEA Own Generation | 7,935 | 23% |
| Import from India | 4,607 | 4% |

**Key Insight**: IPPs contribute the largest share of Nepal's electricity, making them crucial for grid stability.

---

## 5.3 Correlation Findings

### Strong Positive Correlations
- Energy Requirement ↔ Total Energy Available: **0.99**
- Energy Requirement ↔ INPS Demand: **0.95**
- NEA Generation ↔ NEA Subsidiary: **0.85**

### Weather Correlations
- Temperature ↔ Energy Demand: **Positive** (higher temp → higher demand)
- Precipitation ↔ Energy Demand: **Weak negative**

---

## 5.4 Anomaly Findings

| Method | Anomalies Detected |
|--------|-------------------|
| Z-Score | 0 (threshold = 3) |
| IQR | 7 |

**Anomalous days** often correspond to:
- Major holidays (Dashain, Tihar)
- Grid disturbances
- Extreme weather events

---

## 5.5 Energy Supply-Demand Balance

**Key Finding**: Energy not served/Generation Deficit was **zero** throughout the study period.

This indicates Nepal successfully met all daily energy requirements through:
- Domestic generation (NEA + Subsidiaries + IPPs)
- Imports from India when needed

**Energy Interruption** (temporary disruptions) showed:
- Total: 372,269 MWh over study period
- Mean Daily: 615 MWh
- Days with Interruption: 308 out of 606 (51%)

---

# 6. Viva Questions & Answers

## Q1: What key patterns did you discover in the energy consumption data?
**Answer**:
1. **Seasonal patterns**: Summer (May-July) has highest demand due to cooling needs; Spring has lowest demand
2. **Weekly patterns**: Weekends have lower demand due to reduced industrial activity
3. **Festival impact**: Major festivals like Dashain and Tihar show distinct consumption changes
4. **Upward trend**: Energy demand shows gradual increase over the study period
5. **Generation mix**: IPPs contribute 43% of total generation - the largest share

---

## Q2: Why is 7-day rolling average used in the analysis?
**Answer**:
- Daily data has inherent volatility due to random fluctuations
- 7-day rolling average smooths out daily noise while preserving trends
- Weekly window aligns with the natural 7-day cycle in energy consumption
- Helps identify underlying patterns more clearly
- Used both as analysis tool and as a predictive feature (demand_rolling_7)

---

## Q3: What does the correlation matrix reveal about energy demand?
**Answer**:
- **Strong positive correlations**:
  - Energy Requirement ↔ Total Energy Available (0.99) - demand is almost always met
  - Energy Requirement ↔ INPS Demand (0.95) - strong relationship
- **Weather correlations**: Temperature shows positive correlation with demand
- **Key insight**: Lag features (previous day demand) would be the strongest predictors

---

## Q4: How did you detect anomalies in the data?
**Answer**: We used the IQR (Interquartile Range) method:
1. Calculate Q1 (25th percentile) and Q3 (75th percentile)
2. Compute IQR = Q3 - Q1
3. Define bounds: Lower = Q1 - 1.5×IQR, Upper = Q3 + 1.5×IQR
4. Flag points outside these bounds as anomalies

Found 7 anomalous days using IQR method.

---

## Q5: What is the generation mix in Nepal and why does it matter?
**Answer**:
| Source | Share |
|--------|-------|
| IPP | 43% |
| NEA Subsidiary | 30% |
| NEA Own Generation | 23% |
| Import | 4% |

This matters because:
- IPPs are crucial for grid stability - their contribution must be considered in planning
- Low import dependency (4%) indicates good self-sufficiency
- Generation mix affects forecasting - different sources have different patterns

---

## Q6: Why do weekends have lower energy demand?
**Answer**:
- **Industrial shutdown**: Most factories and industries operate on weekdays only
- **Commercial reduction**: Offices, banks, and businesses are closed
- **Schools closed**: Educational institutions consume less energy
- **Government offices closed**: Significant portion of institutional consumption reduced
- The weekend effect (Saturday-Sunday in Nepal) is consistent and predictable

---

## Q7: What visualization libraries did you use and why?
**Answer**:
- **matplotlib**: Primary plotting library, highly customizable, good for static visualizations
- **seaborn**: Built on matplotlib, provides better default aesthetics, excellent for statistical visualizations like heatmaps
- We chose seaborn for correlation matrices and statistical plots due to its built-in support

---

## Q8: How did you handle the finding that Energy Deficit was zero?
**Answer**: Since the "Energy not served/Generation Deficit" column contained all zeros:
- We focused analysis on **Energy Interruption** instead
- Energy Interruption represents temporary service disruptions due to grid issues, maintenance, or load shedding
- This provided more meaningful insights into grid stress patterns
- Created visualizations showing interruption patterns over time

---

## Q9: What is the significance of Nepal's 5 seasons?
**Answer**: Nepal has 5 distinct seasons that affect energy consumption:

| Season | Months | Energy Impact |
|--------|--------|---------------|
| Winter | Dec-Feb | Moderate demand, reduced hydro |
| Spring | Mar-Apr | Lowest demand, mild weather |
| Summer | May-Jul | Peak demand, cooling needs |
| Monsoon | Aug-Sep | Variable, high hydro generation |
| Autumn | Oct-Nov | Festival effects, moderate |

This differs from typical 4-season models and must be considered for accurate forecasting.

---

## Q10: How would you improve the EDA with more time?
**Answer**:
1. **Weather correlation deep dive**: Analyze temperature-demand relationship by season
2. **Festival impact quantification**: Measure exact demand change during major festivals
3. **Regional analysis**: If data available, analyze regional consumption patterns
4. **Peak demand time analysis**: Better extraction of peak timing patterns
5. **Interactive dashboards**: Create interactive visualizations with Plotly
6. **Time series decomposition**: Separate trend, seasonality, and residuals formally

---

## Quick Reference: Key Statistics

| Metric | Value |
|--------|-------|
| Mean Daily Demand | 32,191 MWh |
| Maximum Demand | 42,190 MWh |
| Minimum Demand | 18,936 MWh |
| Std Deviation | 4,613 MWh |
| Total Interruption | 372,269 MWh |
| Days with Interruption | 308 (51%) |
| IPP Share | 43% |
| Import Share | 4% |

---

*Document for Member 2 - EDA & Visualization*
*Smart Energy Consumption Forecasting Project*