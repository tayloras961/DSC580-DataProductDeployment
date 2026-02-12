# Simple Health Analytics Data Product

## Overview
This project is a lightweight web-based health analytics data product developed as part of a graduate data science course. The application allows users to upload health-related datasets and perform exploratory data analysis, preprocessing, visualization, and simple non-clinical risk assessment through an interactive browser interface.

The product demonstrates a complete data analytics workflow, including ingestion, cleaning, analysis, visualization, and report generation.
---

## Features
The application supports the following capabilities:

- Upload CSV datasets
- Explore dataset structure and quality
- Apply common data cleaning methods
- Perform selectable analysis methods
- Visualize trends and relationships
- Generate downloadable HTML reports
- Export cleaned datasets
- Interactive workflow navigation
- Built-in help documentation

### Analysis Methods
- Trend summaries
- Correlation scans
- Simple rule-based health risk scoring

### Visualization Options
- Time-series trends
- Rolling averages
- Histograms
- Scatterplots
- Correlation heatmaps

Reports summarize dataset characteristics, analysis results, and simplified health guidance.

---

## Installation and Local Execution

### Requirements
Python 3.9 or later is recommended.

Required Python libraries:

- streamlit
- pandas
- numpy
- matplotlib
- requests

Install dependencies:

```bash
pip install -r requirements.txt
```
## Run the Application

Launch locally using:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser.

---

## Data Format Requirements

Datasets must be provided in CSV format with column headers.

Recommended columns include:

- `date`
- `weight_lbs`
- `steps`
- `calories`
- `sleep_hours`

Additional columns may also be included.

Template and sample CSV files are available for download within the application.

---

## Deployment

The application is deployed on a cloud-based platform, allowing browser access without local installation. All features available locally are also accessible through the hosted version.

Users access the deployed product through the URL provided in course submission materials.

---

## Usage Workflow

Recommended workflow:

1. Load dataset  
2. Explore data  
3. Apply cleaning if needed  
4. Run analysis  
5. Generate visualizations  
6. Create and download reports  

Users are encouraged to follow these steps sequentially for best results.

---

## Security and Privacy Notes

Uploaded data is processed only in memory and is not permanently stored by the system. Users should avoid uploading sensitive personal health information, as this application is intended strictly for demonstration purposes.

---

## Testing

Testing was performed using sample datasets to verify:

- Data ingestion  
- Cleaning operations  
- Analysis execution  
- Visualization rendering  
- Report generation  
- System stability  

Testing results are documented in course submission materials.

---

## Limitations

This product is intentionally narrow in scope and does not replace clinical health analytics tools. Risk scoring and guidance outputs are simplified and intended only for educational demonstration.

---

## Future Improvements

Potential future enhancements include:

- Improved predictive modeling  
- Multi-user data persistence  
- User authentication  
- Expanded reporting features  
- Advanced analytics modules  

