# Ticketing_system
Develop a machine learning pipeline that classifies customer support tickets by their issue type and urgency level, and extracts key entities (e.g., product names, dates, complaint keywords). 
# Ticket Processing System

## Overview

This project is an NLP-based ticket processing system that:
- Predicts ticket `issue_type` and `urgency_level` (classification)
- Extracts entities such as product names, dates, and complaint keywords from ticket text
- Provides an interactive Gradio web interface for real-time predictions

## Key Design Choices

- **Data Preprocessing:** 
  - Text normalization (lowercasing, removing special characters/numbers)
  - Tokenization, stopword removal, and lemmatization using NLTK
  - Missing value handling for required columns
- **Feature Engineering:** 
  - TF-IDF vectorization of cleaned ticket text
  - Additional features: ticket length and sentiment score (`nltk.vader`)
  - Features are combined using `scipy.sparse.hstack`
- **Modeling:** 
  - Two separate Random Forest classifiers for `issue_type` and `urgency_level`
  - Models trained and tested on an 80/20 split
- **Entity Extraction:** 
  - Regex-based extraction for product names and dates
  - Keyword matching for common complaint terms
- **Interface:** 
  - Gradio app for user-friendly web-based ticket processing

## Model Evaluation

- **Metrics Used:** 
  - Classification report including precision, recall, F1-score for both labels
  - Confusion matrix for error analysis

Example Output:
```
Issue Type Classification Report: 
              precision    recall  f1-score   support
...
Urgency Level Classification Report: 
              precision    recall  f1-score   support
...
```
- **Observations:**  
  - (Fill in with your actual metrics after running)

## Limitations

- **Data Bias:** The model is only as good as the labeled data provided. If the dataset is unbalanced, model performance may vary across classes.
- **Entity Extraction:** The approach is regex/keyword-based and may miss variations or unusual patterns.
- **Generalization:** Performance may decrease on tickets very different from the training set.
- **Dependency:** Requires NLTK downloads and may need internet access for first run.
- **Scalability:** Random Forest and TF-IDF are not optimal for very large datasets.

## Instructions to Run

1. **Install Requirements:**
   ```
   pip install pandas scikit-learn nltk textblob gradio openpyxl
   ```
2. **Prepare the Data:**
   - Place your XLS file in the correct path or adjust the file path in the script.

3. **Run the Script:**
   - Jupyter Notebook: Run all cells
   - Python Script: `python ticket_processing.py`

4. **Gradio Web App:**
   - The Gradio interface will launch in your browser.
   - Input raw ticket text to see the predicted issue type, urgency, and extracted entities.



## Example Usage

**Input:**
```
I ordered SmartWatch V2 but got PowerMax Battery instead. My order number is #65084.
```

**Output:**
```json
{
    "predicted_issue_type": "...",
    "predicted_urgency_level": "...",
    "extracted_entities": {
        "product_names": ["SmartWatch V2", "PowerMax Battery"],
        "dates": [],
        "complaints": []
    }
}
```

---

