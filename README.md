
# 🏛️ Bank-Predicting-Ticket-Resolution-with-NLP

-----

## "Ever felt like your bank complaint was just another drop in the ocean? Our system helps banks hear you louder and clearer\!"

Sometimes, the wait for a ticket resolution feels just like a long line at the bank. Our project aims to speed things up, even if it's just by giving us a heads-up.

-----

## Features

  * **Text Analysis**: Dive deep into complaint narratives with advanced NLP processing.
  * **Prediction Engine**: Leverage a robust **Logistic Regression** model for accurate resolution predictions.
  * **Probability Scores**: Gain confidence with detailed probability scores for each prediction.
  * **Batch Processing**: Efficiently analyze multiple complaints simultaneously.
  * **Real-time Predictions**: Get instant insights on new complaints, empowering swift action.

-----

## Tech Stack & The Brains Behind the Operation

Here are the core technologies powering the Bank Complaint Prediction System:

<img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"><img alt="Python" src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"><img alt="NLTK" src="https://img.shields.io/badge/NLTK-464A53?style=for-the-badge&logo=n-logo&logoColor=white"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"><img alt="NumPy" src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white"><img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3882C7?style=for-the-badge&logo=matplotlib&logoColor=white"><img alt="Flask" src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"><img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white"><img alt="HTML5" src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white"><img alt="CSS3" src="https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white"><img alt="NLTK" src="https://img.shields.io/badge/NLTK-464A53?style=for-the-badge&logo=n-logo&logoColor=white">
<img alt="Pandas" src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img alt="NumPy" src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3882C7?style=for-the-badge&logo=matplotlib&logoColor=white">
<img alt="Flask" src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white">
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">

-----

## Project Structure (A Peek Under the Hood)

```
your_project_folder/
├── app.py                      # The Flask web application for predictions
├── save_model.py               # Script to train and save the model/vectorizer
├── model.pkl                   # Saved Logistic Regression model
├── Consumer_Complaints.csv     # Your dataset of bank complaints
└── templates/
    └── index.html              # Frontend for the web interface
```

-----

## Quick Start: Get It Running in No Time

### Prerequisites

  * Python 3.8+
  * `pip` package manager
  * NLTK data files (`stopwords`, `wordnet`, `omw-1.4`)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/bank-complaint-prediction.git
    cd bank-complaint-prediction
    ```
2.  **Create and activate your virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate       # On Windows: venv\Scripts\activate
    ```
3.  **Install project dependencies**
    ```bash
    pip install flask pandas numpy scikit-learn nltk
    ```
4.  **Download NLTK essential data**
    ```bash
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
    ```

### Training the Model (The Brains Learn)

Before making predictions, you need to train and save your model.

```bash
python save_model.py
```

### Running the Web Application (Bringing It to Life\!)

Once the model is trained, fire up the Flask application:

```bash
python app.py
```

Then, point your browser to **`http://localhost:5000`** to interact with the prediction interface.

-----

## User Interface

\<div align="center"\>
\<img src="[https://github.com/vidhi-sys/Bank-Predicting-Ticket-Resolution-with-NLP/blob/main/image.png](https://www.google.com/search?q=https://github.com/vidhi-sys/Bank-Predicting-Ticket-Resolution-with-NLP/blob/main/image.png)" alt="A screenshot of the bank complaint prediction web interface." width="500"\>
\</div\>

Here's a glimpse of the web interface where users can submit their bank complaints for prediction.

-----

## Model Performance: How Smart Is It?

Our current Logistic Regression model is quite adept, achieving:

  * **Accuracy**: Approximately **85%** on unseen test data.
  * **Precision**: **0.87** for correctly identifying "Solved" complaints.
  * **Recall**: **0.83** for capturing most of the actual "Solved" complaints.
  * **F1-Score**: An overall balanced **0.85**.

This means our system is not just guessing; it's making informed decisions\!

-----

## API Usage: Integrate with Ease

The Flask application exposes a user-friendly RESTful API for both single and batch predictions.

### Single Complaint Prediction

```python
import requests

# Example complaint text
complaint_text = 'The bank charged unexpected fees on my account without notification. This is unacceptable.'

response = requests.post('http://localhost:5000/predict', json={'complaint': complaint_text})
print(response.json())
```

### Batch Complaint Prediction

```python
import requests

complaints_list = [
    "Issue with mortgage payment processing, it's consistently late due to bank error.",
    "Unauthorized credit card transaction appeared on my statement, I did not make this purchase.",
    "Problem with online banking access, I cannot log in to view my accounts."
]

response = requests.post('http://localhost:5000/batch_predict', json={'complaints': complaints_list})
print(response.json())
```

-----

## Prediction Results: What You Get

The API returns detailed predictions, including the predicted class, confidence level, and probabilities for each outcome:

```json
{
    "prediction": "Solved",
    "confidence": 0.87,
    "probabilities": {
        "solved": 0.87,
        "not_solved": 0.13
    }
}
```

-----

## How It Works: The Workflow at a Glance

1.  **Text Preprocessing**: We meticulously clean, tokenize, lemmatize, and remove stopwords from each complaint.
2.  **Feature Extraction**: The cleaned text is transformed into numerical vectors using **TF-IDF**, capturing word importance and context (even with n-grams\!).
3.  **Model Prediction**: These features are fed into our trained **Logistic Regression** classifier, which makes its prediction.
4.  **Result Interpretation**: The model's output is then translated into human-readable predictions, complete with confidence scores and probabilities.

-----

## Data Source: Fueling the Intelligence

The backbone of our model's intelligence comes from the comprehensive **CFPB Consumer Complaint Database**. This treasure trove contains thousands of real-world banking complaints and their corresponding resolution statuses, allowing our model to learn from actual customer experiences.

-----

## Contributing: Join the Mission\!

We're always looking for fellow innovators to enhance this system\! Your contributions, ideas, and feedback are incredibly valuable.

1.  Fork the project.
2.  Create your feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request, outlining your fantastic additions\!

-----

## License

This project is open-source and available under the **MIT License**. See the `LICENSE` file for full details.

-----

## Acknowledgments

  * A huge shout-out to the **Consumer Financial Protection Bureau (CFPB)** for making the invaluable complaint data publicly available.
  * Gratitude to the **Scikit-learn** and **NLTK** communities for their robust and well-documented libraries.
  * Thanks to all open-source contributors for maintaining the essential Python packages that make projects like this possible.

***Note**: This model is built for educational and demonstration purposes. While highly accurate, always validate its predictions with human domain experts and established business protocols before making critical operational decisions.*
