# AI-Enabled Phishing Detection System

The AI-Enabled Phishing Detection System is designed to detect and prevent phishing attempts using machine learning algorithms. This project integrates a front-end interface with JavaScript and a back-end that leverages AI for phishing detection, ensuring user safety against phishing attacks.

## Features

- **Real-Time Phishing Detection**: Automatically detects phishing attempts based on user inputs (e.g., email content, links).
- **AI-Powered Classifier**: Uses machine learning models trained on phishing datasets to distinguish between legitimate and phishing attempts.
- **User-Friendly Interface**: Front-end built with JavaScript to provide a seamless user experience.
- **Customizable Alerts**: Sends warnings or notifications if a phishing attempt is detected.

## Requirements

To run this project, you need the following:

- Python 3.x
- `Flask` (for back-end API)
- `pandas`
- `scikit-learn`
- `JavaScript` for front-end
- `HTML/CSS` for the interface

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/phishing-detection-system.git
    cd phishing-detection-system
    ```

2. Install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up the phishing detection model by running:
    ```bash
    python train_model.py
    ```

4. Run the Flask back-end:
    ```bash
    python app.py
    ```

5. Open the `index.html` file in your browser to access the front-end interface.

## How it Works

1. **Input**: Users can input email content or suspicious URLs through the front-end interface.
2. **Detection**: The system analyzes the input using a machine learning model (trained using phishing datasets).
3. **Output**: The system classifies the input as either **legitimate** or **phishing** and alerts the user.

## Model Training

The machine learning model is trained using the following techniques:

- **Text Vectorization**: Convert email or URL content into numerical features using `TF-IDF` or `CountVectorizer`.
- **Classification Algorithm**: A Support Vector Machine (SVM) or Random Forest model is used for classification.
- **Training Dataset**: The model is trained on a phishing dataset containing both legitimate and phishing samples.

## Customization

You can extend the system by:

- Adding more features for detection, such as domain reputation or historical phishing data.
- Integrating a database to store and track detected phishing attempts.
- Enhancing the model's accuracy by incorporating additional training data.

## Known Issues

- The accuracy of the model depends on the quality of the training data.
- A strong internet connection is required for some front-end features (e.g., loading external JavaScript libraries).



