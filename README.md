# Named Entity Recognition (NER) Web Application

A web application that identifies and classifies named entities (like **persons**, **locations**, and **organizations**) in user-provided text. This project uses a **Bidirectional LSTM model (Bi-LSTM)** built with **TensorFlow/Keras** and is served via a **Flask API**.

![Demo GIF of the application in use]
> *(Optional: It's highly recommended to record a short GIF of your app working and add it here!)*

---

## Features

* **Real-time NER:** Instantly highlights entities in the input text.
* **Bi-LSTM Model:** Utilizes a robust deep learning architecture for sequence tagging.
* **REST API Backend:** The model is served via a Flask API, decoupling the model from the frontend.
* **Interactive UI:** A clean and simple user interface built with HTML and Tailwind CSS.

---

## Tech Stack

* **Backend:** Python, Flask, TensorFlow/Keras
* **Frontend:** HTML, Tailwind CSS, JavaScript
* **Dataset:** Kaggle NER Dataset (`ner_dataset.csv`)

---

## How to Run Locally

Follow these steps to set up and run the project on your local machine.

### Prerequisites

* Python 3.8+
* Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/v-oathkeeper/named-entity-recognition-api.git
    cd named-entity-recognition-api
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the model (one-time setup):**
    This script will train the Bi-LSTM model and generate the necessary `model_weights.weights.h5` and `ner_mappings.json` files.
    ```bash
    python train_model.py
    ```

5.  **Run the Flask server:**
    ```bash
    python app.py
    ```
    The server will be running at `http://127.0.0.1:5000`.

6.  **View the application:**
    Open the `templates/index.html` file directly in your web browser.