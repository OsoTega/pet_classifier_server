# Pet Classification Flask API

This repository contains a Flask API for classifying pet images using a pre-trained model.

## Author Information

- **Author**: Tega Osowa
- **Email**: stevetega.osowa11@gmail.com
- **GitHub**: [OsoTega](https://github.com/OsoTega)

## Instructions

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/OsoTega/pet_classifier_server.git
   ```

2. Install dependencies:

   ```bash
   pip install flask flask_cors PIL opencv
   ```

### Running the Flask API

To start the Flask API, run the following command:

```bash
python app.py
```

The API will be accessible at `http://localhost:5000`.

### Endpoints

- `GET /api/hello`: Returns a simple greeting message.
- `POST /api/classify`: Accepts JSON data containing image data and returns the classification prediction.

## Usage

You can send a POST request to the `/api/classify` endpoint with JSON data containing image data. The API will return the classification prediction.

Example usage:

```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [0.1, 0.2, ..., 0.9]}' http://localhost:5000/api/classify
```

### Pre-trained Model

The Flask API uses a pre-trained model stored in `pet_classification.h5` for classifying pet images. Make sure this file is present in the same directory as `app.py`.

## Previous Python Code

The repository also contains Python code for training the pet classification model. To train the model and generate the pre-trained `pet_classification.h5` file, follow the instructions below:

1. Ensure you have TensorFlow installed:

   ```bash
   pip install tensorflow
   ```

2. Run the training script:

   ```bash
   python image_deep_learning.py
   ```
