# Flask Coin Detection App

## Description
This project is a Flask-based application designed to detect coins in images using a machine learning model.

## Installation
Clone the repository and build the Docker image:

git clone https://github.com/alperenerol/coin-detection-flask-app.git

cd coin-detection-flask-app

docker build -t flask-app .      

## Running the Application
Run the Docker container using:
docker run -p 5001:5001 flask-app

## Usage
Submit an image for processing:

curl -X POST -F 'file=@coin-dataset/175_1479423456_jpg.rf.0723ceef6a241da65f4f36db2132002b.jpg' http://localhost:5001/upload

curl -X POST -F 'filename=175_1479423456_jpg.rf.0723ceef6a241da65f4f36db2132002b.jpg' http://localhost:5001/retrieve

curl -X POST \
  -F "filename=175_1479423456_jpg.rf.0723ceef6a241da65f4f36db2132002b.jpg" \
  -F "object_id=object_14" \
  http://localhost:5001/details

  curl -X POST -F 'filename=175_1479423456_jpg.rf.0723ceef6a241da65f4f36db2132002b.jpg' http://localhost:5001/save

  # Masked Image Display
  python display_app.py coin-dataset/175_1479423456_jpg.rf.0723ceef6a241da65f4f36db2132002b.jpg