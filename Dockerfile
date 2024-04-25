# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents and utils.py into the container at /app
COPY ./flask_app/ /app/
COPY ./utils.py /app/
COPY ./requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5001

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]