# Use a Debian-based Python image
FROM python:3.10-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to use Docker's caching mechanism
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your repository's files into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run the app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]