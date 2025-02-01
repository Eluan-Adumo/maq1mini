FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Print Python and pip versions for debugging
RUN python --version
RUN pip --version

# Print the contents of requirements.txt for debugging
RUN cat requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5100

# Command to run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5100", "app:app"]