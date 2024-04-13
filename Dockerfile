FROM docker.io/library/python:3.9

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install required system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip

# Install numpy first
RUN pip install numpy

# Install remaining Python packages
RUN pip install \
    Cython \
    pandas \
    matplotlib \
    LunarCalendar \
    convertdate \
    holidays==0.10.5 \
    fbprophet

# Set the entry point
ENTRYPOINT [ "python3" ]

# Command to run the application
CMD [ "app.py" ]
