# leverage the renci python base image
FROM ghcr.io/translatorsri/renci-python-image:3.12.4

#Build from this branch.  Assume master for this repo
ARG BRANCH_NAME=main

# update the container
RUN apt-get update

# Create and set working directory
RUN mkdir /repo
WORKDIR /repo

# get the latest code
RUN git clone --branch $BRANCH_NAME --single-branch https://github.com/RENCI-NER/pred-mapping.git

# Set working directory to the cloned repo
WORKDIR /repo/pred-mapping

## Copy project files into the container
#COPY ./ /repo

# Ensure permissions
RUN chmod 777 -R .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn

# Switch to non-root user
USER nru

# Expose FastAPI port
EXPOSE 6380

# Start the app
ENTRYPOINT ["bash", "main.sh"]