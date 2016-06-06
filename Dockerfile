FROM python:2-onbuild

# Install Python Setuptools
RUN apt-get install -y python-setuptools

# Bundle app source
ADD . /src

# Expose
EXPOSE 5000

# Run
CMD [ "python", "./src/CRS.py" ]