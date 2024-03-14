FROM python:3.8

RUN apt-get -y update \
    && apt-get install --no-install-recommends -y \
    libglu1-mesa-dev libgl1-mesa-dev libosmesa6-dev \
    xvfb unzip patchelf ffmpeg cmake swig

# gymnasium==1.0.0a1 solve the issue that mujoco>=3.0.0 change attribute name
RUN pip install mujoco==3.1.3 \
    pip install gymnasium==1.0.0a1 \
    pip install numpy \
    pip install scipy \
    pip install imageio

ENV DISPLAY=:0

RUN mkdir /mujoco_example
COPY . /mujoco_example
WORKDIR /mujoco_example

CMD ["bash"]
