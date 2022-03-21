FROM python:3.8
WORKDIR /app
RUN pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
RUN pip3 install torch_sparse torch_scatter torch_geometric
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . /app/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
ENTRYPOINT [ "streamlit", "run" ]
CMD [ "main.py" ]