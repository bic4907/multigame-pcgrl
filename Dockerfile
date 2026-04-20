FROM bic4907/pcgrl:cu12

RUN pip install timm

ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
RUN apt-get update && apt-get install -y build-essential

# update latest wandb
RUN pip install --upgrade wandb