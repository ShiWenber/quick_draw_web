FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# 创建工作目录
RUN mkdir -p /usr/src/quick_draw_backend

# 将所有文件递归复制到工作目录
ADD . /usr/src/quick_draw_backend

WORKDIR /usr/src/quick_draw_backend

# RUN ls -a

# RUN conda env create -n quick_draw_backend -f ./environment.yml
# 检查环境是否存在，存在则更新，不存在则创建 
RUN conda env update -n quick_draw_backend -f ./environment.yml || conda env create -n quick_draw_backend -f ./environment.yml

# RUN conda init bash

# 激活环境由于docker 容器中使用 conda init bash 无法激活环境，所以常常使用下面的方式激活环境
SHELL ["conda", "run", "-n", "quick_draw_backend", "/bin/bash", "-c"]
# RUN conda activate quick_draw_backend
# RUN echo "source activate quick_draw_backend" > ~/.bashrc


# 安装 gunicorn
RUN pip install gunicorn
# 安装 opengl 补全opencv-python 的依赖
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 运行服务
RUN chmod +x ./run.sh
# RUN bash ./run.sh


