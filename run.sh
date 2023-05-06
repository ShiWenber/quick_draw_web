# 经测试，在intel i5-10400 @ 2.90GHz，NVIDIA RTX 2060，32G内存硬件上，使用 2 进程并发性能最佳
gunicorn -c gunicorn.conf.py main:app    