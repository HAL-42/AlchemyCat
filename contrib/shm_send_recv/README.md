# py2 Client py3 Server用法

### 快速入门
视python2程序为client，python3程序为server。二者分两个进程独立运行。

client程序通过`client_send(obj)`将python对象发送出去。
server程序通过`obj = server_recv()`接收上述对象。

server程序通过`server_send(obj)`将python对象发送出去。
client程序通过`obj = client_recv()`接收上述对象。

支持发送的python对象有：所有python内置类型，以及numpy array。

#### 案例
demo参见src/tasks/py2_client_py3_server/py3_server.py，src/tasks/py2_client_py3_server/py2_client.py程序。
- client为python2下的ROS程序，从裁判系统接收图片。接收图片后，通过`client_send(image)`将图片发送给server程序。
- server为python3下的模型推理程序。在`while True`循环中，`image = server_recv()`接收上述图像。
- server程序推理模型，得到预测结果。通过`server_send(predict)`将predict发送出去。
- client程序通过`predict = client_recv()`接收预测结果。

#### 同步性
`send`, `recv`函数都是“阻塞”的。只有当发送或接收成功后，才会继续运行。因此，client和server中的收发函数必须成对出现，有发必有收。

注意，若前一个`send`函数发送的对象还没被接收，则下一个`send`函数会被阻塞，直到之前发送的内容被`recv`接收。

`send`, `recv`函数均支持设置`time_out`参数。若超过`time_out`时间，收发仍没有完成，则报错。

#### 优雅地退出server程序
若同时运行多个server程序，相互间会冲突。server程序若被强行终止，会在/dev/shm中留下残留文件，可能会影响之后的程序运行。因此，推荐在server程序开始时，运行`init_server()`，其可以保证：
1. 只有一个server在运行。
2. 若通过kill %n，Ctrl+C等方法终止server，server程序可以完成的必要的清理工作后退出。

### 原理
程序借助Linux的shared memory目录/dev/shm来传输文件。该目录实际位置在物理内存上，故而可以快速读写。

发送时，程序将python对象保存到上述目录中，并发送“保存成功信号”。接收程序循环检查“保存成功信号”是否被发送，收到信号后读取python对象，并清除上述文件和信号。

### 性能
经过测量，发送一张512x1024 RBG图像，接收一张同尺寸分割图，收发共计耗时0.005s。可运行
```shell
nohup python src/tasks/py2_client_py3_server/py3_server.py &
python src/tasks/py2_client_py3_server/py2_client.py
```
自行测量。

### 正确性
运行下面指令，验证程序正确性。
```shell
nohup python src/tasks/py2_client_py3_server/py3_server_complex.py &
python src/tasks/py2_client_py3_server/py2_client_complex.py
```
