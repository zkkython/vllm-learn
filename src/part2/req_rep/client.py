import zmq
import time


def echo_client():
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # REQ socket
    socket.connect("tcp://localhost:5555")  # 连接到服务端

    # 发送消息
    msg = "Hello, ZeroMQ!" + time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"发送消息: {msg}")

    socket.send(msg.encode())  # 发送消息
    socket.send(msg.encode())  # 发送消息
    reply = socket.recv()  # 接收回复
    print(f"收到回显: {reply.decode()}")


if __name__ == "__main__":
    echo_client()
