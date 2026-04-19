import zmq
import time
import uuid
import random


def client(client_id):
    context = zmq.Context.instance()
    socket = context.socket(zmq.DEALER)

    # 为每个客户端设置唯一身份标识（ZeroMQ 会自动使用此身份进行路由）
    identity = f"Client-{client_id}".encode("utf-8")
    socket.setsockopt(zmq.IDENTITY, identity)

    socket.connect("tcp://localhost:6666")
    print(f"[Client {client_id}] 启动，连接到引擎...")

    # 发送多个异步请求
    for i in range(5):
        request = {
            "msg_id": str(uuid.uuid4()),
            "query": f"请求 {i} 来自 {client_id}",
            "timestamp": time.time(),
        }
        socket.send_json(request)
        print(f"[Client {client_id}] 发送请求: {request['msg_id']}")

        # 不等待响应，继续发送下一个请求（异步）
        time.sleep(0.1)

    # 接收响应（异步接收，顺序不一定与发送一致）
    for _ in range(5):
        try:
            response = socket.recv_json(flags=zmq.NOBLOCK)  # 非阻塞接收
            print(f"[Client {client_id}] 收到响应: {response}")
        except zmq.Again:
            time.sleep(0.1)
            continue

    socket.close()


if __name__ == "__main__":
    client_id = random.randint(1, 1000)
    client(client_id)
