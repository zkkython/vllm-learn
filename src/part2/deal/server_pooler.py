import time
import zmq


def engine():
    context = zmq.Context.instance()
    frontend = context.socket(zmq.ROUTER)  # 接收客户端请求
    frontend.bind("tcp://*:6666")

    print("推理引擎启动，监听端口 6666...")

    def process_inference(data):
        """模拟异步推理任务"""
        time.sleep(0.5)  # 模拟计算延迟
        return {
            "status": "success",
            "result": f"推理完成: {data['query']}",
            "processed_at": time.time(),
        }

    poller = zmq.Poller()
    poller.register(frontend, zmq.POLLIN)  # 监听是否有消息可读
    try:
        while True:
            # 阻塞最多 1 秒，检查是否有事件就绪
            # 返回格式: {socket: event_mask}
            socks = dict(poller.poll(timeout=1000))  # 超时单位：毫秒

            # 检查 frontend 是否有来自客户端的消息
            if frontend in socks and socks[frontend] == zmq.POLLIN:
                multipart = frontend.recv_multipart()
                identity = multipart[0]
                message = multipart[-1]
                request = zmq.utils.jsonapi.loads(message)

                print(f"引擎收到来自 {identity.decode()} 的请求: {request['msg_id']}")

                # 模拟处理（实际中可转发给 backend 或线程池）
                result = process_inference(request)

                # 构造响应并返回
                response = {
                    "msg_id": request["msg_id"],
                    "reply": result,
                    "from_engine": "Engine-0",
                }
                frontend.send_multipart(
                    [identity, zmq.utils.jsonapi.dumps(response)]
                )  # 阻塞最多 1 秒，检查是否有事件就绪

    except KeyboardInterrupt:
        print("\n引擎关闭.")
    finally:
        frontend.close()


if __name__ == "__main__":

    engine()
