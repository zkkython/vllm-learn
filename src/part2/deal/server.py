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

    try:
        while True:
            # ROUTER 接收格式: [identity, delimiter, message]
            # 注意：DEALER 发送的消息会被 ROUTER 自动加上客户端 identity
            multipart = frontend.recv_multipart()
            if not multipart:
                continue

            identity = multipart[0]
            message = multipart[-1]  # 中间可能有空帧（如果用了 delimiter），但我们忽略
            request = zmq.utils.jsonapi.loads(message)

            print(f"引擎收到来自 {identity.decode()} 的请求: {request['msg_id']}")

            # 异步处理推理（这里简化为同步模拟）
            result = process_inference(request)

            # 构造响应：先发 identity，再发响应内容（ROUTER 自动路由）
            response = {
                "msg_id": request["msg_id"],
                "reply": result,
                "from_engine": "Engine-0",
            }
            frontend.send_multipart([identity, zmq.utils.jsonapi.dumps(response)])

    except KeyboardInterrupt:
        print("\n引擎关闭.")
    finally:
        frontend.close()


if __name__ == "__main__":

    engine()
