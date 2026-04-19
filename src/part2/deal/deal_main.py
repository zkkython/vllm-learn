import threading
import time
from server_pooler import engine
from client import client

if __name__ == "__main__":
    # 启动引擎线程
    engine_thread = threading.Thread(target=engine, daemon=True)
    engine_thread.start()

    time.sleep(1)  # 等待引擎启动

    # 启动多个客户端
    client_threads = []
    for i in range(3):
        t = threading.Thread(target=client, args=(i,), daemon=True)
        t.start()
        client_threads.append(t)

    try:
        # 等待所有客户端完成
        for t in client_threads:
            t.join()
        time.sleep(2)
    except KeyboardInterrupt:
        print("\n主程序退出.")
