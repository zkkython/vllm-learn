import queue
import threading
import time
from enum import Enum


class RequestType(Enum):
    ADD = b"add"
    ABORT = b"abort"


class SimpleEngineCore:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.requests = {}

    def process_input_socket(self):
        simulated_requests = [
            (RequestType.ADD, {"id": "req_1", "prompt": "Hello"}),
            (RequestType.ADD, {"id": "req_2", "prompt": "World"}),
            (RequestType.ABORT, ["req_1"]),
        ]
        for req_type, data in simulated_requests:
            time.sleep(0.1)
            self.input_queue.put_nowait((req_type, data))
            print(f"[InputThread] 收到 {req_type.name}: {data}")

    def run_busy_loop(self):
        for _ in range(3):
            req_type, data = self.input_queue.get(timeout=2.0)
            self._handle_request(req_type, data)
            while not self.input_queue.empty():
                req_type, data = self.input_queue.get_nowait()
                self._handle_request(req_type, data)

    def _handle_request(self, req_type, data):
        if req_type == RequestType.ADD:
            self.requests[data["id"]] = data
            print(f"[BusyLoop] 添加请求: {data['id']}")
        elif req_type == RequestType.ABORT:
            for rid in data:
                self.requests.pop(rid, None)
                print(f"[BusyLoop] 中止请求: {rid}")


engine = SimpleEngineCore()
threading.Thread(target=engine.process_input_socket, daemon=True).start()
engine.run_busy_loop()
