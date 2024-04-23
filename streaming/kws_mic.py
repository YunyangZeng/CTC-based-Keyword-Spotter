import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from listener import MicrophoneListener
from config import ListenerConfig, DetectorConfig
from detector import KeywordDetector
import signal
import time
    
def signal_handler(sig, frame):
    print(f"Received signal {sig}. Stopping all processes...")
    for process in multiprocessing.active_children():
        process.terminate()

def microphone_listener(q, stop_event):
    Listener = MicrophoneListener(q, ListenerConfig())
    Listener.listen()

def keyword_detector(q, stop_event):
    Detector = KeywordDetector(DetectorConfig())
    Detector.display_init()
    try:
        start_time = time.time()
        while True:
            data = q.get()
            if data is None:
                break
            else:
                Detector.run(data, start_time)
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping detector...")
        Detector.abort_display()
    finally:
        print("Detector stopped.")

def main():
    stop_event = multiprocessing.Event()
    signal.signal(signal.SIGINT, signal_handler)
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=microphone_listener, args=(q, stop_event))
    p2 = multiprocessing.Process(target=keyword_detector, args=(q, stop_event))
    p1.start()
    p2.start()
    try:
        p1.join()
        q.put(None)
        p2.join()
    except KeyboardInterrupt:
        print("KeyboardInterrupt: Stopping all processes...")
        p1.terminate()
        p1.join()
        p2.terminate()
        p2.join()

if __name__ == "__main__":
    main()
