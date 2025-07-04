import queue
import threading
import time
import asyncio
import os
import math
import numpy as np

test_img_dir = "photos"

print()

contents = os.listdir(test_img_dir)
photos = []
for content in contents:
    content_path = os.path.join(test_img_dir, content)
    if os.path.isfile(content_path):
        photos.append(content_path)

# print(photos)

a1 = np.array([1, 2, 8])
a2 = np.array([4, 5, 6])


print((a1 + a2) // 2)
print(math.sqrt(256))

q = queue.Queue()


def test():
    while True:
        a = q.get()
        if a is False:
            break
        print(a)


thread = threading.Thread(target=test)
thread.start()

print("a")
q.put("Hello")
print("c")
time.sleep(2)
print("d")
q.put(False)
thread.join()
