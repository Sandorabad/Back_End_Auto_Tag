import requests
import os
url = "http://localhost:8002/pred/"


print(os.getenv("TEST_IMG_PATH"))
response = requests.post(url, files={'file': open(os.getenv("TEST_IMG_PATH"), "rb")}, timeout=30)
print(response)
