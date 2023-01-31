import requests
import os
<<<<<<< HEAD:tests/test_api.py
url = "http://localhost:8000/pred"
=======
url = "http://localhost:8002/pred/"
>>>>>>> e3ea46019acb054c2889c518b563c585d4ed2379:automatic_tagging/api/test_api.py


print(os.getenv("TEST_IMG_PATH"))
response = requests.post(url, files={'file': open(os.getenv("TEST_IMG_PATH"), "rb")}, timeout=30)
print(response)
