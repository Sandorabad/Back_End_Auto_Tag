import requests
import os
<<<<<<< HEAD:automatic_tagging/api/test_api.py
url = "http://localhost:8002/pred/"
=======
url = "http://localhost:8000/pred"
>>>>>>> 095bd45a17907ca5b2b7c2f008e8614eeaf1cfcf:tests/test_api.py


print(os.getenv("TEST_IMG_PATH"))
response = requests.post(url, files={'file': open(os.getenv("TEST_IMG_PATH"), "rb")}, timeout=30)
print(response)
