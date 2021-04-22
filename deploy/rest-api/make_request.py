import requests

url = 'http://localhost:3000/predict'

r = requests.post(url,json={'imageUrl': 'https://upload.wikimedia.org/wikipedia/commons/2/27/Finnish_Spitz_600.jpg'})
print(r.json())
