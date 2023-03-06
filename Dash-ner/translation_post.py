import requests
import json

url = "http://127.0.0.1:7204/onlineTranserver/serverapi/goTextTranslate"

payload = json.dumps({
  "field": "PV",
  "from": "CHINESE",
  "origin": "你好",
  "to": "ENGLISH"
})

headers = {
  'Accept': 'text/plain',
  'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)
print(response.text)