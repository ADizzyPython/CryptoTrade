import requests

proxies = {
    "http": "http://brd-customer-hl_c0565a8d-zone-serp_api1:atpvk7cmhpj2@brd.superproxy.io:33335",
    "https": "http://brd-customer-hl_c0565a8d-zone-serp_api1:atpvk7cmhpj2@brd.superproxy.io:33335"
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

url = "https://www.google.com/search?q=pizza&brd_json=1"

response = requests.get(url, headers=headers, proxies=proxies, verify=False)
print(response.text)