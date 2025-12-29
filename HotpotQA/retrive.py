import requests
import json


class Retriever:
    def __init__(self, url):
        self.url = url

    def send_request(self, payload, headers=None):
        if headers is None:
            headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(self.url, headers=headers, data=json.dumps(payload))
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                error_message = f"Error occurred with status code: {response.status_code}, response text: {response.text}"
                print(error_message)
                return response.text
        except Exception as e:
            print(f"An error occurred: {e}")
            return str(e)

    def retrieve(self, entity, data):
        request_config = {
            'entity': entity,
            'data': data
        }
        response = self.send_request(request_config)
        return response
