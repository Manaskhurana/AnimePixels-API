import requests

class AnimePixelsAPI:
    def __init__(self, base_url="https://your-api-domain.com"):
        self.base_url = base_url

    def random(self):
        return requests.get(f"{self.base_url}/random").json()

    def random_image(self, category=None):
        if category:
            return requests.get(f"{self.base_url}/random/image/{category}").json()
        return requests.get(f"{self.base_url}/random/image").json()

    def random_gif(self, category=None):
        if category:
            return requests.get(f"{self.base_url}/random/gif/{category}").json()
        return requests.get(f"{self.base_url}/random/gif").json()

    def search(self, query, media="image"):
        return requests.get(f"{self.base_url}/search/{media}", params={"query": query}).json()
