import requests
from io import BytesIO
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import urllib.parse

# 모델 로드 (이전에 FaceForensics++ 데이터셋으로 훈련된 모델)
model = load_model('path_to_your_trained_model.h5')

# Google에서 이미지를 검색하는 함수
def search_google_images(search_term, num_results=10):

    API_KEY = 'YOUR_GOOGLE_API_KEY'
    SEARCH_ENGINE_ID = 'YOUR_SEARCH_ENGINE_ID'
    
    url = f"https://www.googleapis.com/customsearch/v1?q={search_term}&cx={SEARCH_ENGINE_ID}&searchType=image&key={API_KEY}&num={num_results}"
    
    response = requests.get(url)
    
    results = response.json()
    
    image_urls = [item['link'] for item in results.get('items', [])]
    
    return image_urls

# Bing에서 이미지를 검색하는 함수
def search_bing_images(search_term, num_results=10):
    
    API_KEY = 'YOUR_BING_API_KEY'
    
    url = f"https://api.bing.microsoft.com/v7.0/images/search?q={urllib.parse.quote(search_term)}&count={num_results}&offset=0&mkt=en-us&safeSearch=Moderate"
    
    headers = {'Ocp-Apim-Subscription-Key': API_KEY}
    
    response = requests.get(url, headers=headers)
    
    results = response.json()
    
    image_urls = [item['contentUrl'] for item in results['value']]
    
    return image_urls

# DuckDuckGo에서 이미지를 검색하는 함수
def search_duckduckgo_images(search_term, num_results=10):
    
    url = f"https://duckduckgo.com/?q={urllib.parse.quote(search_term)}&t=h_&iar=images&iax=images&ia=images"
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(url, headers=headers)
    
    results = response.text

    # 이미지 URL을 파싱하는 간단한 방법 (BeautifulSoup을 사용할 수도 있음)
    image_urls = []
    
    for line in results.splitlines():
    
        if 'vqd=' in line:
    
            vqd = line.split('vqd=')[1].split('&')[0]
    
        if 'imgurl=' in line:
    
            img_url = line.split('imgurl=')[1].split('&')[0]
    
            image_urls.append(urllib.parse.unquote(img_url))
    
            if len(image_urls) >= num_results:
    
                break

    return image_urls

# URL에서 이미지를 불러오는 함수
def load_image_from_url(url):
    
    response = requests.get(url)
    
    img = Image.open(BytesIO(response.content))
    
    img = img.resize((224, 224))  # 모델의 입력 크기에 맞게 조정
    
    img = np.array(img)
    
    img = np.expand_dims(img, axis=0)  # 모델 입력 형식에 맞게 차원 확장
    
    return img

# 딥페이크를 탐지하는 함수
def detect_deepfake(images, model):
    
    predictions = model.predict(images)
    
    return predictions > 0.5  # 0.5 이상이면 딥페이크로 판정

def main():
    
    # 사용자로부터 키워드 입력받기
    
    search_term = input("이미지 검색을 위한 키워드를 입력하세요: ")

    # 각 검색 엔진에서 이미지 검색
    google_images = search_google_images(search_term)
    
    bing_images = search_bing_images(search_term)
    
    duckduckgo_images = search_duckduckgo_images(search_term)

    # 모든 검색 엔진에서 얻은 이미지 URL을 합치기
    all_image_urls = google_images + bing_images + duckduckgo_images
    
    deepfake_count = 0
    
    total_images = len(all_image_urls)
    
    for url in all_image_urls:
    
        try:
    
            img = load_image_from_url(url)
    
            is_deepfake = detect_deepfake(img, model)
    
            print(f"Image URL: {url}")
            print(f"Deepfake Detected: {is_deepfake}")
    
            if is_deepfake:
    
                deepfake_count += 1
    
        except Exception as e:
    
            print(f"Error processing image: {url} - {str(e)}")
    
    print(f"Total Images: {total_images}")
    print(f"Detected Deepfakes: {deepfake_count}")
    
    accuracy = (deepfake_count / total_images) * 100 if total_images > 0 else 0
    
    print(f"Detection Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    
    main()
