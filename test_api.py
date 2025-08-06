# test_api.py (Phiên bản cuối cùng, tương thích với mô hình V5)
import requests
import json

# Địa chỉ của API server
API_URL = "http://127.0.0.1:5000/predict"

def get_prediction(data):
    """Gửi dữ liệu đến API và in ra kết quả."""
    try:
        response = requests.post(API_URL, json=data)
        
        # Kiểm tra xem request có thành công không (status code 200)
        if response.status_code == 200:
            print("Dự đoán thành công!")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Lỗi! Server trả về mã lỗi: {response.status_code}")
            print("Nội dung lỗi:", response.text)
            
    except requests.exceptions.ConnectionError as e:
        print(f"Lỗi kết nối: Không thể kết nối đến server tại {API_URL}.")
        print("Hãy đảm bảo bạn đã chạy 'python app.py' trong một cửa sổ terminal khác.")

if __name__ == '__main__':
    # --- Kịch bản 1: Anime mới, chỉ biết các thông tin bắt buộc ---
    # API sẽ tự động dùng các giá trị mặc định cho 'type', 'episodes', 'duration'
    print("\n--- KỊCH BẢN 1: ANIME MỚI (THÔNG TIN TỐI THIỂU) ---")
    anime_1_data = {
        "genres": ["Action", "Fantasy", "Adventure"],
        "source": "Light novel",
        "studios": "J.C.Staff" # Studio này thuộc Top 30
    }
    get_prediction(anime_1_data)

    # --- Kịch bản 2: Phim lẻ Kimetsu no Yaiba mới (Dự án bom tấn) ---
    print("\n--- KỊCH BẢN 2: PHIM LẺ KIMETSU NO YAIBA MỚI ---")
    anime_2_data = {
        "genres": ["Action", "Fantasy", "Adventure", "Demons"],
        "source": "Manga",
        "studios": "ufotable", # Studio thuộc Top 30, nổi tiếng với chất lượng cao
        "type": "Movie",
        "episodes": 1,
        "duration_per_episode_sec": 9300 # 155 phút
    }
    get_prediction(anime_2_data)

    # --- Kịch bản 3: 'Kimi no Na wa.' (Kiệt tác nguyên tác) ---
    print("\n--- KỊCH BẢN 3: KIỂM TRA 'KIMI NO NA WA.' (STUDIO KHÔNG THUỘC TOP 30) ---")
    anime_3_data = {
        "genres": ["Supernatural", "Drama", "Romance", "School"],
        "source": "Original",
        "studios": "CoMix Wave Films", # Studio này KHÔNG thuộc Top 30 -> sẽ bị coi là 'Other'
        "type": "Movie",
        "episodes": 1,
        "duration_per_episode_sec": 6420 # 107 phút
    }
    get_prediction(anime_3_data)

    # --- Kịch bản 4: 'Kaoru Hana wa Rin to Saku' (Dự án được mong chờ) ---
    # Chỉnh lại genre "School Life" thành "School" để khớp với dữ liệu huấn luyện
    print("\n--- KỊCH BẢN 4: KAORU HANA WA RIN TO SAKU (STUDIO KHÔNG THUỘC TOP 30) ---")
    anime_4_data = {
        "genres": ["Romance", "School", "Drama"], # Sửa lại "School Life" -> "School"
        "source": "Manga",
        "studios": "CloverWorks", # Studio này KHÔNG thuộc Top 30 -> sẽ bị coi là 'Other'
        "type": "TV",
        "episodes": 12,
        "duration_per_episode_sec": 1440
    }
    get_prediction(anime_4_data)
    
    # --- Kịch bản 5: Anime với thể loại hoàn toàn mới (Kiểm tra logic Genre_Other) ---
    print("\n--- KỊCH BẢN 5: ANIME VỚI THỂ LOẠI MỚI ---")
    anime_5_data = {
        "genres": ["Action", "Sci-Fi", "Cyberpunk", "Mecha"], # "Cyberpunk" và "Mecha" là thể loại mới
        "source": "Original",
        "studios": "Trigger", # Studio này không thuộc top 30
        "type": "ONA",
        "episodes": 6,
        "duration_per_episode_sec": 1500 # 25 phút
    }
    get_prediction(anime_5_data)