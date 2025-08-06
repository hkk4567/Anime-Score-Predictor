# app_web.py

import joblib
import pandas as pd
import numpy as np
import logging
from flask import Flask, request, jsonify, render_template

# --- 1. KHỞI TẠO CÁC THÀNH PHẦN ---
app = Flask(__name__)
# Cấu hình logging để in ra console với định dạng đẹp hơn
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [WEB APP] - %(message)s')

# Tải TẤT CẢ các thành phần cần thiết
try:
    model = joblib.load('models/anime_score_predictor_v5.pkl')
    genres_encoder = joblib.load('models/genres_encoder_v5.pkl')
    model_columns = joblib.load('models/model_features_v5.pkl')
    top_30_studios = joblib.load('models/top_30_studios_v5.pkl')
    core_genres = joblib.load('models/core_genres_v5.joblib')
    studio_mean_score_map = joblib.load('models/studio_mean_score_map.joblib')
    source_mean_score_map = joblib.load('models/source_mean_score_map.joblib')
    
    global_mean_score = studio_mean_score_map.mean()
    
    all_sources = sorted(source_mean_score_map.index.tolist())
    all_types = ['TV', 'Movie', 'OVA', 'ONA', 'Special', 'Music']
    all_genres = sorted(list(core_genres))

    logging.info("Tất cả các mô hình và dữ liệu cho web đã được tải thành công.")
except FileNotFoundError as e:
    logging.error(f"Lỗi tải mô hình: {e}. Hãy đảm bảo bạn đã chạy file huấn luyện để tạo ra các file V5.")
    model = None # Gán là None để tránh lỗi khi chạy app

# --- 2. HÀM HELPER ---
def process_genres_with_other(genre_list, core_genres_set):
    """
    Xử lý danh sách genre:
    - Lọc ra các genre có trong core_genres.
    - Nếu có bất kỳ genre nào không nằm trong core_genres, thêm 'Genre_Other'.
    """
    if not genre_list:
        return []
    input_genres_set = set(genre_list)
    # Lấy phần giao giữa genre người dùng nhập và các core genre
    processed_list = list(input_genres_set.intersection(core_genres_set))
    # Nếu có genre nào đó của người dùng không thuộc core_genres -> đó là genre "khác"
    if not input_genres_set.issubset(core_genres_set):
        processed_list.append('Genre_Other')
    return processed_list

def get_prediction(data):
    """Hàm lõi để thực hiện dự đoán, được tái sử dụng bởi cả API và Web."""
    input_df = pd.DataFrame([data])
    
    # Tính toán các feature uy tín
    studio = input_df.loc[0, 'studios']
    source = input_df.loc[0, 'source']
    input_df['studio_avg_score'] = studio_mean_score_map.get(studio, global_mean_score)
    input_df['source_avg_score'] = source_mean_score_map.get(source, global_mean_score)

    # Xử lý genres
    core_genres_set = set(core_genres) # Chuyển sang set để tối ưu
    final_genres = process_genres_with_other(data['genres'], core_genres_set)
    
    # Chuyển đổi list genre thành DataFrame được mã hóa
    # Chú ý: genres_encoder.transform yêu cầu đầu vào là một list của list
    genres_encoded = pd.DataFrame(genres_encoder.transform([final_genres]), columns=genres_encoder.classes_)
    
    # Xử lý các feature chữ còn lại (One-Hot Encoding)
    input_df['Studios_temp'] = input_df['studios'].where(input_df['studios'].isin(top_30_studios), 'Other')
    categorical_input = pd.get_dummies(input_df[['type', 'source', 'Studios_temp']], drop_first=False, dtype=int)
    
    # Lấy các feature số
    numerical_input = input_df[['episodes', 'duration_per_episode_sec', 'studio_avg_score', 'source_avg_score']]

    # Đổi tên cột cho khớp với lúc train (nếu cần)
    # (Bước này có thể không cần nếu tên đã chuẩn)

    # Kết hợp và sắp xếp lại các cột
    final_input_df = pd.concat([numerical_input, categorical_input, genres_encoded], axis=1)
    
    # Reindex để đảm bảo tất cả các cột mô hình cần đều có mặt, và đúng thứ tự
    final_input_df = final_input_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(final_input_df)
    return prediction[0]

# --- 3. ĐỊNH NGHĨA CÁC ROUTE (ENDPOINT) ---

@app.route('/', methods=['GET'])
def home():
    if model is None:
        return "Lỗi: Không thể tải mô hình. Vui lòng kiểm tra file log và khởi động lại server.", 500
    return render_template('index.html', all_sources=all_sources, all_types=all_types, all_genres=all_genres)

@app.route('/predict-web', methods=['POST'])
def predict_web():
    try:
        # === BƯỚC 1: Lấy dữ liệu từ form ===
        form_data = {
            "genres": request.form.getlist('genres'),
            "source": request.form.get('source'),
            "studios": request.form.get('studios'),
            "type": request.form.get('type'),
            "episodes": int(request.form.get('episodes', 12)), # Thêm giá trị mặc định
            "duration_per_episode_sec": int(request.form.get('duration_per_episode_sec', 1440)) # Thêm giá trị mặc định
        }

        # === BƯỚC 2: (SỬA LỖI QUAN TRỌNG) Xử lý checkbox "Genre_Other" ===
        genre_other_checked = request.form.get('genre_other_checkbox')
        form_data['genre_other_checked'] = bool(genre_other_checked) # Lưu lại trạng thái để render lại
        
        if genre_other_checked:
            # Thêm một genre không tồn tại vào list để trigger logic "Genre_Other"
            # trong hàm process_genres_with_other. Đây là một mẹo nhỏ.
            form_data['genres'].append('USER_ADDED_RARE_GENRE')

        # === BƯỚC 3: In dữ liệu ra console để debug ===
        # Cách 1: Dùng logging (khuyên dùng)
        logging.info(f"Dữ liệu nhận được từ Form Web: {form_data}")

        # Cách 2: Dùng print (đơn giản, dễ thấy)
        print("\n" + "="*50)
        print("||   DỮ LIỆU NHẬN ĐƯỢC TỪ FORM WEB   ||")
        print("="*50)
        import json
        print(json.dumps(form_data, indent=2)) # Dùng json.dumps để in cho đẹp
        print("="*50 + "\n")
        
        # Gọi hàm dự đoán
        predicted_score = get_prediction(form_data)
        
        # Trả về trang web với kết quả
        return render_template('index.html', 
                               prediction=predicted_score,
                               user_input=form_data, # Gửi lại toàn bộ dữ liệu người dùng đã nhập
                               all_sources=all_sources, 
                               all_types=all_types, 
                               all_genres=all_genres)

    except Exception as e:
        logging.error(f"Lỗi xử lý form: {e}", exc_info=True)
        return "Đã có lỗi xảy ra trong quá trình dự đoán. Vui lòng kiểm tra log của server.", 500

@app.route('/predict-api', methods=['POST'])
def predict_api():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Không nhận được dữ liệu"}), 400
    
    try:
        # In dữ liệu nhận từ API
        logging.info(f"Dữ liệu nhận được từ API: {data}")
        predicted_score = get_prediction(data)
        return jsonify({'predicted_score': round(predicted_score, 2)})
    except Exception as e:
        logging.error(f"Lỗi xử lý API: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

# --- 4. CHẠY SERVER ---
if __name__ == '__main__':
    # host='0.0.0.0' để có thể truy cập từ máy khác trong cùng mạng
    # debug=True để server tự khởi động lại khi có thay đổi code
    app.run(host='0.0.0.0', port=5000, debug=True)