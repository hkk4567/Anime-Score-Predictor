# app.py (Phiên bản cuối cùng, hỗ trợ Feature Uy Tín & Genre_Other)

import joblib
import pandas as pd
import numpy as np
import logging
from flask import Flask, request, jsonify

# --- 1. KHỞI TẠO CÁC THÀNH PHẦN ---
app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải TẤT CẢ các thành phần cần thiết cho mô hình mạnh nhất
try:
    model = joblib.load('models/anime_score_predictor_v5.pkl')
    genres_encoder = joblib.load('models/genres_encoder_v5.pkl')
    model_columns = joblib.load('models/model_features_v5.pkl')
    top_30_studios = joblib.load('models/top_30_studios_v5.pkl')
    core_genres = joblib.load('models/core_genres_v5.joblib')
    studio_mean_score_map = joblib.load('models/studio_mean_score_map.joblib')
    source_mean_score_map = joblib.load('models/source_mean_score_map.joblib')
    
    # Tính giá trị trung bình toàn cục để điền cho các studio/source mới
    global_mean_score = studio_mean_score_map.mean()
    
    logging.info("Tất cả các mô hình và bộ biến đổi V5 đã được tải thành công.")
except FileNotFoundError as e:
    logging.error(f"Lỗi tải mô hình: {e}. Hãy đảm bảo bạn đã chạy file huấn luyện để tạo ra các file V5.")
    exit()

# Định nghĩa các giá trị mặc định
default_episodes = 12
default_duration_sec = 1440
default_type = 'TV'


# --- 2. HÀM HELPER (Tái sử dụng từ file huấn luyện) ---
def process_genres_with_other(genre_list, core_genres):
    input_genres_set = set(genre_list)
    processed_list = list(input_genres_set.intersection(core_genres))
    if not input_genres_set.issubset(core_genres):
        processed_list.append('Genre_Other')
    return processed_list


# --- 3. ĐỊNH NGHĨA API ENDPOINT ---
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Không nhận được dữ liệu"}), 400
    logging.info(f"Nhận được yêu cầu: {data}")

    required_fields = ['genres', 'source', 'studios']
    if not all(field in data for field in required_fields):
        return jsonify({"error": f"Thiếu các trường bắt buộc: {required_fields}"}), 400

    try:
        # 1. Tạo DataFrame và xử lý input
        input_data = {
            'Type': [data.get('type', default_type)],
            'Source': [data['source']],
            'Studios': [data['studios']],
            'Episodes': [data.get('episodes', default_episodes)],
            'duration_per_episode_sec': [data.get('duration_per_episode_sec', default_duration_sec)]
        }
        input_df = pd.DataFrame(input_data)
        
        genres_list_input = data['genres']
        if not isinstance(genres_list_input, list):
             return jsonify({"error": "Trường 'genres' phải là một danh sách (list)."}), 400

        # 2. BIẾN ĐỔI DỮ LIỆU ĐẦY ĐỦ
        
        ### BƯỚC MỚI: TÍNH TOÁN CÁC FEATURE "UY TÍN" ###
        studio = input_df.loc[0, 'Studios']
        source = input_df.loc[0, 'Source']
        # Dùng .get() để xử lý các studio/source mới một cách an toàn
        input_df['studio_avg_score'] = studio_mean_score_map.get(studio, global_mean_score)
        input_df['source_avg_score'] = source_mean_score_map.get(source, global_mean_score)

        ### BƯỚC MỚI: ÁP DỤNG LOGIC "GENRE_OTHER" ###
        final_genres_for_model = process_genres_with_other(genres_list_input, core_genres)
        genres_input_encoded = pd.DataFrame(genres_encoder.transform([final_genres_for_model]), columns=genres_encoder.classes_)
        
        # Xử lý các feature chữ còn lại
        input_df['Studios'] = input_df['Studios'].where(input_df['Studios'].isin(top_30_studios), 'Other')
        categorical_input = pd.get_dummies(input_df[['Type', 'Source', 'Studios']], drop_first=True, dtype=int)
        
        # Lấy các feature số (bao gồm cả feature uy tín)
        numerical_input = input_df[['Episodes', 'duration_per_episode_sec', 'studio_avg_score', 'source_avg_score']]

        # Kết hợp tất cả lại
        final_input_df = pd.concat([numerical_input, categorical_input, genres_input_encoded], axis=1)
        final_input_df = final_input_df.reindex(columns=model_columns, fill_value=0)
        
        # 3. Dự đoán
        prediction = model.predict(final_input_df)
        
        # 4. Trả về kết quả
        result = {'predicted_score': round(prediction[0], 2)}
        logging.info(f"Trả về kết quả dự đoán: {result}")
        return jsonify(result)

    except Exception as e:
        logging.error(f"Lỗi trong quá trình dự đoán: {e}", exc_info=True)
        return jsonify({"error": "Đã xảy ra lỗi trong quá trình xử lý"}), 500

@app.route('/', methods=['GET'])
def index():
    return "<h1>Anime Score Predictor API (V5 - Final Model) is running!</h1>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)