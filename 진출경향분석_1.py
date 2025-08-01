import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import defaultdict
import warnings
import math
from pyvis.network import Network
import streamlit.components.v1 as components
import mysql.connector
from mysql.connector import Error
warnings.filterwarnings('ignore')

from PIL import Image

import base64 
logo = Image.open('./assets/logo1.jpg')  # 또는 'assets/logo.png'
def get_base64_image(image_path):
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_base64_image('./assets/logo1.jpg')





# 페이지 설정
st.set_page_config(
    page_title="문학 작품 해외 수출국가 및 장르 추천 시스템",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------------- # 
# DB 연결 설정
DB_HOST = st.secrets["database"]["host"]
DB_NAME = st.secrets["database"]["database"]
DB_USER = st.secrets["database"]["user"]
DB_PASSWORD = st.secrets["database"]["password"]

# ------------------------------------------------------------------------- # 


class LiteratureExportAnalyzer:
    def __init__(self):
        self.df = None
        self.hub_scores = {}
        self.genre_fit_scores = {}
        self.transition_matrix = {}
        self.genre_transition_matrix = {}
        
    def load_data_from_db(self, host, database, user, password):
        """DB에서 데이터 로드 및 전처리"""
        try:
            connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            
            if connection.is_connected():
                # SQL 쿼리 (테이블명과 컬럼명은 실제 DB에 맞게 수정)
                query = """
                SELECT book_id, 발간일, genre1, genre2, genre3, genre4, 국가, 원작여부
                FROM literature_books
                """
                
                self.df = pd.read_sql(query, connection)
                
                # 기존 전처리 로직과 동일
                required_cols = ['book_id', '발간일', 'genre1', '국가', '원작여부']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    st.error(f"누락된 컬럼: {missing_cols}")
                    return False
                
                # 장르 컬럼 확인
                self.genre_columns = ['genre1']
                for genre_col in ['genre2', 'genre3']:
                    if genre_col in self.df.columns:
                        self.genre_columns.append(genre_col)
                
                # 발간일을 datetime으로 변환
                self.df['발간일'] = pd.to_datetime(self.df['발간일'])
                
                # 결측치 제거
                self.df = self.df.dropna(subset=required_cols)
                
                connection.close()
                return True
                
        except Error as e:
            st.error(f"DB 연결 오류: {e}")
            return False
    
    def load_wave_data_from_db(self, host, database, user, password):
        """DB에서 wave 데이터 로드"""
        try:
            connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            
            if connection.is_connected():
                # wave_details 테이블에서 데이터 가져오기
                query = """
                SELECT book_id, country, wave, source_country 
                FROM literature_books_wave
                """
                
                wave_df = pd.read_sql(query, connection)
                connection.close()
                return wave_df
                
        except Error as e:
            st.error(f"Wave 데이터 DB 연결 오류: {e}")
            return None

    # --------------- # 

    def get_all_genres(self):
        """모든 장르 컬럼에서 고유 장르 목록 추출"""
        all_genres = set()
        for genre_col in self.genre_columns:
            genres = self.df[genre_col].dropna().unique()
            all_genres.update(genres)
        return sorted(list(all_genres))
    
    def get_books_by_genre(self, selected_genre):
        """특정 장르를 포함한 모든 작품 반환 (genre1, genre2, genre3 중 어디든)"""
        mask = False
        for genre_col in self.genre_columns:
            mask |= (self.df[genre_col] == selected_genre)
        return self.df[mask]
    
    def calculate_hub_scores(self):
        """거점 지수 계산 - 원작 기준, 다중 장르 지원"""
        book_patterns = {}
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                # 원작이 여러 국가에 있다면 가장 빠른 날짜 선택
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_country = original_record['국가']
                original_date = original_record['발간일']
                
                # 모든 장르 정보 수집
                genres = []
                for genre_col in self.genre_columns:
                    if not pd.isna(original_record[genre_col]):
                        genres.append(original_record[genre_col])
                
                # 원작 이후의 모든 진출 국가들
                subsequent_records = group[
                    (group['발간일'] > original_date) | 
                    ((group['발간일'] == original_date) & (group['국가'] != original_country))
                ].sort_values('발간일')
                
                if len(subsequent_records) > 0:
                    subsequent_countries = subsequent_records['국가'].tolist()
                    
                    book_patterns[book_id] = {
                        'original_country': original_country,
                        'original_date': original_date,
                        'subsequent_countries': subsequent_countries,
                        'genres': genres,  # 리스트로 저장
                        'subsequent_count': len(subsequent_countries)
                    }
        
        # 원작 국가별 거점 점수 계산
        hub_analysis = defaultdict(lambda: {'total_books': 0, 'total_subsequent': 0})
        
        for pattern in book_patterns.values():
            original_country = pattern['original_country']
            subsequent_count = pattern['subsequent_count']
            
            hub_analysis[original_country]['total_books'] += 1
            hub_analysis[original_country]['total_subsequent'] += subsequent_count
        
        # 거점 지수 계산 (최소 3개 작품 이상)
        for country, data in hub_analysis.items():
            if data['total_books'] >= 3:
                hub_index = data['total_subsequent'] / data['total_books']
                self.hub_scores[country] = {
                    'hub_index': hub_index,
                    'total_books': data['total_books'],
                    'avg_subsequent': hub_index
                }
        """거점 지수 계산 - 원작 기준"""
        book_patterns = {}
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                # 원작이 여러 국가에 있다면 가장 빠른 날짜 선택
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_country = original_record['국가']
                original_date = original_record['발간일']
                genre = original_record['genre1']
                
                # 원작 이후의 모든 진출 국가들 (원작 포함하지 않음)
                subsequent_records = group[
                    (group['발간일'] > original_date) | 
                    ((group['발간일'] == original_date) & (group['국가'] != original_country))
                ].sort_values('발간일')
                
                if len(subsequent_records) > 0:
                    subsequent_countries = subsequent_records['국가'].tolist()
                    
                    book_patterns[book_id] = {
                        'original_country': original_country,
                        'original_date': original_date,
                        'subsequent_countries': subsequent_countries,
                        'genre': genre,
                        'subsequent_count': len(subsequent_countries)
                    }
        
        # 원작 국가별 거점 점수 계산
        hub_analysis = defaultdict(lambda: {'total_books': 0, 'total_subsequent': 0})
        
        for pattern in book_patterns.values():
            original_country = pattern['original_country']
            subsequent_count = pattern['subsequent_count']
            
            hub_analysis[original_country]['total_books'] += 1
            hub_analysis[original_country]['total_subsequent'] += subsequent_count
        
        # 거점 지수 계산 (최소 3개 작품 이상)
        for country, data in hub_analysis.items():
            if data['total_books'] >= 3:
                hub_index = data['total_subsequent'] / data['total_books']
                self.hub_scores[country] = {
                    'hub_index': hub_index,
                    'total_books': data['total_books'],
                    'avg_subsequent': hub_index
                }
    
    def calculate_genre_fit(self):
        """장르별 국가 적합도 계산 - 다중 장르 지원"""
        # 각 장르별로 국가 출현 횟수 계산
        genre_country_counts = defaultdict(lambda: defaultdict(int))
        genre_totals = defaultdict(int)
        country_totals = defaultdict(int)
        
        for _, row in self.df.iterrows():
            country = row['국가']
            for genre_col in self.genre_columns:
                if not pd.isna(row[genre_col]):
                    genre = row[genre_col]
                    genre_country_counts[genre][country] += 1
                    genre_totals[genre] += 1
                    country_totals[country] += 1
        
        # 적합도 점수 계산
        for genre in genre_country_counts:
            self.genre_fit_scores[genre] = {}
            total_genre_count = genre_totals[genre]
            max_country_count = max(country_totals.values()) if country_totals else 1
            
            for country, count in genre_country_counts[genre].items():
                genre_ratio = count / total_genre_count
                country_activity = min(country_totals[country] / max_country_count, 1.0)
                
                fit_score = genre_ratio * 0.7 + country_activity * 0.3
                self.genre_fit_scores[genre][country] = fit_score
    
    def calculate_transition_matrix(self):
        """원작 기준 국가 간 전이 확률 계산 (전체)"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_country = original_record['국가']
                original_date = original_record['발간일']
                
                # 원작 이후 진출 국가들
                subsequent_records = group[
                    (group['발간일'] > original_date) | 
                    ((group['발간일'] == original_date) & (group['국가'] != original_country))
                ].sort_values('발간일')
                
                if len(subsequent_records) > 0:
                    # 원작 → 첫 번째 후속 국가
                    first_subsequent = subsequent_records.iloc[0]['국가']
                    transitions[original_country][first_subsequent] += 1
                    
                    # 후속 국가들 간의 전이도 계산
                    subsequent_countries = subsequent_records['국가'].tolist()
                    for i in range(len(subsequent_countries) - 1):
                        from_country = subsequent_countries[i]
                        to_country = subsequent_countries[i + 1]
                        transitions[from_country][to_country] += 1
        
        # 확률로 변환
        for from_country in transitions:
            total = sum(transitions[from_country].values())
            if total > 0:
                self.transition_matrix[from_country] = {}
                for to_country, count in transitions[from_country].items():
                    if count >= 2:  # 최소 2회 이상 전이
                        self.transition_matrix[from_country][to_country] = count / total

    def calculate_genre_transition_matrix(self):
        """장르별 원작 기준 국가 간 전이 확률 계산 - 다중 장르 지원"""
        genre_transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_country = original_record['국가']
                original_date = original_record['발간일']
                
                # 원작 작품의 모든 장르들
                original_genres = []
                for genre_col in self.genre_columns:
                    if not pd.isna(original_record[genre_col]):
                        original_genres.append(original_record[genre_col])
                
                # 원작 이후 진출 국가들
                subsequent_records = group[
                    (group['발간일'] > original_date) | 
                    ((group['발간일'] == original_date) & (group['국가'] != original_country))
                ].sort_values('발간일')
                
                if len(subsequent_records) > 0:
                    # 각 장르에 대해 전이 패턴 기록
                    for genre in original_genres:
                        # 원작 → 첫 번째 후속 국가
                        first_subsequent = subsequent_records.iloc[0]['국가']
                        genre_transitions[genre][original_country][first_subsequent] += 1
                        
                        # # 후속 국가들 간의 전이도 계산 (선택한 국가가 경유국가인 경우에도 카운트 됨!!)
                        # subsequent_countries = subsequent_records['국가'].tolist()
                        # for i in range(len(subsequent_countries) - 1):
                        #     from_country = subsequent_countries[i]
                        #     to_country = subsequent_countries[i + 1]
                        #     genre_transitions[genre][from_country][to_country] += 1
        
        # 확률로 변환
        for genre in genre_transitions:
            self.genre_transition_matrix[genre] = {}
            for from_country in genre_transitions[genre]:
                total = sum(genre_transitions[genre][from_country].values())
                if total > 0:
                    self.genre_transition_matrix[genre][from_country] = {}
                    
                    for to_country, count in genre_transitions[genre][from_country].items():
                        probability = count / total
                        
                        if count >= 1:  # 최소 1회 이상 전이
                            self.genre_transition_matrix[genre][from_country][to_country] = {
                                'probability': probability,
                                'count': count,
                                'total_transitions': total
                            }

    def recommend_next_countries(self, start_country, genre, prob_weight=0.7, genre_weight=0.2, conf_weight=0.1, top_k=10):

        """특정 국가에서 특정 장르 원작로 시작했을 때 다음 진출 국가 추천"""

        recommendations = []
        message = None 

        # 🔍 디버깅 정보를 문자열로 구성
        debug_info = f"\n🔍 원작 기준 추천: {start_country} → {genre}\n"
        debug_info += f"장르별 전이 매트릭스에 {genre} 있나? {genre in self.genre_transition_matrix}\n"
        
        if genre in self.genre_transition_matrix:
            debug_info += f"{genre}에서 {start_country} 있나? {start_country in self.genre_transition_matrix[genre]}\n"
            if start_country in self.genre_transition_matrix[genre]:
                debug_info += f"장르별 데이터: {list(self.genre_transition_matrix[genre][start_country].keys())}\n"
        
        debug_info += f"전체 전이 매트릭스에 {start_country} 있나? {start_country in self.transition_matrix}\n"
        if start_country in self.transition_matrix:
            debug_info += f"전체 전이 데이터: {list(self.transition_matrix[start_country].keys())}\n"
        
        # (1) 장르별 원작 전이 데이터가 있는 경우
        if (genre in self.genre_transition_matrix and 
            start_country in self.genre_transition_matrix[genre]):
            
            transitions = self.genre_transition_matrix[genre][start_country]
            
            for next_country, data in transitions.items():
                probability = data['probability'] # 진출 확률
                count = data['count']
                total_transitions = data['total_transitions']
                
                # 신뢰도 계산 (전이 횟수 기반)
                confidence = min(count / 12, 1.0)  # 24회 이상이면 최대 신뢰도
                
                # 장르 적합도 추가
                genre_fit = 0
                if genre in self.genre_fit_scores and next_country in self.genre_fit_scores[genre]:
                    genre_fit = self.genre_fit_scores[genre][next_country]
                
                # 최종 점수 계산 - 가중치 매개변수 사용
                final_score = (probability * prob_weight) + (genre_fit * genre_weight) + (confidence * conf_weight)
        
                
                recommendations.append({
                    'country': next_country,
                    'probability': probability * 100,
                    'confidence': confidence * 100,
                    'transition_count': count,
                    'total_from_start': total_transitions,
                    'genre_fit': genre_fit * 100,
                    'final_score': final_score * 100
                })
        
        # (2) 장르별 전이 데이터가 없는 경우 - 폴백(fallback), 전체 전이 확률 사용
        elif start_country in self.transition_matrix:
            transitions = self.transition_matrix[start_country]
            message = f"⚠️ {start_country}에서 {genre} 장르 원작의 수출 이력이 부족하여 전체 장르 평균을 표시합니다."
            for next_country, probability in transitions.items():
                # 장르 적합도 추가
                genre_fit = 0
                if genre in self.genre_fit_scores and next_country in self.genre_fit_scores[genre]:
                    genre_fit = self.genre_fit_scores[genre][next_country]
                
                # 최종 점수 계산 - 가중치 매개변수 사용  
                final_score = (probability * prob_weight) + (genre_fit * genre_weight)
        
                
                recommendations.append({
                    'country': next_country,
                    'probability': probability * 100,
                    'confidence': 50,  # 장르별 데이터가 없으므로 중간 신뢰도
                    'transition_count': '일반 전이 확률 기반',
                    'total_from_start': '전체',
                    'genre_fit': genre_fit * 100,
                    'final_score': final_score * 100
                })
        else:
            message = f"⚠️ {start_country}에서 원작로 출간된 진출 데이터가 충분하지 않습니다."

        # 정렬
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # ✅ 종합점수 통계를 문자열로 구성
        stats_info = ""
        if recommendations:
            final_scores = [rec['final_score'] for rec in recommendations]
            avg_score = sum(final_scores) / len(final_scores)
            
            stats_info = f"\n📊 종합점수 통계 - {start_country} → {genre}\n"
            stats_info += f"   추천 국가 수: {len(recommendations)}개\n"
            stats_info += f"   종합점수 평균: {avg_score:.2f}\n"
            stats_info += f"   최고점: {max(final_scores):.2f}\n"
            stats_info += f"   최저점: {min(final_scores):.2f}\n"
            stats_info += f"   점수 범위: {max(final_scores) - min(final_scores):.2f}\n"
        
        # 시간순 진출 패턴 계산 추가
        time_progression, timing_summary = self.calculate_time_based_progression(start_country, genre)
        
        # 기존 recommendations에 시간 정보 추가
        for rec in recommendations:
            country = rec['country']
            if country in timing_summary:
                rec['avg_days_from_original'] = timing_summary[country]['avg_days']
                rec['timing_rank'] = next((i+1 for i, (c, _) in enumerate(time_progression) if c == country), None)
            else:
                rec['avg_days_from_original'] = None
                rec['timing_rank'] = None


        return recommendations[:top_k], message, stats_info, time_progression 

    def get_genre_country_stats(self, selected_genres):
        """선택된 장르들의 국가별 진출 건수 반환 - 다중 장르 지원"""
        if not selected_genres:
            return pd.DataFrame()
        
        genre_country_stats = []
        
        for genre in selected_genres:
            # 해당 장르를 포함한 모든 기록 찾기 (genre1, genre2, genre3 중 어디든)
            genre_books = self.get_books_by_genre(genre)
            country_counts = genre_books['국가'].value_counts()
            
            for country, count in country_counts.items():
                genre_country_stats.append({
                    'genre': genre,
                    'country': country,
                    'count': count
                })
        
        return pd.DataFrame(genre_country_stats)
    


    def create_network_graph(self, start_country, recommendations, genre, time_progression=None):
        """네트워크 그래프 생성 (pyvis 사용) - 원작 기준 + 시간순 정보 + 범례"""
        if not recommendations:
            return None

        # pyvis 네트워크 객체 생성
        net = Network(
            height='650px',
            width='100%',
            bgcolor='#f8f9fa',
            font_color='black',
            notebook=True,
            cdn_resources='in_line'
        )

        # 물리 엔진 설정
        net.barnes_hut(
            gravity=-10000,
            central_gravity=0.3,
            spring_length=250,
            spring_strength=0.05,
            damping=0.09,
            overlap=0
        )

        # 중심 노드 (원작 출간 국가) 추가
        start_node_title = f"""
        {start_country}
        원작 출간 국가
        장르: {genre}
        """
        net.add_node(
            start_country,
            label=start_country,
            color='#FF6B6B',
            size=50,
            title=start_node_title,
            font={'size': 24, 'face': 'Arial Black', 'color': 'white'}
        )

        # 종합 점수 범위 계산
        if recommendations:
            final_scores = [rec['final_score'] for rec in recommendations]
            min_score = min(final_scores)
            max_score = max(final_scores)
            score_range = max_score - min_score if max_score > min_score else 1
        else:
            min_score, max_score, score_range = 0, 100, 100

        # 노드 크기 범위 설정
        MIN_NODE_SIZE = 20
        MAX_NODE_SIZE = 45

        # 추천 국가 노드 및 엣지 추가
        for i, rec in enumerate(recommendations):
            next_country = rec['country']
            prob = rec['probability']
            final_score = rec['final_score']
            
            # 종합점수에 따른 노드 크기 계산
            if score_range > 0:
                normalized_score = (final_score - min_score) / score_range
            else:
                normalized_score = 0.5
            
            size = int(MIN_NODE_SIZE + (normalized_score ** 0.7) * (MAX_NODE_SIZE - MIN_NODE_SIZE))
            size = max(MIN_NODE_SIZE, min(MAX_NODE_SIZE, size))

            # 확률에 따른 노드 색상 설정
            if prob >= 40:
                color = '#4ECDC4'  # 청록색 - 높은 확률
            elif prob >= 25:
                color = '#45B7D1'  # 파란색 - 중간 확률
            elif prob >= 15:
                color = '#96CEB4'  # 연두색 - 낮은 확률
            else:
                color = '#FFEAA7'  # 노란색 - 매우 낮은 확률

            # 노드 호버 정보에 시간 정보 추가
            timing_info = ""
            if rec.get('avg_days_from_original') is not None:
                avg_days = rec['avg_days_from_original']
                timing_rank = rec.get('timing_rank', '?')
                timing_info = f"""
            ⏰ 평균 진출 시점: {avg_days:.0f}일 후
            🏃 출간 시점 순위: {timing_rank}위"""

            hover_text = f"""
            ✈︎ {rec['country']}
            ────────────────────
            🎯 종합 점수: {final_score:.1f}
            📊 진출 확률: {prob:.1f}%
            🔄 전이 횟수: {rec['transition_count']}회
                (신뢰도: {rec['confidence']:.1f}%)
            📚 장르 적합도: {rec['genre_fit']:.1f}%
            📍 순위: {i+1}위{timing_info}
            🎯 원작 기준 분석
            """

            # 노드 라벨에도 출간 시점 순위 표시
            label_text = f"{next_country}\n({final_score:.0f}점)"
            if rec.get('timing_rank'):
                label_text = f"{next_country}\n({final_score:.0f}점)\n⏰{rec['timing_rank']}순"

            # 노드 추가
            net.add_node(
                next_country,
                label=label_text,
                color=color,
                size=size,
                title=hover_text,
                font={'size': max(14, int(size/3)), 'face': 'Arial'},
                borderWidth=2,
                borderWidthSelected=4
            )

            # 엣지 추가
            edge_width = max(2, final_score * 0.15)
            net.add_edge(
                start_country,
                next_country,
                value=edge_width,
                title=f"종합점수: {final_score:.1f} | 진출 확률: {prob:.1f}%",
                color={'color': '#888888', 'highlight': '#000000'}
            )

        # HTML 생성
        try:
            source_code = net.generate_html()
            
            # 🔴 NEW: 범례 추가를 위한 HTML/CSS/JS 수정
            legend_html = """
            <div id="legend" style="
                position: absolute;
                top: 20px;
                right: 20px;
                background: white;
                padding: 15px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                font-family: Arial, sans-serif;
                font-size: 14px;
                z-index: 1000;
                min-width: 200px;
            ">
                <h3 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">진출 확률 범례</h3>
                
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <div style="width: 20px; height: 20px; background: #FF6B6B; border-radius: 50%; margin-right: 10px;"></div>
                    <span>원작 출간 국가</span>
                </div>
                
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <div style="width: 20px; height: 20px; background: #4ECDC4; border-radius: 50%; margin-right: 10px;"></div>
                    <span>높은 확률 (40% 이상)</span>
                </div>
                
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <div style="width: 20px; height: 20px; background: #45B7D1; border-radius: 50%; margin-right: 10px;"></div>
                    <span>중간 확률 (25-40%)</span>
                </div>
                
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <div style="width: 20px; height: 20px; background: #96CEB4; border-radius: 50%; margin-right: 10px;"></div>
                    <span>낮은 확률 (15-25%)</span>
                </div>
                
                <div style="display: flex; align-items: center; margin: 8px 0;">
                    <div style="width: 20px; height: 20px; background: #FFEAA7; border-radius: 50%; margin-right: 10px;"></div>
                    <span>매우 낮음 (15% 미만)</span>
                </div>
                
                <hr style="margin: 12px 0; border: none; border-top: 1px solid #eee;">
                
                <div style="font-size: 12px; color: #666;">
                    💡 노드 크기: 종합점수 반영<br>
                    ⏰ 숫자: 출간 시점 순위
                </div>
            </div>
            """
            
            # 기존 HTML에 범례 삽입
            # body 태그 내부에 범례 추가
            if '<body>' in source_code:
                source_code = source_code.replace('<body>', f'<body>{legend_html}')
            else:
                # body 태그가 없는 경우 div 컨테이너 다음에 추가
                source_code = source_code.replace(
                    '<div id="mynetworkid"',
                    f'{legend_html}<div id="mynetworkid"'
                )
            
            # 반응형 디자인을 위한 추가 CSS
            responsive_css = """
            <style>
            @media (max-width: 768px) {
                #legend {
                    position: relative !important;
                    top: 0 !important;
                    right: 0 !important;
                    margin: 10px;
                    width: calc(100% - 20px);
                }
            }
            
            /* 범례 토글 버튼 (모바일용) */
            #legend-toggle {
                display: none;
                position: absolute;
                top: 10px;
                right: 10px;
                background: #4ECDC4;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px 10px;
                cursor: pointer;
                z-index: 1001;
            }
            
            @media (max-width: 768px) {
                #legend-toggle {
                    display: block;
                }
                
                #legend {
                    display: none;
                }
                
                #legend.show {
                    display: block;
                }
            }
            </style>
            """
            
            # 토글 버튼과 JavaScript 추가
            toggle_js = """
            <button id="legend-toggle" onclick="toggleLegend()">📊 범례</button>
            
            <script>
            function toggleLegend() {
                const legend = document.getElementById('legend');
                legend.classList.toggle('show');
            }
            </script>
            """
            
            # CSS와 JavaScript 삽입
            if '</head>' in source_code:
                source_code = source_code.replace('</head>', f'{responsive_css}</head>')
            
            if '<body>' in source_code:
                source_code = source_code.replace('<body>', f'<body>{toggle_js}')
            
            # 파일 저장
            with open('pyvis_graph.html', 'w', encoding='utf-8') as f:
                f.write(source_code)
            
            return source_code
            
        except Exception as e:
            st.error(f"그래프 생성 중 오류 발생: {e}")
            return None

    def get_start_country_stats(self, start_country, genre):
        """원작 기준 시작 국가의 통계 정보 - 다중 장르 지원"""
        # 먼저 해당 국가 + 원작 조건으로 필터링
        country_original_books = self.df[
            (self.df['국가'] == start_country) & 
            (self.df['원작여부'] == 'original')
        ]
        debug_msgs = []
        debug_msgs.append(f"🔍 디버그: {start_country} - {genre}")

        
        # 그 중에서 해당 장르를 포함한 작품 수 계산
        genre_books = set()
        for _, row in country_original_books.iterrows():
            book_id = row['book_id']
            # 해당 작품이 선택한 장르를 포함하는지 확인
            for genre_col in self.genre_columns:
                if not pd.isna(row[genre_col]) and row[genre_col] == genre:
                    genre_books.add(book_id)
                    break  # 하나라도 매치되면 충분
        
        original_books_count = len(genre_books)
        
        # 원작 기준 후속 진출 횟수 계산
        transition_count = 0
        # # 
        debug_msgs.append(f"장르 전이 매트릭스에 '{genre}' 있나? {genre in self.genre_transition_matrix}")
        
        if genre in self.genre_transition_matrix:
            debug_msgs.append(f"'{genre}'에서 '{start_country}' 있나? {start_country in self.genre_transition_matrix[genre]}")
            
            if start_country in self.genre_transition_matrix[genre]:
                transitions = self.genre_transition_matrix[genre][start_country]
                debug_msgs.append(f"전이 데이터: {transitions}")
                
                for next_country, data in transitions.items():
                    count = data['count']
                    debug_msgs.append(f"  {start_country} → {next_country}: {count}회")
                    transition_count += count
                
                debug_msgs.append(f"총 후속 진출 건수: {transition_count}")
            else:
                debug_msgs.append("해당 시작 국가 데이터 없음")
        else:
            debug_msgs.append("해당 장르 데이터 없음")
        
        return {
            'total_original_books': original_books_count,
            'total_transitions': transition_count,
            'transition_rate': (transition_count / original_books_count * 100) if original_books_count > 0 else 0,
            'debug_info': '\n'.join(debug_msgs)  # 디버그 정보 추가
        }, country_original_books
  
    def analyze_all(self):
        """전체 분석 실행"""
        self.calculate_hub_scores()
        self.calculate_genre_fit()
        self.calculate_transition_matrix()
        self.calculate_genre_transition_matrix()

    # 추가 (수출 시점 요소)
    def calculate_time_based_progression(self, start_country, genre):
        """원작 기준 시간순 진출 패턴 계산"""
        time_progressions = []
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_country = original_record['국가']
                original_date = original_record['발간일']
                
                # 해당 원작의 장르 확인
                original_genres = []
                for genre_col in self.genre_columns:
                    if not pd.isna(original_record[genre_col]):
                        original_genres.append(original_record[genre_col])
                
                # 선택한 국가와 장르가 맞는지 확인
                if original_country == start_country and genre in original_genres:
                    # 원작 이후 진출 국가들을 시간순으로 정렬
                    subsequent_records = group[
                        (group['발간일'] > original_date) | 
                        ((group['발간일'] == original_date) & (group['국가'] != original_country))
                    ].sort_values('발간일')
                    
                    if len(subsequent_records) > 0:
                        progression = []
                        for _, record in subsequent_records.iterrows():
                            progression.append({
                                'country': record['국가'],
                                'date': record['발간일'],
                                'days_from_original': (record['발간일'] - original_date).days
                            })
                        time_progressions.append(progression)
        
        # 국가별 평균 진출 시점 계산
        country_avg_timing = defaultdict(list)
        
        for progression in time_progressions:
            for step in progression:
                country_avg_timing[step['country']].append(step['days_from_original'])
        
        # 평균 계산
        country_timing_summary = {}
        for country, days_list in country_avg_timing.items():
            country_timing_summary[country] = {
                # 'avg_days': sum(days_list) / len(days_list), # 평균
                'avg_days': np.median(days_list),  #  median으로 변경
                'count': len(days_list),
                'min_days': min(days_list),
                'max_days': max(days_list)
            }
        
        # 평균 진출 시점 기준으로 정렬
        sorted_countries = sorted(country_timing_summary.items(), 
                                key=lambda x: x[1]['avg_days'])
        
        return sorted_countries, country_timing_summary

# 메인 앱
def main():
    import plotly.graph_objects as go
    import pandas as pd
    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;">
            <h1>문학 작품 해외 수출 추천 시스템</h1>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("👀**원작 출간을 기준으로 후속 진출 국가 및 장르를 분석합니다.**")
    st.caption(f"*데이터 출처: Goodreads, GoogleSearch*")

    st.markdown("---")
    


    # 사이드바
    st.sidebar.markdown("***")  
    st.sidebar.header("⚙️ 데이터 로딩")
    # 데이터 로드 버튼
    if st.sidebar.button("🔄 DB에서 데이터 불러오기", type="primary"):
        st.session_state.load_data = True

    # 파일 경로 설정
    

    
    # 분석기 초기화
    if 'analyzer_page1' not in st.session_state:
        st.session_state.analyzer_page1 = LiteratureExportAnalyzer()

    analyzer = st.session_state.analyzer_page1

    # 데이터 로드 상태 확인 - 페이지별로 관리 (추가)
    if 'data_loaded_page2' not in st.session_state:
        st.session_state.data_loaded_page2 = False

    # 데이터 로드 (변경)
    if not st.session_state.data_loaded_page2:  # ← analyzer.df is None에서 변경
        if st.session_state.get('load_data', False):
            with st.spinner("DB에서 데이터 로딩 중..."):
                success = analyzer.load_data_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
                if success:

                    # st.success(f"✅ DB 데이터 로드 완료: {len(analyzer.df):,}행")
                    original_count = len(analyzer.df[analyzer.df['원작여부'] == 'original'])
                    # st.success(f"✅ 원작 출간 기록: {original_count:,}건")
                    analyzer.analyze_all()
                    # st.success("✅ 원작 기준 분석 완료")
                    st.session_state.data_loaded_page2 = True  # 페이지2 로드 완료 플래그 (추가)
                    st.session_state.load_data = False  # 플래그 리셋
                else:
                    st.session_state.load_data = False
                    st.stop()
        else: 
            st.info("👆 사이드바에서 '데이터 불러오기' 버튼을 클릭하세요.")
            st.stop()

    # analyzer.df가 None인지 추가 체크 (추가)
    if analyzer.df is None:
        st.error("데이터가 로드되지 않았습니다. 데이터 불러오기 버튼을 다시 클릭해주세요.")
        st.stop()

    # 기본 통계
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("◎ 총 출간 기록", f"{len(analyzer.df):,}")
    with col2:
        original_count = len(analyzer.df[analyzer.df['원작여부'] == 'original'])
        st.metric("◎ 원작 출간", f"{original_count:,}")
    with col3:
        st.metric("◎ 진출 국가 수", f"{analyzer.df['국가'].nunique()}")
    with col4:
        st.metric("◎ 장르 수", f"{analyzer.df['genre1'].nunique()}")
    
    st.markdown("---")
    
    # 🔽 생키 다이어그램 섹션 - 조건부 표시
    if analyzer.df is not None:  # 메인 데이터가 로드된 후에만
        wave_df = analyzer.load_wave_data_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
        
        if wave_df is not None and len(wave_df) > 0:
            st.header("0️⃣ 확산 패턴 시각화")
            st.markdown("✨도서 출간 이후 확산 패턴을 다이어그램으로 확인하세요.")
            
            try:
                # st.success(f"✅ Wave 데이터 로드 완료: {len(wave_df):,}행")
                
                # Sankey Diagram에 필요한 데이터 형태로 가공
                # source_country가 없는 경우(원작)는 제외하고, source -> target 흐름을 집계
                sankey_data = wave_df.dropna(subset=['source_country'])
                sankey_data = sankey_data.groupby(['source_country', 'country']).size().reset_index(name='value')
                sankey_data = sankey_data.rename(columns={'source_country': 'source', 'country': 'target'})

                # 상위 30개 흐름만 선택 (데이터가 너무 많으면 다이어그램이 복잡해짐)
                sankey_data = sankey_data.sort_values(by='value', ascending=False).head(30)
                
                # 데이터가 있는지 확인
                if len(sankey_data) > 0:
                    # 전체 노드(국가) 리스트 생성 및 매핑
                    all_nodes = pd.concat([sankey_data['source'], sankey_data['target']]).unique()
                    node_map = {node: i for i, node in enumerate(all_nodes)}

                    # 링크(흐름) 데이터 생성
                    link_data = dict(
                        source=sankey_data['source'].map(node_map).tolist(),
                        target=sankey_data['target'].map(node_map).tolist(),
                        value=sankey_data['value'].tolist()
                    )

                    # Sankey Diagram 객체 생성
                    fig = go.Figure(data=[go.Sankey(
                        node=dict(
                            pad=15,
                            thickness=20,
                            line=dict(width=0),  # 테두리 제거
                            label=all_nodes.tolist(),
                            color="lightblue"
                        ),
                        link=link_data
                    )])

                    # 폰트 설정은 layout에서만 가능
                    fig.update_layout(
                        title_text="국가 간 도서 확산 흐름 (Sankey Diagram)", 
                        font=dict(
                            family="Arial, sans-serif",
                            size=20,  # 이 설정이 노드 라벨에도 적용됨
                            color="blue"
                        ),
                        height=900
                    )
                    
                    # Streamlit에서 시각화 (fig.show() 대신 사용)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 통계 정보 표시
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("◎ 총 흐름 수", f"{len(sankey_data)}")
                    with col2:
                        st.metric("◎ 관련 국가 수", f"{len(all_nodes)}")
                    with col3:
                        st.metric("◎ 총 이동 건수", f"{sankey_data['value'].sum():,}")
                        
                else:
                    st.warning("생키 다이어그램을 생성할 수 있는 데이터가 없습니다.")
                    
            except Exception as e:
                st.error(f"Wave 데이터 처리 오류: {e}")
                st.info("데이터 형식을 확인해주세요.")
            
        # wave 데이터가 없어도 에러 없이 넘어감 (else 블록 없음)
    
    st.markdown("---")
    # 🔼 여기까지 추가

    # 장르별 국가 진출 현황
    st.header("1️⃣ 장르별 국가 진출 현황 확인")
    st.markdown("✨분석하고 싶은 장르를 선택하여 해당 장르의 국가별 진출 건수를 확인하세요.")
    # 장르 체크박스
    available_genres = sorted(set().union(*[analyzer.df[f'genre{i}'].dropna().unique() for i in range(1, 5)]))

    # 장르 코드 매핑 딕셔너리
    # genre_mapping = {
    # "A": "Environment, Climate Disaster, Disaster",
    # "B": "Mystery, Thriller, Crime, Horror",
    # "C": "Science Fiction, Fantasy",
    # "D": "Capitalism, Labor, Poverty, Development, Urbanization, Democracy",
    # "E": "Diaspora, Migration, Refugees, Colonialism, Imperialism, War",
    # "F": "LGBTQ, Gender Equality, Disability",
    # "G": "Religion, Mythology",
    # "H": "Relationships (Healing), Family, Neighbors, Friendship, Coming-of-age",
    # "I": "Romance",
    # "J": "History",
    # "미분류": "미분류"
    # }
    genre_mapping = {
    "A": "환경, 기후재난, 재난",
    "B": "미스터리, 스릴러, 범죄, 호러",
    "C": "SF, 판타지",
    "D": "자본주의, 노동, 빈곤, 개발, 도시화, 민주주의",
    "E":  "이산, 이주, 난민, 식민주의, 제국주의, 전쟁",
    "F":  "LGBTQ, 성평등, 장애",
    "G":  "종교, 신화",
    "H": "관계(힐링), 가족, 이웃, 우정, 성장",
    "I":  "로맨스",
    "J":  "역사",
    "미분류":  "기타"
    }

    # 🔥 역매핑 딕셔너리 추가 (한국어명 → 알파벳 코드)
    reverse_genre_mapping = {v: k for k, v in genre_mapping.items()}

    # available_genres를 실제 장르명으로 변환
    available_genres = [genre_mapping.get(code, code) for code in available_genres]

    st.subheader("📚 장르 선택")

    selected_genres = st.multiselect(
    "확인할 장르를 선택하세요 (여러 개 선택 가능)",
    available_genres,
    default=[],
    help="드롭다운에서 여러 장르를 선택할 수 있습니다. 최대한 많이 선택해도 됩니다."
    )

    # 선택된 장르 표시
    if selected_genres:
        st.success(f"✅ 선택된 장르 ({len(selected_genres)}개): {', '.join(selected_genres)}")

    # 선택된 장르 분석
    if selected_genres:
        # 🔥 한국어 장르명을 알파벳 코드로 변환
        selected_genre_codes = [reverse_genre_mapping.get(genre, genre) for genre in selected_genres]
        
        # 변환된 코드로 분석 실행
        genre_country_df = analyzer.get_genre_country_stats(selected_genre_codes)
        
        if not genre_country_df.empty:
            st.subheader(f"📈 선택된 장르의 **국가별 출간** 건수")
            # st.markdown(f"**선택된 장르**: {', '.join(selected_genres)}")
            
            if len(selected_genres) == 1:
                genre_code = selected_genre_codes[0]  # 코드 사용
                genre_data = genre_country_df[genre_country_df['genre'] == genre_code].nlargest(15, 'count')
                
                fig = px.bar(
                    genre_data,
                    x='country',
                    y='count',
                    title=f'🕮{selected_genres[0]} 장르의 국가별 출간 건수 (상위 15개국)',  # 한국어명 표시
                    labels={'country': '국가', 'count': '출간 건수'},
                    color='count',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
        else:
            comparison_data = []
            for i, genre_code in enumerate(selected_genre_codes):
                top_countries = genre_country_df[
                    genre_country_df['genre'] == genre_code
                ].nlargest(10, 'count')
                
                # 🔥 차트에서는 한국어명으로 표시하기 위해 변환
                top_countries = top_countries.copy()
                top_countries['genre'] = selected_genres[i]  # 한국어명으로 변경
                
                comparison_data.extend(top_countries.to_dict('records'))
            
            comparison_df = pd.DataFrame(comparison_data)
            
            fig = px.bar(
                comparison_df,
                x='country',
                y='count',
                color='genre',
                title=f'선택된 장르들의 국가별 출간 건수 비교 (각 장르별 상위 10개국)',
                labels={'country': '국가', 'count': '출간 건수', 'genre': '장르'},
                barmode='group'
            )
            fig.update_xaxes(tickangle=45)
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)




    st.markdown("---")
    
    # 원작 기준 추천 시스템        
    st.header("2️⃣ 후속 진출 국가 추천")
    st.markdown("✨선택한 국가에서 해당 장르를 **원작으로 출간**했을 때, 다음에 어느 국가로 진출하는 경향이 있는지 분석합니다.")
    

    st.markdown("---")
    # 사이드바

    st.sidebar.markdown("---")
    # 가중치 설정 추가
    st.sidebar.subheader("⚖️ 종합점수 가중치 설정")
    st.sidebar.markdown("**🕸️** **후속 진출 국가 네트워크**의 종합점수에 가중치를 설정합니다.")
    prob_weight = st.sidebar.slider("진출 확률 가중치", 0.0, 1.0, 0.7, 0.1)
    genre_weight = st.sidebar.slider("장르 적합도 가중치", 0.0, 1.0, 0.2, 0.1)
    conf_weight = st.sidebar.slider("신뢰도 가중치", 0.0, 1.0, 0.1, 0.1)

    # 가중치 합계 확인
    total_weight = prob_weight + genre_weight + conf_weight
    if total_weight != 1.0:
        st.sidebar.warning(f"⚠️ 가중치 합계: {total_weight:.1f} (권장: 1.0)")
    else:
        st.sidebar.success("✅ 가중치 합계: 1.0")


    # ------------- #
    col1, col2 = st.columns([2, 1])

    with col1:
        
        
        available_genre_codes = analyzer.get_all_genres()  # 장르 코드들 (A, B, C, ...)
        available_genre_names = [genre_mapping.get(code, code) for code in available_genre_codes]  # 실제 장르명으로 변환
        
        selected_genre_name = st.selectbox(
            "📖 장르 선택", 
            available_genre_names,
            index=0 if 'Romance' not in available_genre_names else available_genre_names.index('Romance')
        )
        
        # 선택된 장르명에서 다시 코드로 변환 (분석에 사용)
        reverse_mapping = {v: k for k, v in genre_mapping.items()}
        selected_genre = reverse_mapping.get(selected_genre_name, selected_genre_name)
        
        st.caption(f"*장르 출처: GoogleSearch*")

    with col2:
        available_countries = sorted(analyzer.df['국가'].unique())
        start_country = st.selectbox("🚀 원작 출간 국가 선택", available_countries)

    # 추천 실행
    if st.button("🔍 분석 실행", type="primary"):
        with st.spinner("원작 기준 후속 진출 패턴 분석 중..."):
            # 시작 국가 통계 (selected_genre는 코드 사용)
            start_stats, df = analyzer.get_start_country_stats(start_country, selected_genre)
            
            # st.write(df)
            
            # 후속 국가 추천
            recommendations, warning_message, debug_stats, time_progression = analyzer.recommend_next_countries(
                    start_country, selected_genre, 
                    prob_weight=prob_weight, 
                    genre_weight=genre_weight, 
                    conf_weight=conf_weight,
                    top_k=8
                )
            
            # 시작 국가 정보 표시 (화면에는 장르명 표시)
            st.subheader(f"🌍 {start_country} » 📍 ({selected_genre_name} 장르)")
            
            info_col1, info_col2, info_col3 = st.columns(3)
            with info_col1:
                st.metric("원작 작품 수", f"{start_stats['total_original_books']}개")
            with info_col2:
                st.metric("후속 진출 건수", f"{start_stats['total_transitions']}건")
            with info_col3:
                st.metric("후속 진출률", f"{start_stats['transition_rate']:.1f}%")
            
            st.markdown("---")
            
            # 추천 결과 표시
            if recommendations:

                # 🔴 NEW: 시간순 진출 패턴 표시 추가
                if time_progression:
                    st.subheader("⏰ 시간순 진출 패턴")
                    timing_text = " → ".join([f"{country} ({data['avg_days']:.0f}일)" 
                                            for country, data in time_progression[:5]])
                    st.info(f"평균 진출 순서: {timing_text}")
                # 네트워크 그래프 생성 및 표시
                st.subheader("🕸️ 후속 진출 국가 네트워크")
                st.write(f"  ᯓ ✈︎ **{start_country}에서 {selected_genre} 장르를 원작로 출간한 후 진출 경향성**")
                
                # 경고메세지가 있으면 출력
                if warning_message:
                    st.warning(warning_message)
                
                # 컬럼 분할: 네트워크 그래프와 텍스트 정보
                graph_col, info_col = st.columns([5, 2])
                
                with graph_col:
                    network_html = analyzer.create_network_graph(start_country, recommendations, selected_genre)
                    if network_html:
                        components.html(network_html, height=660, scrolling=False)
                
                with info_col:
                    st.markdown("### 📋 지표 설명")
                    st.write("""
                        🎯 **종합 점수:** 모든 요소를 종합한 최종 추천 점수 (0~100점)\n
                        📊 **진출 확률:** 원작 출간 후 해당 국가로 실제 진출한 비율\n
                        🔄 **전이 횟수:** 실제로 원작 국가 → 후속 국가로 진출한 작품의 총 건수\n
                        (신뢰도: 통계적 신뢰성, 12건 이상이면 100%)\n
                        📚 **장르 적합도:** 해당 장르가 목표 국가에서 얼마나 인기있는지 점수\n
                        ⏰ **평균 진출 시점:** 작품당 국가A->국가B로 진출하기까지 걸린 일 수의 중앙값\n 
                        🏃 **출간 시점 순위:** 평균 진출 시점 순위\n 
                        📍 **순위:** 종합 점수를 기준으로 한 추천 우선순위
                        """)
                    st.text(debug_stats)
                    
                st.markdown("---")
                
                # 추가 차트
                st.subheader("📈 추가 시각화")
                
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["원작 기준 진출 확률", "종합 분석", "시간순 분석"])

                
                with chart_tab1:
                    # 진출 확률 막대 차트
                    chart_data = recommendations[:8]
                    fig = px.bar(
                        x=[rec['country'] for rec in chart_data],
                        y=[rec['probability'] for rec in chart_data],
                        title=f"{start_country}에서 {selected_genre} 출간 후 후속 진출 확률",
                        labels={'x': '후속 진출 국가', 'y': '진출 확률 (%)'},
                        color=[rec['probability'] for rec in chart_data],
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_tab2:
                    # 진출 확률 vs 신뢰도 산점도
                    fig = px.scatter(
                        x=[rec['confidence'] for rec in recommendations[:8]],
                        y=[rec['probability'] for rec in recommendations[:8]],
                        text=[rec['country'] for rec in recommendations[:8]],
                        title="신뢰도 vs 진출 확률",
                        labels={'x': '신뢰도 (%)', 'y': '진출 확률 (%)'},
                        size=[rec['final_score'] for rec in recommendations[:8]],
                        color=[rec['genre_fit'] for rec in recommendations[:8]],
                        color_continuous_scale="plasma"
                    )
                    fig.update_traces(textposition="top center")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                with chart_tab3:
                    if time_progression:
                        # 시간순 진출 막대 차트
                        timing_data = time_progression[:8]
                        fig = px.bar(
                            x=[country for country, _ in timing_data],
                            y=[data['avg_days'] for _, data in timing_data],
                            title=f"{start_country}에서 {selected_genre} 출간 후 평균(중앙값) 진출 시점",
                            labels={'x': '진출 국가', 'y': '평균 진출 시점 (일)'},
                            color=[data['avg_days'] for _, data in timing_data],
                            color_continuous_scale="viridis"
                        )
                        fig.update_layout(showlegend=False, height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 상세 시간 정보 테이블
                        timing_df = pd.DataFrame([
                            {
                                '순위': i+1,
                                '국가': country,
                                '평균 진출 시점': f"{data['avg_days']:.0f}일",
                                '진출 건수': data['count'],
                                '최빠른 진출': f"{data['min_days']}일",
                                '가장 늦은 진출': f"{data['max_days']}일"
                            }
                            for i, (country, data) in enumerate(timing_data)
                        ])
                        st.dataframe(timing_df, use_container_width=True)
            else:
                st.warning(f"⚠️ {start_country}에서 {selected_genre} 장르 출간 후 후속 진출 데이터가 충분하지 않습니다.")
                st.info("다른 국가나 장르를 선택해보세요.")
    
    # 전체 분석 결과
    st.markdown("---")
    st.header("📋 전체 분석 결과")
    
    analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["원작 거점 국가", "장르별 분포", "국가별 분포"])
    
    with analysis_tab1:
        if analyzer.hub_scores:
            st.subheader("🏆 거점 국가 순위")
            st.markdown("*◎ 원작 출간 후 평균적으로 많은 국가로 진출하는 거점 역할을 하는 국가들*")
            st.markdown("◎ 거점 지수 = 총 후속 진출 건수 / 원작 작품 수")
            hub_df = pd.DataFrame([
                {
                    '순위': i+1,
                    '국가': country,
                    '거점 지수': f"{data['hub_index']:.2f}",
                    '원작 작품 수': data['total_books'],
                    '평균 후속 진출': f"{data['avg_subsequent']:.1f}개국"
                }
                for i, (country, data) in enumerate(
                    sorted(analyzer.hub_scores.items(), key=lambda x: x[1]['hub_index'], reverse=True)
                )
            ])
            st.dataframe(hub_df, use_container_width=True)
    
    with analysis_tab2:
        st.subheader("📚 장르별 작품 분포")
    

        # 원작만 필터링한 후 장르 카운트
        original_df = analyzer.df[analyzer.df['원작여부'] == 'original']
        all_genre_counts = defaultdict(int)
        for genre_col in analyzer.genre_columns:
            genre_counts = original_df[genre_col].value_counts()  # ← 원작만 사용
            for genre, count in genre_counts.items():
                all_genre_counts[genre] += count
        
        # 장르 코드를 장르명으로 변환
        mapped_genre_counts = {}
        for genre_code, count in all_genre_counts.items():
            genre_name = genre_mapping.get(genre_code, genre_code)  # 매핑되지 않으면 원래 코드 사용
            mapped_genre_counts[genre_name] = mapped_genre_counts.get(genre_name, 0) + count
        
        # 상위 10개 장르만 표시
        top_genres = dict(sorted(mapped_genre_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig = px.pie(
            values=list(top_genres.values()),
            names=list(top_genres.keys()),
            title="상위 10개 장르별 작품 분포"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    
    with analysis_tab3:
        st.subheader("🌍 국가별 진출 현황")
        country_counts = analyzer.df['국가'].value_counts().head(15)
        fig = px.bar(
            x=country_counts.index,
            y=country_counts.values,
            title="상위 15개국 진출 작품 수",
            labels={'x': '국가', 'y': '작품 수'}
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()