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
import math
import os 
from PIL import Image
import os
import platform
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


def get_secrets():
    """Secrets 정보를 안전하게 가져오는 함수"""
    try:
        # Streamlit Cloud secrets 접근
        return {
            'app_password': st.secrets["app_password"],
            'db_host': st.secrets["database"]["host"],
            'db_name': st.secrets["database"]["database"],
            'db_user': st.secrets["database"]["user"],
            'db_password': st.secrets["database"]["password"]
        }
    except Exception as e:
        st.error(f"Secrets 접근 오류: {e}")
        st.info("Streamlit Cloud Secrets 설정을 확인해주세요.")
        return None





class LiteratureExportAnalyzer:
    def __init__(self):
        self.df = None
        self.hub_scores = {}
        self.genre_fit_scores = {}
        self.transition_matrix = {}
        self.genre_transition_matrix = {}
        
    def load_data_from_db(self, host, database, user, password, start_date=None, end_date=None):
        """DB에서 데이터 로드 및 전처리"""
        try:
            connection = mysql.connector.connect(
                host=host,
                database=database,
                user=user,
                password=password
            )
            
            if connection.is_connected():
                # 기본 쿼리 - 제목 컬럼 추가
                base_query = """
                    SELECT book_id, 발간일, genre1, genre2, genre3, genre4, 국가, 원작여부, 출판사명, 작가명, `ISBN(13)`, URL, 에디션_제목, 
                    COALESCE(원작_제목, book_id) as title
                    FROM literature_books
                """
                
                # 날짜 필터 조건 추가
                where_conditions = []
                if start_date:
                    where_conditions.append(f"발간일 >= '{start_date}'")
                if end_date:
                    where_conditions.append(f"발간일 <= '{end_date}'")
                
                if where_conditions:
                    query = base_query + " WHERE " + " AND ".join(where_conditions)  + " AND book_id LIKE 'ko_%' "
                else:
                    query = base_query + " WHERE book_id LIKE 'ko_%' "
                
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
                FROM literature_books_wave_ko
                """
                
                wave_df = pd.read_sql(query, connection)
                connection.close()
                return wave_df
                
        except Error as e:
            st.error(f"Wave 데이터 DB 연결 오류: {e}")
            return None

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
    
    def get_all_original_books(self):
        """모든 원작 작품 목록을 반환 (장르 정보 포함)"""
        original_books = self.df[self.df['원작여부'] == 'original'].copy()
        
        # book_id별로 그룹화하여 중복 제거 (같은 책이 여러 국가에서 원작으로 출간된 경우)
        unique_originals = original_books.groupby('book_id').first().reset_index()
        
        # 모든 장르 정보 수집
        def get_book_genres(row):
            genres = []
            for genre_col in self.genre_columns:
                if not pd.isna(row[genre_col]):
                    genres.append(row[genre_col])
            return ', '.join(genres) if genres else '미분류'
        
        unique_originals['genres'] = unique_originals.apply(get_book_genres, axis=1)
        
        # return unique_originals[['book_id', 'title', '발간일', '국가', 'genres']].sort_values('발간일', ascending=False)
        return unique_originals.sort_values('발간일', ascending=False)
    
    def get_original_books_by_genre(self, selected_genre):
        """특정 장르의 원작 작품 목록을 반환"""
        genre_books = self.get_books_by_genre(selected_genre)
        original_books = genre_books[genre_books['원작여부'] == 'original']
        
        # book_id별로 그룹화하여 중복 제거 (같은 책이 여러 국가에서 원작으로 출간된 경우)
        unique_originals = original_books.groupby('book_id').first().reset_index()
        
        # return unique_originals[['book_id', 'title', '발간일', '국가']].sort_values('발간일', ascending=False)
        return unique_originals.sort_values('발간일', ascending=False)
    
    def get_book_export_path(self, book_id):
        """특정 책의 발간 흐름 분석"""
        book_data = self.df[self.df['book_id'] == book_id].sort_values('발간일')
        
        if len(book_data) == 0:
            return None, None, None
        
        # 원작 정보 찾기
        original_records = book_data[book_data['원작여부'] == 'original']
        if len(original_records) == 0:
            return None, None, None
        
        original_record = original_records.iloc[0]
        original_country = original_record['국가']
        original_date = original_record['발간일']
        
        # 발간 흐름 추적
        export_path = []
        for _, record in book_data.iterrows():
            export_path.append({
                'country': record['국가'],
                'date': record['발간일'],
                'is_original': record['원작여부'] == 'original',
                'days_from_original': (record['발간일'] - original_date).days, 
                # 추가 
                '출판사명' : record['출판사명'], 
                '에디션_제목' : record['에디션_제목'],
                'ISBN(13)' : record['ISBN(13)'], 
                'URL' : record['URL']
            })
        
        # 장르 정보 수집
        genres = []
        for genre_col in self.genre_columns:
            if not pd.isna(original_record[genre_col]):
                genres.append(original_record[genre_col])
        
        book_info = {
            'book_id': book_id,
            'title': original_record.get('title', book_id),
            'original_country': original_country,
            'original_date': original_date,
            'genres': genres,
            'total_countries': len(book_data['국가'].unique()),
            'total_days': (book_data['발간일'].max() - original_date).days
        }
        
        return book_info, export_path, book_data
    

    # (1) 
    def create_book_export_network(self, book_info, export_path):
        """개별 책의 발간 흐름 네트워크 그래프 생성"""
        if not export_path or len(export_path) <= 1:
            return None
        
        net = Network(
            height='500px',
            width='100%',
            bgcolor='#f8f9fa',
            font_color='black',
            notebook=True,
            cdn_resources='in_line'
        )
        
        net.barnes_hut(
            gravity=-5000,
            central_gravity=0.3,
            spring_length=200,
            spring_strength=0.05,
            damping=0.09,
            overlap=0
        )
        
        # 노드 및 엣지 추가
        for i, step in enumerate(export_path):
            country = step['country']
            is_original = step['is_original']
            days_from_original = step['days_from_original']
            
            # 노드 색상 및 크기 설정
            if is_original:
                color = '#FF6B6B'  # 빨간색 - 원작
                size = 50
                label_text = f"{country}\n(원작)"
            else:
                color = '#4ECDC4'  # 청록색 - 수출
                size = 35
                label_text = f"{country}\n({days_from_original}일)"
            
            hover_text = f"""
            📍 {country}
            ────────────────
            📅 출간일: {step['date'].strftime('%Y-%m-%d')}
            {'🏠 원작 출간국' if is_original else f'📈 수출 ({days_from_original}일 후)'}
            📊 순서: {i+1}번째
            """
            
            net.add_node(
                country,
                label=label_text,
                color=color,
                size=size,
                title=hover_text,
                font={'size': max(14, int(size/3)), 'face': 'Arial'}
            )
            
            # 이전 국가와 연결 (시간순)
            if i > 0:
                prev_country = export_path[i-1]['country']
                prev_date = export_path[i-1]['date']  # 직전 국가의 날짜
                current_date = step['date']  # 현재 국가의 날짜
                
                # 직전 국가로부터 며칠 후인지 계산
                days_from_prev = (current_date - prev_date).days

                edge_width = max(3, 20 - i*2)  # 초기 수출일수록 굵게
                
                net.add_edge(
                    prev_country,
                    country,
                    value=edge_width,
                    label=f"+{days_from_prev}일",  # 엣지 라벨 추가
                    title=f"{prev_country} → {country} ({days_from_prev}일 후)",
                    color={'color': '#888888', 'highlight': '#000000'}
                )
        
        # 범례 추가
        legend_html = f"""
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
            <h3 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">{book_info['title']}</h3>
            
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <div style="width: 20px; height: 20px; background: #FF6B6B; border-radius: 50%; margin-right: 10px;"></div>
                <span>원작 출간국</span>
            </div>
            
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <div style="width: 20px; height: 20px; background: #4ECDC4; border-radius: 50%; margin-right: 10px;"></div>
                <span>수출 국가</span>
            </div>
            
            <hr style="margin: 12px 0; border: none; border-top: 1px solid #eee;">
            
            <div style="font-size: 12px; color: #666;">
                📚 총 {book_info['total_countries']}개국 진출<br>
                ⏰ 총 {book_info['total_days']}일 소요<br>
                🎭 장르: {', '.join(book_info['genres'])}
            </div>
        </div>
        """
        
        try:
            source_code = net.generate_html()
            
            # 범례 추가
            if '<body>' in source_code:
                source_code = source_code.replace('<body>', f'<body>{legend_html}')
            else:
                source_code = source_code.replace(
                    '<div id="mynetworkid"',
                    f'{legend_html}<div id="mynetworkid"'
                )
            
            return source_code
            
        except Exception as e:
            st.error(f"그래프 생성 중 오류 발생: {e}")
            return None


    # ------------------------------------------------------------ # 
    
    # # (3-1)
    def create_book_export_timeline(self, book_info, export_path):
        """화살표 형태의 타임라인 차트 생성 (지그재그 형태로 배치)"""
        if not export_path or len(export_path) <= 1:
            return None
        
        # 데이터 준비
        timeline_data = []
        for i, step in enumerate(export_path):
            timeline_data.append({
                'order': i,
                'country': step['country'],
                'date': step['date'],
                'is_original': step['is_original'],
                'days_from_original': step['days_from_original']
            })
        
        df = pd.DataFrame(timeline_data)
        
        # 지그재그 위치 계산
        positions = []
        max_per_row = 5
        total_rows = (len(df) - 1) // max_per_row + 1
        
        for i in range(len(df)):
            row = i // max_per_row  # 현재 행 (0, 1, 2, ...)
            col_in_row = i % max_per_row   # 현재 행에서의 위치 (0, 1, 2, 3, 4)
            
            # 지그재그 패턴: 홀수 행은 역순으로 배치
            if row % 2 == 0:  # 짝수 행 (0, 2, 4, ...): 왼쪽에서 오른쪽으로
                x_pos = col_in_row
            else:  # 홀수 행 (1, 3, 5, ...): 오른쪽에서 왼쪽으로
                x_pos = max_per_row - 1 - col_in_row
            
            y_pos = total_rows - 1 - row  # 위에서부터 아래로 (역순)
            
            positions.append({'x': x_pos, 'y': y_pos, 'row': row, 'col': col_in_row})
        
        # 색상 설정
        colors = ['#FF6B6B' if is_orig else '#4ECDC4' for is_orig in df['is_original']]
        
        # 기본 산점도 생성
        fig = go.Figure()
        
        # 국가별 점 추가
        fig.add_trace(go.Scatter(
            x=[pos['x'] for pos in positions],
            y=[pos['y'] for pos in positions],
            mode='markers+text',
            marker=dict(
                color=colors,
                size=[25 if is_orig else 20 for is_orig in df['is_original']],
                line=dict(width=2, color='white')
            ),
            text=df['country'],
            textposition='top center',
            textfont=dict(size=16, color='black'),
            hovertemplate='<b>%{text}</b><br>' +
                        '날짜: %{customdata[0]}<br>' +
                        '경과일수: %{customdata[1]}일<br>' +
                        '<extra></extra>',
            customdata=list(zip(df['date'].dt.strftime('%Y-%m-%d'), df['days_from_original'])),
            showlegend=False
        ))
        
        # 화살표 선 추가 (지그재그 패턴 고려)
        for i in range(len(df) - 1):
            current_pos = positions[i]
            next_pos = positions[i + 1]
            current_row = current_pos['row']
            next_row = next_pos['row']
            
            if current_row == next_row:
                # 같은 행 내에서의 이동
                # 짝수 행: 왼쪽→오른쪽, 홀수 행: 오른쪽→왼쪽
                if current_row % 2 == 0:
                    # 짝수 행: 오른쪽으로
                    arrow_start_x = current_pos['x'] + 0.1
                    arrow_end_x = next_pos['x'] - 0.1
                    arrow_symbol = "▶"
                else:
                    # 홀수 행: 왼쪽으로
                    arrow_start_x = current_pos['x'] - 0.1
                    arrow_end_x = next_pos['x'] + 0.1
                    arrow_symbol = "◀"
                
                # 화살표 선
                fig.add_shape(
                    type="line",
                    x0=arrow_start_x, y0=current_pos['y'],
                    x1=arrow_end_x, y1=next_pos['y'],
                    line=dict(color="#666666", width=1)
                )
                
                # 화살표 머리
                fig.add_annotation(
                    x=arrow_end_x,
                    y=next_pos['y'],
                    text=arrow_symbol,
                    showarrow=False,
                    font=dict(color="#666666", size=12),
                    xanchor="center", yanchor="middle"
                )
            else:
                # 다른 행으로 이동 (행 끝에서 다음 행 시작으로)
                mid_y = (current_pos['y'] + next_pos['y']) / 2
                
                # 현재 행이 짝수면 오른쪽 끝에서, 홀수면 왼쪽 끝에서 내려감
                if current_row % 2 == 0:
                    # 짝수 행 끝 (오른쪽 끝)
                    start_offset = 0.1
                else:
                    # 홀수 행 끝 (왼쪽 끝)
                    start_offset = -0.1
                
                # 수직 선 (아래로)
                fig.add_shape(
                    type="line",
                    x0=current_pos['x'], y0=current_pos['y'] - 0.1,
                    x1=current_pos['x'], y1=mid_y,
                    line=dict(color="#666666", width=1)
                )
                
                # 수평 선 (다음 위치로)
                fig.add_shape(
                    type="line",
                    x0=current_pos['x'], y0=mid_y,
                    x1=next_pos['x'], y1=mid_y,
                    line=dict(color="#666666", width=1)
                )
                
                # 수직 선 (위로)
                fig.add_shape(
                    type="line",
                    x0=next_pos['x'], y0=mid_y,
                    x1=next_pos['x'], y1=next_pos['y'] + 0.1,
                    line=dict(color="#666666", width=1)
                )
                
                # 최종 화살표 머리 (아래쪽을 향함)
                fig.add_annotation(
                    x=next_pos['x'],
                    y=next_pos['y'] + 0.05,
                    text="▼",
                    showarrow=False,
                    font=dict(color="#666666", size=12),
                    xanchor="center", yanchor="middle"
                )
        
        # 레이아웃 설정
        fig.update_layout(
            title=dict(
                text=f"📗'{book_info.get('title', '제목 없음')}' 출간 경로 타임라인",
                x=0,  # 수평 위치
                y=0.95, # 세로 위치 (0~1, 1이 최상단)
                font=dict(size=16, color='#2C3E50')
            ),
            xaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.5, max_per_row - 0.5]
            ),
            yaxis=dict(
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-0.5, total_rows - 0.5]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=200 + (total_rows * 100),  # 행 수에 따라 동적 높이 조정
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=[
                dict(
                    text="🔴 원본 출간 &nbsp;&nbsp; 🔵 번역/재출간",
                    x=0.5, y=-0.15,
                    xref='paper', yref='paper',
                    showarrow=False,
                    font=dict(size=10, color='#7F8C8D'),
                    xanchor='center'
                )
            ]
        )
        
        return fig

    # ------------------------------------------------------------ # 

    def calculate_hub_scores(self):
        """거점 지수 계산 - 원작 기준, 베이지안 평활화 적용"""
        book_patterns = {}
        
        # 1️⃣ 각 책별로 원작 → 후속 진출 패턴 수집
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
        
        # 2️⃣ 원작 국가별 거점 점수 계산
        hub_analysis = defaultdict(lambda: {'total_books': 0, 'total_subsequent': 0})
        
        for pattern in book_patterns.values():
            original_country = pattern['original_country']
            subsequent_count = pattern['subsequent_count']
            
            hub_analysis[original_country]['total_books'] += 1
            hub_analysis[original_country]['total_subsequent'] += subsequent_count
        
        # 거점 지수 최종 계산 (베이지안 평활화 적용)
        total_books_all = 0
        total_subsequent_all = 0
        
        for country, data in hub_analysis.items():
            if data['total_books'] >= 1:  # 최소 1개 작품
                total_books_all += data['total_books']
                total_subsequent_all += data['total_subsequent']
        
        # 전체 평균 거점지수 μ
        mu = total_subsequent_all / total_books_all if total_books_all > 0 else 0
        
        # Step 2: 임계치 설정
        m = 50  # 임계치
        
        # Step 3: 각 국가별 베이지안 평활화된 거점지수 계산
        for country, data in hub_analysis.items():
            if data['total_books'] >= 10:  # 최소 작품 갯수 설정
                # 해당 국가의 원시 거점지수 r
                r = data['total_subsequent'] / data['total_books']
                
                # 해당 국가의 작품 수 v
                v = data['total_books']
                
                # 베이지안 평활화된 거점지수 계산
                bayesian_hub_index = (mu * m + r * v) / (m + v)
                
                self.hub_scores[country] = {
                    'hub_index': bayesian_hub_index,  # 베이지안 평활화된 값
                    'raw_hub_index': r,              # 원시 거점지수 (참고용)
                    'total_books': data['total_books'],
                    'avg_subsequent': bayesian_hub_index,
                    'global_avg': mu                 # 전체 평균 (참고용)
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
    
    def analyze_all(self):
        """전체 분석 실행"""
        self.calculate_hub_scores()
        self.calculate_genre_fit()

# 메인 앱
def main():
    # 비밀번호 입력
    secret_key_user = st.text_input(':closed_lock_with_key: **Secret Key**',
                                    placeholder='비밀번호를 입력해주세요.',
                                    type="password")
    st.write('asdf')
    # 플랫폼에 따른 설정 가져오기
    if platform.system() == "Linux":
        # 리눅스 환경 (서버 환경) - 환경변수 사용
        try:
            from dotenv import load_dotenv
            load_dotenv()
            correct_password = os.environ.get("APP_PASSWORD")
            DB_HOST = os.environ.get("DB_HOST")
            DB_NAME = os.environ.get("DB_NAME") 
            DB_USER = os.environ.get("DB_USER")
            DB_PASSWORD = os.environ.get("DB_PASSWORD")
        except:
            st.error("환경변수 설정을 찾을 수 없습니다.")
            st.stop()
    else:
        # 비리눅스 환경 (Streamlit Cloud) - secrets 사용
        st.write("🔍 DEBUG: 비리눅스 환경 감지, secrets 접근 시도")
        st.write(f"🔍 DEBUG: st.secrets 키 목록: {list(st.secrets.keys())}")
        
        secrets = get_secrets()
        st.write(f"🔍 DEBUG: get_secrets() 결과: {secrets is not None}")
        
        if secrets:
            correct_password = secrets['app_password']
            DB_HOST = secrets['db_host']
            DB_NAME = secrets['db_name']
            DB_USER = secrets['db_user']
            DB_PASSWORD = secrets['db_password']
            st.write("✅ DEBUG: secrets에서 설정값 로드 완료")
        else:
            st.write("❌ DEBUG: secrets 로드 실패")
            st.error("Streamlit Cloud Secrets 설정을 찾을 수 없습니다.")
            st.stop()
    
    # 비밀번호 확인
    if secret_key_user != correct_password:
        st.warning("올바른 비밀번호를 입력해주세요.")
        st.stop()
    


    st.markdown(f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;">
            <h1>개별 작품 발간 흐름 분석</h1>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("👀**개별 작품별 한국 문학 발간 흐름을 상세히 분석합니다.**")
    st.caption(f"*데이터 출처: Goodreads, GoogleSearch*")

    st.markdown("---")

    # 사이드바
    st.sidebar.header("⚙️ 데이터 로딩")
    
    # 기간 설정 추가
    st.sidebar.subheader("📅 분석 기간 설정")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        start_date = st.date_input(
            "시작일",
            value=datetime(2003, 1, 1),
            help="분석할 데이터의 시작 날짜"
        )

    with col2:
        end_date = st.date_input(
            "종료일", 
            # value=datetime.now(),
            value=datetime(2027, 1, 1),
            help="분석할 데이터의 종료 날짜"
        )

    # 데이터 로드 버튼
    if st.sidebar.button("🔄 데이터 불러오기", type="primary"):
        st.session_state.load_data = True
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date

    # 분석기 초기화
    if 'analyzer_book' not in st.session_state:
        st.session_state.analyzer_book = LiteratureExportAnalyzer()

    analyzer = st.session_state.analyzer_book

    # 데이터 로드 상태 확인
    if 'data_loaded_book' not in st.session_state:
        st.session_state.data_loaded_book = False

    # 데이터 로드
    if not st.session_state.data_loaded_book:
        if st.session_state.get('load_data', False):
            with st.spinner("DB에서 데이터 로딩 중..."):
                start_date = st.session_state.get('start_date')
                end_date = st.session_state.get('end_date')
                
                success = analyzer.load_data_from_db(
                    DB_HOST, DB_NAME, DB_USER, DB_PASSWORD,
                    start_date=start_date.strftime('%Y-%m-%d') if start_date else None,
                    end_date=end_date.strftime('%Y-%m-%d') if end_date else None
                )
                
                if success:
                    if start_date and end_date:
                        st.success(f"✅ DB 데이터 로드 완료 (기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
                    else:
                        st.success(f"✅ DB 데이터 로드 완료")
                        
                    analyzer.analyze_all()
                    st.session_state.data_loaded_book = True
                    st.session_state.load_data = False
                else:
                    st.session_state.load_data = False
                    st.stop()
        else: 
            st.info("👆 사이드바에서 데이터 기간을 설정하고 '데이터 불러오기' 버튼을 클릭하세요.")
            st.stop()

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
    
    # 장르 매핑
    genre_mapping = {
        "A": "A - 환경·재난",
        "B": "B - 미스터리·스릴러", 
        "C": "C - SF·판타지",
        "D": "D - 사회·정치",
        "E": "E - 이주·전쟁",
        "F": "F - 젠더·다양성",
        "G": "G - 종교·신화",
        "H": "H - 관계·성장",
        "I": "I - 로맨스",
        "J": "J - 역사",
        "미분류": "기타"
    }
    
    # # 모든 원작 작품 목록
    # if analyzer.df is not None:
    #     wave_df = analyzer.load_wave_data_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
        
    #     if wave_df is not None and len(wave_df) > 0:
    #         st.header("1️⃣ 전체 확산 패턴 분석")
    #         st.markdown("✨전체 도서 출간 후 확산 패턴을 다이어그램으로 확인하세요.")
            
    #         try:
    #             # Sankey Diagram에 필요한 데이터 형태로 가공
    #             sankey_data = wave_df.dropna(subset=['source_country'])
    #             sankey_data = sankey_data.groupby(['source_country', 'country']).size().reset_index(name='value')
    #             sankey_data = sankey_data.rename(columns={'source_country': 'source', 'country': 'target'})

    #             # 상위 30개 흐름만 선택
    #             sankey_data = sankey_data.sort_values(by='value', ascending=False).head(30)
                
    #             if len(sankey_data) > 0:
    #                 # 전체 노드(국가) 리스트 생성 및 매핑
    #                 all_nodes = pd.concat([sankey_data['source'], sankey_data['target']]).unique()
    #                 node_map = {node: i for i, node in enumerate(all_nodes)}

    #                 # 링크(흐름) 데이터 생성
    #                 link_data = dict(
    #                     source=sankey_data['source'].map(node_map).tolist(),
    #                     target=sankey_data['target'].map(node_map).tolist(),
    #                     value=sankey_data['value'].tolist()
    #                 )

    #                 # Sankey Diagram 객체 생성
    #                 fig = go.Figure(data=[go.Sankey(
    #                     node=dict(
    #                         pad=15,
    #                         thickness=20,
    #                         line=dict(width=0),
    #                         label=all_nodes.tolist(),
    #                         color="lightblue"
    #                     ),
    #                     link=link_data
    #                 )])

    #                 fig.update_layout(
    #                     title_text="국가 간 도서 확산 흐름 (Sankey Diagram)", 
    #                     font=dict(
    #                         family="Arial, sans-serif",
    #                         size=20,
    #                         color="blue"
    #                     ),
    #                     height=900
    #                 )
                    
    #                 st.plotly_chart(fig, use_container_width=True)
                    
    #                 # 통계 정보 표시
    #                 col1, col2, col3 = st.columns(3)
    #                 with col1:
    #                     st.metric("◎ 총 흐름 수", f"{len(sankey_data)}")
    #                 with col2:
    #                     st.metric("◎ 관련 국가 수", f"{len(all_nodes)}")
    #                 with col3:
    #                     st.metric("◎ 총 이동 건수", f"{sankey_data['value'].sum():,}")
                        
    #             else:
    #                 st.warning("생키 다이어그램을 생성할 수 있는 데이터가 없습니다.")
                    
    #         except Exception as e:
    #             st.error(f"Wave 데이터 처리 오류: {e}")
    #             st.info("데이터 형식을 확인해주세요.")


    st.header("1️⃣ 원작 작품 선택")
    st.markdown("✨원하는 작품을 클릭하여 발간 흐름을 확인하세요.")
    
    # 모든 원작 작품 가져오기
    all_original_books = analyzer.get_all_original_books()
    
    if len(all_original_books) > 0:
        # 필터링 옵션 추가
        st.subheader("🔍 필터 옵션")

        all_genre_codes = analyzer.get_all_genres()
        genre_filter_options = ["전체"] + [genre_mapping.get(code, code) for code in all_genre_codes]
        selected_genre_filter = st.selectbox("📖 장르 필터", genre_filter_options, index=0)

        
        # 검색 기능
        titles = all_original_books['title'].unique().tolist()  
        search_term = st.selectbox(
                "🔍 작품 검색", 
                options=titles,
                index=None, 
                placeholder="작품을 검색(선택)하세요."
            )
        # search_term = st.text_input("🔍 제목 검색", placeholder="작품 제목을 입력하세요...")
        

        # 필터 적용
        filtered_books = all_original_books.copy()
        
        # 장르 필터 적용
        if selected_genre_filter != "전체":
            reverse_mapping = {v: k for k, v in genre_mapping.items()}
            selected_genre_code = reverse_mapping.get(selected_genre_filter, selected_genre_filter)
            filtered_books = filtered_books[filtered_books['genres'].str.contains(selected_genre_code, na=False)]
        
    
        filtered_books = filtered_books[filtered_books['국가'] =='대한민국' ]
        
        # 검색 필터 적용
        if search_term:
            filtered_books = filtered_books[filtered_books['title'].str.contains(search_term, case=False, na=False)]
        
        st.subheader(f"📚 원작 작품 목록 ({len(filtered_books)}개)")
        
        if len(filtered_books) > 0:
            # 장르 정보를 사용자 친화적으로 변환
            def format_genres(genre_str):
                if pd.isna(genre_str) or genre_str == '':
                    return '미분류'
                genre_codes = [g.strip() for g in genre_str.split(',')]
                formatted_genres = [genre_mapping.get(code, code) for code in genre_codes]
                return ', '.join(formatted_genres)
            
            # 작품 목록을 데이터프레임으로 표시
            display_df = filtered_books.copy()
            display_df['발간일'] = display_df['발간일'].dt.strftime('%Y-%m-%d')
            display_df['장르'] = display_df['genres'].apply(format_genres)
            display_df = display_df.rename(columns={
                'book_id': '작품 ID',
                'title': '제목',
                '발간일': '원작 발간일',
                '국가': '원작 출간국',
                '장르': '장르', 
                'ISBN(13)' : 'ISBN-13',
                '작가명' : '작가', 
                '출판사명':'원작 출판사명'
            })
            
            # 컬럼 순서 조정
            # display_df = display_df[['제목', '작가', '원작 발간일', '원작 출판사명',  '장르', 'ISBN-13']]
            display_df['선택'] = False

            # 컬럼 순서 조정 (체크박스를 맨 앞에)
            display_df = display_df[['선택', '제목', '작가', '원작 발간일', '원작 출판사명', '장르', 'ISBN-13']]
            display_df = display_df.sort_values('제목', ascending=True)

            # 데이터프레임 표시 (체크박스로)
            edited_df = st.data_editor(
                display_df,
                hide_index=True,
                column_config={
                    "선택": st.column_config.CheckboxColumn(
                        "선택",
                        help="분석할 작품을 선택하세요",
                        default=False,
                    )
                },
                disabled=["제목", "작가", "원작 발간일", "원작 출판사명", "장르", "ISBN-13"],
                use_container_width=True,
                key="book_selection_editor"
            )

            # 선택된 작품이 있는 경우
            selected_books = edited_df[edited_df['선택'] == True]
            if len(selected_books) > 0:
                selected_book = selected_books.iloc[0]  # 첫 번째 선택된 작품만 사용
                
                # 원본 데이터에서 해당 작품 정보 찾기
                book_mask = (filtered_books['title'] == selected_book['제목']) & \
                        (filtered_books['작가명'] == selected_book['작가'])
                
                if book_mask.any():
                    selected_book_id = filtered_books.loc[book_mask, 'book_id'].iloc[0]
                    selected_book_title = selected_book['제목']
                    
                    st.markdown("---")
                    st.header("3️⃣ 개별 작품별 한국 문학 발간 흐름 분석")
                    
                    # 선택된 작품의 발간 흐름 분석
                    book_info, export_path, book_data = analyzer.get_book_export_path(selected_book_id)
                    
                    if book_info and export_path:
                        st.subheader(f"📖 {selected_book_title}")
                        
                        # 기본 정보 표시
                        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                        with info_col1:
                            st.metric("원작 출간국", book_info['original_country'])
                        with info_col2:
                            st.metric("총 진출 국가", f"{book_info['total_countries']}개국")
                        with info_col3:
                            st.metric("총 소요 기간", f"{book_info['total_days']}일")
                        with info_col4:
                            st.metric("장르", f"{len(book_info['genres'])}개")
                        
                        # 발간 흐름 테이블
                        st.subheader("📈 발간 흐름 상세")
                        
                        path_df = pd.DataFrame(export_path)
                        path_df['순서'] = range(1, len(path_df) + 1)
                        path_df['출간일'] = path_df['date'].dt.strftime('%Y-%m-%d')
                        path_df['구분'] = path_df['is_original'].apply(lambda x: '원작' if x else '수출')
                        path_df['경과일수'] = path_df['days_from_original']
                        
                        display_path_df = path_df[['순서', 'country', '출간일', '구분', '경과일수', '에디션_제목','출판사명','ISBN(13)']].rename(columns={
                            'country': '국가',
                            '에디션_제목': '번역 출간 제목',
                            '출판사명':'출판사',
                            'ISBN(13)': 'ISBN-13'
                        })
            
                        display_path_df = display_path_df[['순서','구분', '국가','출간일','번역 출간 제목','출판사','경과일수','ISBN-13']]
                        display_path_df_2 = display_path_df.copy()
                        # 순서를 0부터 시작하도록 변경 (기존 인덱스를 그대로 사용)
                        display_path_df_2['순서'] = display_path_df_2.index
                        display_path_df_2 = display_path_df_2.drop(columns='구분')

                        # 0번 행(첫 번째 행)을 제외하고 표시
                        display_path_df_2 = display_path_df_2[display_path_df_2['순서'] != 0]

                        # 그래프 출력
                        st.dataframe(display_path_df_2, use_container_width=True, hide_index=True)
                    # st.dataframe(display_path_df_2, use_container_width=True, hide_index=True)
                    



                    # --------------------------------- # 
                    # 네트워크 그래프
                    st.subheader("📈 발간 흐름 차트 - ver1")

                    timeline_fig = analyzer.create_book_export_timeline(book_info, export_path)  # 또는 create_book_export_flow
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                    else:
                        st.info("해외 발간 이력이 없습니다.")
                    # --------------------------------- # 

                    st.subheader("🕸️ 발간 흐름 차트 - ver2")

                    col_graph, col_info = st.columns([3, 1])
                    
                    with col_graph:
                        network_html = analyzer.create_book_export_network(book_info, export_path)
                        if network_html:
                            components.html(network_html, height=520, scrolling=False)
                        else:
                            st.info("발간 흐름가 충분하지 않아 네트워크를 생성할 수 없습니다.")
                    
                    with col_info:
                        st.markdown("### 📋 경로 정보")
                        st.write(f"**📚 작품명:** {book_info['title']}")
                        st.write(f"**🎭 장르:** {', '.join([genre_mapping.get(g, g) for g in book_info['genres']])}")
                        st.write(f"**🏠 원작국:** {book_info['original_country']}")
                        st.write(f"**📅 원작일:** {book_info['original_date'].strftime('%Y-%m-%d')}")
                        st.write(f"**🌍 진출국:** {book_info['total_countries']}개국")
                        st.write(f"**⏰ 총 기간:** {book_info['total_days']}일")
                        
                        # 수출 속도 분석
                        if len(export_path) > 1:
                            export_only = [p for p in export_path if not p['is_original']]
                            if export_only:
                                avg_gap = np.mean([p['days_from_original'] for p in export_only])
                                st.write(f"**📊 평균 수출 간격:** {avg_gap:.0f}일")
                    

                    # ------------------------------------- #  
                    # 시간별 수출 진행 차트
                    st.subheader("📊 시간별 수출 진행")
                    
                    if len(export_path) > 1:
                        # 시간순 진행 차트
                        timeline_df = pd.DataFrame(export_path)
                        timeline_df['날짜'] = timeline_df['date']
                        timeline_df['누적국가수'] = range(1, len(timeline_df) + 1)
                        
                        fig = px.line(
                            timeline_df, 
                            x='날짜', 
                            y='누적국가수',
                            title=f"{selected_book_title} - 시간별 진출국 누적",
                            markers=True
                        )
                        
                        # 원작 시점 표시
                        original_date = book_info['original_date']
          
                        # Timestamp를 문자열로 변환하여 전달
                        # X축 데이터와 동일한 타입으로 맞춤
                        if hasattr(original_date, 'normalize'):
                            original_datetime = original_date.normalize()  # pandas Timestamp (00:00:00)
                        else:
                            original_datetime = pd.to_datetime(original_date).normalize()
        
                        # 원작 출간 시점에 특별한 점 추가
                        fig.add_scatter(
                            x=[original_datetime],
                            y=[0],  # 또는 적절한 y값
                            mode='markers+text',
                            marker=dict(color='red', size=15, symbol='diamond'),
                            text=['원작 출간'],
                            textposition="top center",
                            name='원작 출간',
                            showlegend=False
                        )
                                    
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 수출 간격 분석
                        export_intervals = []
                        prev_date = None
                        
                        for step in export_path:
                            if prev_date is not None:
                                interval = (step['date'] - prev_date).days
                                export_intervals.append({
                                    'from_country': prev_country,
                                    'to_country': step['country'],
                                    'interval_days': interval,
                                    'date': step['date']
                                })
                            prev_date = step['date']
                            prev_country = step['country']
                        
                        
                            
                    
                else:
                    st.warning("선택한 작품의 발간 흐름 데이터를 찾을 수 없습니다.")
                    
            else:
                st.info("👆 위 테이블에서 분석하고 싶은 작품을 클릭하세요.")
        
        else:
            st.warning("필터 조건에 맞는 작품이 없습니다. 필터를 조정해보세요.")
    
    else:
        st.warning("원작 작품 데이터가 없습니다.")

    

if __name__ == "__main__":
    main()