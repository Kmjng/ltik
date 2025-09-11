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
    page_title="문학 작품 해외 수출 권역별 분석 시스템",
    page_icon="🌍",
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

class RegionLiteratureExportAnalyzer:
    def __init__(self):
        self.df = None
        self.region_scores = {}
        self.transition_matrix = {}
        
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
                # 권역 테이블에서 데이터 로드
                base_query = """
                SELECT book_id, 발간일, 권역, 국가, 원작여부
                FROM literature_books_region
                """
                
                # 날짜 필터 조건 추가
                where_conditions = []
                if start_date:
                    where_conditions.append(f"발간일 >= '{start_date}'")
                if end_date:
                    where_conditions.append(f"발간일 <= '{end_date}'")
                
                if where_conditions:
                    query = base_query + " WHERE " + " AND ".join(where_conditions)
                else:
                    query = base_query
                
                self.df = pd.read_sql(query, connection)
                
                # 필수 컬럼 확인
                required_cols = ['book_id', '발간일', '권역', '국가', '원작여부']
                missing_cols = [col for col in required_cols if col not in self.df.columns]
                if missing_cols:
                    st.error(f"누락된 컬럼: {missing_cols}")
                    return False
                
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

    def get_all_regions(self):
        """모든 고유 권역 목록 추출"""
        return sorted(self.df['권역'].dropna().unique())
    
    def get_books_by_region(self, selected_region):
        """특정 권역의 모든 원작 작품 반환"""
        return self.df[(self.df['권역'] == selected_region) & (self.df['원작여부'] == 'original')]
    
    def calculate_region_hub_scores(self):
        """권역별 거점 지수 계산 - 원작 권역 → edition 국가 진출 패턴"""
        book_patterns = {}
        
        # 1️⃣ 각 책별로 원작 권역 → 후속 진출 패턴 수집
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기 (권역 기준)
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                # 원작이 여러 개 있다면 가장 빠른 날짜 선택
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_region = original_record['권역']
                original_date = original_record['발간일']
                
                # 원작 이후의 모든 edition 진출 국가들
                subsequent_records = group[
                    (group['발간일'] > original_date) | 
                    ((group['발간일'] == original_date) & (group['원작여부'] == 'edition'))
                ].sort_values('발간일')
                
                if len(subsequent_records) > 0:
                    subsequent_countries = subsequent_records['국가'].tolist()
                    
                    book_patterns[book_id] = {
                        'original_region': original_region,
                        'original_date': original_date,
                        'subsequent_countries': subsequent_countries,
                        'subsequent_count': len(subsequent_countries)
                    }
        
        # 2️⃣ 원작 권역별 거점 점수 계산
        region_analysis = defaultdict(lambda: {'total_books': 0, 'total_subsequent': 0})
        
        for pattern in book_patterns.values():
            original_region = pattern['original_region']
            subsequent_count = pattern['subsequent_count']
            
            region_analysis[original_region]['total_books'] += 1
            region_analysis[original_region]['total_subsequent'] += subsequent_count
        
        # 3️⃣ 거점 지수 최종 계산 (베이지안 평활화 적용)
        
        # Step 1: 전체 평균 거점지수(μ) 계산
        total_books_all = 0
        total_subsequent_all = 0
        
        for region, data in region_analysis.items():
            if data['total_books'] >= 1:  # 최소 1개 작품
                total_books_all += data['total_books']
                total_subsequent_all += data['total_subsequent']
        
        # 전체 평균 거점지수 μ
        mu = total_subsequent_all / total_books_all if total_books_all > 0 else 0
        
        # Step 2: 임계치 설정
        m = 50  # 임계치
        
        # Step 3: 각 권역별 베이지안 평활화된 거점지수 계산
        for region, data in region_analysis.items():
            if data['total_books'] >= 5:  # 최소 작품 갯수 설정
                # 해당 권역의 원시 거점지수 r
                r = data['total_subsequent'] / data['total_books']
                
                # 해당 권역의 작품 수 v
                v = data['total_books']
                
                # 베이지안 평활화된 거점지수 계산
                bayesian_hub_index = (mu * m + r * v) / (m + v)
                
                self.region_scores[region] = {
                    'hub_index': bayesian_hub_index,
                    'raw_hub_index': r,
                    'total_books': data['total_books'],
                    'avg_subsequent': bayesian_hub_index,
                    'global_avg': mu
                }
    
    def calculate_region_transition_matrix(self):
        """권역 기준 원작 → edition 국가 전이 확률 계산"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_region = original_record['권역']
                original_date = original_record['발간일']
                
                # 원작 이후 edition 진출 국가들
                subsequent_records = group[
                    (group['발간일'] > original_date) | 
                    ((group['발간일'] == original_date) & (group['원작여부'] == 'edition'))
                ].sort_values('발간일')
                
                if len(subsequent_records) > 0:
                    # 원작 권역 → 각 edition 국가로의 진출 기록
                    for _, record in subsequent_records.iterrows():
                        edition_country = record['국가']
                        transitions[original_region][edition_country] += 1
        
        # 확률로 변환
        for region in transitions:
            total = sum(transitions[region].values())
            if total > 0:
                self.transition_matrix[region] = {}
                for country, count in transitions[region].items():
                    if count >= 1:  # 최소 1회 이상 전이
                        self.transition_matrix[region][country] = {
                            'probability': count / total,
                            'count': count,
                            'total_transitions': total
                        }

    def recommend_countries_from_region(self, start_region, prob_weight=0.7, conf_weight=0.3, top_k=10):
        """특정 권역에서 원작 출간 후 다음 진출 국가 추천"""
        recommendations = []
        message = None
        
        debug_info = f"\n🔍 권역 기준 추천: {start_region} 권역 원작 → edition 국가\n"
        debug_info += f"전이 매트릭스에 {start_region} 있나? {start_region in self.transition_matrix}\n"
        
        if start_region in self.transition_matrix:
            debug_info += f"전이 데이터: {list(self.transition_matrix[start_region].keys())}\n"
        
        # 권역별 전이 데이터가 있는 경우
        if start_region in self.transition_matrix:
            transitions = self.transition_matrix[start_region]
            
            for country, data in transitions.items():
                probability = data['probability']
                count = data['count']
                total_transitions = data['total_transitions']
                
                # 전이 횟수가 3회 미만이면 제외
                if count < 3:
                    continue
                
                # 신뢰도 계산 (전이 횟수 기반)
                confidence = min(count / 12, 1.0)  # 12회 이상이면 최대 신뢰도
                
                # 최종 점수 계산
                final_score = (probability * prob_weight) + (confidence * conf_weight)
                
                recommendations.append({
                    'country': country,
                    'probability': probability * 100,
                    'confidence': confidence * 100,
                    'transition_count': count,
                    'total_from_region': total_transitions,
                    'final_score': final_score * 100
                })
        else:
            message = f"⚠️ {start_region} 권역에서 원작 출간 후 edition 진출 데이터가 충분하지 않습니다."
        
        # 정렬
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 통계 정보
        stats_info = ""
        if recommendations:
            final_scores = [rec['final_score'] for rec in recommendations]
            avg_score = sum(final_scores) / len(final_scores)
            
            stats_info = f"\n📊 종합점수 통계 - {start_region} 권역\n"
            stats_info += f"   추천 국가 수: {len(recommendations)}개\n"
            stats_info += f"   종합점수 평균: {avg_score:.2f}\n"
            stats_info += f"   최고점: {max(final_scores):.2f}\n"
            stats_info += f"   최저점: {min(final_scores):.2f}\n"
            stats_info += f"   점수 범위: {max(final_scores) - min(final_scores):.2f}\n"
        
        # 시간순 진출 패턴 계산
        time_progression, timing_summary = self.calculate_time_based_progression(start_region)
        
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

    def get_region_country_stats(self, selected_regions):
        """선택된 권역들의 국가별 진출 건수 반환"""
        if not selected_regions:
            return pd.DataFrame()
        
        region_country_stats = []
        
        for region in selected_regions:
            # 해당 권역의 원작 작품들
            region_books = self.get_books_by_region(region)
            
            # 각 원작 작품의 edition들이 어느 국가에 진출했는지 계산
            for book_id in region_books['book_id'].unique():
                book_group = self.df[self.df['book_id'] == book_id]
                edition_records = book_group[book_group['원작여부'] == 'edition']
                
                for country in edition_records['국가'].unique():
                    count = len(edition_records[edition_records['국가'] == country])
                    region_country_stats.append({
                        'region': region,
                        'country': country,
                        'count': count
                    })
        
        return pd.DataFrame(region_country_stats)

    def create_network_graph(self, start_region, recommendations):
        """네트워크 그래프 생성 (pyvis 사용) - 권역 → 국가 진출 패턴"""
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

        # 중심 노드 (원작 권역) 추가
        start_node_title = f"""
        {start_region} 권역
        원작 출간 권역
        """
        net.add_node(
            start_region,
            label=start_region,
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
            country = rec['country']
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
            """

            hover_text = f"""
            ✈️ {rec['country']}
            ────────────────────
            🎯 종합 점수: {final_score:.1f}
            📊 진출 확률: {prob:.1f}%
            🔄 전이 횟수: {rec['transition_count']}회
                (신뢰도: {rec['confidence']:.1f}%)
            📍 순위: {i+1}위{timing_info}
            🌍 권역 → 국가 분석
            """

            # 노드 추가
            net.add_node(
                country,
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
                start_region,
                country,
                value=edge_width,
                title=f"종합점수: {final_score:.1f} | 진출 확률: {prob:.1f}%",
                color={'color': '#888888', 'highlight': '#000000'}
            )

        # HTML 생성
        try:
            source_code = net.generate_html()
            
            # 범례 추가
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
                    <span>원작 출간 권역</span>
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
                    🌍 권역 → 국가 진출 분석
                </div>
            </div>
            """
            
            # 기존 HTML에 범례 삽입
            if '<body>' in source_code:
                source_code = source_code.replace('<body>', f'<body>{legend_html}')
            else:
                source_code = source_code.replace(
                    '<div id="mynetworkid"',
                    f'{legend_html}<div id="mynetworkid"'
                )
            
            # 파일 저장
            with open('pyvis_graph.html', 'w', encoding='utf-8') as f:
                f.write(source_code)
            
            return source_code
            
        except Exception as e:
            st.error(f"그래프 생성 중 오류 발생: {e}")
            return None

    def get_region_stats(self, start_region):
        """권역 기준 시작 권역의 통계 정보"""
        # 해당 권역의 원작 작품 수
        region_original_books = self.df[
            (self.df['권역'] == start_region) & 
            (self.df['원작여부'] == 'original')
        ]
        
        original_books_count = len(region_original_books['book_id'].unique())
        
        # 권역 기준 후속 진출 횟수 계산
        transition_count = 0
        
        if start_region in self.transition_matrix:
            transitions = self.transition_matrix[start_region]
            
            for country, data in transitions.items():
                count = data['count']
                transition_count += count
        
        return {
            'total_original_books': original_books_count,
            'total_transitions': transition_count,
            'transition_rate': (transition_count / original_books_count * 100) if original_books_count > 0 else 0
        }, region_original_books

    def calculate_time_based_progression(self, start_region):
        """권역 기준 시간순 진출 패턴 계산"""
        time_progressions = []
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_region = original_record['권역']
                original_date = original_record['발간일']
                
                # 선택한 권역이 맞는지 확인
                if original_region == start_region:
                    # 원작 이후 edition 국가들을 시간순으로 정렬
                    subsequent_records = group[
                        (group['발간일'] > original_date) | 
                        ((group['발간일'] == original_date) & (group['원작여부'] == 'edition'))
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
                'avg_days': np.median(days_list),
                'count': len(days_list),
                'min_days': min(days_list),
                'max_days': max(days_list)
            }
        
        # 평균 진출 시점 기준으로 정렬
        sorted_countries = sorted(country_timing_summary.items(), 
                                key=lambda x: x[1]['avg_days'])
        
        return sorted_countries, country_timing_summary
            
    def analyze_all(self):
        """전체 분석 실행"""
        self.calculate_region_hub_scores()
        self.calculate_region_transition_matrix()

    def get_region_time_series_data(self, selected_region):
        """권역별 시간에 따른 진출 패턴 데이터 생성"""
        time_series_data = []
        
        for book_id, group in self.df.groupby('book_id'):
            # 원작 발간 기록 찾기
            original_records = group[group['원작여부'] == 'original']
            
            if len(original_records) > 0:
                original_record = original_records.loc[original_records['발간일'].idxmin()]
                original_region = original_record['권역']
                original_date = original_record['발간일']
                
                # 선택한 권역이 맞는지 확인
                if original_region == selected_region:
                    # 원작 이후 edition 국가들을 시간순으로 정렬
                    subsequent_records = group[
                        (group['발간일'] > original_date) | 
                        ((group['발간일'] == original_date) & (group['원작여부'] == 'edition'))
                    ].sort_values('발간일')
                    
                    if len(subsequent_records) > 0:
                        for _, record in subsequent_records.iterrows():
                            days_from_original = (record['발간일'] - original_date).days
                            time_series_data.append({
                                'book_id': book_id,
                                'country': record['국가'],
                                'date': record['발간일'],
                                'days_from_original': days_from_original,
                                'year': record['발간일'].year,
                                'month': record['발간일'].month
                            })
        
        return pd.DataFrame(time_series_data)
    

# 메인 앱
def main():
    st.markdown(f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{logo_base64}" width="50" style="margin-right: 10px;">
        <h1>권역별 도서 진출 경향 분석</h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("👀**권역에서 발생한 도서가 어느 국가로 진출하는 경향이 있는지 분석합니다.**")
    st.caption(f"*데이터 출처: Goodreads, GoogleSearch*")

    st.markdown("---")
    # session_state 초기화
    if 'initiated' not in st.session_state:
        st.session_state['initiated'] = False
    
    # 사이드바에 비밀번호 입력
    with st.sidebar.form(key='설정'):
        # --- Secret key input --- #
        secret_key_user = st.text_input(':closed_lock_with_key: **Secret Key**',
                                        placeholder='비밀번호를 입력해주세요.',
                                        type="password")
        # --- Secret key input --- #
        
        submit_prerequisite = st.form_submit_button('**✅ 확인하기**', use_container_width=True)

    if submit_prerequisite:
        if secret_key_user == st.secrets.get("app_password", "your_password"):
            initiated = st.sidebar.success('`Secret Key`가 확인되었습니다', icon="✅")
            st.session_state['initiated'] = True
        else:
            st.sidebar.warning('올바른 `Secret Key`를 입력해 주세요', icon="🚨")
            st.stop()
    
    # 인증 상태에 따른 메시지 표시
    if st.session_state.get('initiated') and not submit_prerequisite:
        st.sidebar.success('`Secret Key`가 확인되었습니다', icon="✅")
    if not st.session_state.get('initiated'):
        st.sidebar.info('`Secret Key`를 입력해 주세요', icon="ℹ️")
        st.stop()
    
    st.sidebar.markdown("---")

    import plotly.graph_objects as go
    import pandas as pd

    # 사이드바
    st.sidebar.header("⚙️ 데이터 로딩")
    
    # 기간 설정 추가
    st.sidebar.subheader("📅 분석 기간 설정")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        start_date = st.date_input(
            "시작일",
            value=datetime(1901, 1, 1),
            help="분석할 데이터의 시작 날짜"
        )

    with col2:
        end_date = st.date_input(
            "종료일", 
            value=datetime.now(),
            help="분석할 데이터의 종료 날짜"
        )

    # 데이터 로드 버튼
    if st.sidebar.button("🔄 데이터 불러오기", type="primary"):
        st.session_state.load_data = True
        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
    
    # 분석기 초기화 - RegionLiteratureExportAnalyzer 사용
    if 'analyzer_region' not in st.session_state:
        st.session_state.analyzer_region = RegionLiteratureExportAnalyzer()

    analyzer = st.session_state.analyzer_region

    # 데이터 로드 상태 확인
    if 'data_loaded_region' not in st.session_state:
        st.session_state.data_loaded_region = False

    # 데이터 로드
    if not st.session_state.data_loaded_region:
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
                        
                    original_count = len(analyzer.df[analyzer.df['원작여부'] == 'original'])
                    analyzer.analyze_all()
                    st.session_state.data_loaded_region = True
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
        st.metric("◎ 권역 수", f"{analyzer.df['권역'].nunique()}")
    
    st.markdown("---")
    
    # # Wave 데이터 기반 Sankey Diagram
    # if analyzer.df is not None:
    #     wave_df = analyzer.load_wave_data_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
        
    #     if wave_df is not None and len(wave_df) > 0:
    #         st.header("1️⃣ 확산 패턴 분석")
    #         st.markdown("✨도서 출간 후 확산 패턴을 다이어그램으로 확인하세요.")
            
    #         try:
    #             sankey_data = wave_df.dropna(subset=['source_country'])
    #             sankey_data = sankey_data.groupby(['source_country', 'country']).size().reset_index(name='value')
    #             sankey_data = sankey_data.rename(columns={'source_country': 'source', 'country': 'target'})
    #             sankey_data = sankey_data.sort_values(by='value', ascending=False).head(30)
                
    #             if len(sankey_data) > 0:
    #                 all_nodes = pd.concat([sankey_data['source'], sankey_data['target']]).unique()
    #                 node_map = {node: i for i, node in enumerate(all_nodes)}

    #                 link_data = dict(
    #                     source=sankey_data['source'].map(node_map).tolist(),
    #                     target=sankey_data['target'].map(node_map).tolist(),
    #                     value=sankey_data['value'].tolist()
    #                 )

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
    
    # st.markdown("---")
    
    # 권역 기준 추천 시스템
    st.header("1️⃣ 권역별 후속 진출 국가 분석")
    st.markdown("✨선택한 권역에서 출간된 원작이 어느 국가로 진출하는 경향이 있는지 분석합니다.")

    st.sidebar.markdown("---")

    # 권역 선택
    available_regions = analyzer.get_all_regions()
    
    if available_regions:
        selected_region = st.selectbox(
            "🌍 권역 선택", 
            available_regions,
            index=0
        )
        
        # 추천 실행
        if st.button("🔍 분석 실행", type="primary"):
            with st.spinner("권역 기준 후속 진출 패턴 분석 중..."):
                # 권역 통계
                region_stats, region_books = analyzer.get_region_stats(selected_region)
                
                # 후속 국가 추천
                recommendations, warning_message, debug_stats, time_progression = analyzer.recommend_countries_from_region(
                    selected_region, top_k=8
                )
                
                # 권역 정보 표시
                st.subheader(f"🌍 {selected_region} 권역 분석")
                
                info_col1, info_col2, info_col3 = st.columns(3)
                with info_col1:
                    st.metric("원작 작품 수", f"{region_stats['total_original_books']}개")
                with info_col2:
                    st.metric("후속 진출 건수", f"{region_stats['total_transitions']}건")
                with info_col3:
                    # st.metric("후속 진출률", f"{region_stats['transition_rate']:.1f}%")
                    st.write('')
                st.markdown("---")
                
                # 추천 결과 표시
                if recommendations:
                    # 시간순 진출 패턴 표시
                    if time_progression:
                        st.subheader("⏰ 시간순 진출 패턴")
                        recommended_countries = [rec['country'] for rec in recommendations]
                        filtered_progression = [(country, data) for country, data in time_progression 
                                            if country in recommended_countries]
                        if filtered_progression:
                            timing_text = " → ".join([f"{country} ({data['avg_days']:.0f}일)" 
                                                    for country, data in filtered_progression[:5]])
                            st.info(f"평균 진출순서 (진출시차): {timing_text}")   

                    # 네트워크 그래프 생성 및 표시
                    st.subheader("🕸️ 후속 진출 국가 네트워크")
                    st.write(f"  ᯓ ✈︎ **{selected_region} 권역에서 원작으로 출간한 후 진출 경향성**")
                    
                    if warning_message:
                        st.warning(warning_message)
                    
                    # 컬럼 분할: 네트워크 그래프와 텍스트 정보
                    graph_col, info_col = st.columns([5, 2])
                    
                    with graph_col:
                        network_html = analyzer.create_network_graph(selected_region, recommendations)
                        if network_html:
                            components.html(network_html, height=660, scrolling=False)
                    
                    with info_col:
                        st.markdown("### 📋 지표 설명")
                        st.write("""
                            🎯 **종합 점수:** 모든 요소를 종합한 최종 추천 점수 (0~100점)\n
                            📊 **진출 확률:** 권역에서 원작 출간 후 해당 국가로 실제 진출한 비율\n
                            🔄 **전이 횟수:** 실제로 원작 권역 → 후속 국가로 진출한 작품의 총 건수\n
                            (신뢰도: 통계적 신뢰성, 12건 이상이면 100%)\n
                            ⏰ **평균 진출 시점:** 권역에서 국가로 진출하기까지 걸린 일 수의 중앙값\n 
                            🏃 **출간 시점 순위:** 평균 진출 시점 순위\n 
                            📍 **순위:** 종합 점수를 기준으로 한 추천 우선순위
                            """)
                    
                    st.markdown("---")
                    
                    # 추가 차트
                    st.subheader("📈 추가 분석 - 권역 기준 진출 확률")
                    
                    # chart_tab1, chart_tab2 = st.tabs(["권역 기준 진출 확률", "종합 분석"])

                    # 진출 확률 막대 차트
                    chart_data = recommendations[:8]
                    fig = px.bar(
                        x=[rec['country'] for rec in chart_data],
                        y=[rec['probability'] for rec in chart_data],
                        title=f"{selected_region} 권역발 작품 진출 확률",
                        labels={'x': '후속 진출 국가', 'y': '진출 확률 (%)'},
                        color=[rec['probability'] for rec in chart_data],
                        color_continuous_scale="viridis"
                    )
                    fig.update_traces(
                        hovertemplate='<b>%{x}</b><br>진출 확률: %{y:.2f}%<extra></extra>'
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # with chart_tab2:
                    #     # 시간에 따른 진출 패턴 시계열 차트
                    #     time_series_df = analyzer.get_region_time_series_data(selected_region)
                        
                    #     if not time_series_df.empty:
                            
                    #         # 연도별 진출 건수 집계 
                    #         yearly_data = time_series_df.groupby([time_series_df['date'].dt.to_period('Y'), 'country']).size().reset_index(name='count')
                    #         yearly_data['date'] = yearly_data['date'].dt.to_timestamp()
                        
                    #         top_countries = time_series_df['country'].value_counts().index.tolist()
                    #         filtered_data = yearly_data[yearly_data['country'].isin(top_countries)]

                    #         ##
                    #         if not filtered_data.empty:
                    #             fig = px.line(
                    #                 filtered_data, 
                    #                 x='date', 
                    #                 y='count',
                    #                 color='country',
                    #                 title=f"{selected_region} 권역 → 국가별 시간에 따른 진출 패턴",
                    #                 labels={'date': '날짜', 'count': '진출 건수', 'country': '국가'},
                    #                 markers=True
                    #             )
                        
                    #             # fig.update_layout(height=400, xaxis_title="날짜", yaxis_title="연도별 진출 건수")
                    #             #  범례 설정 개선
                    #             fig.update_layout(
                    #                 height=500,  # 높이 증가
                    #                 xaxis_title="날짜", 
                    #                 yaxis_title="연도별 진출 건수",
                    #                 legend=dict(
                    #                     orientation="v",  # 세로 방향
                    #                     yanchor="top",
                    #                     y=1,
                    #                     xanchor="left",
                    #                     x=1.02,  # 차트 오른쪽 바깥쪽에 배치
                    #                     bgcolor="rgba(255,255,255,0.8)",  # 반투명 배경
                    #                     bordercolor="Black",
                    #                     borderwidth=1,
                    #                     font=dict(size=10)  # 폰트 크기 조정
                    #                 ),
                    #                 margin=dict(r=150)  # 오른쪽 마진 증가하여 범례 공간 확보
                    #             )
            

                        
                    #             ## 
                    #             st.plotly_chart(fig, use_container_width=True)
                                
                    #             # 추가 정보 테이블
                    #             st.subheader("국가별 상세 진출 통계")
                    #             country_stats = time_series_df.groupby('country').agg({
                    #                 'book_id': 'nunique',
                    #                 'days_from_original': ['mean', 'median', 'min', 'max'],
                    #                 'country': 'size'
                    #             }).round(1)
                                
                    #             country_stats.columns = ['진출작품수', '평균소요일', '중간소요일', '최소소요일', '최대소요일', '총진출건수']
                    #             country_stats = country_stats.sort_values('총진출건수', ascending=False)
                                
                    #             st.dataframe(country_stats, use_container_width=True)
                                
                    #             st.caption(f"분석 기간 내 {selected_region} 권역에서 원작 출간 후 후속 진출 패턴")

                    #             ## 


                    #         else:
                    #             st.info("시계열 데이터가 충분하지 않습니다.")
                    #     else:
                    #         st.info("해당 권역의 시계열 데이터가 없습니다.")

                else:
                    st.warning(f"⚠️ {selected_region} 권역에서의 후속 진출 데이터가 충분하지 않습니다.")
                    st.info("다른 권역을 선택해보세요.")
    else:
        st.error("권역 데이터를 찾을 수 없습니다. 데이터를 확인해주세요.")


if __name__ == "__main__":
    main()