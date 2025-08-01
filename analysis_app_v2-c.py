import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import defaultdict
import warnings
import mysql.connector
from mysql.connector import Error

warnings.filterwarnings('ignore')

DB_HOST = st.secrets["database"]["host"]
DB_NAME = st.secrets["database"]["database"]
DB_USER = st.secrets["database"]["user"]
DB_PASSWORD = st.secrets["database"]["password"]

# 페이지 설정
st.set_page_config(
    page_title="한국 문학 해외 수출 추천 시스템",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class LiteratureExportAnalyzer:
    def __init__(self):
        self.df = None
        self.hub_scores = {}
        self.genre_fit_scores = {}
        self.transition_matrix = {}
        
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
                # SQL 쿼리
                query = """
                SELECT book_id, 발간일, genre1, 국가
                FROM literature_books
                """
                
                self.df = pd.read_sql(query, connection)
                
                # 기존 전처리 로직과 동일
                required_cols = ['book_id', '발간일', 'genre1', '국가']
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
    
    
    def calculate_hub_scores(self):
        """거점 지수 계산"""
        book_patterns = {}
        
        for book_id, group in self.df.groupby('book_id'):
            sorted_group = group.sort_values('발간일')
            
            if len(sorted_group) > 1:  # 다국가 진출 작품만
                countries = sorted_group['국가'].tolist()
                dates = sorted_group['발간일'].tolist()
                genre = sorted_group['genre1'].iloc[0]
                
                book_patterns[book_id] = {
                    'countries': countries,
                    'dates': dates,
                    'genre': genre,
                    'first_country': countries[0],
                    'subsequent_count': len(countries) - 1
                }
        
        # 첫 번째 진출 국가별 거점 점수 계산
        hub_analysis = defaultdict(lambda: {'total_books': 0, 'total_subsequent': 0})
        
        for pattern in book_patterns.values():
            first_country = pattern['first_country']
            subsequent_count = pattern['subsequent_count']
            
            hub_analysis[first_country]['total_books'] += 1
            hub_analysis[first_country]['total_subsequent'] += subsequent_count
        
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
        """장르별 국가 적합도 계산"""
        genre_country_counts = self.df.groupby(['genre1', '국가']).size().reset_index(name='count')
        genre_totals = self.df.groupby('genre1').size()
        country_totals = self.df.groupby('국가').size()
        
        for genre in self.df['genre1'].unique():
            self.genre_fit_scores[genre] = {}
            
            genre_data = genre_country_counts[genre_country_counts['genre1'] == genre]
            total_genre_count = genre_totals[genre]
            
            for _, row in genre_data.iterrows():
                country = row['국가']
                count = row['count']
                
                genre_ratio = count / total_genre_count
                country_activity = min(country_totals[country] / country_totals.max(), 1.0)
                
                fit_score = genre_ratio * 0.7 + country_activity * 0.3
                self.genre_fit_scores[genre][country] = fit_score
    
    def calculate_transition_matrix(self):
        """국가 간 전이 확률 계산"""
        transitions = defaultdict(lambda: defaultdict(int))
        
        for book_id, group in self.df.groupby('book_id'):
            sorted_group = group.sort_values('발간일')
            countries = sorted_group['국가'].tolist()
            
            for i in range(len(countries) - 1):
                from_country = countries[i]
                to_country = countries[i + 1]
                transitions[from_country][to_country] += 1
        
        # 확률로 변환
        for from_country in transitions:
            total = sum(transitions[from_country].values())
            if total > 0:
                self.transition_matrix[from_country] = {}
                for to_country, count in transitions[from_country].items():
                    if count >= 2:  # 최소 2회 이상 전이
                        self.transition_matrix[from_country][to_country] = count / total
    
    def recommend_countries(self, genre, target_country=None, top_k=10):
        """국가 추천 함수"""
        recommendations = []
        all_countries = set(self.df['국가'].unique())
        
        for country in all_countries:
            # 거점 점수 (40%)
            hub_score = 0
            if country in self.hub_scores:
                normalized_hub = min(self.hub_scores[country]['hub_index'] / 10, 1.0)
                hub_score = normalized_hub * 0.4
            
            # 장르 적합도 (30%)
            genre_score = 0
            if genre in self.genre_fit_scores and country in self.genre_fit_scores[genre]:
                genre_score = self.genre_fit_scores[genre][country] * 0.3
            
            # 시장 크기 (20%)
            country_count = len(self.df[self.df['국가'] == country])
            max_count = self.df['국가'].value_counts().max()
            market_score = (country_count / max_count) * 0.2
            
            # 전이 확률 (10%)
            transition_score = 0
            if target_country and country in self.transition_matrix:
                if target_country in self.transition_matrix[country]:
                    transition_score = self.transition_matrix[country][target_country] * 0.1
            else:
                transition_score = 0.05
            
            total_score = hub_score + genre_score + market_score + transition_score
            
            recommendations.append({
                'country': country,
                'total_score': total_score,
                'probability': min(total_score * 100, 95),
                'hub_index': self.hub_scores.get(country, {}).get('hub_index', 0),
                'genre_fit': self.genre_fit_scores.get(genre, {}).get(country, 0) * 100,
                'market_size': (country_count / max_count) * 100,
                'transition_prob': transition_score * 10,
                'book_count': self.hub_scores.get(country, {}).get('total_books', 0)
            })
        
        recommendations.sort(key=lambda x: x['total_score'], reverse=True)
        return recommendations[:top_k]
    
    def analyze_all(self):
        """전체 분석 실행"""
        self.calculate_hub_scores()
        self.calculate_genre_fit()
        self.calculate_transition_matrix()

# 메인 앱
def main():
    st.title("📚 한국 문학 해외 수출 추천 시스템")
    st.markdown("---")
    st.markdown("**문학번역원 • 데이터 기반 최적 진출 국가 분석**")
    st.markdown("**1️⃣ 목표 장르/국가에 도달하기 위한 경유지 추천**")

    # 사이드바
    st.sidebar.header("⚙️ 데이터 로딩")

    # 데이터 로드 버튼
    if st.sidebar.button("🔄 DB에서 데이터 불러오기", type="primary"):
        st.session_state.load_data = True
    
    # 분석기 초기화
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = LiteratureExportAnalyzer()

    analyzer = st.session_state.analyzer

    # 데이터 로드 (한 번만 실행)
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

    if not st.session_state.data_loaded:
        if st.session_state.get('load_data', False):
            with st.spinner("DB에서 데이터 로딩 중..."):
                success = analyzer.load_data_from_db(DB_HOST, DB_NAME, DB_USER, DB_PASSWORD)
                if success:
                    st.success(f"✅ DB 데이터 로드 완료: {len(analyzer.df):,}행")
                    analyzer.analyze_all()
                    st.success("✅ 분석 완료")
                    st.session_state.data_loaded = True  # 데이터 로드 완료 플래그
                    st.session_state.load_data = False
                else:
                    st.session_state.load_data = False
                    st.stop()
        else: 
            st.info("👆 사이드바에서 '데이터 불러오기' 버튼을 클릭하세요.")
            st.stop()

    # 장르 코드 매핑 딕셔너리
    genre_mapping = {
        "A": "Environment, Climate Disaster, Disaster",
        "B": "Mystery, Thriller, Crime, Horror",
        "C": "Science Fiction, Fantasy",
        "D": "Capitalism, Labor, Poverty, Development, Urbanization, Democracy",
        "E": "Diaspora, Migration, Refugees, Colonialism, Imperialism, War",
        "F": "LGBTQ, Gender Equality, Disability",
        "G": "Religion, Mythology",
        "H": "Relationships (Healing), Family, Neighbors, Friendship, Coming-of-age",
        "I": "Romance",
        "J": "History",
        "Other": "Cases that do not fall under the above categories", 
        "Z": "미분류"
    }

    # 역매핑 딕셔너리 (한국어명 → 알파벳 코드)
    reverse_genre_mapping = {v: k for k, v in genre_mapping.items()}

    # 데이터가 로드된 경우에만 UI 표시
    if st.session_state.get('data_loaded', False) and analyzer.df is not None:
        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 출간 기록", f"{len(analyzer.df):,}")
        with col2:
            st.metric("고유 작품 수", f"{analyzer.df['book_id'].nunique():,}")
        with col3:
            st.metric("진출 국가 수", f"{analyzer.df['국가'].nunique()}")
        with col4:
            st.metric("장르 수", f"{analyzer.df['genre1'].nunique()}")
        
        st.markdown("---")
        
        # 추천 시스템
        st.header("🎯 추천 시스템")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 장르 코드를 실제 장르명으로 변환
            available_genre_codes = sorted(analyzer.df['genre1'].unique())
            available_genre_names = [genre_mapping.get(code, code) for code in available_genre_codes]
            
            selected_genre_name = st.selectbox(
                "📖 장르 선택", 
                available_genre_names,
                index=0 if 'Romance' not in available_genre_names else available_genre_names.index('Romance'),
                key="genre_selectbox"
            )
            
            # 선택된 장르명에서 다시 코드로 변환 (분석에 사용)
            selected_genre = reverse_genre_mapping.get(selected_genre_name, selected_genre_name)
            
            st.caption(f"*장르 출처:Goodreads, GoogleSearch*")
        
        with col2:
            available_countries = ['선택하지 않음'] + sorted(analyzer.df['국가'].unique())
            target_country = st.selectbox(
                "🎯 목표 국가 (선택사항)", 
                available_countries,
                key="target_country_selectbox"
            )
            st.caption("※ 목표국가 선택 시: 어떤 국가에 책을 먼저 출간했을 때, 그 다음 '목표국가'로 진출할 확률이 높은지를 분석합니다.")

            target_country = None if target_country == '선택하지 않음' else target_country
        
        # 추천 실행
        if st.button("🚀 추천 분석 실행", type="primary"):
            with st.spinner("추천 분석 중..."):
                recommendations = analyzer.recommend_countries(
                    selected_genre, 
                    target_country, 
                    top_k=10
                )
                
                # 결과 표시 (화면에는 장르명 표시)
                st.subheader(f"📈 {selected_genre_name} 장르 추천 결과")
                
                # 상위 5개국 카드 형태로 표시
                for i in range(min(5, len(recommendations))):
                    rec = recommendations[i]
                    
                    # 등급 결정
                    if rec['probability'] >= 80:
                        color = "🟢"
                        level = "매우 높음"
                    elif rec['probability'] >= 60:
                        color = "🟡"
                        level = "높음"
                    else:
                        color = "🟠"
                        level = "보통"
                    
                    with st.container():
                        st.markdown(f"""
                        <div style="
                            border: 2px solid #e0e0e0; 
                            border-radius: 10px; 
                            padding: 15px; 
                            margin: 10px 0;
                            background-color: {'#f0f8f0' if i == 0 else '#f8f9fa'}
                        ">
                            <h4>{color} {i+1}위. {rec['country']} (성공확률: {rec['probability']:.1f}%)</h4>
                            <p><strong>추천 등급:</strong> {level}</p>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                                <div>• 거점 지수: <strong>{rec['hub_index']:.2f}</strong></div>
                                <div>• 장르 적합도: <strong>{rec['genre_fit']:.1f}%</strong></div>
                                <div>• 시장 규모: <strong>{rec['market_size']:.1f}%</strong></div>
                                <div>• 기여 작품: <strong>{rec['book_count']}개</strong></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # 차트 생성
                st.subheader("📊 시각화 분석")
                
                chart_tab1, chart_tab2, chart_tab3 = st.tabs(["성공 확률", "거점 분석", "장르 적합도"])
                
                with chart_tab1:
                    # 성공 확률 막대 차트
                    chart_data = recommendations[:8]
                    fig = px.bar(
                        x=[rec['country'] for rec in chart_data],
                        y=[rec['probability'] for rec in chart_data],
                        title=f"{selected_genre_name} 장르 국가별 성공 확률",
                        labels={'x': '국가', 'y': '성공 확률 (%)'},
                        color=[rec['probability'] for rec in chart_data],
                        color_continuous_scale="viridis"
                    )
                    fig.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with chart_tab2:
                    # 거점 지수 vs 작품 수 산점도
                    if analyzer.hub_scores:
                        hub_countries = list(analyzer.hub_scores.keys())
                        hub_indices = [analyzer.hub_scores[c]['hub_index'] for c in hub_countries]
                        book_counts = [analyzer.hub_scores[c]['total_books'] for c in hub_countries]
                        
                        fig = px.scatter(
                            x=book_counts,
                            y=hub_indices,
                            text=hub_countries,
                            title="거점 지수 vs 기여 작품 수",
                            labels={'x': '기여 작품 수', 'y': '거점 지수'},
                            size=[10] * len(hub_countries),
                            color=hub_indices,
                            color_continuous_scale="plasma"
                        )
                        fig.update_traces(textposition="top center")
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                with chart_tab3:
                    # 장르 적합도 히트맵 (상위 10개 국가)
                    top_countries = [rec['country'] for rec in recommendations[:10]]
                    top_genres = analyzer.df['genre1'].value_counts().head(6).index.tolist()
                    
                    heatmap_data = []
                    for genre in top_genres:
                        row = []
                        for country in top_countries:
                            if genre in analyzer.genre_fit_scores and country in analyzer.genre_fit_scores[genre]:
                                row.append(analyzer.genre_fit_scores[genre][country] * 100)
                            else:
                                row.append(0)
                        heatmap_data.append(row)
                    
                    fig = px.imshow(
                        heatmap_data,
                        x=top_countries,
                        y=top_genres,
                        title="장르별 국가 적합도 히트맵",
                        labels={'x': '국가', 'y': '장르', 'color': '적합도 (%)'},
                        color_continuous_scale="blues"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # 전체 분석 결과
        st.markdown("---")
        st.header("📋 전체 분석 결과")
        
        analysis_tab1, analysis_tab2, analysis_tab3 = st.tabs(["거점 국가 순위", "장르별 분포", "국가별 분포"])
        
        with analysis_tab1:
            if analyzer.hub_scores:
                st.subheader("🏆 거점 국가 순위")
                hub_df = pd.DataFrame([
                    {
                        '순위': i+1,
                        '국가': country,
                        '거점 지수': f"{data['hub_index']:.2f}",
                        '기여 작품 수': data['total_books'],
                        '평균 후속 진출': f"{data['avg_subsequent']:.1f}개국"
                    }
                    for i, (country, data) in enumerate(
                        sorted(analyzer.hub_scores.items(), key=lambda x: x[1]['hub_index'], reverse=True)
                    )
                ])
                st.dataframe(hub_df, use_container_width=True)
        
        with analysis_tab2:
            st.subheader("📚 장르별 작품 분포")
            genre_counts = analyzer.df['genre1'].value_counts().head(10)
            
            # 장르 코드를 장르명으로 변환
            mapped_genre_names = [genre_mapping.get(code, code) for code in genre_counts.index]
            
            fig = px.pie(
                values=genre_counts.values,
                names=mapped_genre_names,
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
    
    else:
        # 데이터가 로드되지 않은 경우에만 표시
        if not st.session_state.get('data_loaded', False):
            st.info("👆 사이드바에서 '데이터 불러오기' 버튼을 클릭하세요.")
            
            # 샘플 데이터 구조 설명
            st.markdown("### 📋 필요한 데이터 구조")
            st.markdown("""
            DB에서 불러올 데이터에는 다음 컬럼들이 포함되어야 합니다:
            
            - **book_id**: 동일 작품을 나타내는 고유 ID
            - **발간일**: 출간 날짜 (YYYY-MM-DD 형태)
            - **genre1**: 주요 장르
            - **국가**: 진출 국가명
            
            추가 컬럼들은 자동으로 무시됩니다.
            """)

if __name__ == "__main__":
    main()