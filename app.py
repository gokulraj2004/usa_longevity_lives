import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import base64
from branca.colormap import LinearColormap
import os
from PIL import Image
import pickle
import logging

# Set page config
st.set_page_config(page_title="U.S. Life Expectancy Explorer", layout="wide")

# Optimize data loading
@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, "data", "chr_census_feature_engineered_final.csv")
    logo_path = os.path.join(BASE_DIR, "assets", "omdena_logo.png")
    logo = Image.open(logo_path)
    #st.image(logo, width=300)

    st.sidebar.image(logo, width=250)
    st.sidebar.write("")
    try:
        df = pd.read_csv(data_path, usecols=[
            'FIPS', 'year', 'geo_name', 'longitude', 'latitude', 'state',
            'life_expectancy', 'population', 'premature_mortality',
            'median_household_income', 'pct_65_and_older', 'driving_alone_to_work',
            'injury_deaths', 'adult_obesity', 'frequent_mental_distress',
            'single_parent_households', 'air_pollution_particulate_matter',
            'median_age', 'adult_smoking', 'mammography_screening',
            'housing_cost_challenges', 'social_associations',
            'excessive_drinking', 'insufficient_sleep', 'pct_under_18',
            'high_school_graduation', 'ratio_of_pop_to_dentists', 'uninsured_adults',
            'preventable_hospital_stays', 'homeownership', 'poverty',
            'children_in_poverty', 'unemployment_rate'
        ])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Load data at app startup
df = load_data()
if df is None:
    st.error("Unable to proceed due to data loading error. Please check the data file and try again.")
    st.stop()

# Optimize model loading
@st.cache_resource
def load_model_and_scaler():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, "models", "xgboost_best_selected_model.pkl")
    scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model, scaler = load_model_and_scaler()

# Define selected features
selected_features = [
    'premature_mortality', 'median_household_income', 'pct_65_and_older', 
    'driving_alone_to_work', 'injury_deaths', 'adult_obesity', 'frequent_mental_distress', 
    'single_parent_households', 'air_pollution_particulate_matter', 'median_age', 
    'adult_smoking', 'mammography_screening', 'housing_cost_challenges', 
    'social_associations', 'excessive_drinking', 'insufficient_sleep', 'pct_under_18', 
    'high_school_graduation', 'ratio_of_pop_to_dentists', 'uninsured_adults', 
    'preventable_hospital_stays', 'homeownership', 'poverty', 
    'children_in_poverty', 'unemployment_rate'
]

# Optimize feature range calculation
@st.cache_data
def get_feature_ranges():
    feature_ranges = {}
    for feature in selected_features:
        min_val = df[feature].min()
        max_val = df[feature].max()
        feature_ranges[feature] = (min_val, max_val)
    return feature_ranges

# Optimize video background setting
@st.cache_data
def get_base64_video(video_file):
    with open(video_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_video_background(video_file):
    video_base64 = get_base64_video(video_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:video/mp4;base64,{video_base64});
            background-size: cover;
        }}
        #myVideo {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
        }}
        </style>
        <video autoplay muted loop id="myVideo">
            <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
        </video>
        """,
        unsafe_allow_html=True
    )

# Optimize polar area chart creation
@st.cache_data
def get_polar_area_chart(input_data):
    feature_ranges = get_feature_ranges()
    normalized_data = {feature: (value - feature_ranges[feature][0]) / (feature_ranges[feature][1] - feature_ranges[feature][0])
                       for feature, value in input_data.items()}

    fig = go.Figure(go.Barpolar(
        r=list(normalized_data.values()),
        theta=list(normalized_data.keys()),
        opacity=0.8,
        marker=dict(
            color=list(normalized_data.values()),
            colorscale='Magma',
            showscale=True,
            colorbar=dict(title='Normalized Value')
        )
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), angularaxis=dict(direction="clockwise")),
        showlegend=False,
        height=600,
        margin=dict(l=80, r=80, t=20, b=20),
        title="Community Health Indicators"
    )
    return fig

# Optimize life expectancy prediction
def predict_life_expectancy():
    st.title("🧬 Life Expectancy Predictor")
    st.write("Explore how community health indicators affect life expectancy. Adjust the sliders to see real-time changes in the prediction.")
    
    feature_ranges = get_feature_ranges()
    input_dict = {}
    
    st.sidebar.header("Community Health Indicators")
    for feature, (min_val, max_val) in feature_ranges.items():
        default_val = df[feature].median()
        input_dict[feature] = st.sidebar.slider(
            feature.replace('_', ' ').title(),
            min_value=float(min_val),
            max_value=float(max_val),
            value=float(default_val),
            key=feature
        )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Community Health Indicators Visualization")
        polar_area_chart = get_polar_area_chart(input_dict)
        st.plotly_chart(polar_area_chart, use_container_width=True)
    
    with col2:
        st.subheader("Life Expectancy Prediction")
        if st.button("Predict Life Expectancy", key="predict_button"):
            input_array = np.array([input_dict[feature] for feature in selected_features]).reshape(1, -1)
            scaled_input = scaler.transform(input_array)
            prediction = model.predict(scaled_input)[0]
            st.markdown(f"<h1 style='text-align: center; color: #4e79a7;'>{prediction:.2f} years</h1>", unsafe_allow_html=True)
            st.info("This prediction is based on community health indicators and should not be used as a substitute for professional medical advice.")
        
        st.markdown("### How it works")
        st.write("1. Adjust the sliders in the sidebar to input community health indicators.")
        st.write("2. The polar area chart updates in real-time to visualize your inputs.")
        st.write("3. Click 'Predict Life Expectancy' to see the estimated life expectancy based on your inputs.")
        st.write("4. Experiment with different combinations to see how various factors affect life expectancy.")

# Main function to run the app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page", ["Overview", "Life Expectancy Insights", "Disparities and Impact", "Life Expectancy Predictor"])
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0);
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    if page == "Overview":
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bluezonevideo.mp4")
        set_video_background(video_path)

        st.title("Exploring Life Expectancy in the U.S.")
        st.write("""
        This app explores factors influencing life expectancy across the United States, 
        with a focus on disparities between rural and urban areas. Key factors include 
        healthcare access, environmental conditions, lifestyle choices, and socioeconomic status.
        """)

        selected_state = st.sidebar.selectbox("Select a State", ["All"] + sorted(df['state'].unique()))
        selected_year = st.sidebar.slider("Select a Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))

        filtered_df = df[df['year'] == selected_year]
        if selected_state != "All":
            filtered_df = filtered_df[filtered_df['state'] == selected_state]

        national_avg = filtered_df['life_expectancy'].mean()

        st.subheader("Life Expectancy Across U.S. Counties")
                
        col1, col2, col3 = st.columns(3)
        col1.metric("National Average Life Expectancy", f"{national_avg:.1f} years")
        col2.metric("Highest Life Expectancy", f"{filtered_df['life_expectancy'].max():.1f} years")
        col3.metric("Lowest Life Expectancy", f"{filtered_df['life_expectancy'].min():.1f} years")

        fig = px.scatter_mapbox(filtered_df, 
                                lat="latitude", 
                                lon="longitude", 
                                color="life_expectancy",
                                hover_name="geo_name",
                                zoom=3, 
                                mapbox_style="carto-positron",
                                color_continuous_scale="Viridis",
                                size_max=18,
                                width=1200,
                                height=800)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

    elif page == "Life Expectancy Insights":
        st.title("Life Expectancy Insights")

        selected_state = st.sidebar.selectbox("Select a State", ["All"] + sorted(df['state'].unique()))
        selected_year = st.sidebar.slider("Select a Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))

        filtered_df = df[df['year'] == selected_year]
        if selected_state != "All":
            filtered_df = filtered_df[filtered_df['state'] == selected_state]

        st.header("Statistical Summaries")
        
        stats_df = filtered_df.groupby('state')['life_expectancy'].agg(['mean', 'median']).round(2).reset_index()
        
        for stat in ['mean', 'median']:
            stats_df[f'{stat}_county'] = stats_df.apply(lambda row: 
                filtered_df[filtered_df['state'] == row['state']].iloc[
                    (filtered_df[filtered_df['state'] == row['state']]['life_expectancy'] - row[stat]).abs().argsort()[:1]
                ]['geo_name'].values[0], axis=1)

        st.write(stats_df)

        st.subheader("Statistical Summaries Map")

        col1, col2 = st.columns([3, 1])

        with col1:
            mean_data = stats_df[['state', 'mean', 'mean_county']].merge(filtered_df[['geo_name', 'longitude', 'latitude']], left_on='mean_county', right_on='geo_name')
            median_data = stats_df[['state', 'median', 'median_county']].merge(filtered_df[['geo_name', 'longitude', 'latitude']], left_on='median_county', right_on='geo_name')

            fig = go.Figure()

            fig.add_trace(go.Scattergeo(
                lon=mean_data['longitude'],
                lat=mean_data['latitude'],
                text=[f"State: {row['state']}<br>Mean: {row['mean']:.2f}<br>County: {row['mean_county']}" for _, row in mean_data.iterrows()],
                mode='markers',
                name='Mean Life Expectancy',
                marker=dict(
                    size=10,
                    color=mean_data['mean'],
                    colorscale='RdYlGn',
                    colorbar=dict(title='Life Expectancy (years)', x=1.1),
                    cmin=min(stats_df['mean'].min(), stats_df['median'].min()),
                    cmax=max(stats_df['mean'].max(), stats_df['median'].max()),
                    symbol='circle'
                ),
                hoverinfo='text'
            ))

            fig.add_trace(go.Scattergeo(
                lon=median_data['longitude'],
                lat=median_data['latitude'],
                text=[f"State: {row['state']}<br>Median: {row['median']:.2f}<br>County: {row['median_county']}" for _, row in median_data.iterrows()],
                mode='markers',
                name='Median Life Expectancy',
                marker=dict(
                    size=10,
                    color=median_data['median'],
                    colorscale='RdYlGn',
                    cmin=min(stats_df['mean'].min(), stats_df['median'].min()),
                    cmax=max(stats_df['mean'].max(), stats_df['median'].max()),
                    symbol='diamond'
                ),
                hoverinfo='text'
            ))

            fig.update_layout(
                geo=dict(
                    scope='usa',
                    projection_type='albers usa',
                    showland=True,
                    landcolor='rgb(20, 20, 20)',
                    countrycolor='rgb(40, 40, 40)',
                    showlakes=True,
                    lakecolor='rgb(20, 20,20)',
                    subunitcolor='rgb(40, 40, 40)'
                ),
                paper_bgcolor='rgb(10, 10, 10)',
                plot_bgcolor='rgb(10, 10, 10)',
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(0,0,0,0.5)',
                    font=dict(color='white')
                ),
                height=600,
                font=dict(color='white'),
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Legend")
            
            st.markdown("""
            The map shows two key statistics for each state:
            - **Circles**: Mean Life Expectancy
            - **Diamonds**: Median Life Expectancy
            
            Colors indicate life expectancy values:
            - Red: Lower life expectancy
            - Yellow: Medium life expectancy
            - Green: Higher life expectancy
            
            Hover over points to see detailed information.
            """)
            
            st.write("### Summary Statistics")
            st.write(f"Lowest Life Expectancy: {min(stats_df['mean'].min(), stats_df['median'].min()):.1f} years")
            st.write(f"Highest Life Expectancy: {max(stats_df['mean'].max(), stats_df['median'].max()):.1f} years")
            st.write(f"Average Life Expectancy: {((stats_df['mean'].mean() + stats_df['median'].mean()) / 2):.1f} years")
            
            if selected_state != "All":
                st.write(f"### Current Filter")
                st.write(f"State: {selected_state}")
            st.write(f"Year: {selected_year}")

        st.header("Correlation and Regression Analysis")
        
        factors = [
            'premature_mortality', 'median_household_income', 'pct_65_and_older',
            'driving_alone_to_work', 'injury_deaths', 'adult_obesity',
            'frequent_mental_distress', 'single_parent_households',
            'air_pollution_particulate_matter', 'median_age', 'adult_smoking',
            'mammography_screening', 'housing_cost_challenges', 'social_associations'
        ]

        selected_factors = st.multiselect("Select factors for correlation analysis", factors, default=factors)
        
        @st.cache_data
        def get_correlation_matrix(df, factors):
            return df[factors + ['life_expectancy']].corr()

        corr_matrix = get_correlation_matrix(filtered_df, selected_factors)
        fig_heatmap = px.imshow(corr_matrix, 
                                labels=dict(color="Correlation"),
                                x=selected_factors + ['life_expectancy'],
                                y=selected_factors + ['life_expectancy'],
                                color_continuous_scale="RdBu_r")
        fig_heatmap.update_layout(title="Correlation Heatmap", width=800, height=600)
        st.plotly_chart(fig_heatmap)

        st.subheader("Regression Analysis")
        selected_factor_regression = st.selectbox("Select a factor for regression analysis", factors)
        
        @st.cache_data
        def perform_regression(df, x_col, y_col):
            x = df[x_col]
            y = df[y_col]
            return stats.linregress(x, y)

        slope, intercept, r_value, p_value, std_err = perform_regression(filtered_df, selected_factor_regression, 'life_expectancy')
        
        fig_regression = px.scatter(filtered_df, x=selected_factor_regression, y='life_expectancy', trendline="ols")
        fig_regression.update_layout(title=f"Regression: {selected_factor_regression} vs Life Expectancy", 
                                    width=1000, height=700)
        st.plotly_chart(fig_regression)

        st.header("Key Insights")
        st.write("""
        - Statistical summaries show variations in life expectancy across states.
        - The map highlights states with the highest and lowest life expectancy statistics.
        - Factors such as income, education, and healthcare access tend to have strong correlations with life expectancy.
        - The impact of various factors on life expectancy can vary significantly between states and counties.
        """)

    elif page == "Disparities and Impact":
        st.title("Understanding Disparities")

        selected_state = st.sidebar.selectbox("Select a State", ["All"] + sorted(df['state'].unique()))
        selected_year = st.sidebar.slider("Select a Year", int(df['year'].min()), int(df['year'].max()), int(df['year'].max()))
        population_threshold = st.sidebar.slider("Population threshold for Urban classification", 10000, 100000, 50000)

        filtered_df = df[df['year'] == selected_year]
        if selected_state != "All":
            filtered_df = filtered_df[filtered_df['state'] == selected_state]
        filtered_df['area_type'] = np.where(filtered_df['population'] > population_threshold, 'Urban', 'Rural')

        factors = [
            'premature_mortality', 'median_household_income', 'pct_65_and_older',
            'driving_alone_to_work', 'injury_deaths', 'adult_obesity',
            'frequent_mental_distress', 'single_parent_households',
            'air_pollution_particulate_matter', 'median_age', 'adult_smoking',
            'mammography_screening', 'housing_cost_challenges', 'social_associations'
        ]

        st.header("Feature Distribution Map")
        selected_feature = st.selectbox("Select a feature to visualize", factors)
        
        st.write(f"Data range for {selected_feature}: {filtered_df[selected_feature].min()} to {filtered_df[selected_feature].max()}")
        st.write(f"Number of non-null values: {filtered_df[selected_feature].count()}")
        
        if pd.api.types.is_numeric_dtype(filtered_df[selected_feature]):
            fig = px.scatter_mapbox(filtered_df, 
                                    lat="latitude", 
                                    lon="longitude", 
                                    color=selected_feature,
                                    size=selected_feature,
                                    size_max=10,
                                    zoom=3, 
                                    center={"lat": 37.0902, "lon": -95.7129},
                                    mapbox_style="carto-positron",
                                    hover_name="geo_name",
                                    hover_data=[selected_feature, 'area_type'],
                                    color_continuous_scale="Viridis",
                                    title=f"Distribution of {selected_feature} Across Counties")
            
            fig.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
            st.plotly_chart(fig)
        else:
            st.error(f"The selected feature '{selected_feature}' is not numeric. Please choose a numeric feature for visualization.")

        st.header("Rural vs. Urban Disparities")
        
        fig = go.Figure()
        for area in ['Rural', 'Urban']:
            fig.add_trace(go.Box(y=filtered_df[filtered_df['area_type'] == area][selected_feature], name=area))
        fig.update_layout(title=f"{selected_feature} in Rural vs Urban Areas", width=800, height=500)
        st.plotly_chart(fig)

        st.header("Case Study Analysis")
        
        low_county = filtered_df.loc[filtered_df[selected_feature].idxmin()]
        high_county = filtered_df.loc[filtered_df[selected_feature].idxmax()]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"Lowest {selected_feature}")
            st.write(f"County: {low_county['geo_name']}")
            st.write(f"{selected_feature}: {low_county[selected_feature]:.2f}")
            st.write("Other Key Factors:")
            for factor in [f for f in factors if f != selected_feature][:5]:
                st.write(f"- {factor}: {low_county[factor]:.2f}")

        with col2:
            st.subheader(f"Highest {selected_feature}")
            st.write(f"County: {high_county['geo_name']}")
            st.write(f"{selected_feature}: {high_county[selected_feature]:.2f}")
            st.write("Other Key Factors:")
            for factor in [f for f in factors if f != selected_feature][:5]:
                st.write(f"- {factor}: {high_county[factor]:.2f}")

        st.header("Feature Relationships Heatmap")
        
        @st.cache_data
        def get_correlation_data(df, factors, selected_feature):
            return df[factors].corr()[selected_feature].sort_values(ascending=False)

        correlation_data = get_correlation_data(filtered_df, factors, selected_feature)
        
        fig = px.imshow(correlation_data.values.reshape(-1, 1),
                        y=correlation_data.index,
                        color_continuous_scale="RdBu_r",
                        labels=dict(color="Correlation"),
                        title=f"Correlation of {selected_feature} with Other Features")
        fig.update_layout(width=800, height=600)
        st.plotly_chart(fig)

        st.header("Recommendations")
        st.write("""
        Based on our analysis, here are some data-driven recommendations to improve life expectancy:
        1. Improve healthcare access in rural areas
        2. Invest in education and job opportunities to reduce poverty
        3. Implement environmental policies to reduce air pollution
        4. Promote healthy lifestyle choices and access to exercise opportunities
        5. Address mental health issues and provide better access to mental health services
        """)

        st.header("Key Insights")
        st.write(f"""
        - The map reveals significant geographic variations in {selected_feature} across counties.
        - Rural and urban areas show distinct patterns in {selected_feature} distribution.
        - Extreme cases highlight the range of disparities and potential areas for targeted interventions.
        - The heatmap reveals strong correlations between certain factors and {selected_feature}, which can guide policy decisions.
        - Addressing these disparities requires a multi-faceted approach, considering local contexts and needs.
        - Policy interventions should be tailored to the specific challenges faced by each community, as the impact of various factors can differ between regions.
        """)

    elif page == "Life Expectancy Predictor":
        predict_life_expectancy()

if __name__ == "__main__":
    main()