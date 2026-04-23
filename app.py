import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")
st.title("🏠 House Price Predictor")
st.caption("Trained on real Kaggle housing data — Linear Regression vs Random Forest")

FEATURES = [
    'OverallQual',
    'GrLivArea',
    'GarageCars',
    'TotalBsmtSF',
    'FullBath',
    'YearBuilt',
    'BedroomAbvGr',
    'LotArea',
]
FEATURE_LABELS = {
    'OverallQual':  'Overall Quality (1–10)',
    'GrLivArea':    'Living Area (sq ft)',
    'GarageCars':   'Garage Capacity (cars)',
    'TotalBsmtSF':  'Basement Area (sq ft)',
    'FullBath':     'Full Bathrooms',
    'YearBuilt':    'Year Built',
    'BedroomAbvGr': 'Bedrooms',
    'LotArea':      'Lot Size (sq ft)',
}
TARGET = 'SalePrice'

# ── Load & train ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('train.csv')
    return df[FEATURES + [TARGET]].dropna()

@st.cache_resource
def train_models(data):
    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    metrics = {}
    for name, model in [('Linear Regression', lr), ('Random Forest', rf)]:
        pred = model.predict(X_test)
        metrics[name] = {
            'R2':     r2_score(y_test, pred),
            'MAE':    mean_absolute_error(y_test, pred),
            'pred':   pred,
            'actual': y_test.values,
        }
    return lr, rf, metrics

data = load_data()
lr, rf, metrics = train_models(data)

# ── Sidebar — prediction inputs ───────────────────────────────────────────────
st.sidebar.header("🏡 Enter House Details")

qual     = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 7)
area     = st.sidebar.slider("Living Area (sq ft)",    300, 5000, 1500, step=50)
garage   = st.sidebar.selectbox("Garage Capacity (cars)", [0, 1, 2, 3, 4])
basement = st.sidebar.slider("Basement Area (sq ft)",  0, 3000, 800, step=50)
baths    = st.sidebar.selectbox("Full Bathrooms",      [0, 1, 2, 3, 4])
year     = st.sidebar.slider("Year Built",             1872, 2010, 2000)
beds     = st.sidebar.selectbox("Bedrooms",            [0, 1, 2, 3, 4, 5, 6])
lot      = st.sidebar.slider("Lot Size (sq ft)",       1000, 50000, 8000, step=500)
model_choice = st.sidebar.radio("Model", ["Random Forest", "Linear Regression"])

input_df = pd.DataFrame([[qual, area, garage, basement, baths, year, beds, lot]],
                        columns=FEATURES)
chosen = rf if model_choice == "Random Forest" else lr
prediction = chosen.predict(input_df)[0]

st.sidebar.markdown("---")
st.sidebar.metric("Predicted Sale Price", f"${prediction:,.0f}")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 EDA — Explore the Data", "🤖 Model Comparison", "🔮 Predict"])

# ── Tab 1 : EDA ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Exploratory Data Analysis")
    st.write(f"Dataset: **{len(data)} houses** · **{len(FEATURES)} features** · Target: `SalePrice`")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Sale Price Distribution")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.histplot(data[TARGET], bins=50, kde=True, color='steelblue', ax=ax)
        ax.set_xlabel("Sale Price ($)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig)
        plt.close()

    with col2:
        st.markdown("#### Sale Price vs Living Area")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sc = ax.scatter(data['GrLivArea'], data[TARGET],
                        c=data['OverallQual'], cmap='RdYlGn', alpha=0.5, s=12)
        plt.colorbar(sc, ax=ax, label='Quality')
        ax.set_xlabel("Living Area (sq ft)")
        ax.set_ylabel("Sale Price ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig)
        plt.close()

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Sale Price by Overall Quality")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        quality_means = data.groupby('OverallQual')[TARGET].median()
        ax.bar(quality_means.index, quality_means.values,
               color=plt.cm.RdYlGn(np.linspace(0.1, 0.9, len(quality_means))))
        ax.set_xlabel("Overall Quality (1–10)")
        ax.set_ylabel("Median Sale Price ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("#### Sale Price by Bedrooms")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        bed_means = data.groupby('BedroomAbvGr')[TARGET].median()
        ax.bar(bed_means.index, bed_means.values, color='coral')
        ax.set_xlabel("Bedrooms Above Ground")
        ax.set_ylabel("Median Sale Price ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig)
        plt.close()

    # Row 3
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("#### Sale Price vs Year Built")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.scatter(data['YearBuilt'], data[TARGET], alpha=0.3, color='mediumseagreen', s=10)
        ax.set_xlabel("Year Built")
        ax.set_ylabel("Sale Price ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        st.pyplot(fig)
        plt.close()

    with col6:
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = data.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                    mask=mask, linewidths=0.5, annot_kws={'size': 7})
        ax.set_xticklabels([FEATURE_LABELS.get(c, c).replace(' ', '\n') for c in corr.columns],
                           fontsize=6)
        ax.set_yticklabels([FEATURE_LABELS.get(c, c) for c in corr.index], fontsize=6)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Raw Data Sample")
    display = data.head(10).copy()
    display.columns = [FEATURE_LABELS.get(c, c) for c in display.columns]
    st.dataframe(display, use_container_width=True)

# ── Tab 2 : Model Comparison ──────────────────────────────────────────────────
with tab2:
    st.subheader("Model Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Linear Regression")
        st.metric("R² Score", f"{metrics['Linear Regression']['R2']:.4f}")
        st.metric("MAE",      f"${metrics['Linear Regression']['MAE']:,.0f}")
        st.caption("Fast and simple. Works well when relationships are roughly linear.")
    with col2:
        st.markdown("#### Random Forest")
        st.metric("R² Score", f"{metrics['Random Forest']['R2']:.4f}")
        st.metric("MAE",      f"${metrics['Random Forest']['MAE']:,.0f}")
        st.caption("Handles complex patterns. Usually wins on accuracy.")

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### R² Score (higher = better)")
        fig, ax = plt.subplots(figsize=(5, 3))
        names = list(metrics.keys())
        r2s = [metrics[n]['R2'] for n in names]
        bars = ax.bar(names, r2s, color=['steelblue', 'coral'])
        ax.set_ylim(0, 1)
        ax.set_ylabel("R² Score")
        for bar, val in zip(bars, r2s):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.3f}', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()

    with col4:
        st.markdown("#### MAE (lower = better)")
        fig, ax = plt.subplots(figsize=(5, 3))
        maes = [metrics[n]['MAE'] for n in names]
        bars = ax.bar(names, maes, color=['steelblue', 'coral'])
        ax.set_ylabel("Mean Absolute Error ($)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x() + bar.get_width()/2, val + 200,
                    f'${val:,.0f}', ha='center', fontweight='bold', fontsize=8)
        st.pyplot(fig)
        plt.close()

    st.markdown("#### Actual vs Predicted — Random Forest")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(metrics['Random Forest']['actual'],
               metrics['Random Forest']['pred'],
               alpha=0.4, color='coral', s=15)
    mn = min(metrics['Random Forest']['actual'].min(), metrics['Random Forest']['pred'].min())
    mx = max(metrics['Random Forest']['actual'].max(), metrics['Random Forest']['pred'].max())
    ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, label='Perfect prediction')
    ax.set_xlabel("Actual Sale Price ($)")
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown("#### Feature Importance — Random Forest")
    importances = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['coral' if v == importances.max() else 'steelblue' for v in importances]
    importances.plot.barh(ax=ax, color=colors)
    ax.set_xlabel("Importance Score")
    ax.set_title("Which features matter most?")
    ax.set_yticklabels([FEATURE_LABELS[f] for f in importances.index])
    st.pyplot(fig)
    plt.close()

# ── Tab 3 : Predict ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Your House Prediction")
    st.info("Adjust the sliders in the sidebar and see the price update instantly.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Quality",   f"{qual}/10")
    col2.metric("Living Area", f"{area:,} sq ft")
    col3.metric("Year Built", year)
    col4.metric("Bedrooms",  beds)

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Bathrooms", baths)
    col6.metric("Garage",    f"{garage} cars")
    col7.metric("Basement",  f"{basement:,} sq ft")
    col8.metric("Lot Size",  f"{lot:,} sq ft")

    st.markdown("---")

    both = {
        'Linear Regression': lr.predict(input_df)[0],
        'Random Forest':     rf.predict(input_df)[0],
    }

    col_a, col_b = st.columns(2)
    col_a.metric("Linear Regression", f"${both['Linear Regression']:,.0f}")
    col_b.metric("Random Forest",     f"${both['Random Forest']:,.0f}")

    fig, ax = plt.subplots(figsize=(5, 3))
    bars = ax.bar(both.keys(), both.values(), color=['steelblue', 'coral'])
    ax.set_ylabel("Predicted Sale Price ($)")
    ax.set_title("Both Models — Your House")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    for bar, val in zip(bars, both.values()):
        ax.text(bar.get_x() + bar.get_width()/2, val + 500,
                f'${val:,.0f}', ha='center', fontweight='bold', fontsize=9)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.markdown("#### How does your house compare to the dataset?")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(data[TARGET], bins=50, color='lightgray', edgecolor='white', label='All houses')
    ax.axvline(prediction, color='red', lw=2.5, linestyle='--',
               label=f'Your house: ${prediction:,.0f}')
    ax.set_xlabel("Sale Price ($)")
    ax.set_ylabel("Count")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend()
    st.pyplot(fig)
    plt.close()
