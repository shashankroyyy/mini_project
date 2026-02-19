import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

from route_planner import get_route_with_future_crowd

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Hyderabad Metro | Smart Route Planner",
    page_icon="🚇",
    layout="wide"
)

# -------------------------------------------------
# METRO GRAPH
# -------------------------------------------------
metro_graph = {
    "Miyapur": [("JNTU College", 3)],
    "JNTU College": [("Miyapur", 3), ("KPHB Colony", 3)],
    "KPHB Colony": [("JNTU College", 3), ("SR Nagar", 3)],
    "SR Nagar": [("KPHB Colony", 3), ("Ameerpet", 3)],
    "Ameerpet": [("SR Nagar", 3), ("LB Nagar", 5), ("MGBS", 5), ("Raidurg", 6)],
    "LB Nagar": [("Ameerpet", 5)],
    "Nagole": [("Uppal", 3)],
    "Uppal": [("Nagole", 3), ("Stadium", 3)],
    "Stadium": [("Uppal", 3), ("MGBS", 4)],
    "MGBS": [("Stadium", 4), ("Ameerpet", 5), ("JBS Parade Ground", 4)],
    "Raidurg": [("Ameerpet", 6)],
    "JBS Parade Ground": [("MGBS", 4)],
}

station_coords = {
    "Miyapur": (0, 0),
    "JNTU College": (1, 0),
    "KPHB Colony": (2, 0),
    "SR Nagar": (3, 0),
    "Ameerpet": (4, 0),
    "LB Nagar": (5, -1),
    "Raidurg": (6, 0),
    "MGBS": (4, 1),
    "Stadium": (3, 2),
    "Uppal": (4, 2),
    "Nagole": (5, 2),
    "JBS Parade Ground": (3, 3),
}

# -------------------------------------------------
# METRO MAP VISUALIZATION
# -------------------------------------------------
def build_metro_map(result):
    fig = go.Figure()

    red = ["Miyapur","JNTU College","KPHB Colony","SR Nagar","Ameerpet","LB Nagar"]
    blue = ["Nagole","Uppal","Stadium","MGBS","Ameerpet","Raidurg"]
    green = ["JBS Parade Ground","MGBS"]

    def draw_line(stations, color, name, dash=None):
        fig.add_trace(go.Scatter(
            x=[station_coords[s][0] for s in stations],
            y=[station_coords[s][1] for s in stations],
            mode="lines",
            line=dict(width=6, color=color, dash=dash),
            name=name,
            hoverinfo="none"
        ))

    draw_line(red, "#E53935", "Red Line")
    draw_line(blue, "#1E88E5", "Blue Line")
    draw_line(green, "#43A047", "Green Line", dash="dot")

    # Stations
    fig.add_trace(go.Scatter(
        x=[station_coords[s][0] for s in station_coords],
        y=[station_coords[s][1] for s in station_coords],
        mode="markers+text",
        text=list(station_coords.keys()),
        textposition="top center",
        marker=dict(size=14, color="white", line=dict(width=3, color="#263238")),
        name="Stations"
    ))

    # Selected route
    fig.add_trace(go.Scatter(
        x=[station_coords[s][0] for s in result["route"]],
        y=[station_coords[s][1] for s in result["route"]],
        mode="lines+markers",
        line=dict(color="#FFD600", width=10),
        marker=dict(size=18, color="#FFD600"),
        name="Selected Route"
    ))

    # Crowd overlay
    color_map = {"Low":"#00E676","Medium":"#FFD54F","High":"#FF5252"}

    fig.add_trace(go.Scatter(
        x=[station_coords[s["station"]][0] for s in result["station_crowd"]],
        y=[station_coords[s["station"]][1] for s in result["station_crowd"]],
        mode="markers",
        marker=dict(
            size=22,
            color=[color_map[s["predicted_crowd"]] for s in result["station_crowd"]],
            line=dict(width=3, color="black")
        ),
        text=[
            f"{s['station']}<br>Arrival: {s['arrival_time']}<br>Crowd: {s['predicted_crowd']}"
            for s in result["station_crowd"]
        ],
        hoverinfo="text",
        name="Crowd at Arrival"
    ))

    fig.update_layout(
        height=600,
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font=dict(color="white"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )

    return fig

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown(
    """
    <div style="padding:20px;border-radius:12px;
    background:linear-gradient(90deg,#0F2027,#203A43,#2C5364)">
    <h1 style="color:white">🚇 Hyderabad Metro Rail Commute Analytics</h1>
    <p style="color:#B0BEC5">Predictive Crowd Analytics · Dijkstra · LSTM</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# LOAD LSTM MODEL
# -------------------------------------------------
lstm_model = load_model("lstm_crowd_model.h5", compile=False)

station_crowd_history = {
    "Miyapur": [0, 1, 1],          # residential
    "JNTU College": [1, 1, 2],     # IT + students
    "KPHB Colony": [1, 2, 2],      # IT residential
    "SR Nagar": [1, 1, 1],
    "Ameerpet": [2, 2, 2],         # interchange (very crowded)
    "LB Nagar": [1, 1, 0],
    "MGBS": [2, 2, 1],             # major interchange
    "Raidurg": [2, 2, 2],          # IT hub
    "Uppal": [1, 1, 0],
    "Nagole": [0, 1, 1],
    "Stadium": [0, 0, 1],
    "JBS Parade Ground": [1, 2, 1]
}


# -------------------------------------------------
# JOURNEY PLANNER
# -------------------------------------------------
st.markdown("## 🧭 Plan Your Journey")

c1, c2, c3 = st.columns(3)
with c1:
    source = st.selectbox("Source Station", metro_graph.keys())
with c2:
    destination = st.selectbox("Destination Station", metro_graph.keys(), index=4)
with c3:
    hour = st.slider("Start Hour", 6, 22, 9)

plan_journey = st.button("🚀 Plan Journey", use_container_width=True)


# -------------------------------------------------
# RESULT SECTION
# -------------------------------------------------
if plan_journey:
    result = get_route_with_future_crowd(
        metro_graph,
        source,
        destination,
        hour,
        model=lstm_model,
        station_crowd_history=station_crowd_history
    )

    df = pd.DataFrame(result["station_crowd"])
    crowd_map = {"Low":0,"Medium":1,"High":2}
    df["crowd_score"] = df["predicted_crowd"].map(crowd_map)

    st.success("✅ Optimal route and future crowd predicted successfully")

    # ---------------- KPI DASHBOARD ----------------
    st.markdown("## 📊 Journey Dashboard")
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("⏱ Travel Time", f"{result['total_time']} min")
    k2.metric("🚉 Stations", len(result["route"]))
    k3.metric("📈 Avg Crowd", round(df["crowd_score"].mean(),2))
    k4.metric("⚠️ Peak Station", df.loc[df["crowd_score"].idxmax(),"station"])

    # ---------------- DISTRIBUTION ----------------
    st.markdown("### 🚦 Crowd Distribution")
    fig1 = px.bar(
        df,
        x="predicted_crowd",
        color="predicted_crowd",
        color_discrete_map={
            "Low":"#00E676","Medium":"#FFD54F","High":"#FF5252"
        },
        text_auto=True
    )
    fig1.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font_color="white"
    )
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- TREND ----------------
    st.markdown("### 📉 Crowd Trend Along Route")
    fig2 = px.line(df, x="station", y="crowd_score", markers=True)
    fig2.update_layout(
        plot_bgcolor="#0E1117",
        paper_bgcolor="#0E1117",
        font_color="white",
        yaxis=dict(
            tickvals=[0,1,2],
            ticktext=["Low","Medium","High"]
        )
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- TABLE ----------------
    st.markdown("### 📍 Station-wise Arrival & Crowd")
    st.dataframe(
        df[["station","arrival_time","predicted_crowd"]],
        use_container_width=True
    )

    # ---------------- MAP ----------------
    st.markdown("## 🗺 Smart Metro Route Map")
    st.caption("Highlighted path = optimal route · Colors = predicted crowd at arrival")
    st.plotly_chart(build_metro_map(result), use_container_width=True)
