import heapq

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ============================
# STREAMLIT PAGE CONFIG
# ============================
st.set_page_config(page_title="Hyderabad Metro Planner", layout="wide")
      
# ============================
# 1. BUILD SYNTHETIC DATASET
# ============================
def build_dataset():
    stations_info = {
        "Miyapur": "residential",
        "JNTU College": "IT_residential",
        "KPHB Colony": "IT_residential",        
        "Ameerpet": "interchange",
        "LB Nagar": "residential",
        "Nagole": "residential",
        "Uppal": "residential",
        "Stadium": "residential",
        "MGBS": "interchange",
        "Raidurg": "IT_hub",
        "JBS Parade Ground": "interchange",
    }

    day_types = ["weekday", "weekend"]
    rows = []

    for station, stype in stations_info.items():
        for day in day_types:
            for hour in range(6, 23):  # 6 AM to 10 PM
                # Rule-based crowd assignment
                if day == "weekday" and (8 <= hour <= 10 or 17 <= hour <= 20):
                    # Peak hours on weekday
                    if stype in ["interchange", "IT_hub", "IT_residential"]:
                        crowd = "High"
                    else:
                        crowd = "Medium"
                elif day == "weekday" and (11 <= hour <= 16):
                    # Midday weekday
                    if stype in ["interchange", "IT_hub", "IT_residential"]:
                        crowd = "Medium"
                    else:
                        crowd = "Low"
                elif day == "weekday":
                    # Early morning / late evening weekday
                    crowd = "Low"
                else:
                    # Weekend pattern: generally lower
                    if 10 <= hour <= 13 and stype in ["interchange", "IT_hub"]:
                        crowd = "Medium"
                    else:
                        crowd = "Low"

                rows.append(
                    {
                        "station": station,
                        "hour": hour,
                        "day_type": day,
                        "crowd_level": crowd,
                    }
                )

    df = pd.DataFrame(rows)
    return df


# ============================
# 2. TRAIN ML MODEL
# ============================
def train_crowd_model(df: pd.DataFrame):
    # Encode target
    crowd_mapping = {"Low": 0, "Medium": 1, "High": 2}
    df["crowd_encoded"] = df["crowd_level"].map(crowd_mapping)

    # One-hot encode station + day_type
    X = pd.get_dummies(df[["station", "hour", "day_type"]])
    Y = df["crowd_encoded"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42, stratify=Y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, Y_train)

    return model, X, crowd_mapping


# ============================
# 3. METRO GRAPH + COORDINATES
# ============================
metro_graph = {
    "Miyapur": [("JNTU College", 3)],
    "JNTU College": [("Miyapur", 3), ("KPHB Colony", 3)],
    "KPHB Colony": [("JNTU College", 3), ("SR Nagar", 3)],
    "SR Nagar": [("KPHB Colony", 3), ("Ameerpet", 3)],
    "Ameerpet": [("SR Nagar", 3), ("LB Nagar", 5), ("MGBS", 5), ("Raidurg", 6)],
    "LB Nagar": [("Ameerpet", 5)],  # Red line

    "Nagole": [("Uppal", 3)],
    "Uppal": [("Nagole", 3), ("Stadium", 3)],
    "Stadium": [("Uppal", 3), ("MGBS", 4)],
    "MGBS": [("Stadium", 4), ("Ameerpet", 5), ("JBS Parade Ground", 4)],  # Blue+Green

    "Raidurg": [("Ameerpet", 6)],

    "JBS Parade Ground": [("MGBS", 4)],
}

# Schematic coordinates for a simple metro map (not real GPS)
station_coords = {
    "Miyapur": (0, 0),
    "JNTU College": (1, 0),
    "KPHB Colony": (2, 0),
    "SR Nagar": (3, 0),
    "Ameerpet": (4, 0),
    "LB Nagar": (5, -1),
    "Nagole": (5, 2),
    "Uppal": (4, 2),
    "Stadium": (3, 2),
    "MGBS": (4, 1),
    "Raidurg": (5, 0),
    "JBS Parade Ground": (3, 3),
}


def dijkstra_shortest_path(graph, source, destination):
    heap = [(0, source, [source])]
    visited = {}

    while heap:
        current_time, station, path = heapq.heappop(heap)

        if station in visited and visited[station] <= current_time:
            continue

        visited[station] = current_time

        if station == destination:
            return current_time, path

        for neighbor, travel_time in graph.get(station, []):
            new_time = current_time + travel_time
            new_path = path + [neighbor]
            heapq.heappush(heap, (new_time, neighbor, new_path))

    return None, []


# ============================
# 4. PREDICTION HELPERS
# ============================
def make_predict_crowd(model, X, crowd_mapping):
    inv_mapping = {v: k for k, v in crowd_mapping.items()}

    def predict_crowd(station_name, hour, day_type):
        sample = pd.DataFrame(
            [{"station": station_name, "hour": hour, "day_type": day_type}]
        )
        sample_encoded = pd.get_dummies(sample)
        sample_encoded = sample_encoded.reindex(columns=X.columns, fill_value=0)
        pred_encoded = model.predict(sample_encoded)[0]
        return inv_mapping[pred_encoded]

    return predict_crowd


def get_route_with_crowd(graph, source, destination, hour, day_type, predict_crowd_fn):
    total_time, path = dijkstra_shortest_path(graph, source, destination)
    if not path:
        return None

    station_crowd = []
    crowd_to_score = {"Low": 0, "Medium": 1, "High": 2}
    total_crowd_score = 0

    for station in path:
        level = predict_crowd_fn(station, hour, day_type)
        score = crowd_to_score[level]
        station_crowd.append((station, level))
        total_crowd_score += score

    avg_crowd_score = total_crowd_score / len(path)

    return {
        "source": source,
        "destination": destination,
        "total_time": total_time,
        "path": path,
        "station_crowd": station_crowd,
        "avg_crowd_score": avg_crowd_score,
    }


# ============================
# 5. PLOTLY METRO MAP
# ============================
def build_metro_map(result):
    fig = go.Figure()

    # Define line sequences
    red_line = ["Miyapur", "JNTU College", "KPHB Colony", "SR Nagar", "Ameerpet", "LB Nagar"]
    blue_line = ["Nagole", "Uppal", "Stadium", "MGBS", "Ameerpet", "Raidurg"]
    green_line = ["JBS Parade Ground", "MGBS"]

    def add_line_trace(stations, color_name, name):
        xs = [station_coords[s][0] for s in stations]
        ys = [station_coords[s][1] for s in stations]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                line=dict(width=3, color=color_name),
                name=name,
                hoverinfo="none",
            )
        )

    add_line_trace(red_line, "firebrick", "Red Line")
    add_line_trace(blue_line, "royalblue", "Blue Line")
    add_line_trace(green_line, "seagreen", "Green Line")

    # All stations as labels
    all_x = [station_coords[s][0] for s in station_coords.keys()]
    all_y = [station_coords[s][1] for s in station_coords.keys()]
    all_names = list(station_coords.keys())

    fig.add_trace(
        go.Scatter(
            x=all_x,
            y=all_y,
            mode="markers+text",
            marker=dict(size=10, color="lightgrey"),
            text=all_names,
            textposition="top center",
            name="Stations",
            hoverinfo="text",
        )
    )

    # Highlight selected route
    route = result["path"]
    route_x = [station_coords[s][0] for s in route]
    route_y = [station_coords[s][1] for s in route]

    fig.add_trace(
        go.Scatter(
            x=route_x,
            y=route_y,
            mode="lines+markers",
            line=dict(width=5, color="black"),
            marker=dict(size=12, color="black"),
            name="Selected Route",
        )
    )

    # Station markers colored by crowd
    crowd_color_map = {"Low": "green", "Medium": "orange", "High": "red"}
    scatter_x, scatter_y, scatter_colors, scatter_text = [], [], [], []

    for station, crowd in result["station_crowd"]:
        scatter_x.append(station_coords[station][0])
        scatter_y.append(station_coords[station][1])
        scatter_colors.append(crowd_color_map.get(crowd, "grey"))
        scatter_text.append(f"{station}: {crowd}")

    fig.add_trace(
        go.Scatter(
            x=scatter_x,
            y=scatter_y,
            mode="markers",
            marker=dict(size=16, color=scatter_colors, line=dict(width=1, color="black")),
            name="Crowd Level",
            text=scatter_text,
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title="Hyderabad Metro – Schematic Route & Crowd Map",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=True,
        height=600,
    )

    return fig


# ============================
# 6. STREAMLIT APP
# ============================
def main():
    st.title("🚇 Hyderabad Metro Rail Commute analytics")
    st.caption("B.Tech Mini Project – Data Science + Machine Learning + Graph Algorithms")

    st.markdown(
        """
        This tool predicts **crowd levels** for Hyderabad Metro stations and finds the **best route**
        between two stations using a combination of **Machine Learning** and **Dijkstra's shortest path algorithm**.
        """
    )

    # ---- Model preparation section ----
    st.subheader("Model Preparation")
    df = build_dataset()
    model, X, crowd_mapping = train_crowd_model(df)
    predict_crowd_fn = make_predict_crowd(model, X, crowd_mapping)
    st.success("Crowd prediction model trained successfully on synthetic metro data.")

    # ---- Journey planning section ----
    st.subheader("Plan Your Metro Journey")

    left_col, right_col = st.columns(2)

    with left_col:
        stations = list(metro_graph.keys())
        source = st.selectbox("Source station", stations, index=0)
        destination = st.selectbox("Destination station", stations, index=3)
        hour = st.slider("Hour of travel (24-hour format)", min_value=6, max_value=22, value=9)
        day_type = st.radio("Day type", ["weekday", "weekend"], index=0)
        go = st.button("Plan Journey")

    with right_col:
        st.info(
            """
            **Usage Tips:**
            - Try **9 AM** or **6 PM** on a **weekday** to see peak-hour crowd at IT and interchange stations.
            - Try **2 PM** to observe lower crowd.
            - Weekends generally show lower crowd levels.
            """
        )

    if go:
        if source == destination:
            st.error("Source and destination must be different.")
            return

        result = get_route_with_crowd(
            metro_graph, source, destination, hour, day_type, predict_crowd_fn
        )

        if result is None:
            st.error("No route found between the selected stations.")
            return

        st.markdown("### Route Summary")
        st.write(f"**Source:** {result['source']}")
        st.write(f"**Destination:** {result['destination']}")
        st.write(f"**Total Travel Time:** {result['total_time']} minutes")
        st.write("**Route:** " + " → ".join(result["path"]))

        st.markdown("### Station-wise Crowd Prediction")
        crowd_rows = [{"Station": stn, "Predicted Crowd": lvl} for stn, lvl in result["station_crowd"]]
        st.table(pd.DataFrame(crowd_rows))

        avg = result["avg_crowd_score"]
        st.write(f"**Average Crowd Score (0=Low, 1=Medium, 2=High):** {avg:.2f}")
        if avg < 0.7:
            st.success("This route is mostly comfortable (Low crowd).")
        elif avg < 1.4:
            st.warning("This route has Medium crowd levels.")
        else:
            st.error("This route is quite crowded (High crowd). Consider traveling at a different time.")

        st.markdown("### Metro Route Visualization")
        fig = build_metro_map(result)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
