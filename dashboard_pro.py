import streamlit as st
from data_loader import load_data
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import onnxruntime as rt
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_map_data():
    return pd.read_csv("aggregated_map_data.csv")

def get_display_genre(row):
    tags = [t.strip().lower() for t in str(row["top3_genres"]).split(",")]
    skip = {"pop", ""}
    for t in tags:
        if t not in skip:
            return t
    return tags[0] if tags else "unknown"


@st.cache_data
def get_metrics(music):
    return {
        "songs": music["title"].nunique(),
        "artists": music["artist"].nunique(),
        "regions": music["region"].nunique(),
        "streams": music["streams"].sum()
    }



_DATASET_START = pd.Timestamp("2017-01-01")


@st.cache_resource
def _load_clf_session():
    return rt.InferenceSession("model_classification.onnx")


@st.cache_resource
def _load_reg_session():
    return rt.InferenceSession("model_regression.onnx")


@st.cache_resource(show_spinner=False)
def _load_prediction_assets():
    # --- Classification assets (exact same encoding as Classification_Experiment_3_XGBClassifier.py) ---
    clf_df = pd.read_csv("music_classification.csv", dtype={
        "rank": "int16", "streams": "float32", "is_hit": "int8",
        "year": "int16", "month": "int8", "day": "int8", "weekday": "int8",
        "quarter": "int8", "is_weekend": "int8", "days_since_start": "int32",
        "artist_count": "int32",
    })
    clf_train  = clf_df[clf_df["year"] < 2021].copy()
    clf_global = float(clf_train["is_hit"].mean())
    clf_artist = clf_train.groupby("artist")["is_hit"].mean()
    clf_genre  = clf_train.groupby("main_genre")["is_hit"].mean()
    clf_region = clf_train.groupby("region")["is_hit"].mean()

    clf_train["artist_te"] = clf_train["artist"].map(clf_artist).fillna(clf_global)
    clf_train["genre_te"]  = clf_train["main_genre"].map(clf_genre).fillna(clf_global)
    clf_train["region_te"] = clf_train["region"].map(clf_region).fillna(clf_global)

    X_clf  = clf_train.drop(columns=["streams", "is_hit", "artist", "main_genre", "region"])
    scaler = StandardScaler()
    scaler.fit(X_clf)

    # --- Regression assets (exact same encoding as Regression_Experiment_13_XGBRegressor.py) ---
    reg_df = pd.read_csv("music_regression.csv", dtype={
        "rank": "int16", "streams": "float32",
        "year": "int16", "month": "int8", "day": "int8", "weekday": "int8",
        "quarter": "int8", "is_weekend": "int8", "days_since_start": "int32",
        "artist_count": "int32",
    })
    reg_train  = reg_df[reg_df["year"] < 2021].copy()
    reg_global = float(np.log1p(reg_train["streams"]).mean())
    reg_artist = reg_train.groupby("artist")["streams"].apply(lambda x: np.log1p(x).mean())
    reg_genre  = reg_train.groupby("main_genre")["streams"].apply(lambda x: np.log1p(x).mean())
    reg_region = reg_train.groupby("region")["streams"].apply(lambda x: np.log1p(x).mean())

    return {
        "clf_global":   clf_global,
        "clf_artist":   clf_artist.to_dict(),
        "clf_genre":    clf_genre.to_dict(),
        "clf_region":   clf_region.to_dict(),
        "artist_count": clf_df.groupby("artist")["artist_count"].first().to_dict(),
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "reg_global":   reg_global,
        "reg_artist":   reg_artist.to_dict(),
        "reg_genre":    reg_genre.to_dict(),
        "reg_region":   reg_region.to_dict(),
    }


def pro_dashboard():
    st.title("Professional Dashboard")

    # with st.spinner("Loading music data..."):
    #     music = load_music()
    #
    # c1, c2, c3, c4 = st.columns(4)
    #
    # c1.metric("Songs", music["title"].nunique(), border=True)
    # c2.metric("Artists", music["artist"].nunique(), border=True)
    # c3.metric("Regions", music["region"].nunique(), border=True)
    # c4.metric("streams", music['streams'].sum(), border=True)


    with st.spinner("Loading data..."):
            music = load_data()

    m = get_metrics(music)

    c1, c2, c3, c4 = st.columns(4)


    c1.metric("Songs", m["songs"], border=True)
    c2.metric("Artists", m["artists"], border=True)
    c3.metric("Regions", m["regions"], border=True)
    c4.metric("Streams", m['streams'] , border=True)




    col_left, col_right = st.columns([2, 1] , border=True)

    with col_left:
        r1, r2 = st.columns([3, 2], border=True)
        with r1:
                st.subheader("🗺️ Global Music Map")

                map_df = load_map_data()
                map_df = map_df[map_df["region"] != "Global"].copy()
                map_df["display_genre"] = map_df.apply(get_display_genre, axis=1)

                # Toggle
                genre_view = st.radio(
                    "Genre View",
                    ["Top Genre", "Top Local Genre"],
                    horizontal=True,
                    key="map_genre_view"
                )

                if genre_view == "Top Genre":
                    map_df["map_genre"] = map_df["top_genre"]
                else:
                    map_df["map_genre"] = map_df["display_genre"]

                f1, f2, f3 = st.columns(3)
                all_genres = sorted(map_df["map_genre"].unique().tolist())
                selected_genre = f1.selectbox("Genre", ["All"] + all_genres, key="map_genre_filter")
                selected_streams = f2.selectbox("Streams", ["All", "Top 10", "Top 11–30", "31+"], key="map_streams")
                search = f3.text_input("Search country", placeholder="e.g. Brazil", key="map_search")

                filtered = map_df.copy()
                sorted_by_streams = filtered.sort_values("total_streams", ascending=False).reset_index(drop=True)
                sorted_by_streams["rank"] = sorted_by_streams.index + 1
                filtered = filtered.merge(sorted_by_streams[["region", "rank"]], on="region")

                if selected_genre != "All":
                    filtered = filtered[filtered["map_genre"] == selected_genre]
                if selected_streams == "Top 10":
                    filtered = filtered[filtered["rank"] <= 10]
                elif selected_streams == "Top 11–30":
                    filtered = filtered[(filtered["rank"] > 10) & (filtered["rank"] <= 30)]
                elif selected_streams == "31+":
                    filtered = filtered[filtered["rank"] > 30]
                if search:
                    filtered = filtered[filtered["region"].str.lower().str.contains(search.lower())]

                fig_map = px.choropleth(
                    filtered,
                    locations="region",
                    locationmode="country names",
                    color="map_genre",
                    hover_name="region",
                    hover_data={"total_streams": ":,", "unique_artists": ":,", "unique_songs": ":,",
                                "top3_genres": True, "map_genre": False, "region": False},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={"map_genre": "Genre", "total_streams": "Total Streams", "unique_artists": "Unique Artists",
                            "unique_songs": "Unique Songs", "top3_genres": "Top 3 Genres"},
                )

                fig_map.add_scattergeo(
                    locations=filtered["region"],
                    locationmode="country names",
                    marker=dict(
                        size=filtered["total_streams"] / filtered["total_streams"].max() * 40 + 5,
                        color=filtered["total_streams"],
                        colorscale="Viridis",
                        showscale=False,
                        opacity=0.6,
                        line=dict(width=0),
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="",
                )

                fig_map.update_geos(
                    showframe=False,
                    showcoastlines=True,
                    coastlinecolor="rgba(255,255,255,0.1)",
                    showland=True, landcolor="#1a1a2e",
                    showocean=True, oceancolor="#0d0d1a",
                    showlakes=False,
                    bgcolor="rgba(0,0,0,0)",
                    projection_type="natural earth",
                )

                fig_map.update_layout(
                    height=380,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="v", x=1.01, y=0.5, font=dict(size=10), title=dict(text="Genre")),
                )

                st.plotly_chart(fig_map, use_container_width=True)

                s1, s2, s3 = st.columns(3)
                s1.metric("Countries shown", len(filtered))
                s2.metric("Total Streams", f"{filtered['total_streams'].sum() / 1e9:.1f}B")
                s3.metric("Top Genre",filtered.groupby("map_genre")["total_streams"].sum().idxmax() if len(filtered) > 0 else "—")
        with r2:
            st.subheader("🎵 Top Genres by Streams")

            music_copy = music.copy()
            music_copy["date"] = pd.to_datetime(music_copy["date"])
            music_copy["year"] = music_copy["date"].dt.year

            g1, g2, g3 = st.columns(3)

            region_options = ["Global"] + sorted(
                music_copy[music_copy["region"] != "Global"]["region"].unique().tolist())
            selected_region = g1.selectbox("Region", region_options, key="genre_region")

            year_options = ["All"] + sorted(music_copy["year"].dropna().unique().astype(int).tolist())
            selected_year = g2.selectbox("Year", year_options, key="genre_year")

            artist_options = ["All"] + sorted(music_copy["artist"].dropna().unique().tolist())
            selected_artist = g3.selectbox("Artist", artist_options, key="genre_artist")

            # Apply filters
            genre_df = music_copy.copy()

            if selected_region != "Global":
                genre_df = genre_df[genre_df["region"] == selected_region]

            if selected_year != "All":
                genre_df = genre_df[genre_df["year"] == int(selected_year)]

            if selected_artist != "All":
                genre_df = genre_df[genre_df["artist"] == selected_artist]

            genre_df = (
                genre_df.groupby("main_genre", as_index=False)["streams"]
                .sum()
                .sort_values("streams", ascending=False)
                .head(10)
            )

            fig_genre = px.bar(
                genre_df,
                x="streams",
                y="main_genre",
                orientation="h",
                color="main_genre",
                color_discrete_sequence=px.colors.qualitative.Bold,
                labels={"main_genre": "Genre", "streams": "Total Streams"},
            )

            fig_genre.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                yaxis_title="",
                xaxis_title="Streams",
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                height=320,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(fig_genre, use_container_width=True)


        r3, r4 = st.columns([2, 3], border=True)
        with r3:
            st.subheader("📊 Market Overview")

            # toggle between views
            view = st.radio("View by", ["Genre", "Artist", "Song"], horizontal=True, key="pro_view")

            # filters
            p1, p2, p3 = st.columns(3)

            pro_copy = music.copy()
            pro_copy["date"] = pd.to_datetime(pro_copy["date"])
            pro_copy["year"] = pro_copy["date"].dt.year

            selected_year = p1.selectbox("Year",
                                         ["All"] + sorted(pro_copy["year"].dropna().unique().astype(int).tolist()),
                                         key="pro_year")
            selected_region = p2.selectbox("Region", ["Global"] + sorted(
                pro_copy[pro_copy["region"] != "Global"]["region"].unique().tolist()), key="pro_region")
            top_n = p3.selectbox("Top", [5, 10, 20], key="pro_topn")

            # apply filters
            filtered = pro_copy.copy()
            if selected_year != "All":
                filtered = filtered[filtered["year"] == int(selected_year)]
            if selected_region != "Global":
                filtered = filtered[filtered["region"] == selected_region]

            # group by selected view
            if view == "Genre":
                col = "main_genre"
            elif view == "Artist":
                col = "artist"
            else:
                col = "title"

            grouped = (
                filtered.groupby(col)["streams"]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )

            # pie chart
            fig = px.pie(
                grouped,
                names=col,
                values="streams",
                hole=0.3,
            )

            fig.update_layout(
                height=450,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
            )

            st.plotly_chart(fig, use_container_width=True)
        with r4:
            st.subheader("Ranking of songs")
            songs_ranking = music.copy()

            col1, col2, col3 = st.columns(3)

            songs_ranking["date"] = pd.to_datetime(songs_ranking["date"])

            song = col1.selectbox(
                "Song",
                ["Rockabye (feat. Sean Paul & Anne-Marie)"] +
                songs_ranking[
                    songs_ranking["title"] != "Rockabye (feat. Sean Paul & Anne-Marie)"
                ]["title"].unique().tolist()
            )
            with col2:
                d = st.date_input(
                    "Duration",
                    value=(songs_ranking["date"].min().date(), songs_ranking["date"].max().date()),
                    min_value=songs_ranking["date"].min().date(),
                    max_value=songs_ranking["date"].max().date(),
                    format="DD.MM.YYYY",
                )

            region = col3.selectbox(
                "Region",
                ["Global"] + songs_ranking[songs_ranking["region"] != "Global"]["region"].unique().tolist()
            )

            if len(d) == 2:
                songs_ranking = songs_ranking[
                    (songs_ranking["date"] >= pd.Timestamp(d[0])) &
                    (songs_ranking["date"] <= pd.Timestamp(d[1]))
                    ]

            songs_ranking = songs_ranking[songs_ranking["region"] == region]
            songs_ranking = songs_ranking[songs_ranking["title"] == song]

            songs_ranking = songs_ranking.sort_values(by='date', ascending=True)

            # st.line_chart(
            #     songs_ranking,
            #     x="date",
            #     y=["rank"],
            #     color=["#FF0000"],
            # )
            line = alt.Chart(songs_ranking).mark_line(color="#7777ff").encode(
                x="date:T",
                y=alt.Y("rank:Q", scale=alt.Scale(reverse=True), title="Rank")
            )

            # points = alt.Chart(songs_ranking).mark_point(color="#7777ff", size=80, filled=False).encode(
            points = alt.Chart(songs_ranking).mark_point(color="#7777ff", size=80, filled=True).encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("rank:Q", scale=alt.Scale(reverse=True))
            )

            st.altair_chart((line + points).interactive(), use_container_width=True)


    with col_right:
        st.subheader("Which artists dominate the market?")

        pro_copy2 = music.copy()
        pro_copy2["date"] = pd.to_datetime(pro_copy2["date"])
        pro_copy2["year"] = pro_copy2["date"].dt.year

        col1, col2 = st.columns(2)
        selected_year = col1.selectbox(
            "Year",
            ["All"] + sorted(pro_copy2["year"].dropna().unique().astype(int).tolist()),
            key="dom_year"
        )
        selected_genre = col2.selectbox(
            "Genre",
            ["All"] + sorted(pro_copy2["main_genre"].dropna().unique().tolist()),
            key="dom_genre"
        )

        filtered_pro = pro_copy2.copy()
        if selected_year != "All":
            filtered_pro = filtered_pro[filtered_pro["year"] == selected_year]
        if selected_genre != "All":
            filtered_pro = filtered_pro[filtered_pro["main_genre"] == selected_genre]

        top_artists = (
            filtered_pro.groupby("artist", as_index=False)["streams"]
            .sum()
            .sort_values("streams", ascending=False)
            .head(10)
        )

        fig = px.bar(
            top_artists,
            x="streams",
            y="artist",
            orientation="h",
            color_discrete_sequence=["#6b068a"]
        )
        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            yaxis_title="",
            xaxis_title="Streams",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            top_artists.rename(columns={
                "artist": "Artist",
                "streams": "Total Streams"
            }),
            use_container_width=True,
            hide_index=True
        )

    st.divider()
    st.subheader("🔬 Predictive Analysis")
    st.caption("Enter song parameters to generate hit prediction and stream forecast.")

    _g_opts = sorted([g for g in music["main_genre"].dropna().unique()
                      if isinstance(g, str) and not g.strip().isdigit()])
    _r_opts = sorted(music[music["region"] != "Global"]["region"].dropna().unique())

    with st.form("pro_pred_form"):
        _c1, _c2 = st.columns(2)
        _song   = _c1.text_input("Song Name",   placeholder="e.g. God's Plan")
        _artist = _c2.text_input("Artist Name", placeholder="e.g. Drake")

        _p1, _p2, _p3 = st.columns(3)
        _genre  = _p1.selectbox("Genre",        _g_opts, key="pp_genre")
        _region = _p2.selectbox("Region",       _r_opts, key="pp_region")
        _date   = _p3.date_input("Release Date", value=pd.Timestamp("today").date(), key="pp_date")

        _submitted = st.form_submit_button("🔍 Run Prediction", use_container_width=True, type="primary")

    if _submitted:
        with st.spinner("Running predictions..."):
            _assets = _load_prediction_assets()
            _dt     = pd.Timestamp(_date)
            _key    = _artist.strip() or "unknown"
            _iw     = int(_dt.dayofweek >= 5)
            _dss    = max(0, (_dt - _DATASET_START).days)
            _td, _tu, _tn, _ts = 0, 0, 1, 0
            _ac     = int(_assets["artist_count"].get(_key, 1))

            # --- Classification ---
            _gm   = _assets["clf_global"]
            _feat_clf = np.array([[
                50, _dt.year, _dt.month, _dt.day, _dt.dayofweek, _dt.quarter,
                _iw, _dss, _td, _tu, _tn, _ts, _ac,
                float(_assets["clf_artist"].get(_key,    _gm)),
                float(_assets["clf_genre"].get(_genre,   _gm)),
                float(_assets["clf_region"].get(_region, _gm)),
            ]], dtype=np.float32)

            _mean   = np.array(_assets["scaler_mean"],  dtype=np.float32)
            _scale  = np.array(_assets["scaler_scale"], dtype=np.float32)
            _feat_clf_s = ((_feat_clf - _mean) / _scale).astype(np.float32)

            _lbl, _probs = _load_clf_session().run(None, {"input": _feat_clf_s})
            _is_hit   = int(_lbl[0])
            _hit_prob = float(_probs[0][1])

            # --- Regression (no scaler) ---
            _gm_r = _assets["reg_global"]
            _feat_reg = np.array([[
                50, _dt.year, _dt.month, _dt.day, _dt.dayofweek, _dt.quarter,
                _iw, _dss, _td, _tu, _tn, _ts, _ac,
                float(_assets["reg_artist"].get(_key,    _gm_r)),
                float(_assets["reg_genre"].get(_genre,   _gm_r)),
                float(_assets["reg_region"].get(_region, _gm_r)),
            ]], dtype=np.float32)

            _res          = _load_reg_session().run(None, {"float_input": _feat_reg})
            _pred_streams = max(0, int(np.expm1(float(_res[0][0][0]))))

        _song_lbl   = f"*{_song}*"        if _song.strip()   else "this song"
        _artist_lbl = f" by **{_artist}**" if _artist.strip() else ""
        st.markdown(f"### Prediction for: {_song_lbl}{_artist_lbl}")

        _col_a, _col_b = st.columns(2)

        with _col_a:
            st.markdown("##### 🎯 Hit Prediction")
            if _is_hit == 1:
                st.success("✅ Predicted: **HIT**")
            else:
                st.warning("❌ Predicted: **Not a Hit**")
            st.metric("Hit Probability", f"{_hit_prob * 100:.1f}%")

        with _col_b:
            st.markdown("##### 📈 Stream Forecast")
            _disp = f"{_pred_streams / 1_000_000:.2f}M" if _pred_streams >= 1_000_000 else f"{_pred_streams / 1_000:.0f}K"
            st.metric("Predicted Streams", _disp)

