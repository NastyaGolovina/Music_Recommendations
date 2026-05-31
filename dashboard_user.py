import streamlit as st
import pandas as pd
import numpy as np
from data_loader import load_data
import plotly.express as px
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


@st.cache_resource(show_spinner=False)
def _load_prediction_assets():
    df = pd.read_csv("music_classification.csv", dtype={
        "rank":             "int16",
        "streams":          "float32",
        "is_hit":           "int8",
        "year":             "int16",
        "month":            "int8",
        "day":              "int8",
        "weekday":          "int8",
        "quarter":          "int8",
        "is_weekend":       "int8",
        "days_since_start": "int32",
        "artist_count":     "int32",
    })

    train = df[df["year"] < 2021].copy()

    global_mean = float(train["is_hit"].mean())
    artist_mean = train.groupby("artist")["is_hit"].mean()
    genre_mean  = train.groupby("main_genre")["is_hit"].mean()
    region_mean = train.groupby("region")["is_hit"].mean()

    train["artist_te"] = train["artist"].map(artist_mean).fillna(global_mean)
    train["genre_te"]  = train["main_genre"].map(genre_mean).fillna(global_mean)
    train["region_te"] = train["region"].map(region_mean).fillna(global_mean)

    drop_cols = ["streams", "is_hit", "artist", "main_genre", "region"]
    X_train   = train.drop(columns=drop_cols)

    scaler = StandardScaler()
    scaler.fit(X_train)

    return {
        "global_mean":  global_mean,
        "artist_te":    artist_mean.to_dict(),
        "genre_te":     genre_mean.to_dict(),
        "region_te":    region_mean.to_dict(),
        "artist_count": df.groupby("artist")["artist_count"].first().to_dict(),
        "scaler_mean":  scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }


def user_dashboard():
    st.title("User Dashboard")

    with st.spinner("Loading data..."):
            music = load_data()

    m = get_metrics(music)

    c1, c2, c3, c4 = st.columns(4)


    c1.metric("Songs", m["songs"], border=True)
    c2.metric("Artists", m["artists"], border=True)
    c3.metric("Regions", m["regions"], border=True)
    c4.metric("Streams", m['streams'] , border=True)


    col_left, col_right = st.columns([1, 2] , border=True)

    with col_left:

        st.subheader("Top of songs")
        sorted_songs_titles = music.copy()
        # sorted_songs_titles = music[['title', 'artist', 'streams', 'date', 'region', 'main_genre']].copy()  # use this if low RAM

        top_song_col_left, top_song_col_right = st.columns([1, 2])

        top_song = top_song_col_left.selectbox("Top", [5,10,20,100])

        col1, col2, col3 = st.columns(3)

        sorted_songs_titles["date"] = pd.to_datetime(sorted_songs_titles["date"])
        with col1:
            d = st.date_input(
                "Duration",
                value=(sorted_songs_titles["date"].min().date(), sorted_songs_titles["date"].max().date()),
                min_value=sorted_songs_titles["date"].min().date(),
                max_value=sorted_songs_titles["date"].max().date(),
                format="DD.MM.YYYY",
            )

        region = col2.selectbox(
            "Region",
            ["Global"] + sorted_songs_titles[sorted_songs_titles["region"] != "Global"]["region"].unique().tolist()
        )

        genre = col3.selectbox("Genre", ["All"] + sorted_songs_titles["main_genre"].unique().tolist())


        if len(d) == 2:
            sorted_songs_titles = sorted_songs_titles[
                (sorted_songs_titles["date"] >= pd.Timestamp(d[0])) &
                (sorted_songs_titles["date"] <= pd.Timestamp(d[1]))
                ]

        sorted_songs_titles = sorted_songs_titles[sorted_songs_titles["region"] == region]


        if genre != "All":
            sorted_songs_titles = sorted_songs_titles[sorted_songs_titles["main_genre"] == genre]



        sorted_songs_titles = (
            sorted_songs_titles.groupby("title", as_index=False)["streams"]
            .sum()
            .sort_values(by="streams", ascending=False)
        )

        fig = px.bar(
            sorted_songs_titles.head(top_song),
            x="streams",
            y="title",
            orientation="h",
            color_discrete_sequence=["#6b068a"]
            # ,title="Top Songs by Streams"
            # ,hover_data = {
            #     "title": False,
            #     "streams": True,
            #     "artist": True,
            # }
        )

        fig.update_layout(
            yaxis=dict(categoryorder="total ascending"),
            yaxis_title="",
            xaxis_title="Streams",
            xaxis=dict(tickangle=-45)
        )

        st.plotly_chart(fig, use_container_width=True)


    with col_right:

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
                showland=True, landcolor="#e8e8e8",  # ← light grey land
                showocean=True, oceancolor="#a8c8e8",  # ← light blue ocean
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
            s3.metric("Top Genre",
                      filtered.groupby("map_genre")["total_streams"].sum().idxmax() if len(filtered) > 0 else "—")
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


        r3, r4 = st.columns([5, 6], border=True)
        with r3:
            st.subheader("🎧 Top Artists by Streams")

            f1, f2, f3 = st.columns(3)

            top_n = f1.selectbox("Top", [5, 10, 20, 100], key="top_artists")

            region_selected = f2.selectbox(
                "Region",
                ["Global"] + sorted(
                    music.loc[music["region"] != "Global", "region"].unique().tolist()
                ),
                key="region_artists"
            )

            _dates = pd.to_datetime(music["date"])
            date_min = _dates.min().date()
            date_max = _dates.max().date()

            date_range = f3.date_input(
                "Time",
                value=(date_min, date_max),
                min_value=date_min,
                max_value=date_max,
                format="DD.MM.YYYY",
                key="date_artists"
            )

            mask = music["region"] == region_selected
            if len(date_range) == 2:
                mask &= (_dates >= pd.Timestamp(date_range[0])) & \
                        (_dates <= pd.Timestamp(date_range[1]))

            artists_df = (
                music.loc[mask, ["artist", "streams"]]
                .groupby("artist", as_index=False)["streams"]
                .sum()
                .sort_values(by="streams", ascending=False)
            )

            fig_artists = px.bar(
                artists_df.head(top_n),
                x="streams",
                y="artist",
                orientation="h",
                color_discrete_sequence=["#2ec4b6"],
                hover_data={"artist": False, "streams": ":,"}
            )

            fig_artists.update_layout(
                yaxis=dict(categoryorder="total ascending"),
                yaxis_title="",
                xaxis_title="Streams",
                height=350,
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )

            st.plotly_chart(fig_artists, use_container_width=True)
        with r4:
            st.subheader("Mean Statistics")

            df = music.copy()
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year

            col_f1, col_f2 = st.columns(2)
            selected_region = col_f1.selectbox(
                "Region",
                ["All"] + sorted(df["region"].dropna().unique().tolist()),
                key="mean_region"
            )
            selected_year = col_f2.selectbox(
                "Year",
                ["All"] + sorted(df["year"].dropna().unique().astype(int).tolist()),
                key="mean_year"
            )

            filtered = df.copy()
            if selected_region != "All":
                filtered = filtered[filtered["region"] == selected_region]
            if selected_year != "All":
                filtered = filtered[filtered["year"] == selected_year]

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Streams",
                          f"{filtered['streams'].mean():,.0f}")
                st.metric("Mean Rank",
                          f"{filtered['rank'].mean():.1f}")
            with col2:
                st.metric("Mean Streams/Artist",
                          f"{filtered.groupby('artist')['streams'].mean().mean():,.0f}")
                st.metric("Mean Streams/Region",
                          f"{filtered.groupby('region')['streams'].mean().mean():,.0f}")

            mean_region = (
                filtered.groupby('region', as_index=False)['streams']
                .mean()
                .sort_values(by='streams', ascending=False)
                .head(8)
            )

            fig = px.bar(
                mean_region,
                x='streams',
                y='region',
                orientation='h',
                color_discrete_sequence=['#6b068a']
            )
            fig.update_layout(
                yaxis=dict(categoryorder='total ascending'),
                yaxis_title='',
                xaxis_title='Mean Streams'
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("🎯 Will This Song Be a Hit?")
    st.caption("Enter song details to predict whether it will become a hit.")

    _genre_opts  = sorted([g for g in music["main_genre"].dropna().unique()
                           if isinstance(g, str) and not g.strip().isdigit()])
    _region_opts = sorted(music[music["region"] != "Global"]["region"].dropna().unique())

    with st.form("user_hit_form"):
        _c1, _c2 = st.columns(2)
        _song   = _c1.text_input("Song Name",   placeholder="e.g. Anti-Hero")
        _artist = _c2.text_input("Artist Name", placeholder="e.g. Taylor Swift")

        _p1, _p2, _p3 = st.columns(3)
        _genre  = _p1.selectbox("Genre",        _genre_opts,  key="u_pred_genre")
        _region = _p2.selectbox("Region",       _region_opts, key="u_pred_region")
        _date   = _p3.date_input("Release Date", value=pd.Timestamp("today").date(), key="u_pred_date")

        _submitted = st.form_submit_button("🎵 Predict", use_container_width=True, type="primary")

    if _submitted:
        with st.spinner("Predicting..."):
            _assets = _load_prediction_assets()
            _dt     = pd.Timestamp(_date)
            _key    = _artist.strip() or "unknown"
            _gm     = _assets["global_mean"]

            _feat = np.array([[
                50,
                _dt.year, _dt.month, _dt.day, _dt.dayofweek, _dt.quarter,
                int(_dt.dayofweek >= 5),
                max(0, (_dt - _DATASET_START).days),
                0, 0, 1, 0,
                int(_assets["artist_count"].get(_key, 1)),
                float(_assets["artist_te"].get(_key,    _gm)),
                float(_assets["genre_te"].get(_genre,   _gm)),
                float(_assets["region_te"].get(_region, _gm)),
            ]], dtype=np.float32)

            _mean   = np.array(_assets["scaler_mean"],  dtype=np.float32)
            _scale  = np.array(_assets["scaler_scale"], dtype=np.float32)
            _feat_s = ((_feat - _mean) / _scale).astype(np.float32)

            _lbl, _probs = _load_clf_session().run(None, {"input": _feat_s})
            _is_hit      = int(_lbl[0])
            _hit_prob    = float(_probs[0][1])

        _song_lbl   = f"*{_song}*"        if _song.strip()   else "this song"
        _artist_lbl = f" by **{_artist}**" if _artist.strip() else ""
        st.markdown(f"### Result for: {_song_lbl}{_artist_lbl}")

        if _is_hit == 1:
            st.success("🎉 This song is predicted to be a **HIT**!")
        else:
            st.warning("📉 This song is predicted **not** to be a hit.")

        _m1, _m2 = st.columns(2)
        _m1.metric("Hit Probability", f"{_hit_prob * 100:.1f}%")
        _m2.metric("Prediction", "HIT ✅" if _is_hit else "Not a Hit ❌")