from dashboard_pro import pro_dashboard
from dashboard_user import user_dashboard
import streamlit as st

USERS = {
    "pro": {"password": "pro4321", "role": "pro"},
    "user": {"password": "user123", "role": "user"}
}


st.set_page_config(layout="wide")

def creds_entered():
    username = st.session_state["user"].strip()
    password = st.session_state["passwd"].strip()

    if  username in USERS and USERS[username]["password"] == password:
        st.session_state["authenticated"] = True
        st.session_state["role"] = USERS[username]["role"]
    else:
        st.session_state["authenticated"] = False
        if not st.session_state["passwd"]:
            st.warning("Please enter password.")
        elif not st.session_state["user"]:
            st.warning("Please enter username.")
        else:
            st.error("Invalid Username/Password")

def authenticate_user():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if "authenticated" not in st.session_state:
            st.session_state["authenticated"] = False

        if not st.session_state["authenticated"]:
            st.text_input(label="Username :", value="", key="user")
            st.text_input(label="Password :", value="", key="passwd", type="password")
            if st.button("Log in"):
                creds_entered()
                if st.session_state["authenticated"]:
                    st.rerun()
            st.stop()
            # return False
        else:
            return True


# def authenticate_user():
#     col1, col2, col3 = st.columns([1, 2, 1])
#
#     with col2:
#         if "authenticated" not in st.session_state:
#             st.text_input(label="Username :", value="", key="user", on_change=creds_entered)
#             st.text_input(label="Password :", value="", key="passwd", type="password", on_change=creds_entered)
#             return False
#         else:
#             if st.session_state["authenticated"]:
#                 return True
#             else:
#                 st.text_input(label="Username :", value="", key="user", on_change=creds_entered)
#                 st.text_input(label="Password :", value="", key="passwd", type="password", on_change=creds_entered)
#                 return False

def logout():
    # for key in st.session_state.keys():
    #     del st.session_state[key]
    st.session_state["authenticated"] = False
    st.session_state["user"] = ""
    st.session_state["passwd"] = ""
    st.cache_data.clear()
    # st.rerun()



if authenticate_user():
    st.sidebar.button("Logout", on_click=logout)

    if st.session_state["role"] == "pro":
        pro_dashboard()
    else:
        user_dashboard()
