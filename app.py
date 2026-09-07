import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier

# -----------------------------
# Load trained model
# -----------------------------
model = CatBoostClassifier()
model.load_model("titanic_model.cbm")


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

st.title("Titanic Survival Prediction")
st.write("Predict whether a Titanic passenger would have survived.")


# -----------------------------
# Passenger details
# -----------------------------
st.header("Passenger Details")

name = st.text_input(
    "Passenger Name",
    value="Braund, Mr. Owen Harris"
)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
)

pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings / Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents / Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    max_value=600.0,
    value=30.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)

ticket = st.text_input(
    "Ticket Number",
    value="A/5 21171"
)

cabin = st.text_input(
    "Cabin",
    value=""
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival", type="primary"):

    # Title
    title = "Unknown"

    if "," in name and "." in name:
        title = name.split(",")[1].split(".")[0].strip()

    title_map = {
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs"
    }

    title = title_map.get(title, title)

    common_titles = ["Mr", "Miss", "Mrs", "Master"]

    if title not in common_titles:
        title = "Rare"

    # Family features
    family_size = sibsp + parch + 1
    is_alone = int(family_size == 1)

    # Surname
    surname = name.split(",")[0].strip() if "," in name else "Unknown"

    # Ticket prefix
    ticket_prefix = (
        "".join(
            c for c in ticket
            if not c.isdigit()
        )
        .replace(".", "")
        .replace("/", "")
        .replace(" ", "")
        .strip()
    )

    if ticket_prefix == "":
        ticket_prefix = "NONE"

    # Ticket number
    ticket_number = np.nan

    parts = ticket.strip().split()

    if parts:
        try:
            ticket_number = float(parts[-1])
        except ValueError:
            ticket_number = np.nan

    # Cabin
    cabin_deck = cabin[0] if cabin else "Unknown"
    has_cabin = int(bool(cabin))

    # Missing-value indicators
    age_missing = 0
    fare_missing = 0

    # Other engineered features
    is_child = int(age < 16)

    is_mother = int(
        sex == "female"
        and parch > 0
        and age > 18
        and title == "Mrs"
    )

    # For a single prediction, these are estimated as 1
    ticket_group_size = 1
    surname_group_size = 1
    fare_per_person = fare

    sex_pclass = f"{sex}_{pclass}"
    title_pclass = f"{title}_{pclass}"

    # -----------------------------
    # Create model input
    # -----------------------------
    passenger = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "Title": title,
        "FamilySize": family_size,
        "IsAlone": is_alone,
        "Surname": surname,
        "TicketPrefix": ticket_prefix,
        "TicketNumber": ticket_number,
        "CabinDeck": cabin_deck,
        "HasCabin": has_cabin,
        "AgeMissing": age_missing,
        "FareMissing": fare_missing,
        "IsChild": is_child,
        "IsMother": is_mother,
        "TicketGroupSize": ticket_group_size,
        "SurnameGroupSize": surname_group_size,
        "FarePerPerson": fare_per_person,
        "Sex_Pclass": sex_pclass,
        "Title_Pclass": title_pclass
    }])

    # Fill missing numeric values
    passenger["TicketNumber"] = passenger["TicketNumber"].fillna(0)

    # Make categorical columns strings
    categorical_cols = [
        "Sex",
        "Embarked",
        "Title",
        "Surname",
        "TicketPrefix",
        "CabinDeck",
        "Sex_Pclass",
        "Title_Pclass"
    ]

    for col in categorical_cols:
        passenger[col] = passenger[col].astype(str)

    # -----------------------------
    # Predict
    # -----------------------------
    probability = model.predict_proba(passenger)[0, 1]

    # Same threshold used in Kaggle
    prediction = int(probability >= 0.47)

    st.divider()

    if prediction == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: DID NOT SURVIVE")

    st.metric(
        "Survival Probability",
        f"{probability * 100:.1f}%"
    )
