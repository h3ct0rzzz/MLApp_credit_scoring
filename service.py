import catboost as cb
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np


def corr_res(val):
    if (val >= 0.3) and (val < 0.5):
        return "Умеренная"
    elif (val >= 0.5) and (val < 0.7):
        return "Заметная"
    elif (val >= 0.7) and (val < 0.9):
        return "Высокая"
    elif (val >= 0.9):
        return "Весьма высокая"
    else:
        return "Слабая"


def age_vectorize(val):
    age = np.zeros((3, ))
    if val == "от 35 до 50":
        age[0] = 1
    if val == "от 50 до 65":
        age[1] = 1
    if val == "от 65":
        age[2] = 1
    return age

def loans_vectorize(val):
    arr = np.zeros((4, ))
    if val == "от 2 до 4":
        arr[0] = 1
    if val == "от 4 до 6":
        arr[1] = 1
    if val == "от 6 до 8":
        arr[2] = 1
    if val == "Свыше 8":
        arr[3] = 1
    return arr


data = pd.read_csv("data/credit_scoring.csv")

with open("cols_map.json", "r", encoding="utf-8") as json_file:
    col_map = json.load(json_file)
col_df = pd.DataFrame(columns=["col", "description"])
col_df["col"] = col_map.keys()
col_df["description"] = col_map.values()

st.set_page_config(
    page_title="Прогноз платежеспособности клиента банка",
    layout="wide"
)
st.title("Прогноз платежеспособности клиента банка")
placeholder = st.empty()

with placeholder.container():
    st.write("## Краткая справка по архивным данным.")

    st.write("Исходные данные.")
    st.table(data.head(5))

    raws, columns, memory_usage = st.columns(3)
    raws.metric(
        label="Кол-во строк",
        value=data.shape[0],
        delta=-22081,
        delta_color="inverse"
    )
    columns.metric(
        label="Кол-во столбцов",
        value=data.shape[1],
        delta="+4",
        delta_color="inverse"
    )
    memory_usage.metric(
        label="Потребление памяти",
        value=f"{round(data.memory_usage().sum() / 1024 ** 2, 1)}Мб",
        delta="-7.9Мб",
        delta_color="inverse"
    )

    selector = st.selectbox("Выберите признак", data.columns)
    st.table(col_df[col_df["col"] == selector])

    st.write("Зависимость переменных")
    val, res = st.columns(2)
    val.metric(
        label="Значение",
        value=(
            round(data[[selector, "SeriousDlqin2yrs"]].corr().values[0, 1], 5)
            if selector not in ["RealEstateLoansOrLines", "GroupAge"] else "Категория"
        )
    )
    res.metric(
        label="Описание",
        value=(
            corr_res(round(data[[selector, "SeriousDlqin2yrs"]].corr().values[0, 1]))
            if selector not in ["RealEstateLoansOrLines", "GroupAge"] else "Нельзя посчитать"
        )
    )

    
    st.write("## Введите параметры:")

    radio1, radio2 = st.columns(2)
    with radio1:
        radio1 = st.radio(
            "Возраст клиента:",
            ("от 21 до 35", "от 35 до 50", "от 50 до 65", "от 65")
        )

    with radio2:
        radio2 = st.radio(
            "Количество кредитов:",
            ("до 2", "от 2 до 4", "от 4 до 6", "от 6 до 8", "свыше 8")
        )

    st.write("Была ли просрочка два года назад?")
    radio3, radio4, radio5 = st.columns(3)
    with radio3:
        radio3 = st.radio(
            "От 30 до 59 дней?",
            ("Да", "Нет")
        )

    with radio4:
        radio4 = st.radio(
            "ОТ 60 до 89 дней?",
            ("Да", "Нет")
        )

    with radio5:
        radio5 = st.radio(
            "Свыше 90 дней?",
            ("Да", "Нет")
        )

    num1 = st.number_input("Общий баланс средств:")
    num2 = st.number_input("Ежемесячный доход:")
    num23 = st.number_input("Ежемесячные расходы:")
    if num2 != 0:
        num3 = num23 / num2
    else:
        num3 = num23 / 1
    num4 = st.number_input("Количество открытых кредитных продуктов (кредитов, кредитных карт и т.д):")
    num5 = st.number_input("Количество иждивенцев на попечении (супруги, дети и др):")


    full_arr = np.asarray([
        num1,
        1 if radio3 == "Да" else 0,
        num3,
        num2,
        num4,
        1 if radio5 == "Да" else 0,
        1 if radio4 == "Да" else 0,
        num5
    ])
    full_arr = np.hstack([full_arr, loans_vectorize(radio2), age_vectorize(radio1)])

    clf = cb.CatBoostClassifier()
    clf.load_model("weights/clf")
    prob = clf.predict_proba(full_arr)[1]
    if (prob >= 0.5) and (prob < 0.75):
        st.warning(f"С веротяностью {round(prob * 100, 2)} у клиента будет  просрочка в 90 или 90+ дней", icon="⚠️")
    elif prob >= 0.75:
        st.error(f"С веротяностью {round(prob * 100, 2)} у клиента будет  просрочка в 90 или 90+ дней", icon="🚨")
    else:
        st.info(f"С веротяностью {round(prob * 100, 2)} у клиента будет  просрочка в 90 или 90+ дней", icon="ℹ️")
