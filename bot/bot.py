import re
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from config_reader import config
import tabulate

import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram import F
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

TOKEN = config.bot_token.get_secret_value()

# Инициализация бота и диспетчера
bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
storage = MemoryStorage()
dp = Dispatcher(bot=bot, storage=storage)

path_to_database = "data/database.csv"
df_database = pd.read_csv(path_to_database, dtype=str)

model = joblib.load("model.pkl")

def predict_and_save(results, message):
    global df_database
    # создаем датафрейм подходящего для модели формата
    #features = df_database.drop(["date","bg+1:00" ,"real_bg+1:00"], axis=1)
    features = df_database.copy()
    features= features.iloc[0:0] # Очистка
    len_database = len(df_database)

    features.loc[len_database,"user_id"] = str(message.from_user.id) # type: ignore

    features.loc[len_database,"date"] = datetime.now().strftime("%d.%m.%Y")
    features.loc[len_database,"hour"] = str(datetime.now().hour)
    features.loc[len_database,"minute"] = str(datetime.now().minute)

    for feature in results.keys():
        features.loc[len_database,f"{feature}-0:55":f"{feature}-0:00"] = results[feature]

    predicted_bg = model.predict(features.drop(["date","bg+1:00" ,"real_bg+1:00"], axis=1))
    features.loc[len_database,"bg+1:00"] = str(round(predicted_bg[0], 2))

    df_database = pd.concat([df_database, features])
    df_database.to_csv("data/database.csv", index=False)

    return predicted_bg

def delete_notes_from_database(message):
    global df_database
    df_database = df_database.drop(df_database[df_database["user_id"] == message.from_user.id].index)
    df_database.to_csv("data/database.csv", index=False)

# FSM States
class Features_Input(StatesGroup):
    choosing_feature = State()
    inputing_blood_glucose = State()
    inputing_insulin = State()
    inputing_carbohydrate = State()
    inputing_heart_rate = State()
    inputing_steps = State()
    inputing_calories = State()
    inputing_activity = State()

# Клавиатуры

def rules_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Правила")
    
    return builder.as_markup(resize_keyboard=True)

def main_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Добавить запись")
    builder.button(text="Просмотреть историю")
    builder.button(text="Правила")
    builder.adjust(2, 1)

    return builder.as_markup(resize_keyboard=True)#, input_field_placeholder="Воспользуйтесь меню:")

def history_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Удалить все записи")
    builder.button(text="В главное меню")
    builder.adjust(2)

    return builder.as_markup(resize_keyboard=True)

def features_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="Глюкоза")
    builder.button(text="Инсулин")
    builder.button(text="Углеводы")
    builder.button(text="Пульс")
    builder.button(text="Шаги")
    builder.button(text="Калории")
    builder.button(text="Активность")
    builder.button(text="В главное меню")
    builder.button(text="Отправить")
    builder.adjust(3,3,1,2)

    return builder.as_markup(resize_keyboard=True)

def feature_inputing_keyboard():
    builder = ReplyKeyboardBuilder()
    builder.button(text="К выбору признака")
    builder.button(text="В главное меню")
    builder.adjust(2)

    return builder.as_markup(resize_keyboard=True)

# Хендлеры

@dp.message(F.text, Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! \nЯ умею предсказывать уровень глюкозы.\nПеред началом ознакомьтесь с <b>правилами</b>", reply_markup=rules_keyboard())

@dp.message(F.text.lower() == "правила")
async def rules(message: types.Message, state: FSMContext):
    await message.answer("Правила:\n 1. -------\n 2. -------\n 3. -------")
    await back_to_main_menu(message, state)

@dp.message(F.text.lower() == "в главное меню")
async def back_to_main_menu(message: types.Message, state: FSMContext):
    await message.answer("Выберите опцию:", reply_markup=main_keyboard())
    await state.clear()

@dp.message(F.text.lower() == "просмотреть историю")
async def view_history(message: types.Message):
    history = df_database.loc[df_database["user_id"] == str(message.from_user.id), ["date","hour", "minute","bg+1:00","real_bg+1:00"]] # type: ignore
    history_data= [["Дата","Время","Пред","Реал"]]
    for note in history.values:
        time = datetime(2025, 5, 21, hour=int(note[1]), minute=int(note[2])).strftime("%H:%M")
        history_data.append([note[0], f"{time}", note[3]])#, note[4]])
    tabulate.MIN_PADDING = 0
    history_table = tabulate.tabulate(history_data, headers="firstrow", tablefmt="plain")

    await message.answer(f"История:\n<code>{history_table}</code>", reply_markup=history_keyboard())

@dp.message(F.text.lower() == "удалить все записи")
async def delete_history(message: types.Message, state: FSMContext):
    delete_notes_from_database(message)
    await message.answer("История удалена.")
    await back_to_main_menu(message, state)

@dp.message(F.text.lower().in_(["добавить запись","к выбору признака"]))
async def make_note(message: types.Message, state: FSMContext):
    await message.answer("Для большей точности предсказаний рекомендуется заполнить все признаки" \
                            "\nВыберите признак для ввода:", reply_markup=features_keyboard())
    await state.set_state(Features_Input.choosing_feature)

@dp.message(F.text.lower() == "отправить")
async def send_features(message: types.Message, state: FSMContext):
    global df_database
    results = await state.get_data()
    if len(results) !=0:
        predicted_bg = predict_and_save(results, message)
        
        time = (datetime.now() + timedelta(hours=1)).strftime("%H:%M")
        await message.answer(f"Предсказанное значение глюкозы через час (в {time}): {predicted_bg}")
        await back_to_main_menu(message, state)
    else:
        await message.answer("Пожалуйста, введите данные признаков.")

# Ввод признаков

def check_and_transform_feature_input(string_feature, required_type: str): # Функция проверки введенного признака
    NaN_list = ["n", "н", "nan", "нан"]
    activity_list= ["Walk",
                    "Run",
                    "Dancing",
                    "Bike",
                    "Outdoor Bike",
                    "Swim",
                    "Aerobic Workout",
                    "Yoga",
                    "Zumba",
                    "Tennis",
                    "Weights",
                    "Strength training",
                    "Workout",
                    "HIIT",
                    "Hike",
                    "Indoor climbing",
                    "Stairclimber",
                    "Spinning",
                    "Sport"]
    
    list_feature = re.split(r"\s+", string_feature) # Разделение по пробелам
    if len(list_feature) != 12: # Проверка на количество временных промежутков
        return(False)
    try:
        for i in range(len(list_feature)):
            if list_feature[i].lower() in NaN_list:          # Обработка для недостающих данных
                list_feature[i] = np.nan
            else:
                match required_type:                                  # Преобразование в требуемый тип
                    case "float":
                        list_feature[i] = float(list_feature[i])
                    case "integer":
                        list_feature[i] = int(list_feature[i])
                    case "string":
                        list_feature[i] = activity_list[int(list_feature[i])-1]
        return(list_feature)
    
    except Exception:
        return(False)

# Глюкоза

@dp.message(F.text.lower() == "глюкоза")
async def input_blood_glucose(message: Message, state: FSMContext):
    await message.answer("Введите значения глюкозы (ммоль/л) за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_blood_glucose)

@dp.message(Features_Input.inputing_blood_glucose)
async def process_blood_glucose(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "float")
    if list_feature:
        await state.update_data(bg= list_feature)
        await message.answer("Значение глюкозы сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону:" \
                                "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")

# Инсулин

@dp.message(F.text.lower() == "инсулин")
async def input_insulin(message: Message, state: FSMContext):
    await message.answer("Введите значения инсулина за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_insulin)

@dp.message(Features_Input.inputing_insulin)
async def process_insulin(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "float")
    if list_feature:
        await state.update_data(insulin= list_feature)
        await message.answer("Значение инсулина сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону:" \
                                "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")

# Углеводы

@dp.message(F.text.lower() == "углеводы")
async def input_carbohydrate(message: Message, state: FSMContext):
    await message.answer("Введите значения углеводов за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_carbohydrate)

@dp.message(Features_Input.inputing_carbohydrate)
async def process_carbohydrate(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "float")
    if list_feature:
        await state.update_data(carbs= list_feature)
        await message.answer("Значение углеводов сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону:" \
                                "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")

# Пульс

@dp.message(F.text.lower() == "пульс")
async def input_heart_rate(message: Message, state: FSMContext):
    await message.answer("Введите значения пульса за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n20 3 n 56 0 5 0 20 4 наН 1 0")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_heart_rate)

@dp.message(Features_Input.inputing_heart_rate)
async def process_heart_rate(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "float")
    if list_feature:
        await state.update_data(hr= list_feature)
        await message.answer("Значение пульса сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону:" \
                                "\n20 3 n 56 0 5 0 20 4 наН 1 0")

# Шаги

@dp.message(F.text.lower() == "шаги")
async def input_steps(message: Message, state: FSMContext):
    await message.answer("Введите значения шагов за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n20 3 n 56 0 5 0 20 4 наН 1 0")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_steps)

@dp.message(Features_Input.inputing_steps)
async def process_steps(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "integer")
    if list_feature:
        await state.update_data(steps= list_feature)
        await message.answer("Значение шагов сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону:" \
                                "\n20 3 n 56 0 5 0 20 4 наН 1 0")

# Калории

@dp.message(F.text.lower() == "калории")
async def input_calories(message: Message, state: FSMContext):
    await message.answer("Введите значения калорий за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_calories)

@dp.message(Features_Input.inputing_calories)
async def process_calories(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "float")
    if list_feature:
        await state.update_data(cals= list_feature)
        await message.answer("Значение калорий сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону:" \
                                "\n20.4 3.0 n 56.11 0.54 .3 0.0 20.4 4 наН .3 0")

# Активность

@dp.message(F.text.lower() == "активность")
async def input_activity(message: Message, state: FSMContext):
    await message.answer("Введите значения активности за последний час с промежутком 5 минут (12 значений), согласно шаблону:" \
                            "\n7 16 n 4 2 1 1 8 4 наН 15 9")
    await message.answer("Поддерживаемые виды активности: " \
                            "\n1 - Ходьба" \
                            "\n2 - Бег" \
                            "\n3 - Танцы" \
                            "\n4 - Велосипед" \
                            "\n5 - Велосипед на улице" \
                            "\n6 - Плавание" \
                            "\n7 - Аэробика" \
                            "\n8 - Йога" \
                            "\n9 - Зумба" \
                            "\n10 - Теннис" \
                            "\n11 - Тяжелая атлетика" \
                            "\n12 - Силовые тренировки" \
                            "\n13 - Воркаут" \
                            "\n14 - Высокоинтенсивная тренировка" \
                            "\n15 - Поход" \
                            "\n16 - Альпинизм" \
                            "\n17 - Лестничный тренажер" \
                            "\n18 - Силовой велотренажер" \
                            "\n19 - Спорт")
    await message.answer("Для неизвестных данных используйте: \"n\", \"н\", \"nan\", \"нан\".", reply_markup=feature_inputing_keyboard())
    await state.set_state(Features_Input.inputing_activity)

@dp.message(Features_Input.inputing_activity)
async def process_activity(message: Message, state: FSMContext):
    list_feature=check_and_transform_feature_input(message.text, "string")
    if list_feature:
        await state.update_data(activity= list_feature)
        await message.answer("Значение активности сохранено.", reply_markup=features_keyboard())
        await state.set_state(Features_Input.choosing_feature)
    else:
        await message.answer("Пожалуйста, введите значения согласно шаблону и списку активности:" \
                                "\n7 16 n 4 2 1 1 8 4 наН 15 9")

@dp.message()
async def unknown_message(message: types.Message):
    await message.answer("Неизвестная команда")

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
