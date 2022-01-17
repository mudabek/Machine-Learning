from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)


modes = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Brain tumor')
        ],
        [
            KeyboardButton(text='Breast cancer')
        ],
    ],
    resize_keyboard=True
)