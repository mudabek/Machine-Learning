from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)


share_phone = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Share Phone Number ☎️', request_contact=True)
        ]
    ],
    resize_keyboard=True
)