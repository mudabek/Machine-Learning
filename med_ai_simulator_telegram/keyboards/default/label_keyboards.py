from aiogram.types import (ReplyKeyboardMarkup, KeyboardButton)

breat_labels = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Benign')
        ],
        [
            KeyboardButton(text='Malignant')
        ],
        [
            KeyboardButton(text='Normal')
        ]
    ],
    resize_keyboard=True
)

brain_labels = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='Glioma tumor')
        ],
        [
            KeyboardButton(text='Meningioma tumor')
        ],
        [
            KeyboardButton(text='No Tumor')
        ],
        [
            KeyboardButton(text='Pituitary tumor')
        ]
    ],
    resize_keyboard=True
)

polyp_labels = ReplyKeyboardMarkup(
    keyboard=[
        [
            KeyboardButton(text='1 - Model  2 - Ground Truth')
        ],
        [
            KeyboardButton(text='1 - Ground Truth  2 - Model')
        ],
    ],
    resize_keyboard=True
)