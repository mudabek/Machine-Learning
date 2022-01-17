from aiogram import types
from loader import dp
from aiogram.dispatcher.filters import Command, Text
from ai_model.get_image import get_image_path
from keyboards.default import breat_labels, brain_labels, modes
from aiogram.dispatcher import FSMContext
from states.detecton import Detection
from skimage import io
import numpy as np
from ai_model import get_models, transformations, get_gradcam_image
import cv2
import matplotlib.pyplot as plt

@dp.message_handler(Command(commands=['qa']), state=None)
async def begin_test(message: types.Message, state: FSMContext):
    await message.answer("QA")
