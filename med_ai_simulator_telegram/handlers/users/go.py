from aiogram import types
from loader import dp
from aiogram.dispatcher.filters import Command, Text
from ai_model.get_image import get_image_path
from keyboards.default import breat_labels, brain_labels, modes
from aiogram.dispatcher import FSMContext
from states.classification import Classification
from skimage import io
import numpy as np
from ai_model import get_models, transformations, get_gradcam_image
import cv2
import matplotlib.pyplot as plt

RESULT_DATA_DIR = 'results/'

OUTPUTS = {
    0: {
        0: 'Glioma tumor',
        1: 'Meningioma tumor',
        2: 'No Tumor',
        3: 'Pituitary tumor'
    },
    1: {
        0: 'Normal',
        1: 'Benign',
        2: 'Malignant'},

}

OUTPUTS_TO_INT = {
    0: {
        'glioma tumor': 0,
        'meningioma tumor': 1,
        'no tumor': 2,
        'pituitary tumor': 3
    },
    1: {
        'normal': 0,
        'benign': 1,
        'malignant': 2},

}

MODES_DICT = {
    "Brain tumor": 0,
    "Breast cancer": 1
}

labels = {
    0: brain_labels,
    1: breat_labels
}


@dp.message_handler(Command(commands=['classification']), state=None)
async def begin_test(message: types.Message, state: FSMContext):
    await message.answer("Choose training option", reply_markup=modes)
    await Classification.first()


@dp.message_handler(state=Classification.t1, content_types='text')
async def test_1(message: types.Message, state: FSMContext):
    mode = MODES_DICT[message.text]
    print("MODE", mode)
    path, label = get_image_path(mode)
    print("Label", mode)
    img = open(path, 'br')
    await state.update_data({
        'mode': mode,
        't1_image_path': path,
        'target_1': label,
        'model_score': 0,
        'user_score_1': 0,
        'user_score_2': 0
    })
    await message.answer_photo(img, reply_markup=labels[mode])
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t11, content_types='text')
async def test_1(message: types.Message, state: FSMContext):
    await state.update_data({
        't1_answer_1': message.text
    })
    data = await state.get_data()
    path = data['t1_image_path']
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    model, gradcamplusplus = get_models(data['mode'])
    output_class = model(transformed_image).argmax().item()
    print("output_class", output_class)
    if output_class == OUTPUTS_TO_INT[data['mode']][data['target_1']]:
        async with state.proxy() as elem:
            elem['model_score'] += 1
    if data['t1_answer_1'].lower() == data['target_1']:
        async with state.proxy() as elem:
            elem['user_score_1'] += 1
    # print(OUTPUTS[output_class], data['target_1'], data['t1_answer_1'].lower())
    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, gradcamed_image)
    # plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels[data['mode']], caption=f"Model's answer: <b>{OUTPUTS[data['mode']][output_class]}</b>",
                               parse_mode='HTML')
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t12, content_types='text')
async def test_12(message: types.Message, state: FSMContext):
    await message.answer("Answer has been recorded")
    answer_2 = message.text
    await state.update_data({
        't1_answer_2': answer_2
    })
    data = await state.get_data()
    if data['t1_answer_2'].lower() == data['target_1']:
        async with state.proxy() as elem:
            elem['user_score_2'] += 1
    path, label = get_image_path(data['mode'])
    img = open(path, 'br')
    await state.update_data({
        't2_image_path': path,
        'target_2': label
    })
    await message.answer_photo(img, reply_markup=labels[data['mode']])
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t2, content_types='text')
async def test_21(message: types.Message, state: FSMContext):
    await state.update_data({
        't2_answer_1': message.text
    })
    data = await state.get_data()
    path = data['t2_image_path']
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    model, gradcamplusplus = get_models(data['mode'])
    output_class = model(transformed_image).argmax().item()
    if output_class == OUTPUTS_TO_INT[data['mode']][data['target_2']]:
        async with state.proxy() as elem:
            elem['model_score'] += 1
    if data['t2_answer_1'].lower() == data['target_2']:
        async with state.proxy() as elem:
            elem['user_score_1'] += 1
    # print(OUTPUTS[output_class], data['target_2'], data['t2_answer_1'].lower())
    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, gradcamed_image)
    # plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels[data['mode']], caption=f"Model's answer: <b>{OUTPUTS[data['mode']][output_class]}</b>",
                               parse_mode='HTML')
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t21, content_types='text')
async def test_22(message: types.Message, state: FSMContext):
    await message.answer("Answer has been recorded")
    answer_2 = message.text
    await state.update_data({
        't2_answer_2': answer_2
    })
    data = await state.get_data()
    if data['t2_answer_2'].lower() == data['target_2']:
        async with state.proxy() as elem:
            elem['user_score_2'] += 1
    path, label = get_image_path(data['mode'])
    img = open(path, 'br')
    await state.update_data({
        't3_image_path': path,
        'target_3': label
    })
    await message.answer_photo(img, reply_markup=labels[data['mode']])
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t3, content_types='text')
async def test_31(message: types.Message, state: FSMContext):
    await state.update_data({
        't3_answer_1': message.text
    })
    data = await state.get_data()
    path = data['t3_image_path']
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    model, gradcamplusplus = get_models(data['mode'])
    output_class = model(transformed_image).argmax().item()
    if output_class == OUTPUTS_TO_INT[data['mode']][data['target_3']]:
        async with state.proxy() as elem:
            elem['model_score'] += 1
    if data['t3_answer_1'].lower() == data['target_3']:
        async with state.proxy() as elem:
            elem['user_score_1'] += 1
    # print(OUTPUTS[output_class], data['target_3'], data['t3_answer_1'].lower())
    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, gradcamed_image)
    # plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels[data['mode']], caption=f"Model's answer: <b>{OUTPUTS[data['mode']][output_class]}</b>",
                               parse_mode='HTML')
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t31, content_types='text')
async def test_32(message: types.Message, state: FSMContext):
    await message.answer("Answer has been recorded")
    answer_2 = message.text
    await state.update_data({
        't3_answer_2': answer_2
    })
    data = await state.get_data()
    if data['t3_answer_2'].lower() == data['target_3']:
        async with state.proxy() as elem:
            elem['user_score_2'] += 1
    path, label = get_image_path(data['mode'])
    img = open(path, 'br')
    await state.update_data({
        't4_image_path': path,
        'target_4': label
    })
    await message.answer_photo(img, reply_markup=labels[data['mode']])
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t4, content_types='text')
async def test_41(message: types.Message, state: FSMContext):
    await state.update_data({
        't4_answer_1': message.text
    })
    data = await state.get_data()
    path = data['t4_image_path']
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    model, gradcamplusplus = get_models(data['mode'])
    output_class = model(transformed_image).argmax().item()
    if output_class == OUTPUTS_TO_INT[data['mode']][data['target_4']]:
        async with state.proxy() as elem:
            elem['model_score'] += 1
    if data['t4_answer_1'].lower() == data['target_4']:
        async with state.proxy() as elem:
            elem['user_score_1'] += 1
    # print(OUTPUTS[output_class], data['target_4'], data['t4_answer_1'].lower())
    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, gradcamed_image)
    # plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels[data['mode']], caption=f"Model's answer: <b>{OUTPUTS[data['mode']][output_class]}</b>",
                               parse_mode='HTML')
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t41, content_types='text')
async def test_42(message: types.Message, state: FSMContext):
    await message.answer("Answer has been recorded")
    answer_2 = message.text
    await state.update_data({
        't4_answer_2': answer_2
    })
    data = await state.get_data()
    if data['t4_answer_2'].lower() == data['target_4']:
        async with state.proxy() as elem:
            elem['user_score_2'] += 1
    path, label = get_image_path(data['mode'])
    img = open(path, 'br')
    await state.update_data({
        't5_image_path': path,
        'target_5': label
    })
    await message.answer_photo(img, reply_markup=labels[data['mode']])
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t5, content_types='text')
async def test_41(message: types.Message, state: FSMContext):
    await state.update_data({
        't5_answer_1': message.text
    })
    data = await state.get_data()
    path = data['t5_image_path']
    image = io.imread(path, as_gray=False).astype(np.float32)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    transformed_image = transformations(image).unsqueeze(0)
    model, gradcamplusplus = get_models(data['mode'])
    output_class = model(transformed_image).argmax().item()
    if output_class == OUTPUTS_TO_INT[data['mode']][data['target_5']]:
        async with state.proxy() as elem:
            elem['model_score'] += 1
    if data['t5_answer_1'].lower() == data['target_5']:
        async with state.proxy() as elem:
            elem['user_score_1'] += 1
    # print(OUTPUTS[output_class], data['target_5'], data['t5_answer_1'].lower())
    gradcamed_image = get_gradcam_image(image, output_class, gradcamplusplus)
    f = plt.figure()
    save_path = RESULT_DATA_DIR + path.split('/')[2]
    plt.imsave(save_path, gradcamed_image)
    # plt.imsave(save_path, transformed_image)
    f.clear()
    plt.close(f)
    img = open(save_path, 'br')
    await message.answer_photo(img, reply_markup=labels[data['mode']], caption=f"Model's answer: <b>{OUTPUTS[data['mode']][output_class]}</b>",
                               parse_mode='HTML')
    await message.answer("Choose the right answer")
    await Classification.next()


@dp.message_handler(state=Classification.t51, content_types='text')
async def test_52(message: types.Message, state: FSMContext):
    answer_2 = message.text
    await state.update_data({
        't5_answer_2': answer_2
    })
    data = await state.get_data()
    if data['t5_answer_2'].lower() == data['target_5']:
        async with state.proxy() as elem:
            elem['user_score_2'] += 1
    data = await state.get_data()
    await message.answer(
        f" Model's accuracy: <b>{data['model_score']}/5 </b>\n" +
        f" Accuracy of your first answer: <b>{data['user_score_1']}/5 </b>\n" +
        f" Accuracy of your second answer: <b>{data['user_score_2']}/5 </b>\n\n" +
        "Type /classification for classification task practice \n\n Type /segmentation for segmentation task practice \n\n Type /qa for Q&A practice", parse_mode='HTML'
    )
    await state.finish()
