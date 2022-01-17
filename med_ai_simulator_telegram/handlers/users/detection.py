from aiogram import types
from loader import dp
from aiogram.dispatcher.filters import Command, Text
from ai_model.get_image import get_image_for_detection
from keyboards.default import polyp_labels
from aiogram.dispatcher import FSMContext
from states.detecton import Detection
from skimage import io
import numpy as np
from ai_model import get_detector
import cv2
import matplotlib.pyplot as plt
import random


RESULT_DATA_DIR = 'results/'


RESULTS_TO_INT = {
    '1 - Model  2 - Ground Truth': 0,
    '1 - Ground Truth  2 - Model': 1
}


@dp.message_handler(Command(commands=['segmentation']), state=None)
async def begin_test(message: types.Message, state: FSMContext):
    test_set = get_image_for_detection()
    idx = np.random.randint(0, len(test_set))
    model = get_detector()
    model.eval()
    softmax_output = model(test_set[idx][0].unsqueeze(0))[0].detach().numpy()
    pred = np.zeros_like(softmax_output)
    pred[softmax_output > 0.5] = 1
    input_img = test_set[idx][0].numpy()
    gt = test_set[idx][1].numpy()
    pred = cv2.addWeighted(np.repeat(pred, 3, axis=0), 0.2, input_img, 0.8, 0)
    gt = cv2.addWeighted(np.repeat(gt, 3, axis=0), 0.2, input_img, 0.8, 0)
    model_vis = np.concatenate((input_img, pred), axis=1)
    true_vis = np.concatenate((input_img, gt), axis=1)
    save_path_1 = RESULT_DATA_DIR + 'model_' + str(test_set[idx][2]) + '.jpg'
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imsave(save_path_1, model_vis.T)
    ax1 = fig.add_subplot(2, 1, 2)
    save_path_2 = RESULT_DATA_DIR + 'ground_truth' + str(test_set[idx][2]) + '.jpg'
    plt.imsave(save_path_2, true_vis.T)
    img_1 = open(save_path_1, 'br')
    img_2 = open(save_path_2, 'br')
    images = [(img_1, 1), (img_2, 2)]
    random.shuffle(images)
    target = 0
    for idx in range(len(images)):
        await message.answer_photo(images[idx][0], caption=str(idx + 1))
        if images[idx][1] == 2:
            target = idx
    await state.update_data({
        'target_1': target,
        'user_score': 0
    })
    await message.answer("Choose the right answer.", reply_markup=polyp_labels)
    await Detection.first()


@dp.message_handler(state=Detection.t1, content_types='text')
async def test_1(message: types.Message, state: FSMContext):
    user_ans = RESULTS_TO_INT[message.text]
    data = await state.get_data()
    if data['target_1'] == user_ans:
        async with state.proxy() as elem:
            elem['user_score'] += 1
    test_set = get_image_for_detection()
    idx = np.random.randint(0, len(test_set))
    model = get_detector()
    model.eval()
    softmax_output = model(test_set[idx][0].unsqueeze(0))[0].detach().numpy()
    pred = np.zeros_like(softmax_output)
    pred[softmax_output > 0.5] = 1
    input_img = test_set[idx][0].numpy()
    gt = test_set[idx][1].numpy()
    pred = cv2.addWeighted(np.repeat(pred, 3, axis=0), 0.2, input_img, 0.8, 0)
    gt = cv2.addWeighted(np.repeat(gt, 3, axis=0), 0.2, input_img, 0.8, 0)
    model_vis = np.concatenate((input_img, pred), axis=1)
    true_vis = np.concatenate((input_img, gt), axis=1)
    save_path_1 = RESULT_DATA_DIR + 'model_' + str(test_set[idx][2]) + '.jpg'
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imsave(save_path_1, model_vis.T)
    ax1 = fig.add_subplot(2, 1, 2)
    save_path_2 = RESULT_DATA_DIR + 'ground_truth' + str(test_set[idx][2]) + '.jpg'
    plt.imsave(save_path_2, true_vis.T)
    img_1 = open(save_path_1, 'br')
    img_2 = open(save_path_2, 'br')
    images = [(img_1, 1), (img_2, 2)]
    random.shuffle(images)
    target = 0
    for idx in range(len(images)):
        await message.answer_photo(images[idx][0], caption=str(idx + 1))
        if images[idx][1] == 2:
            target = idx
    await state.update_data({
        'target_2': target
    })
    await message.answer("Choose the right answer.", reply_markup=polyp_labels)
    await Detection.next()


@dp.message_handler(state=Detection.t2, content_types='text')
async def test_2(message: types.Message, state: FSMContext):
    user_ans = RESULTS_TO_INT[message.text]
    data = await state.get_data()
    if data['target_2'] == user_ans:
        async with state.proxy() as elem:
            elem['user_score'] += 1
    test_set = get_image_for_detection()
    idx = np.random.randint(0, len(test_set))
    model = get_detector()
    model.eval()
    softmax_output = model(test_set[idx][0].unsqueeze(0))[0].detach().numpy()
    pred = np.zeros_like(softmax_output)
    pred[softmax_output > 0.5] = 1
    input_img = test_set[idx][0].numpy()
    gt = test_set[idx][1].numpy()
    pred = cv2.addWeighted(np.repeat(pred, 3, axis=0), 0.2, input_img, 0.8, 0)
    gt = cv2.addWeighted(np.repeat(gt, 3, axis=0), 0.2, input_img, 0.8, 0)
    model_vis = np.concatenate((input_img, pred), axis=1)
    true_vis = np.concatenate((input_img, gt), axis=1)
    save_path_1 = RESULT_DATA_DIR + 'model_' + str(test_set[idx][2]) + '.jpg'
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imsave(save_path_1, model_vis.T)
    ax1 = fig.add_subplot(2, 1, 2)
    save_path_2 = RESULT_DATA_DIR + 'ground_truth' + str(test_set[idx][2]) + '.jpg'
    plt.imsave(save_path_2, true_vis.T)
    img_1 = open(save_path_1, 'br')
    img_2 = open(save_path_2, 'br')
    images = [(img_1, 1), (img_2, 2)]
    random.shuffle(images)
    target = 0
    for idx in range(len(images)):
        await message.answer_photo(images[idx][0], caption=str(idx + 1))
        if images[idx][1] == 2:
            target = idx
    await state.update_data({
        'target_3': target
    })
    await message.answer("Choose the right answer.", reply_markup=polyp_labels)
    await Detection.next()


@dp.message_handler(state=Detection.t3, content_types='text')
async def test_3(message: types.Message, state: FSMContext):
    user_ans = RESULTS_TO_INT[message.text]
    data = await state.get_data()
    if data['target_3'] == user_ans:
        async with state.proxy() as elem:
            elem['user_score'] += 1
    test_set = get_image_for_detection()
    idx = np.random.randint(0, len(test_set))
    model = get_detector()
    model.eval()
    softmax_output = model(test_set[idx][0].unsqueeze(0))[0].detach().numpy()
    pred = np.zeros_like(softmax_output)
    pred[softmax_output > 0.5] = 1
    input_img = test_set[idx][0].numpy()
    gt = test_set[idx][1].numpy()
    pred = cv2.addWeighted(np.repeat(pred, 3, axis=0), 0.2, input_img, 0.8, 0)
    gt = cv2.addWeighted(np.repeat(gt, 3, axis=0), 0.2, input_img, 0.8, 0)
    model_vis = np.concatenate((input_img, pred), axis=1)
    true_vis = np.concatenate((input_img, gt), axis=1)
    save_path_1 = RESULT_DATA_DIR + 'model_' + str(test_set[idx][2]) + '.jpg'
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imsave(save_path_1, model_vis.T)
    ax1 = fig.add_subplot(2, 1, 2)
    save_path_2 = RESULT_DATA_DIR + 'ground_truth' + str(test_set[idx][2]) + '.jpg'
    plt.imsave(save_path_2, true_vis.T)
    img_1 = open(save_path_1, 'br')
    img_2 = open(save_path_2, 'br')
    images = [(img_1, 1), (img_2, 2)]
    random.shuffle(images)
    target = 0
    for idx in range(len(images)):
        await message.answer_photo(images[idx][0], caption=str(idx + 1))
        if images[idx][1] == 2:
            target = idx
    await state.update_data({
        'target_4': target
    })
    await message.answer("Choose the right answer.", reply_markup=polyp_labels)
    await Detection.next()


@dp.message_handler(state=Detection.t4, content_types='text')
async def test_4(message: types.Message, state: FSMContext):
    user_ans = RESULTS_TO_INT[message.text]
    data = await state.get_data()
    if data['target_4'] == user_ans:
        async with state.proxy() as elem:
            elem['user_score'] += 1
    test_set = get_image_for_detection()
    idx = np.random.randint(0, len(test_set))
    model = get_detector()
    model.eval()
    softmax_output = model(test_set[idx][0].unsqueeze(0))[0].detach().numpy()
    pred = np.zeros_like(softmax_output)
    pred[softmax_output > 0.5] = 1
    input_img = test_set[idx][0].numpy()
    gt = test_set[idx][1].numpy()
    pred = cv2.addWeighted(np.repeat(pred, 3, axis=0), 0.2, input_img, 0.8, 0)
    gt = cv2.addWeighted(np.repeat(gt, 3, axis=0), 0.2, input_img, 0.8, 0)
    model_vis = np.concatenate((input_img, pred), axis=1)
    true_vis = np.concatenate((input_img, gt), axis=1)
    save_path_1 = RESULT_DATA_DIR + 'model_' + str(test_set[idx][2]) + '.jpg'
    fig = plt.figure(figsize=(20, 12))
    ax1 = fig.add_subplot(2, 1, 1)
    plt.imsave(save_path_1, model_vis.T)
    ax1 = fig.add_subplot(2, 1, 2)
    save_path_2 = RESULT_DATA_DIR + 'ground_truth' + str(test_set[idx][2]) + '.jpg'
    plt.imsave(save_path_2, true_vis.T)
    img_1 = open(save_path_1, 'br')
    img_2 = open(save_path_2, 'br')
    images = [(img_1, 1), (img_2, 2)]
    random.shuffle(images)
    target = 0
    for idx in range(len(images)):
        await message.answer_photo(images[idx][0], caption=str(idx + 1))
        if images[idx][1] == 2:
            target = idx
    await state.update_data({
        'target_5': target
    })
    await message.answer("Choose the right answer.", reply_markup=polyp_labels)
    await Detection.next()


@dp.message_handler(state=Detection.t5, content_types='text')
async def test_4(message: types.Message, state: FSMContext):
    user_ans = RESULTS_TO_INT[message.text]
    data = await state.get_data()
    if data['target_5'] == user_ans:
        async with state.proxy() as elem:
            elem['user_score'] += 1
    data = await state.get_data()
    await message.answer(f"Your results: <b>{data['user_score']}/5</b>\n\n" +
                         "Type /classification for classification task practice \n\n Type /segmentation for segmentation task practice \n\n Type /qa for Q&A practice", parse_mode='HTML')
    await state.finish()