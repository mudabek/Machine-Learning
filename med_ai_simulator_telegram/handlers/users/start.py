from aiogram import types
from aiogram.dispatcher.filters.builtin import CommandStart
from states import UserRegistration
from loader import dp
from aiogram.dispatcher import FSMContext
from keyboards.default import share_phone


@dp.message_handler(CommandStart(), state=None)
async def bot_start(message: types.Message):
    await message.answer("Здавствуйте. Введите свое имя")
    await UserRegistration.first()


@dp.message_handler(state=UserRegistration.first_name)
async def get_first_name(message: types.Message, state: FSMContext):
    first_name = message.text
    await state.update_data({
        'first_name': first_name
    })
    await message.answer("Хорошо. А теперь введите фамилию.")
    await UserRegistration.next()


@dp.message_handler(state=UserRegistration.last_name)
async def get_last_name(message: types.Message, state: FSMContext):
    last_name = message.text
    await state.update_data({
        'last_name': last_name
    })
    await message.answer("Для отправки своего номера телефона, пожалуйста, нажмите на конпку ниже.",
                         reply_markup=share_phone)
    await UserRegistration.next()


@dp.message_handler(content_types='contact', state=UserRegistration.phone_number)
async def get_phone_number(message: types.Message, state: FSMContext):
    phone_number = message.contact.phone_number
    await state.update_data({
        'phone_number': phone_number
    })
    registration_data = await state.get_data()
    await message.answer(
        f"First name: {registration_data['first_name']}\nLast name: {registration_data['last_name']}\nPhone number: {registration_data['phone_number']}\n", reply_markup=types.ReplyKeyboardRemove())
    await state.finish()
    await message.answer("Наберите /classification чтобы начать тренировку на классификаторах \n\n Наберите /segmentation чтобы начать тренировку на сегментацию \n\n Наберите /qa чтобы испытать вопросно ответную систему")
