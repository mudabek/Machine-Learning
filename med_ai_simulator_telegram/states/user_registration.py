from aiogram.dispatcher.filters.state import (StatesGroup, State)


class UserRegistration(StatesGroup):
    first_name = State()
    last_name = State()
    phone_number = State()
