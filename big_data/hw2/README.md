# Hive.

## Блок 1. Развёртывание локального Hive.

1) Развернул Hive с помощью https://github.com/tech4242/docker-hadoop-hive-parquet

2) Подключился к Hive с помощью Hue и Python Driver (PyCharm), установив Hive dependencies и пробросив требуемый порт.

3) Скриншоты:

Запуск Hive на docker
![alt text](https://github.com/mudabek/Big_Data/blob/main/hw2/hive_hue_startup.png?raw=true)

Hive на Hue
![alt text](https://github.com/mudabek/Big_Data/blob/main/hw2/hue_interface.png?raw=true)

Hive на PyCharm
- скриншоты во 2 блоке


## Блок 2. Работа с Hive.

1) Таблица artists в Hive
![alt text](https://github.com/mudabek/Big_Data/blob/main/hw2/hive_artists.table.png?raw=true)
![alt text](https://github.com/mudabek/Big_Data/blob/main/hw2/importeb_table_pycharm.png?raw=true)

2) SQL запросы

a) Исполнитель с максимальным числом скробблов
```
SELECT artist_lastfm, scrobbles_lastfm FROM artists
ORDER BY scrobbles_lastfm DESC
LIMIT 1;

Output:
artist_lastfm, scrobbles_lastfm
The Beatles, 517126254
```

d) Малоизвестные исполнители с очень заразными песнями (репитов много, слушателей мало)
```
SELECT DISTINCT artist_lastfm, scrobbles_lastfm/listeners_lastfm as ratio FROM artists
ORDER BY ratio DESC
LIMIT 10;

Output:
artist_lastfm, ratio
BTS, 463.19432498315433
Exo, 228.16363011618802
SHINee, 211.4087390933205
the GazettE, 210.74272727833386
O.S.T.R., 194.22867842646477
Eldo, 187.32135980030904
AKB48, 184.07628425673272
Böhse Onkelz, 175.42030447101763
소녀시대, 173.23673726325217
동방신기, 171.11682933326875
```
