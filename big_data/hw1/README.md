## Блок 1. Развёртывание локального кластера Hadoop.
- Скриншоты вложены в репозиторий

## Блок 2. Работа с HDFS

### Блок 2.1.
1. [2 балла] Создайте папку в корневой HDFS-папке
```
docker exec -it namenode /bin/bash
hdfs dfs -mkdir /folder
```

2. [2 балла] Создайте в созданной папке новую вложенную папку.
```
hdfs dfs -mkdir /folder/subfolder
```

3. [3 балла] Что такое Trash в распределенной FS? Как сделать так, чтобы файлы удалялись сразу, минуя “Trash”?
Trash это куда попадают удалённые файлы. Чтобы они удалялись, минуя “Trash”, нужно добавить флаг -skipTrash при удалении например:
```
Hdfs dfs -rm -skipTrash <filename> 
```

4. [2 балла] Создайте пустой файл в подпапке из пункта 2.
```
hdfs dfs -touchz /folder/subfolder/empty_file
```

5. [2 балла] Удалите созданный файл.
```
hdfs dfs -rm -skipTrash /folder/subfolder/empty_file
```

6. [2 балла] Удалите созданные папки
```
hdfs dfs -rm -r /folder
```

### Блок 2.2.
1. [3 балла] Скопируйте любой файл в новую папку на HDFS
```
docker cp .\borodino.txt namenode:/
docker exec -it namenode /bin/bash
hdfs dfs -mkdir /folder
hdfs dfs -put borodino.txt /
hdfs dfs -cp /borodino.txt /folder
```

2. [3 балла] Выведите содержимое HDFS-файла на экран.
```
hdfs dfs -cat /folder/borodino.txt
```

3. [3 балла] Выведите содержимое нескольких последних строчек HDFS-файла на экран.
```
hdfs dfs -tail /folder/borodino.txt
Гора кровавых тел.

Изведал враг в тот день немало,
Что значит русский бой удалый,
Наш рукопашный бой!..
Земля тряслась — как наши груди,
Смешались в кучу кони, люди,
И залпы тысячи орудий
Слились в протяжный вой…

Вот смерклось. Были все готовы
Заутра бой затеять новый
И до конца стоять…
Вот затрещали барабаны —
И отступили бусурманы.
Тогда считать мы стали раны,
Товарищей считать.

Да, были люди в наше время,
Могучее, лихое племя:
Богатыри — не вы.
Плохая им досталась доля:
Немногие вернулись с поля.
Когда б на то не божья воля,
```

4. [3 балла] Выведите содержимое нескольких первых строчек HDFS-файла на экран.
```
hdfs dfs -head /folder/borodino.txt
Михаил Лермонтов
Бородино

— Скажи-ка, дядя, ведь не даром
Москва, спаленная пожаром,
Французу отдана?
Ведь были ж схватки боевые,
Да, говорят, еще какие!
Недаром помнит вся Россия
Про день Бородина!

— Да, были люди в наше время,
Не то, что нынешнее племя:
Богатыри — не вы!
Плохая им досталась доля:
Немногие вернулись с поля…
Не будь на то господня воля,
Не отдали б Москвы!

Мы долго молча отступали,
Досадно было, боя ждали,
Ворчали старики:
«Что ж мы? на зимние квартиры?
Не смеют, что ли, командиры
Чужие изорвать мундиры
О русские штыки?»
```


5. [3 балла] Переместите копию файла в HDFS на новую локацию.
```
hdfs dfs -mv /folder/borodino.txt /second_folder
```

### Блок 2.3.
1. [4 баллов] Изменить replication factor для файла. Как долго занимает время на увеличение / уменьшение числа реплик для файла?
```
hdfs dfs -setrep -w 2 /second_folder/borodino.txt
Replication 2 set: /second_folder/borodino.txt
Waiting for /second_folder/borodino.txt ...
WARNING: the waiting time may be long for DECREASING the number of replications.
. done

hdfs dfs -setrep -w 3 /second_folder/borodino.txt
Replication 3 set: /second_folder/borodino.txt
Waiting for /second_folder/borodino.txt .... done
```

Вроде бы файл маленький но времени занимает достаточно (около 5 секунд). И уменьшение реплик занимает ещё больше времени чем увеличение.

2. [4 баллов] Найдите информацию по файлу, блокам и их расположениям с помощью “hdfs fsck”
```
hdfs fsck /second_folder/borodino.txt
Connecting to namenode via http://namenode:9870/fsck?ugi=root&path=%2Fsecond_folder%2Fborodino.txt
FSCK started by root (auth:SIMPLE) from /172.19.0.4 for path /second_folder/borodino.txt at Fri Oct 01 11:38:40 UTC 2021

Status: HEALTHY
 Number of data-nodes:  3
 Number of racks:               1
 Total dirs:                    0
 Total symlinks:                0

Replicated Blocks:
 Total size:    4677 B
 Total files:   1
 Total blocks (validated):      1 (avg. block size 4677 B)
 Minimally replicated blocks:   1 (100.0 %)
 Over-replicated blocks:        0 (0.0 %)
 Under-replicated blocks:       0 (0.0 %)
 Mis-replicated blocks:         0 (0.0 %)
 Default replication factor:    3
 Average block replication:     3.0
 Missing blocks:                0
 Corrupt blocks:                0
 Missing replicas:              0 (0.0 %)

Erasure Coded Block Groups:
 Total size:    0 B
 Total files:   0
 Total block groups (validated):        0
 Minimally erasure-coded block groups:  0
 Over-erasure-coded block groups:       0
 Under-erasure-coded block groups:      0
 Unsatisfactory placement block groups: 0
 Average block group size:      0.0
 Missing block groups:          0
 Corrupt block groups:          0
 Missing internal blocks:       0
FSCK ended at Fri Oct 01 11:38:40 UTC 2021 in 3 milliseconds


The filesystem under path '/second_folder/borodino.txt' is HEALTHY
```


3. [4 баллов] Получите информацию по любому блоку из п.2 с помощью "hdfs fsck -blockId”.
Обратите внимание на Generation Stamp (GS number).
```
hdfs fsck /second_folder/borodino.txt -files -blocks -locations
hdfs fsck -blockId blk_1073741835

Connecting to namenode via http://namenode:9870/fsck?ugi=root&blockId=BP-104876619+&path=%2F
FSCK started by root (auth:SIMPLE) from /172.19.0.4 at Fri Oct 01 11:51:18 UTC 2021
Incorrect blockId format: BP-104876619
root@60a2e3b248a3:/# hdfs fsck -blockId blk_1073741835
Connecting to namenode via http://namenode:9870/fsck?ugi=root&blockId=blk_1073741835+&path=%2F
FSCK started by root (auth:SIMPLE) from /172.19.0.4 at Fri Oct 01 11:51:50 UTC 2021

Block Id: blk_1073741835
Block belongs to: /second_folder/borodino.txt
No. of Expected Replica: 3
No. of live Replica: 3
No. of excess Replica: 0
No. of stale Replica: 0
No. of decommissioned Replica: 0
No. of decommissioning Replica: 0
No. of corrupted Replica: 0
Block replica on datanode/rack: 9ad999990bc1/default-rack is HEALTHY
Block replica on datanode/rack: 75a79ba6fe4a/default-rack is HEALTHY
Block replica on datanode/rack: 89fe845e35a1/default-rack is HEALTHY
```

## Блок 3. 
1. см. ноутбук

2. см. ноутбук

3. см. файлы

4. Установил на NN, NM и RM python. Потом запустил следующие команды:
```
mapred streaming -files mapper_mean.py,reducer_mean.py,prices.txt -mapper "python mapper_mean.py" -reducer "python reducer_mean.py" -input prices.txt -output mean

mapred streaming -files mapper_var.py,reducer_var.py,prices.txt -mapper "python mapper_var.py" -reducer "python reducer_var.py" -input prices.txt -o
utput variance
```

5. С помощью следующих команд получил файлы с результатами вычислений:
```
hdfs dfs -cat /user/root/mean/part-00000 > result_mean.txt
hdfs dfs -cat /user/root/variance/part-00000 > result_variance.txt
```
Есть небольшая разница при вычислении с помошью numpy и map-reduce (см. файлы с результатами)

6. см. репозиторий