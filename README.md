# jooble
---
this repository was created for test task

structure:
```
|
|- data        -> folder with external and processed data
|
|- notebooks   -> folder with notebooks
|
|- src         -> folder with necessary functions/scripts
|
|- other       -> folder with test task info
|
 ```
 
to run jupyter notebook with necessary env run in terminal command

```
poetry update
poetry run jupyter notebook
```
to run python script to preprocess test dataset run command
```
poetry update
poetry run python src/preprocess_vacancy.py 
```
Задача:
1. требуется реализовать пакет по предобработке признаков вакансий

notebooks/reference.ipynb

src/preprocess_vacancy.py 
