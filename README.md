# Глубокое обучение, ДЗ 3: Embeddings, Attention, Transformer, ViT

В данной работе требуется с нуля реализовать ключевые компоненты архитектуры Transformer (токенизация, позиционные кодировки, механизм внимания, encoder) и Vision Transformer, а затем применить их к практической задаче — fine-tuning предобученной ViT-Tiny на спутниковых снимках EuroSAT. Вы на практике разберётесь, как attention обрабатывает последовательности произвольной природы (текст, SMILES, патчи изображений) и почему transfer learning стал стандартом в современном DL.

Дополнительно предлагается одно из четырёх бонусных заданий: детекция дрифта эмбеддингов, knowledge distillation ViT, регрессия свойств молекул по SMILES или multi-task learning на спутниковых снимках.

Реализация должна строго соответствовать заданным сигнатурам классов и функций. Проверка корректности решений (численное совпадение с эталонной реализацией, формы тензоров, граничные случаи) осуществляется автоматически с помощью GitHub Actions.

## Структура репозитория

```
modules/                # Основные компоненты (разделы 1-4): реализуете вы
├── tokenization.py     # BPE, SMILES токенизаторы
├── embeddings.py       # Positional encoding, collate, embedding pipeline
├── attention.py        # Scaled dot-product, Multi-Head Attention
├── transformer.py      # Transformer encoder
└── vit.py              # Patch Embedding

bonus/                  # Бонусные задания
├── b1_drift/
├── b2_distillation/
├── b3_smiles/
└── b4_multitask/

scripts/                # Запускающие скрипты для обучения
results/                # Артефакты обучения
docs/                   # Тексты заданий
├── homework.md         # Основное задание (разделы 1-4)
└── bonus/              # Бонусные варианты

dl_sem5_hw03_aie.ipynb  # Colab-ноутбук для обучения с GPU
student.json            # Ваш ИСУ и выбранный бонусный вариант
```

## Задания

- **Основная часть** (обязательная, 20 баллов): [docs/homework.md](docs/homework.md)
- **Бонусная часть** (на выбор, +4 балла): [docs/bonus/README.md](docs/bonus/README.md)

## Как выполнять работу

Перед началом работы необходимо создать свою копию репозитория:
1. Сделать **Fork** репозитория для создания своего репозитория.
2. Клонировать свою копию локально:
```
git clone https://github.com/USERNAME/dl_sem5_hw03_aie.git
cd dl_sem5_hw03_aie
```
3. Добавить основной template-репозиторий как дополнительный remote, чтобы можно было создавать Pull Request
```
git remote add upstream https://github.com/L1st3ner/dl_sem5_hw03_aie.git
git fetch upstream
```
4. Создать ветку для решения задания и работать в ней:
```
git checkout -b my_solution
```
5. Заполнить пропущенные реализации. В файл `student.json` записать свой номер ИСУ и вариант бонусного задания, если вы планируете его делать.
6. После выполнения задания коммитить и пушить изменения в свой репозиторий:
```
git add .
git commit -m "Homework solution"
git push origin my_solution
```

7. Обучение модели и генерация артефактов:

    **7.1 Если у вас нет собственного GPU:**

    7.1.1. Создать **Fine-Grained** Personal Access Token на GitHub и сохранить токен в Colab Secrets.

            GitHub → Settings → Developer settings → Personal access tokens → Fine-grained tokens → Generate new token
            - Repository access: Only select repositories → выбрать свой форк
            - Permissions → Contents: Read and Write

    7.1.2. Открыть `dl_sem5_hw03_aie.ipynb` в Google Colab.

    7.1.3. Последовательно выполнить необходимые ячейки ноутбука с использованием GPU.

    7.1.4. Если при выполнении возникают ошибки:
    - Сначала подтянуть изменения, запушенные из Colab, на локальную машину:
        ```
        git pull origin my_solution
        ```
    - Локально исправить код
    - Запушить исправления:
        ```
        git add . && git commit -m "fix" && git push origin my_solution
        ```
    - В Colab выполнить ячейку `!git pull origin my_solution`
    - Перезапустить нужные ячейки

    7.1.5. После успешного обучения выполнить ячейку сохранения артефактов — она запушит `results/` в ваш форк.

    > После того как Colab запушил `results/` в ваш форк, не забудьте подтянуть эти изменения на локальную машину перед дальнейшей работой:
    > ```
    > git pull origin my_solution
    > ```

    **7.2 Если у вас есть собственный GPU:**

    7.2.1. Запустить обучение ViT локально:
    ```
    python -m scripts.train_vit --epochs 10 --batch-size 64 --lr 1e-3
    ```
    7.2.2. После успешного обучения запушить `results/` в ваш форк:
    ```
    git add results/
    git commit -m "Add training artifacts"
    git push origin my_solution
    ```
    7.2.3. Для бонусных заданий, требующих GPU, локально обучение запускается аналогичным образом.

8. Создать Pull Request из ветки `my_solution` вашего форка в `upstream/main` для запуска автотестов.
9. Дождаться результатов CI. Если тесты прошли, всё хорошо.

## Оценивание

Автотесты запускаются в CI/CD на каждый PR. Таблица с итоговыми баллами
доступна у преподавателя.