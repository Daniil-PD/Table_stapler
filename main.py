import sys
import time

import PyQt5
from PyQt5 import QtWidgets, QtCore
import logging
from pathlib import Path
import os

# модули с графикой
import main_window
import table_form
import column_form

# модули для анализа данных
import pandas as pd
from scipy.stats import zscore

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent


class ColumnListItem(QtWidgets.QListWidgetItem):
    def __init__(self, series: pd.Series):
        super().__init__()
        self.pd_series = series

class TableListItem(QtWidgets.QListWidgetItem):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.pd_data = pd.read_excel(file_path, sheet_name=0, header=1)

    def name(self):
        return os.path.basename(self.file_path)



class MainWindow(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    # Класс главного окна

    def __init__(self):
        # Обязательно нужно вызвать метод супер класса
        QtWidgets.QMainWindow.__init__(self)
        self.setupUi(self)

        self.statusbar.showMessage("Загрузка программы")

        self.listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        self.listWidget.setDefaultDropAction(QtCore.Qt.MoveAction)
        self.listWidget.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)  # Включить прокрутку
        self.listWidget.verticalScrollBar().setSingleStep(15)

        self.pushButton_open_table.clicked.connect(self.open_table)
        self.pushButton_save.clicked.connect(lambda: self.statusbar.showMessage("Файл сохранён", 3000))

        self.statusbar.showMessage("")

        self.statusbarProgressBar = QtWidgets.QProgressBar()
        self.statusbarProgressBar.setFormat("")
        self.statusbarProgressBar.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.statusbar.addPermanentWidget(self.statusbarProgressBar)


    def open_table(self):
        # открытие файла с таблицей
        self.statusbar.showMessage("Открытие файла", 6000)
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home', filter="*.xlsx")[0]
        logging.debug(f"Открытие файла {fname.split('/')[-1]}")
        self.statusbar.showMessage(f"Открытие файл {fname.split('/')[-1]}", 6000)
        self.add_table_to_list(fname)

    def add_table_to_list(self, fpath: str):
        item = None
        try:
            item = TableListItem(fpath)  # Создать объект QListWidgetItem
        except FileNotFoundError as e:
            self.statusbar.showMessage("Ошибка открытия файла", 6000)
            return
        except Exception as e:
            self.statusbar.showMessage(e.__str__(), 6000)
            return
        item.setSizeHint(QtCore.QSize(550, 350))  # Установить размер QListWidgetItem

        form = QtWidgets.QWidget()
        ui = table_form.Ui_Form()
        ui.setupUi(form)
        ui.label_name.setText(item.name())
        ui.label_name.setToolTip(item.name())
        ui.label_info.setText(f"Количество строк в таблице: {item.pd_data.shape[0]}")

        ui.listWidget.setDragDropMode(QtWidgets.QAbstractItemView.DragDrop)
        ui.listWidget.setDefaultDropAction(QtCore.Qt.MoveAction) # Установить действие при перетаскивании
        ui.listWidget.horizontalScrollBar().valueChanged.connect(self.move_scrollbar)  # При изменении прокрутки
        ui.listWidget.setObjectName("listWidget")  # Установить имя
        ui.listWidget.setHorizontalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)  # Включить прокрутку
        scroll_step = 15  # Шаг прокрутки
        ui.listWidget.horizontalScrollBar().setSingleStep(scroll_step)
        ui.listWidget.horizontalScrollBar().rangeChanged.connect(self.set_max_scrollbar)

        ui.pushButton_close.clicked.connect(lambda: self.listWidget.takeItem(self.listWidget.indexFromItem(item).row()))
        ui.pushButton_avto.clicked.connect(lambda: self.automatic_positioning(item))


        self.listWidget.addItem(item)  # Добавить элемент
        self.listWidget.setItemWidget(item, form)  # Установить виджет для элемента


        for col in item.pd_data:
            self.add_rows_to_list(ui.listWidget, item.pd_data.loc[:, col])

        self.set_max_scrollbar()
        self.statusbar.showMessage(f"Файл {item.name()} открыт", 6000)

    def add_rows_to_list(self, list_widget : PyQt5.QtWidgets.QListWidget, series: pd.Series):

        item = ColumnListItem(series)   # Создать объект QListWidgetItem
        item.setSizeHint(QtCore.QSize(250, 260))
        form = QtWidgets.QWidget()
        ui = column_form.Ui_Form()
        ui.setupUi(form)
        ui.label_name.setText(str(item.pd_series.name))
        ui.label_name.setToolTip(str(item.pd_series.name))

        ui.textBrowser.setText(str(item.pd_series.values))  # Выводим информацию об столбце

        ui.pushButton_close.clicked.connect(lambda: list_widget.takeItem(list_widget.indexFromItem(item).row()))

        self.display_column_info(item.pd_series, ui.textBrowser)

        list_widget.addItem(item)  # Добавить элемент
        list_widget.setItemWidget(item, form)  # Установить виджет для элемента

    def display_column_info(self, column, text_browser):
        # Для текстовых столбцов
        if column.dtype == 'object':
            text_data = column.dropna().astype(str)
            text_length_mean = text_data.str.len().mean()
            most_common_words: pd.Series = text_data.str.split().explode().value_counts().head(8)

            text_browser.setText(f"Средняя длина текста: {text_length_mean:.2f}\n"
                                 f"Самые часто встречающиеся слова:\n{most_common_words.to_string(header=False)}")


        # Для числовых столбцов
        elif column.dtype == 'int64' or column.dtype == 'float64':
            numerical_data = column.dropna()
            numerical_mean = numerical_data.mean()
            numerical_max = numerical_data.max()
            numerical_min = numerical_data.min()
            numerical_std = numerical_data.std()
            numerical_quantile_25 = numerical_data.quantile(0.25)
            numerical_median = numerical_data.median()
            numerical_quantile_75 = numerical_data.quantile(0.75)

            text_browser.setText(f"Среднее значение: {numerical_mean}\n"
                                 f"Максимальное значение: {numerical_max}\n"
                                 f"Минимальное значение: {numerical_min}\n"
                                 f"Стандартное отклонение: {numerical_std}\n"
                                 f"Верхние значения первого квартиля: {numerical_quantile_25}\n"
                                 f"Медиана: {numerical_median}\n"
                                 f"Верхние значения третьего квартиля: {numerical_quantile_75}")

        else:
            # Обработка других типов данных
            text_browser.setText("Информация о столбце недоступна для данного типа данных")

    def move_scrollbar(self, value):
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            form_widget = self.listWidget.itemWidget(item)

            # Находим виджет списка
            list_widget = form_widget.findChild(QtWidgets.QListWidget, "listWidget")
            if list_widget:
                list_widget.blockSignals(True)
                list_widget.horizontalScrollBar().setValue(value)
                list_widget.blockSignals(False)

    def set_max_scrollbar(self):
        temporary_maximum = 0
        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            form_widget = self.listWidget.itemWidget(item)

            # Находим виджет списка
            list_widget = None
            if form_widget:
                list_widget = form_widget.findChild(QtWidgets.QListWidget, "listWidget")
            if list_widget:
                temporary_maximum = max(temporary_maximum, list_widget.horizontalScrollBar().maximum())

        for i in range(self.listWidget.count()):
            item = self.listWidget.item(i)
            form_widget = self.listWidget.itemWidget(item)

            # Находим виджет списка
            list_widget = None
            if form_widget:
                list_widget = form_widget.findChild(QtWidgets.QListWidget, "listWidget")
            if list_widget:
                list_widget.blockSignals(True)
                list_widget.horizontalScrollBar().setMaximum(temporary_maximum)
                list_widget.blockSignals(False)

    def calculate_similarity_column(self, column1, column2):
        similarity = 0

        if column1.dtype == 'object' and column2.dtype == 'object':
            # Создаем объект TfidfVectorizer
            vectorizer = TfidfVectorizer()

            # Преобразуем тексты в матрицу TF-IDF признаков
            tfidf_matrix = vectorizer.fit_transform([' '.join(column1.astype(str).to_list()), ' '.join(column2.astype(str).to_list()) ])


            # Рассчитываем сходство между текстами
            similarity_matrix = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])

            # Возвращаем оценку схожести текстов
            return similarity_matrix[0][0]

        elif (column1.dtype == 'int64' or column1.dtype == 'float64') and (
                column2.dtype == 'int64' or column2.dtype == 'float64'):
            # Создаем DataFrame из двух series
            df = pd.DataFrame({'series1': column1, 'series2': column2})

            # Рассчитываем различные метрики для каждой колонки
            metrics = {
                'mean': df.mean(),
                'max': df.max(),
                'min': df.min(),
                'std': df.std(),
                'q1': df.quantile(0.25),
                'median': df.median(),
                'q3': df.quantile(0.75)
            }

            # Нормализуем значения метрик с использованием Z-оценки
            normalized_metrics = {metric: zscore(values) for metric, values in metrics.items()}

            # Вычисляем сумму нормализованных значений метрик
            similarity_score = sum(normalized_metrics.values())

            # Нормализуем сумму в диапазоне от 0 до 1
            similarity = (similarity_score - min(similarity_score)) / (max(similarity_score) - min(similarity_score))

            return similarity

        return similarity

    def create_similarity_dataframe(self, table_item: TableListItem):

        current_index = self.listWidget.indexFromItem(table_item)  # Получаем индекс текущей таблицы
        next_index = current_index.row() + 1  # Получаем индекс следующей таблицы
        next_item = self.listWidget.item(next_index)  # Получаем следующую таблицу

        # Если нет следующей таблицы
        if next_item is None:
            return None


        current_listWidget: QtWidgets.QListWidget = self.listWidget.itemWidget(table_item)\
            .findChild(QtWidgets.QListWidget, "listWidget")
        next_listWidget: QtWidgets.QListWidget = self.listWidget.itemWidget(next_item)\
            .findChild(QtWidgets.QListWidget, "listWidget")

        # Получаем данные текущей таблицы и следующей таблицы и записываем их в датафреймы
        current_data = pd.DataFrame([current_listWidget.item(i).pd_series for i in range(current_listWidget.count())])
        next_data = pd.DataFrame([next_listWidget.item(i).pd_series for i in range(next_listWidget.count())])

        similarity_df = pd.DataFrame(index=range(current_data.shape[0]), columns=range(next_data.shape[0]))

        # Заполняем значениями схожести
        for row in range(current_data.shape[0]):
            for col in range(next_data.shape[0]):
                self.statusbarProgressBar.setValue(int((row / (current_data.shape[0] - 1)) * 100))
                # Вычисляем схожесть колонок
                similarity_score = self.calculate_similarity_column(current_data.iloc[row], next_data.iloc[col])

                # Записываем значение схожести в DataFrame
                similarity_df.at[row, col] = similarity_score

        # Возвращаем DataFrame со значениями схожести
        return similarity_df

    def automatic_positioning(self, table_item: TableListItem):
        self.statusbar.showMessage("Анализ столбцов")
        self.statusbarProgressBar.setValue(0)
        sim_df = self.create_similarity_dataframe(table_item)
        self.statusbarProgressBar.setValue(0)
        print(sim_df)

    def closeEvent(self, event) -> None:
        self.statusbar.showMessage("Закрытие программы")
        event.accept()




def main():

    # проверка на нахождение папки "data" в проекте
    folders = None
    for dirs, folders, files in os.walk(BASE_DIR):
        break

    if not "data" in folders:   # если папки нет, то создаём её
        os.mkdir("data")

    logging.basicConfig(    # настройки для модуля записи логов
        level=logging.DEBUG,
        format='%(asctime)s : %(levelname)s : %(message)s',
        filename=os.path.join(BASE_DIR, "data", "log_file.log"),
        filemode='w',
    )
    logging.debug("Cтарт приложения")

    app = QtWidgets.QApplication([])
    win = MainWindow()

    win.show()

    time.sleep(0.5)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

