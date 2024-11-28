import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QFileDialog, QListWidget, QListWidgetItem, QMessageBox,
                             QTextEdit, QMenu, QShortcut)
from PyQt5.QtGui import QPixmap, QIcon, QDrag, QImage, QKeySequence
from PyQt5.QtCore import Qt, QMimeData, QByteArray, QSize, QUrl
import os
import glob
import re
import json

# Please put the PK_FILE_NAME and all the input images into the OpenING_DIR first
OPENING_DIR = "InputImages"
PK_FILE_NAME = "data_instance_modelAB.json"
SAVE_FILE_NAME = "judge_modelAB_results.json"

class DraggableListWidget(QListWidget):
    def __init__(self, parent=None, is_input_area=True):
        super(DraggableListWidget, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(325, 325))
        self.setSpacing(1)
        self.draggingItem = None
        self.is_input_area = is_input_area  # Add this to distinguish between input and output areas

    def findWidgetById(self, widget_id):
        # Function to find the widget by ID within the application
        for widget in QApplication.allWidgets():
            if id(widget) == widget_id:
                return widget
        return None
    
    def addImageToList(self, file_path, is_input_area=True):
        if self.count() >= 11:
            QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
            return
        item = QListWidgetItem(self)
        widget = ListWidgetItem(file_path, is_input_area)
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)

class ListWidgetItem(QWidget):
    def __init__(self, icon_path=None, is_input_area=True, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.icon_label = QLabel()

        # 如果 icon_path 不为空，则加载图像
        if icon_path:
            self.original_pixmap = QPixmap(icon_path)  # Store the original pixmap for saving
            if self.original_pixmap.isNull():
                QMessageBox.warning(self, "Loading Error", f"Image cannot be loaded because of file corruption: {icon_path}")
            else:
                scaled_pixmap = self.original_pixmap.scaled(325, 325, Qt.KeepAspectRatio)
                self.icon_label.setPixmap(scaled_pixmap)
            self.layout.addWidget(self.icon_label)

        # 始终添加文本编辑框
        self.text_edit = QTextEdit()
        placeholder_text = "Please enter the question or instruction text" if is_input_area else "Please enter the answer text"
        self.text_edit.setPlaceholderText(placeholder_text)
        self.layout.addWidget(self.text_edit)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.save_directory = self.prompt_for_save_directory()  # Prompt user to select a save directory at startup
        self.current_data_i = 0
        self.judge_results = []
        self.load_judge_results()  # Load existing judge results
        self.load_pk_file()
        self.updata_current_file()
        self.currentBox = None
        self.initUI()
        self.check_and_load_current_data()
        self.update_subtask_id_display()  # Ensure initial display is updated
        self.showMaximized()

    def load_judge_results(self):
        judge_results_path = os.path.join(self.save_directory, SAVE_FILE_NAME)
        if os.path.exists(judge_results_path):
            with open(judge_results_path, 'r', encoding='utf-8') as f:
                self.judge_results = json.load(f)

    def prompt_for_save_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select the root folder as the save path", "")
        if not directory:
            QMessageBox.warning(self, "Attention", "No save path is selected, the save function will not be available!")
            exit()
            sys.exit(app.exec_())
        return directory
    
    def load_pk_file(self):
        with open(os.path.join(self.save_directory, OPENING_DIR, PK_FILE_NAME), 'r') as f:
        # with open(os.path.join(self.save_directory, PK_FILE_NAME), 'r') as f:
            pk_data = json.load(f)
            self.pk_list = []
            for data in pk_data:
                # if data['model_A']['name'] != 'SEED-LLaMA' and data['model_B']['name'] != 'SEED-LLaMA':
                self.pk_list.append(data)
                # # 'Gemini1.5+Flux'
                # if data['model_A']['name'] in ['SEED-X', 'Show-o', 'Gemini1.5+Flux'] and data['model_B']['name'] in ['SEED-X', 'Show-o', 'Gemini1.5+Flux']:
                #     self.pk_list.append(data)
                # if data['model_A']['name'] in ['GPT-4o+DALL-E3', 'Human'] and data['model_B']['name'] in ['GPT-4o+DALL-E3', 'Human']:
                #     self.pk_list.append(data)

    def updata_current_file(self):
        if self.current_data_i < len(self.pk_list):
            self.current_data_uid = self.pk_list[self.current_data_i]['data_id']
            self.current_model_A = self.pk_list[self.current_data_i]['model_A']['name']
            self.current_model_B = self.pk_list[self.current_data_i]['model_B']['name']
            self.current_file_path1 = os.path.join(self.save_directory, f"{self.current_model_A}_output", f'{self.current_data_uid}.jsonl')
            self.current_file_path2 = os.path.join(self.save_directory, f"{self.current_model_B}_output", f'{self.current_data_uid}.jsonl')
            return True
        else:
            return False

    def initUI(self):
        self.setWindowTitle("Interleaved Arena: Pairwise annotation interface for human judges.")

        main_layout = QVBoxLayout(self)

        self.current_data_i_label = QLabel(f"Current Item Index: {self.current_data_i}. The winner is: ")
        main_layout.addWidget(self.current_data_i_label)

        # 初始化输入图像列表，并默认隐藏
        self.input_image_list = DraggableListWidget(self, is_input_area=True)
        self.input_image_list.setVisible(False)
        self.input_image_list.setFixedHeight(200)  # 设置高度为原先的一半

        main_layout.addWidget(QLabel("Input Image Area: The question or instruction corresponding to the input"))

        self.placeholder_text_edit = QTextEdit()
        self.placeholder_text = "no corresponding images"
        self.placeholder_text_edit.setPlaceholderText(self.placeholder_text)
        self.placeholder_text_edit.setFixedHeight(200)  # 设置高度为原先的一半
        main_layout.addWidget(self.placeholder_text_edit)

        # 将图像列表添加到布局中
        main_layout.addWidget(self.input_image_list)

        # 输出图像区1
        main_layout.addWidget(QLabel("A Answer Area: corresponds to the output of model A"))
        self.output_image_list = DraggableListWidget(self, is_input_area=False)
        main_layout.addWidget(self.output_image_list)

        main_layout.addWidget(QLabel("B Answer Area: corresponds to the output of model B"))
        self.output_image_list2 = DraggableListWidget(self, is_input_area=False)
        main_layout.addWidget(self.output_image_list2)

        button_layout = QHBoxLayout()
        button_a_better = QPushButton("A is Better")
        button_a_better.setFixedHeight(50)
        button_a_better.clicked.connect(lambda: self.judge("A"))

        button_tie_a = QPushButton("Tie (A is Slightly Better)")
        button_tie_a.setFixedHeight(50)
        button_tie_a.clicked.connect(lambda: self.judge("Tie(A)"))

        button_tie_b = QPushButton("Tie (B is Slightly Better)")
        button_tie_b.setFixedHeight(50)
        button_tie_b.clicked.connect(lambda: self.judge("Tie(B)"))

        button_b_better = QPushButton("B is Better")
        button_b_better.setFixedHeight(50)
        button_b_better.clicked.connect(lambda: self.judge("B"))

        reset_button = QPushButton("Next")
        reset_button.setFixedHeight(50)  # 设置按钮高度
        reset_button.clicked.connect(self.reset_all)
        
        button_layout.addWidget(button_a_better)
        button_layout.addWidget(button_tie_a)
        button_layout.addWidget(button_tie_b)
        button_layout.addWidget(button_b_better)

        prev_button = QPushButton("Previous")
        prev_button.setFixedHeight(50)  # 设置按钮高度
        prev_button.clicked.connect(self.load_previous)
        # 确保这个按钮添加到合适的布局中
        button_layout.addWidget(prev_button)

        button_layout.addWidget(reset_button)

        main_layout.addLayout(button_layout)

        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.reset_all)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.load_previous)

    def judge(self, result):
        current_data = self.pk_list[self.current_data_i].copy()
        current_data['winner'] = result
        self.judge_results.append(current_data)
        self.save_judge_results()
        self.current_data_i_label.setText(f"Current Item Index: {self.current_data_i}.")
        self.reset_all()

    def save_judge_results(self):
        save_path = os.path.join(self.save_directory, SAVE_FILE_NAME)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.judge_results, f, ensure_ascii=False, indent=4)

    def setCurrentBox(self, box):
        self.currentBox = box  # 设置当前选中的图像框

    def update_subtask_id_display(self):
        """Update the display of the subtask ID."""
        self.current_data_i_label.setText(f"Current Item Index: {self.current_data_i}")
        current_winner = next((item['winner'] for item in self.judge_results if item['data_id'] == self.pk_list[self.current_data_i]['data_id'] and item['model_A'] == self.pk_list[self.current_data_i]['model_A'] and item['model_B'] == self.pk_list[self.current_data_i]['model_B']), "无")
        self.current_data_i_label.setText(f"Current Item Index: {self.current_data_i}. The winner is: {current_winner}")

    def add_image(self, list_widget, is_input_area):
        if list_widget.count() >= 11:
            QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
            return
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select images you wanna upload", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        for file_path in file_paths:
            if list_widget.count() <= 10:
                list_widget.addImageToList(file_path, is_input_area)
                if is_input_area:
                    # 隐藏占位文本框，显示图像列表框
                    self.placeholder_text_edit.setVisible(False)
                    self.input_image_list.setVisible(True)
            else:
                QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
                break

    def reset_all(self):
        # Clear existing files for this subtask
        self.input_image_list.clear()
        self.output_image_list.clear()
        # 显示占位文本框，隐藏图像列表框
        self.placeholder_text_edit.setVisible(True)
        self.placeholder_text_edit.clear()
        self.output_image_list2.clear()
        self.input_image_list.setVisible(False)

        self.current_data_i += 1

        if self.current_data_i >= len(self.pk_list):
            QMessageBox.information(self, "Done","All sampled pairwise data has been annotated!")
            self.current_data_i = len(self.pk_list)
            return
        
        self.updata_current_file()
        self.update_subtask_id_display()
        self.check_and_load_current_data()  # Check and load data for the new subtask_id

    def check_and_load_current_data(self):
        """Check for existing data for the current subtask_id and load it if available."""
        self.input_image_list.clear()
        self.output_image_list.clear()
        self.output_image_list2.clear()

        # 加载jsonl文件列表
        file_path = self.current_file_path1
        
        input_files_loaded = False
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as file:
                json_dict = json.load(file)
                meta_task_id = json_dict.get("meta_task_id")
                subtask_id = json_dict.get("subtask_id")
                data_id = json_dict.get("data_id")
                total_uid = generate_total_uid(meta_task_id, subtask_id, data_id)
                for input_step, input_content in enumerate(json_dict['conversations'][0]['input']):
                    if 'image' in input_content and input_content['image']:
                        if len(input_content['image']) > 0:
                            ori_img_path = input_content['image'].split('/')[-1]
                            file_parts = ori_img_path.split('-')
                            num = file_parts[2].split('.')[0]
                            image_path = os.path.join(self.save_directory, OPENING_DIR, f"{total_uid}-i-{num}.jpg")
                            self.add_image_to_list(image_path, True, self.input_image_list)
                            input_files_loaded = True  # Set flag if any input images are loaded
                if len(json_dict['conversations']) > 1:
                    for output_step, output_content in enumerate(json_dict['conversations'][1]['output']):
                        if 'image' in output_content and output_content['image']:
                            if len(output_content['image'].split('/')[-1]) > 0:
                                image_path = os.path.join(self.save_directory, f"{self.current_model_A}_output", output_content['image'].split('/')[-1])
                                self.add_image_to_list(image_path, False, self.output_image_list)
        else:
            QMessageBox.information(self, "Warning", f"No A data from {file_path}")
            return

        file_path2 = self.current_file_path2
        #加载B模型内容           
        if os.path.exists(file_path2):
            with open(file_path2, encoding='utf-8') as file:
                json_dict = json.load(file)
                if len(json_dict['conversations']) > 1:
                    for output_step, output_content in enumerate(json_dict['conversations'][1]['output']):
                        if 'image' in output_content and output_content['image']:
                            if len(output_content['image'].split('/')[-1]) > 0:
                                image_path = os.path.join(self.save_directory, f"{self.current_model_B}_output", output_content['image'].split('/')[-1])
                                self.add_image_to_list(image_path, False, self.output_image_list2)
        else:
            QMessageBox.information(self, "Warning", f"No B data from {file_path}")
            return

        # Adjust visibility based on loaded content
        self.placeholder_text_edit.setVisible(not input_files_loaded)
        self.input_image_list.setVisible(input_files_loaded)

        # Load and distribute text if available
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        with open(file_path2, 'r', encoding='utf-8') as file:
            content2 = json.load(file)
        self.parse_and_load_json(content, content2, input_files_loaded)

    def load_previous(self):
        if self.current_data_i > 0:
            self.current_data_i -= 1
            self.updata_current_file()
        else:
            QMessageBox.warning(self, "Warning","This is already the first item")
            return

        self.input_image_list.clear()
        self.output_image_list.clear()
        self.output_image_list2.clear()
        self.placeholder_text_edit.clear()

        # 加载jsonl文件列表
        file_path = self.current_file_path1
        
        input_files_loaded = False
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as file:
                json_dict = json.load(file)
                meta_task_id = json_dict.get("meta_task_id")
                subtask_id = json_dict.get("subtask_id")
                data_id = json_dict.get("data_id")
                total_uid = generate_total_uid(meta_task_id, subtask_id, data_id)
                for input_step, input_content in enumerate(json_dict['conversations'][0]['input']):
                    if 'image' in input_content and input_content['image']:
                        if len(input_content['image']) > 0:
                            ori_img_path = input_content['image'].split('/')[-1]
                            file_parts = ori_img_path.split('-')
                            num = file_parts[2].split('.')[0]
                            image_path = os.path.join(self.save_directory, OPENING_DIR, f"{total_uid}-i-{num}.jpg")
                            self.add_image_to_list(image_path, True, self.input_image_list)
                            input_files_loaded = True  # Set flag if any input images are loaded
                if len(json_dict['conversations']) > 1:
                    for output_step, output_content in enumerate(json_dict['conversations'][1]['output']):
                        if 'image' in output_content and output_content['image']:
                            if len(output_content['image'].split('/')[-1]) > 0:
                                image_path = os.path.join(self.save_directory, f"{self.current_model_A}_output", output_content['image'].split('/')[-1])
                                self.add_image_to_list(image_path, False, self.output_image_list)
        else:
            QMessageBox.information(self, "Warning", f"No A data from {file_path}")
            return

        file_path2 = self.current_file_path2
        #加载B模型内容           
        if os.path.exists(file_path2):
            with open(file_path2, encoding='utf-8') as file:
                json_dict = json.load(file)
                if len(json_dict['conversations']) > 1:
                    for output_step, output_content in enumerate(json_dict['conversations'][1]['output']):
                        if 'image' in output_content and output_content['image']:
                            if len(output_content['image'].split('/')[-1]) > 0:
                                image_path = os.path.join(self.save_directory, f"{self.current_model_B}_output", output_content['image'].split('/')[-1])
                                self.add_image_to_list(image_path, False, self.output_image_list2)
        else:
            QMessageBox.information(self, "Warning", f"No B data from {file_path2}")
            return

        # Adjust visibility based on loaded content
        self.placeholder_text_edit.setVisible(not input_files_loaded)
        self.input_image_list.setVisible(input_files_loaded)

        # Load and distribute text if available
        with open(file_path, 'r', encoding='utf-8') as file:
            content = json.load(file)
        with open(file_path2, 'r', encoding='utf-8') as file:
            content2 = json.load(file)
        self.parse_and_load_json(content, content2, input_files_loaded)
        
        self.update_subtask_id_display()
            
    def parse_and_load_json(self, content, content2, input_files_loaded):
        
        input_list = []
        output_list_A = []  # A模型的输出
        output_list_B = []  # B模型的输出

        for input_step, input_content in enumerate(content['conversations'][0]['input']):
            input_list.append(input_content['text'])
        # input_list[0] = 'Please show me the steps of the tutorial with interleaved images and text: <BEGIN> ' + input_list[0]
        # input_list[0] = '<BEGIN> ' + input_list[0] + ' Also give interleaved text explanations for generated images.'

        if not input_files_loaded:
            self.placeholder_text_edit.setPlainText(input_list[0].strip())
        else:
            for i in range(len(input_list)):
                input_content = input_list[i].strip()
                self.add_text_to_list(self.input_image_list, input_content, i)

        if len(content['conversations']) > 1:
            for output_step, output_content in enumerate(content['conversations'][1]['output']):
                output_list_A.append(output_content['text'])

            for i in range(len(output_list_A)):
                output_content = output_list_A[i].strip()
                self.add_text_to_list(self.output_image_list, output_content, i)
                # self.add_text_to_list(self.output_image_list, self.current_model_A+output_content, i)
        
        if len(content2['conversations']) > 1:
            for output_step, output_content in enumerate(content2['conversations'][1]['output']):
                output_list_B.append(output_content['text'])

            for i in range(len(output_list_B)):
                output_content = output_list_B[i].strip()
                self.add_text_to_list(self.output_image_list2, output_content, i)
                # self.add_text_to_list(self.output_image_list2, self.current_model_B+output_content, i)

    def add_image_to_list(self, file_path, is_input_area, list_widget, text_only=False):
        print(file_path)
        if list_widget.count() >= 11:
            QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
            return  # 避免超过10张图像
        if is_input_area:
            # 隐藏占位文本框，显示图像列表框
            self.placeholder_text_edit.setVisible(False)
        item = QListWidgetItem(list_widget)
        widget = ListWidgetItem(file_path if not text_only else None, is_input_area)
        item.setSizeHint(widget.sizeHint())
        list_widget.addItem(item)
        list_widget.setItemWidget(item, widget)

    def add_text_to_list(self, list_widget, text, index):
        if index >= list_widget.count():
            item = QListWidgetItem(list_widget)
            widget = ListWidgetItem(None, is_input_area=False)
            widget.text_edit.setPlainText(text)  # 在添加新项时设置文本内容
            item.setSizeHint(widget.sizeHint())
            list_widget.addItem(item)
            list_widget.setItemWidget(item, widget)
        elif index < list_widget.count():
            item = list_widget.item(index)
            widget = list_widget.itemWidget(item)
            widget.text_edit.setPlainText(text)

def generate_total_uid(meta_task_id, subtask_id, data_id):
    # Ensure meta_task_id is 2 digits, subtask_id is 2 digits, and data_id is 3 digits
    return f'{int(meta_task_id):02}{int(subtask_id):02}{int(data_id):03}'

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())