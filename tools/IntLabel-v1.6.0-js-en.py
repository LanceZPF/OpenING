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

        if is_input_area:
            # 文本框初始化
            self.placeholder_text_edit = QTextEdit()
            self.placeholder_text = "If the input does not have the image, please do not upload the input image, directly enter the text of question or instruction in this box."
            self.placeholder_text_edit.setPlaceholderText(self.placeholder_text)
            self.parent().layout().addWidget(self.placeholder_text_edit)  # 这假设了 parent 拥有 QVBoxLayout
    
    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if item:
            contextMenu = QMenu(self)
            deleteAction = contextMenu.addAction("Delete Image")
            action = contextMenu.exec_(self.mapToGlobal(event.pos()))
            if action == deleteAction:
                row = self.row(item)
                self.takeItem(row)
                if self.is_input_area:
                    self.updatePlaceholderVisibility()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            self.parent().setCurrentBox(self)
            self.draggingItem = self.itemAt(event.pos())
            if self.draggingItem:
                drag = QDrag(self)
                mimedata = QMimeData()

                # Encode widget identity, item index, and associated text in MIME data
                associated_text = self.itemWidget(self.draggingItem).text_edit.toPlainText()
                item_index = self.row(self.draggingItem)
                mimedata.setText(f"{id(self)}:{item_index}:{associated_text}")
                pixmap = self.itemWidget(self.draggingItem).icon_label.pixmap()
                mimedata.setImageData(pixmap.toImage())
                drag.setMimeData(mimedata)
                drag.setPixmap(pixmap)
                drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        if event.mimeData().hasImage() or event.mimeData().hasText():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasImage() or event.mimeData().hasText():
            event.setDropAction(Qt.MoveAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage():
            target_item = self.itemAt(event.pos())
            if target_item and self.draggingItem and target_item != self.draggingItem:
                try:
                
                    target_widget = self.itemWidget(target_item)
                    target_original_pixmap = target_widget.original_pixmap

                    source_widget = self.itemWidget(self.draggingItem)
                    source_original_pixmap = source_widget.original_pixmap

                    # Swap the original pixmaps
                    target_widget.icon_label.setPixmap(source_original_pixmap.scaled(325, 325, Qt.KeepAspectRatio))
                    source_widget.icon_label.setPixmap(target_original_pixmap.scaled(325, 325, Qt.KeepAspectRatio))

                    temp_pixmap = target_original_pixmap

                    target_widget.original_pixmap = source_original_pixmap
                    source_widget.original_pixmap = temp_pixmap

                except Exception as e:
                    print(e)
                    print('oops')
                    return
                
            elif event.mimeData().hasText():  # Check if this includes widget ID and item index
                source_data = event.mimeData().text().split(':')
                source_widget_id = int(source_data[0])
                source_index = int(source_data[1])
                source_text = ':'.join(source_data[2:])

                source_widget = self.findWidgetById(source_widget_id)
                if source_widget:
                    item = source_widget.item(source_index)
                    if item:
                        source_widget_item = source_widget.itemWidget(item)
                        if source_widget_item and hasattr(source_widget_item, 'original_pixmap'):
                            original_pixmap = source_widget_item.original_pixmap

                            widget = ListWidgetItem2(original_pixmap, is_input_area=self.is_input_area)
                            widget.text_edit.setPlainText(source_text)

                            new_item = QListWidgetItem()
                            new_item.setSizeHint(widget.sizeHint())
                            self.addItem(new_item)
                            self.setItemWidget(new_item, widget)
                            event.acceptProposedAction()
                        else:
                            print("Error: No valid widget or 'original_pixmap' not found.")
                        other_item = source_widget.takeItem(source_index)
                    else:
                        print("Error: Item not found or already removed.")
                else:
                    print("Error: Source widget not found.")

        elif event.mimeData().hasUrls():
            # Handle drops from external applications
            files = [u.toLocalFile() for u in event.mimeData().urls()]
            for file_path in files:
                if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.addImageToList(file_path, self.is_input_area)
            event.acceptProposedAction()

    def findWidgetById(self, widget_id):
        # Function to find the widget by ID within the application
        for widget in QApplication.allWidgets():
            if id(widget) == widget_id:
                return widget
        return None
    
    def addImageToList(self, file_path, is_input_area=True):
        if self.count() >= 10:
            QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
            return
        item = QListWidgetItem(self)
        widget = ListWidgetItem(file_path, is_input_area)
        item.setSizeHint(widget.sizeHint())
        self.addItem(item)
        self.setItemWidget(item, widget)
        if is_input_area:
            self.updatePlaceholderVisibility()

    def updatePlaceholderVisibility(self):
        return
        # 根据列表中的图像数量更新文本框的显示状态
        # if self.count() > 0:
        #     self.placeholder_text_edit.setVisible(False)
        # else:
        #     self.placeholder_text_edit.setVisible(True)

class ListWidgetItem(QWidget):
    def __init__(self, icon_path, is_input_area=True, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.icon_label = QLabel()
        self.original_pixmap = QPixmap(icon_path)  # Store the original pixmap for saving
        if self.original_pixmap.isNull():
            QMessageBox.warning(self, "Loading Error", "Image cannot be loaded:" + icon_path)
        else:
            scaled_pixmap = self.original_pixmap.scaled(325, 325, Qt.KeepAspectRatio)
            self.icon_label.setPixmap(scaled_pixmap)
        self.text_edit = QTextEdit()
        placeholder_text = "Please enter the question or instruction text" if is_input_area else "Please enter the answer text"
        self.text_edit.setPlaceholderText(placeholder_text)
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text_edit)

class ListWidgetItem2(QWidget):
    def __init__(self, pixmap, is_input_area=False, parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.icon_label = QLabel()
        self.original_pixmap = pixmap  # Store the original pixmap here
        if pixmap.isNull():
            QMessageBox.warning(self, "Loading Error", "No image loaded")
        else:
            self.icon_label.setPixmap(pixmap.scaled(325, 325, Qt.KeepAspectRatio))
        self.text_edit = QTextEdit()
        # 根据是输入还是输出区设置不同的占位文本
        if is_input_area:
            placeholder_text = "Please enter the question or instruction text"
        else:
            placeholder_text = "Please enter the answer text"
        self.text_edit.setPlaceholderText(placeholder_text)
        self.layout.addWidget(self.icon_label)
        self.layout.addWidget(self.text_edit)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.save_directory = self.prompt_for_save_directory()  # Prompt user to select a save directory at startup
        self.subtask_id=1
        self.currentBox = None
        self.initUI()
        self.set_subtask_id()
        self.showMaximized()

    def prompt_for_save_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select the path to save the Task folder that you are annotating", "")
        if not directory:
            QMessageBox.warning(self, "Attention", "No save path is selected, the save function will not be available!")
            sys.exit(app.exec_())
        return directory
    
    def set_subtask_id(self):
        if not self.save_directory:
            self.subtask_id = 1  # Default value if no directory is set
            return
        
        # Initialize subtask_id to 0 and check for existing .jsonl files
        self.subtask_id = 1

        # THIS ONE
        self.check_and_load_current_data()

        # OROROR

        # if os.path.exists(os.path.join(self.save_directory, "1.jsonl")):
        #     while True:
        #         file_path = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        #         if os.path.exists(file_path):
        #             break
        #         self.subtask_id += 1
        # while True:
        #     file_path = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        #     if not os.path.exists(file_path):
        #         break
        #     self.subtask_id += 1

    def initUI(self):
        self.setWindowTitle("IntLabel: An Interleaved Image-Text Annotation Tool")

        main_layout = QVBoxLayout(self)

        self.subtask_id_label = QLabel(f"Current Index: {self.subtask_id} in the Processing Task.")
        main_layout.addWidget(self.subtask_id_label)

        # 初始化输入图像列表，并默认隐藏
        self.input_image_list = DraggableListWidget(self, is_input_area=True)

        main_layout.addWidget(QLabel("Input Image Area: corresponding to question or instruction"))

        # 将图像列表添加到布局中
        main_layout.addWidget(self.input_image_list)

        # 输出图像区
        main_layout.addWidget(QLabel("Output Image Area: corresponding to answers"))
        self.output_image_list = DraggableListWidget(self, is_input_area=False)
        main_layout.addWidget(self.output_image_list)

        button_layout = QHBoxLayout()
        add_input_image_button = QPushButton("Add Input Images")
        add_input_image_button.setFixedHeight(50)  # 设置按钮高度
        add_input_image_button.clicked.connect(lambda: self.add_image(self.input_image_list, True))
        
        add_output_image_button = QPushButton("Add Output Images")
        add_output_image_button.setFixedHeight(50)  # 设置按钮高度
        add_output_image_button.clicked.connect(lambda: self.add_image(self.output_image_list, False))
        
        reset_button = QPushButton("Next")
        reset_button.setFixedHeight(50)  # 设置按钮高度
        reset_button.clicked.connect(self.reset_all)
        
        button_layout.addWidget(add_input_image_button)
        button_layout.addWidget(add_output_image_button)

        prev_button = QPushButton("Previous")
        prev_button.setFixedHeight(50)  # 设置按钮高度
        prev_button.clicked.connect(self.load_previous)
        # 确保这个按钮添加到合适的布局中
        button_layout.addWidget(prev_button)

        button_layout.addWidget(reset_button)

        main_layout.addLayout(button_layout)

        QShortcut(QKeySequence("Ctrl+V"), self, activated=self.pasteImage)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.reset_all)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.load_previous)

    def pasteImage(self):
        clipboard = QApplication.clipboard()
        pixmap = clipboard.pixmap()

        if pixmap.isNull():
            QMessageBox.warning(self, "Warning", 'No pictures on the clipboard')
            return  # 避免超过10张图像

        if self.currentBox == self.input_image_list or self.currentBox == self.output_image_list:
            list_widget = self.currentBox
            if list_widget.count() >= 10:
                QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
                return  # 避免超过10张图像

            item = QListWidgetItem(list_widget)
            widget = ListWidgetItem2(pixmap, is_input_area=False)
            item.setSizeHint(widget.sizeHint())
            list_widget.addItem(item)
            list_widget.setItemWidget(item, widget)

            if self.currentBox == self.input_image_list:
                self.input_image_list.placeholder_text_edit.setVisible(False)
        else:
            QMessageBox.warning(self, "Warning", "Please click the selected Input or output image box before copying the picture")
            return  # 避免超过10张图像

    def setCurrentBox(self, box):
        self.currentBox = box  # 设置当前选中的图像框

    def update_subtask_id_display(self):
        """Update the display of the subtask ID."""
        self.subtask_id_label.setText(f"Current Index: {self.subtask_id} in the Processing Task.")

    def add_image(self, list_widget, is_input_area):
        if list_widget.count() >= 10:
            QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
            return
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select the images you wanna upload", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        for file_path in file_paths:
            if list_widget.count() < 10:
                list_widget.addImageToList(file_path, is_input_area)
                if is_input_area:
                    # 隐藏占位文本框，显示图像列表框
                    self.input_image_list.placeholder_text_edit.setVisible(False)
            else:
                QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
                break

    def save_contents(self):
        if not self.save_directory:
            QMessageBox.warning(self, "Save Error", "No save path was specified, operation could not continue")
            sys.exit(app.exec_())
        
        # Check for empty text fields or no output images
        if self.output_image_list.count() == 0:
            QMessageBox.warning(self, "Save Error", "The output image area cannot be empty")
            return False

        temp_text = self.input_image_list.placeholder_text_edit.toPlainText().strip()
        if self.input_image_list.count() == 0 and not temp_text:
            QMessageBox.warning(self, "Save Error", "The input text cannot be empty")
            return False

        # Prepare the data structure according to the specified format
        data = {
            "meta_task_id": 0,  # Adjust these as necessary for your application context
            "meta_task_name": "temp",
            "subtask_id": 0,
            "subtask_name": self.save_directory.split('/')[-1],
            "data_id": self.subtask_id,  # This should be uniquely generated or incremented as needed
            "conversations": []
        }

        input_conversation = {"input": []}
        output_conversation = {"output": []}

        if self.input_image_list.count() == 0 and temp_text:  # No images, only text
            input_conversation["input"].append({"image": None, "text": temp_text})
        else:
            # Handle inputs with images and text
            for index in range(self.input_image_list.count()):
                item = self.input_image_list.item(index)
                widget = self.input_image_list.itemWidget(item)
                text = widget.text_edit.toPlainText().strip()
                # if not text:
                #     QMessageBox.warning(self, "保存失败", "所有文本框都必须填写内容")
                #     return False
                original_pixmap = widget.original_pixmap
                file_name = f"{self.subtask_id}-i-{index + 1}.jpg"
                original_pixmap.save(os.path.join(self.save_directory, file_name), 'JPEG', 100)
                input_conversation["input"].append({"image": f"./{file_name}", "text": text})

        # Handle outputs with images and text
        for index in range(self.output_image_list.count()):
            item = self.output_image_list.item(index)
            widget = self.output_image_list.itemWidget(item)
            text = widget.text_edit.toPlainText().strip()
            if not text:
                QMessageBox.warning(self, "Save Error", "There is blank in the output text box, please fill in all the output text boxes!")
                return False
            original_pixmap = widget.original_pixmap
            file_name = f"{self.subtask_id}-o-{index + 1}.jpg"
            original_pixmap.save(os.path.join(self.save_directory, file_name), 'JPEG', 100)
            output_conversation["output"].append({"image": f"./{file_name}", "text": text})

        # Append conversations to the data structure
        if input_conversation["input"]:
            data["conversations"].append(input_conversation)
        if output_conversation["output"]:
            data["conversations"].append(output_conversation)

        # Save the data to a JSONL file
        jsonl_path = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

        return True

    def reset_all(self):
        if not self.save_contents():  # Only clear if saving was successful
            return
        # Clear existing files for this subtask
        self.clear_subtask_files()
        self.save_contents()
        self.input_image_list.clear()
        self.output_image_list.clear()
        # 显示占位文本框，隐藏图像列表框
        self.input_image_list.placeholder_text_edit.setVisible(True)
        self.input_image_list.placeholder_text_edit.clear()
        self.input_image_list.setVisible(True)

        self.subtask_id += 1
        self.update_subtask_id_display()
        self.check_and_load_current_data()  # Check and load data for the new subtask_id

    def clear_subtask_files(self):
        """
        Deletes all files related to the current subtask in the save directory,
        including image files and the associated text file.
        """
        pattern = f"{self.subtask_id}-*"
        file_paths = glob.glob(os.path.join(self.save_directory, pattern))
        for file_path in file_paths:
            os.remove(file_path)
        print(f"All files for subtask {self.subtask_id} have been deleted.")

    def check_and_load_current_data(self):
        """Check for existing data for the current subtask_id and load it if available."""
        self.input_image_list.clear()
        self.output_image_list.clear()

        # 加载jsonl文件列表
        file_path = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        
        input_files_loaded = False
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as file:
                json_dict = json.load(file)
                for input_step, input_content in enumerate(json_dict['conversations'][0]['input']):
                    if 'image' in input_content and input_content['image']:
                        if len(input_content['image'].split('/')[-1]) > 0:
                            image_path = os.path.join(self.save_directory, input_content['image'].split('/')[-1])
                            self.add_image_to_list(image_path, True, self.input_image_list)
                            input_files_loaded = True  # Set flag if any input images are loaded
                if len(json_dict['conversations']) > 1:
                    for output_step, output_content in enumerate(json_dict['conversations'][1]['output']):
                        if len(output_content['image'].split('/')[-1]) > 0:
                            image_path = os.path.join(self.save_directory, output_content['image'].split('/')[-1])
                            self.add_image_to_list(image_path, False, self.output_image_list)
        else:
            return

        # Adjust visibility based on loaded content
        self.input_image_list.placeholder_text_edit.setVisible(True)
        # self.input_image_list.placeholder_text_edit.setVisible(not input_files_loaded)
        # self.input_image_list.setVisible(input_files_loaded)

        # Load and distribute text if available
        text_file = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as file:
                content = json.load(file)
                self.parse_and_load_json(content, input_files_loaded)
        else:
            # Clear text if no file for the current subtask_id
            self.input_image_list.placeholder_text_edit.setPlainText("")

    def flush_current_text(self, mode, input_files_loaded, text, index):
        if mode == 'input' and input_files_loaded:
            self.add_text_to_list(self.input_image_list, text.strip(), index)
        elif mode == 'output':
            self.add_text_to_list(self.output_image_list, text.strip(), index)
        elif mode == 'input' and not input_files_loaded:
            self.input_image_list.placeholder_text_edit.setPlainText(text.strip())

    def load_previous(self):
        if self.subtask_id > 1:
            self.subtask_id -= 1
        else:
            QMessageBox.warning(self, "Warning","This is already the first item")
            return

        self.input_image_list.clear()
        self.output_image_list.clear()
        self.input_image_list.placeholder_text_edit.clear()

        # 加载jsonl文件列表
        file_path = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        
        input_files_loaded = False
        if os.path.exists(file_path):
            with open(file_path, encoding='utf-8') as file:
                json_dict = json.load(file)
                for input_step, input_content in enumerate(json_dict['conversations'][0]['input']):
                    if 'image' in input_content and input_content['image']:
                        if len(input_content['image'].split('/')[-1]) > 0:
                            image_path = os.path.join(self.save_directory, input_content['image'].split('/')[-1])
                            self.add_image_to_list(image_path, True, self.input_image_list)
                            input_files_loaded = True  # Set flag if any input images are loaded
                if len(json_dict['conversations']) > 1:
                    for output_step, output_content in enumerate(json_dict['conversations'][1]['output']):
                        if len(output_content['image'].split('/')[-1]) > 0:
                            image_path = os.path.join(self.save_directory, output_content['image'].split('/')[-1])
                            self.add_image_to_list(image_path, False, self.output_image_list)
        else:
            return

        # Adjust visibility based on loaded content
        self.input_image_list.placeholder_text_edit.setVisible(True)
        # self.input_image_list.placeholder_text_edit.setVisible(not input_files_loaded)
        # self.input_image_list.setVisible(input_files_loaded)
        self.input_image_list.setVisible(True)

        # Load and distribute text
        text_file = os.path.join(self.save_directory, f"{self.subtask_id}.jsonl")
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as file:
                content = json.load(file)
                self.parse_and_load_json(content, input_files_loaded)
        
        self.update_subtask_id_display()
            
    def parse_and_load_json(self, content, input_files_loaded):
        
        input_list = []
        output_list = []

        for input_step, input_content in enumerate(content['conversations'][0]['input']):
            input_list.append(input_content['text'])
        # input_list[0] = 'Please show me the steps of the tutorial with interleaved images and text: <BEGIN> ' + input_list[0]
        # input_list[0] = '<BEGIN> ' + input_list[0] + ' Also give interleaved text explanations for generated images.'

        if not input_files_loaded:
            self.input_image_list.placeholder_text_edit.setPlainText(input_list[0].strip())
        else:
            for i in range(len(input_list)):
                input_content = input_list[i].strip()
                self.add_text_to_list(self.input_image_list, input_content, i)

        if len(content['conversations']) > 1:
            for output_step, output_content in enumerate(content['conversations'][1]['output']):
                output_list.append(output_content['text'])

            for i in range(len(output_list)):
                output_content = output_list[i].strip()
                # output_content = output_content.replace('It\'s started', 'It is time to start eating')
                self.add_text_to_list(self.output_image_list, output_content, i)

    def add_image_to_list(self, file_path, is_input_area, list_widget):
        if list_widget.count() >= 10:
            QMessageBox.warning(self, "Limit", "You can upload a maximum of 10 images")
            return  # 避免超过10张图像
        
        if is_input_area:
            # 隐藏占位文本框，显示图像列表框
            self.input_image_list.placeholder_text_edit.setVisible(False)
        item = QListWidgetItem(list_widget)
        widget = ListWidgetItem(file_path, is_input_area)
        item.setSizeHint(widget.sizeHint())
        list_widget.addItem(item)
        list_widget.setItemWidget(item, widget)

    def add_text_to_list(self, list_widget, text, index):
        if index < list_widget.count():
            item = list_widget.item(index)
            widget = list_widget.itemWidget(item)
            widget.text_edit.setPlainText(text)

def sort_key(filename):
    filename = filename.split('\\')[-1]
    match = re.match(r"(\d+)\.jsonl", filename)
    if match:
        return (int(match.group(1)), )
    return (float('inf'), float('inf'))  # 如果文件名不匹配，将它们放在列表的末尾

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())