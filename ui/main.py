import sys
import os
from PySide6.QtWidgets import *
from PySide6.QtCore import *
from PySide6.QtGui import *


class ParkinsonDiagnosisUI(QDialog):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('帕金森病语音诊断系统')
        self.setFixedSize(1000, 700)

        # 设置整体样式
        self.setup_styles()

        # 创建主布局
        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 添加各个功能模块
        main_layout.addWidget(self.create_title_section())
        main_layout.addWidget(self.create_control_section())
        main_layout.addWidget(self.create_content_section())

        self.setLayout(main_layout)

        # 连接信号槽
        self.connect_signals()

    def setup_styles(self):
        """设置样式表，方便修改"""
        self.setStyleSheet("""
            QDialog { background-color: #f5f7fa; }
            QGroupBox {
                font-weight: bold; font-size: 12px;
                border: 2px solid #dcdfe6; border-radius: 8px;
                margin-top: 1ex; padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px;
                padding: 0 5px 0 5px; color: #409eff;
            }
            QPushButton {
                background-color: #409eff; color: white;
                border: none; border-radius: 4px;
                padding: 8px 15px; font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover { background-color: #66b1ff; }
            QPushButton:pressed { background-color: #3a8ee6; }
            QPushButton:disabled {
                background-color: #c0c4cc; color: #909399;
            }
            QProgressBar {
                border: 1px solid #dcdfe6; border-radius: 4px;
                background-color: #f5f7fa; text-align: center;
            }
            QProgressBar::chunk {
                background-color: #67c23a; border-radius: 3px;
            }
            QListWidget, QTextEdit {
                border: 1px solid #dcdfe6; border-radius: 4px;
                background-color: white;
            }
            QLabel { color: #606266; }
        """)

    def create_title_section(self):
        """创建标题区域 - 您可以在这里添加副标题或其他标题元素"""
        title_group = QGroupBox()
        title_group.setStyleSheet("QGroupBox { border: none; }")
        title_layout = QVBoxLayout()

        # 主标题
        self.det_Title_Label = QLabel('帕金森病语音诊断系统')
        self.det_Title_Label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(18)
        title_font.setBold(True)
        self.det_Title_Label.setFont(title_font)
        self.det_Title_Label.setStyleSheet("color: #303133; padding: 10px;")

        # 副标题（您可以添加或修改）
        self.det_Subtitle_Label = QLabel('基于语音分析的帕金森病早期筛查系统')
        self.det_Subtitle_Label.setAlignment(Qt.AlignCenter)
        self.det_Subtitle_Label.setStyleSheet("color: #909399; padding: 5px;")

        title_layout.addWidget(self.det_Title_Label)
        title_layout.addWidget(self.det_Subtitle_Label)
        title_group.setLayout(title_layout)

        return title_group

    def create_control_section(self):
        """创建控制按钮区域 - 您可以在这里添加更多按钮"""
        control_group = QGroupBox("操作控制")
        control_layout = QVBoxLayout()

        # 第一行按钮
        button_row1 = self.create_button_row1()
        # 第二行按钮
        button_row2 = self.create_button_row2()
        # 第三行按钮（预留扩展）
        button_row3 = self.create_button_row3()

        control_layout.addLayout(button_row1)
        control_layout.addLayout(button_row2)
        control_layout.addLayout(button_row3)
        control_group.setLayout(control_layout)

        return control_group

    def create_button_row1(self):
        """创建第一行按钮 - 您可以修改或添加按钮"""
        layout = QHBoxLayout()

        self.det_Upload_File_Button = QPushButton('📁 上传文件')
        self.det_Upload_Folder_Button = QPushButton('📂 上传文件夹')
        self.det_Process_Button = QPushButton('🔧 特征提取')
        self.det_Predict_Button = QPushButton('🔍 开始诊断')

        layout.addWidget(self.det_Upload_File_Button)
        layout.addWidget(self.det_Upload_Folder_Button)
        layout.addWidget(self.det_Process_Button)
        layout.addWidget(self.det_Predict_Button)
        layout.addStretch()

        return layout

    def create_button_row2(self):
        """创建第二行按钮 - 您可以修改或添加按钮"""
        layout = QHBoxLayout()

        self.det_Train_Button = QPushButton('🤖 训练模型')
        self.det_Record_Button = QPushButton('🎤 实时录音')
        self.det_Clear_Button = QPushButton('🗑️ 清除数据')
        self.det_Export_Button = QPushButton('💾 导出结果')

        layout.addWidget(self.det_Train_Button)
        layout.addWidget(self.det_Record_Button)
        layout.addWidget(self.det_Clear_Button)
        layout.addWidget(self.det_Export_Button)
        layout.addStretch()

        return layout

    def create_button_row3(self):
        """创建第三行按钮 - 这是预留的扩展行，您可以添加新按钮"""
        layout = QHBoxLayout()

        # 这里预留扩展按钮的位置
        # 例如：
        # self.det_New_Button1 = QPushButton('新功能1')
        # self.det_New_Button2 = QPushButton('新功能2')
        # layout.addWidget(self.det_New_Button1)
        # layout.addWidget(self.det_New_Button2)

        layout.addStretch()  # 暂时只有拉伸，等您添加按钮

        return layout

    def create_content_section(self):
        """创建主要内容区域 - 您可以在这里添加新的显示组件"""
        content_splitter = QSplitter(Qt.Horizontal)

        # 左侧文件区域
        left_widget = self.create_left_panel()
        # 右侧状态区域
        right_widget = self.create_right_panel()

        content_splitter.addWidget(left_widget)
        content_splitter.addWidget(right_widget)
        content_splitter.setSizes([400, 600])

        return content_splitter

    def create_left_panel(self):
        """创建左侧面板 - 您可以添加文件相关的其他组件"""
        left_widget = QFrame()
        left_layout = QVBoxLayout()

        # 文件列表组
        file_group = QGroupBox("文件列表")
        file_layout = QVBoxLayout()

        self.det_File_List_Widget = QListWidget()
        self.det_File_List_Widget.setMinimumHeight(200)
        file_layout.addWidget(self.det_File_List_Widget)

        # 文件操作按钮（您可以添加更多文件操作）
        file_buttons_layout = QHBoxLayout()
        self.det_Remove_File_Button = QPushButton('移除选中')
        self.det_Clear_Files_Button = QPushButton('清空列表')
        file_buttons_layout.addWidget(self.det_Remove_File_Button)
        file_buttons_layout.addWidget(self.det_Clear_Files_Button)
        file_buttons_layout.addStretch()
        file_layout.addLayout(file_buttons_layout)

        # 文件统计信息
        file_info_layout = QHBoxLayout()
        self.det_File_Count_Label = QLabel('文件数: 0')
        self.det_File_Size_Label = QLabel('总大小: 0 MB')
        file_info_layout.addWidget(self.det_File_Count_Label)
        file_info_layout.addStretch()
        file_info_layout.addWidget(self.det_File_Size_Label)
        file_layout.addLayout(file_info_layout)

        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)

        # 这里可以添加其他左侧组件
        # 例如：数据统计图表、文件预览等
        # left_layout.addWidget(self.create_additional_left_widget())

        left_widget.setLayout(left_layout)
        return left_widget

    def create_right_panel(self):
        """创建右侧面板 - 您可以添加状态和结果显示的其他组件"""
        right_widget = QFrame()
        right_layout = QVBoxLayout()

        # 状态显示组
        status_group = self.create_status_group()
        # 结果显示组
        result_group = self.create_result_group()
        # 详细信息组（扩展）
        detail_group = self.create_detail_group()

        right_layout.addWidget(status_group)
        right_layout.addWidget(result_group)
        right_layout.addWidget(detail_group)

        right_widget.setLayout(right_layout)
        return right_widget

    def create_status_group(self):
        """创建状态显示组 - 您可以添加更多状态信息"""
        status_group = QGroupBox("诊断状态")
        status_layout = QVBoxLayout()

        # 基本状态信息
        status_info_layout = QHBoxLayout()
        self.det_Status_Label = QLabel('🟢 状态: 就绪')
        self.det_Progress_Label = QLabel('进度: 0%')
        status_info_layout.addWidget(self.det_Status_Label)
        status_info_layout.addStretch()
        status_info_layout.addWidget(self.det_Progress_Label)

        # 概率显示
        prob_layout = self.create_probability_display()

        status_layout.addLayout(status_info_layout)
        status_layout.addLayout(prob_layout)
        status_group.setLayout(status_layout)

        return status_group

    def create_probability_display(self):
        """创建概率显示区域 - 您可以修改显示方式"""
        layout = QVBoxLayout()

        prob_title_layout = QHBoxLayout()
        prob_title_layout.addWidget(QLabel('📊 帕金森病预测概率:'))
        prob_title_layout.addStretch()

        prob_display_layout = QHBoxLayout()
        self.det_Probability_Bar = QProgressBar()
        self.det_Probability_Bar.setMinimum(0)
        self.det_Probability_Bar.setMaximum(100)
        self.det_Probability_Bar.setValue(0)
        self.det_Probability_Bar.setFormat("%p%")

        self.det_Probability_Label = QLabel('0%')
        self.det_Probability_Label.setStyleSheet("""
            font-size: 24px; 
            font-weight: bold; 
            color: #409eff;
            min-width: 80px;
        """)

        prob_display_layout.addWidget(self.det_Probability_Bar, 4)
        prob_display_layout.addWidget(self.det_Probability_Label, 1)

        layout.addLayout(prob_title_layout)
        layout.addLayout(prob_display_layout)

        return layout

    def create_result_group(self):
        """创建结果显示组 - 您可以添加更多结果显示方式"""
        result_group = QGroupBox("诊断结果")
        result_layout = QVBoxLayout()

        self.det_Result_Text_Edit = QTextEdit()
        self.det_Result_Text_Edit.setMaximumHeight(120)
        self.det_Result_Text_Edit.setPlaceholderText("诊断结果将显示在这里...")
        result_layout.addWidget(self.det_Result_Text_Edit)

        result_group.setLayout(result_layout)
        return result_group

    def create_detail_group(self):
        """创建详细信息组 - 这是扩展区域，您可以添加详细数据"""
        detail_group = QGroupBox("详细信息")
        detail_layout = QVBoxLayout()

        # 这里可以添加详细数据展示
        # 例如：特征值表格、波形图等
        self.det_Detail_Text_Edit = QTextEdit()
        self.det_Detail_Text_Edit.setMaximumHeight(100)
        self.det_Detail_Text_Edit.setPlaceholderText("详细分析数据将显示在这里...")
        detail_layout.addWidget(self.det_Detail_Text_Edit)

        detail_group.setLayout(detail_layout)
        return detail_group

    def connect_signals(self):
        """连接信号槽 - 您可以在这里添加新按钮的信号连接"""
        # 文件操作按钮
        self.det_Upload_File_Button.clicked.connect(self.upload_audio_file)
        self.det_Upload_Folder_Button.clicked.connect(self.upload_audio_folder)
        self.det_Remove_File_Button.clicked.connect(self.remove_selected_file)
        self.det_Clear_Files_Button.clicked.connect(self.clear_file_list)

        # 处理按钮
        self.det_Process_Button.clicked.connect(self.process_features)
        self.det_Predict_Button.clicked.connect(self.predict_parkinson)
        self.det_Train_Button.clicked.connect(self.train_model)

        # 其他按钮
        self.det_Record_Button.clicked.connect(self.record_audio)
        self.det_Clear_Button.clicked.connect(self.clear_data)
        self.det_Export_Button.clicked.connect(self.export_results)

    # ==================== 以下是功能方法 ====================
    # 这些方法您可以根据需要修改和扩展

    def upload_audio_file(self):
        """上传单个音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, '选择音频文件', '',
            '音频文件 (*.wav *.mp3 *.m4a *.flac)'
        )
        if file_path:
            self.det_File_List_Widget.addItem(file_path)
            self.update_file_info()
            self.det_Status_Label.setText('🟢 状态: 文件上传成功')
            QMessageBox.information(self, '上传成功', f'已上传: {os.path.basename(file_path)}')

    def upload_audio_folder(self):
        """上传包含音频文件的文件夹"""
        folder_path = QFileDialog.getExistingDirectory(self, '选择音频文件夹')
        if folder_path:
            self.det_File_List_Widget.addItem(f"文件夹: {folder_path}")
            self.update_file_info()
            self.det_Status_Label.setText('🟢 状态: 文件夹上传成功')
            QMessageBox.information(self, '上传成功', f'已上传文件夹: {folder_path}')

    def remove_selected_file(self):
        """移除选中的文件"""
        current_row = self.det_File_List_Widget.currentRow()
        if current_row >= 0:
            self.det_File_List_Widget.takeItem(current_row)
            self.update_file_info()

    def clear_file_list(self):
        """清空文件列表"""
        self.det_File_List_Widget.clear()
        self.update_file_info()

    def process_features(self):
        """特征提取处理"""
        if self.det_File_List_Widget.count() == 0:
            QMessageBox.warning(self, '警告', '请先上传音频文件')
            return

        self.det_Status_Label.setText('🟡 状态: 特征提取中...')
        # 这里调用您的特征提取代码
        QMessageBox.information(self, '完成', '特征提取完成')
        self.det_Status_Label.setText('🟢 状态: 特征提取完成')

    def predict_parkinson(self):
        """进行帕金森病预测"""
        self.det_Status_Label.setText('🟡 状态: 诊断中...')

        # 这里调用您的模型预测代码
        import random
        probability = random.randint(0, 100)
        self.update_probability_display(probability)

        # 显示诊断结果
        if probability < 30:
            result_text = "✅ 低风险: 帕金森病可能性较低"
        elif probability < 70:
            result_text = "⚠️ 中风险: 建议进一步检查"
        else:
            result_text = "❌ 高风险: 帕金森病可能性较高，建议就医"

        self.det_Result_Text_Edit.setText(result_text)
        self.det_Status_Label.setText('🟢 状态: 诊断完成')
        QMessageBox.information(self, '诊断完成', f'预测概率: {probability}%')

    def update_probability_display(self, probability):
        """更新概率显示"""
        self.det_Probability_Bar.setValue(probability)
        self.det_Probability_Label.setText(f'{probability}%')

        # 根据概率设置颜色
        if probability < 30:
            color = "#67c23a"  # 绿色
        elif probability < 70:
            color = "#e6a23c"  # 黄色
        else:
            color = "#f56c6c"  # 红色

        self.det_Probability_Label.setStyleSheet(f"""
            font-size: 24px; 
            font-weight: bold; 
            color: {color};
            min-width: 80px;
        """)

    def train_model(self):
        """训练模型"""
        reply = QMessageBox.question(self, '确认训练',
                                     '确定要开始训练模型吗？这可能需要一些时间。')
        if reply == QMessageBox.Yes:
            self.det_Status_Label.setText('🟡 状态: 模型训练中...')
            # 调用训练代码
            QMessageBox.information(self, '完成', '模型训练完成')
            self.det_Status_Label.setText('🟢 状态: 模型训练完成')

    def record_audio(self):
        """录音功能"""
        self.det_Status_Label.setText('🔴 状态: 录音中...')
        QMessageBox.information(self, '录音', '开始录音功能')
        # 实现录音逻辑
        self.det_Status_Label.setText('🟢 状态: 录音完成')

    def clear_data(self):
        """清除所有数据"""
        reply = QMessageBox.question(self, '确认清除', '确定要清除所有数据吗？')
        if reply == QMessageBox.Yes:
            self.clear_file_list()
            self.det_Probability_Bar.setValue(0)
            self.det_Probability_Label.setText('0%')
            self.det_Result_Text_Edit.clear()
            self.det_Detail_Text_Edit.clear()
            self.det_Status_Label.setText('🟢 状态: 就绪')
            self.det_Probability_Label.setStyleSheet("""
                font-size: 24px; 
                font-weight: bold; 
                color: #409eff;
                min-width: 80px;
            """)

    def export_results(self):
        """导出诊断结果"""
        if self.det_Result_Text_Edit.toPlainText() == "":
            QMessageBox.warning(self, '警告', '没有可导出的结果')
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, '导出结果', 'parkinson_diagnosis_result.txt',
            '文本文件 (*.txt)'
        )
        if file_path:
            QMessageBox.information(self, '导出成功', f'结果已导出到: {file_path}')

    def update_file_info(self):
        """更新文件信息"""
        count = self.det_File_List_Widget.count()
        self.det_File_Count_Label.setText(f'文件数: {count}')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ParkinsonDiagnosisUI()
    window.show()
    sys.exit(app.exec())