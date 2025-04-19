from pysift import computeKeypointsAndDescriptors
import sys
from tkinter import filedialog
from PyQt5.QtWidgets import QMainWindow, QApplication, QScrollArea, QWidget ,QVBoxLayout, QRadioButton, QSlider, QLabel, QPushButton, QLineEdit, QCheckBox, QHBoxLayout
from PyQt5 import uic
import cv2
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from FeatureMatching import Matching

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("Task03.ui")
        
        # ##### bc of resolution issues for my laptop, comment out if not needed!!!
        # scroll = QScrollArea()
        # scroll.setWidget(self.ui)
        # self.setCentralWidget(scroll)
        # self.resize(1360, 768) 
        # ######## end of resolution fix 
        
        self.main_widget = self.ui.findChild(QWidget, "widget")
        self.load_btn = self.ui.findChild(QPushButton, "load_button")
        self.load_btn.clicked.connect(self.load_button_clicked)
        
        ## sift flag, to make sure feature matching options are off until sift is done.
        self.sift_done = False
        self.sift_button = self.ui.findChild(QPushButton,"sift_button")
        self.sift_button.clicked.connect(self.sift_button_clicked)
        
        self.ssd_slider = self.ui.findChild(QSlider,"ssd_slider")
        self.ssd_slider.setMinimum(0)
        self.ssd_slider.setMaximum(60000)
        self.ssd_slider.setValue(20000)
        self.ssd_slider.setEnabled(False)
        self.ssd_slider.setSingleStep(5000)
        self.ssd_slider.setPageStep(5000)
        self.ssd_slider.valueChanged.connect(self.ssd_threshold_control)
        self.ssd_slider.valueChanged.connect(lambda value: self.match_features("ssd"))
        
        self.ncc_slider = self.ui.findChild(QSlider,"ncc_slider")
        self.ncc_slider.setMinimum(55)
        self.ncc_slider.setMaximum(100)
        self.ncc_slider.setValue(95)
        self.ncc_slider.setEnabled(False)
        self.ncc_slider.setSingleStep(5)
        self.ncc_slider.setPageStep(5)
        self.ncc_slider.valueChanged.connect(self.ncc_threshold_control)
        self.ncc_slider.valueChanged.connect(lambda value: self.match_features("ncc"))
        
        self.ssd_rb = self.ui.findChild(QRadioButton,"ssd_radio_button")
        self.ssd_rb.setEnabled(False)
        self.ssd_rb.clicked.connect(self.ssd_rb_clicked)
        
        self.ncc_rb = self.ui.findChild(QRadioButton,"ncc_radio_button")
        self.ncc_rb.setEnabled(False)
        self.ncc_rb.clicked.connect(self.ncc_rb_clicked)
        
        self.ssd_threshold = self.ssd_slider.value()
        self.ncc_threshold = self.ncc_slider.value()/100.0
        
        self.ssd_label = self.ui.findChild(QLabel,"ssd_label")
        self.ncc_label = self.ui.findChild(QLabel,"ncc_label")
        
        self.matching_time_label = self.ui.findChild(QLabel,"matching_time_label")
        
    def load_button_clicked(self):
        file_path1 = filedialog.askopenfilename(title="Select Image 1")
        file_path2 = filedialog.askopenfilename(title="Select Image 2")
        if file_path1 and file_path2:
            self.image1_path = file_path1
            self.image2_path = file_path2
            print(f"Selected images: {file_path1}, {file_path2}")
            self.image1 = cv2.imread(file_path1, cv2.IMREAD_COLOR)
            self.image2 = cv2.imread(file_path2, cv2.IMREAD_COLOR)
            if self.image1 is not None and self.image2 is not None:
                self.display_images(self.image1, self.image2)
            else:
                print("Error loading images.")
        else:
            print("No images selected.")

        
    def ssd_threshold_control(self):
        value = self.ssd_slider.value()
        self.ssd_threshold = value
        self.ssd_label.setText(f"{self.ssd_threshold:.0f}") 
        
    
    def ncc_threshold_control(self):
        value = self.ncc_slider.value()
        self.ncc_threshold = value / 100.0
        self.ncc_label.setText(f"{self.ncc_threshold *100:.0f}%")
        
    
    def ssd_rb_clicked(self):
        if self.sift_done:
            self.ssd_slider.setEnabled(True)
            self.ncc_slider.setEnabled(False)
            self.matching_time_label.setText("0 seconds")
            # self.ssd_rb.setChecked(True)
            # self.ncc_rb.setChecked(False)
            self.match_features("ssd")
        else:
            self.ssd_rb.setChecked(False)
            self.ncc_rb.setChecked(False)
            self.ssd_rb.setEnabled(False)
            self.ncc_rb.setEnabled(False)
            self.ssd_slider.setEnabled(False)
            self.ncc_slider.setEnabled(False)
    
    def ncc_rb_clicked(self):
        if self.sift_done:
            self.ncc_slider.setEnabled(True)
            self.ssd_slider.setEnabled(False)
            self.matching_time_label.setText("0 seconds")
            # self.ncc_rb.setChecked(True)
            # self.ssd_rb.setChecked(False)
            self.match_features("ncc")
        else:
            self.ncc_rb.setChecked(False)
            self.ssd_rb.setChecked(False)
            self.ncc_rb.setEnabled(False)
            self.ssd_rb.setEnabled(False)
            self.ncc_slider.setEnabled(False)
            self.ssd_slider.setEnabled(False)
            
    def match_features(self, method):
        # image1, keypoints1, descriptor1, image2, keypoints2, descriptor2 = Matching.extract_sift_features(
        #     self.image1_path, self.image2_path
        # )
        keypoints1, descriptor1 = computeKeypointsAndDescriptors(self.image1)
        keypoints2, descriptor2 = computeKeypointsAndDescriptors(self.image2)
        
        matcher = Matching(self.image1, keypoints1, descriptor1, self.image2, keypoints2, descriptor2, method)
        
        if method == "ssd":
            matches, time = matcher.match_ssd(self.ssd_threshold)
        elif method == "ncc":
            matches, time = matcher.match_ncc(self.ncc_threshold)
        
        self.matching_time_label.setText(f" {time:.2f} seconds")

        layout = self.main_widget.layout()
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)
        else:
            layout = QVBoxLayout()
            self.main_widget.setLayout(layout)
        canvas = matcher.visualize_matches(matches)
        layout.addWidget(canvas)
        self.main_widget.setLayout(layout)
        

    
    def display_images(self, image1, image2):   
        # Clear existing widgets from the layout
        if self.main_widget.layout() is not None:
            while self.main_widget.layout().count():
                item = self.main_widget.layout().takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.setParent(None)

        
        layout = self.main_widget.layout()
        if layout is None:
            layout = QVBoxLayout()
            self.main_widget.setLayout(layout)

        figure = Figure()
        canvas = FigureCanvas(figure)
        layout.addWidget(canvas)

        ax1 = figure.add_subplot(1, 2, 1)
        ax2 = figure.add_subplot(1, 2, 2)

        if len(image1.shape) == 2:
            ax1.imshow(image1, cmap='gray')
        else:
            ax1.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

        if len(image2.shape) == 2:
            ax2.imshow(image2, cmap='gray')
        else:
            ax2.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))

        ax1.set_title("Image 1")
        ax2.set_title("Image 2")

        for ax in [ax1, ax2]:
            ax.axis('off')

        figure.tight_layout()
        canvas.draw()

        
        
    
    def sift_button_clicked(self):
        # Call the SIFT feature extraction method here
        self.sift_done = True
        self.ssd_rb.setEnabled(True)
        self.ncc_rb.setEnabled(True)
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = MainApp()
    window.show()
    sys.exit(app.exec_())