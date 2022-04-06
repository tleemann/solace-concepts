import numpy as np
import json
import torch
import concept_reader
import typing as tp
import random
import os.path

import PyQt5
from PyQt5.QtWidgets import QApplication, QGridLayout, QGroupBox, QHBoxLayout, QRadioButton, QSizePolicy, QVBoxLayout
from PyQt5.QtWidgets import QLabel, QPushButton, QTextEdit, QSlider, QScrollArea
from PyQt5.QtWidgets import QWidget 
from PyQt5.QtGui import *
from PyQt5.QtCore import QParallelAnimationGroup, Qt, QtDebugMsg
import time

## MAIN CONFIGURATION
show_examples = 18  # Number of examples to show

n_clm = 3 # Number of rows
n_row = show_examples // n_clm

concept_view_mode = False   # Do not check answers to questions, do not randomly 
                            # sort concepts if set to True (this can be used just to browse through the dataset)

# Input file or folder here.
path_prefix = "data/concepts_wo_img/"
filenames = ["awa_Shap", "places365_Shap", "awa_kMeans", "places365_kMeans",  
             "places365_mixin5", "places365_mixin10", "awa_mixin5", "awa_mixin10"]

# A dictionary with the words for each algorithm: AlgorithmID-> filename.
word_files = {0: "data/user_study/frcnn_words.json", 1: "data/user_study/align_words_mse_10kwords.json", 2: "data/user_study/crop_align_words_mse_10kwords.json"}
result_file = "annotations.json"

show_top_k = 4 # Top k words shown per algorithm
## END CONFIGURATION

# Probe the provided files
concept_counts = [concept_reader.get_concept_count(path_prefix + filename) for filename in filenames]
tot_count = sum(concept_counts) # total number of concepts
print("Total number of concepts:", tot_count)

words_lists = {k: json.load(open(v, "r")) for k, v in word_files.items()}

imbox_array = [] # An array which contains all the image QLabel objects

if os.path.isfile(result_file):
    results_dict = json.load(open(result_file, "r"))
else:
    results_dict = [] # Lists of dicts with the results. Keys:

# Random shuffle
shuffled_idx = torch.randperm(tot_count)


def fuse_wordlists(file_idx, infile_idx, num_words=show_top_k):
    """ Return a list of the words from all algorithms. """
    word_dict = {}
    for alg_id, v in words_lists.items():

        for item in v:
            if item["source_file"] == filenames[file_idx] and item["source_id"] == infile_idx:
                num_items = 0
                for witem in item["word_list"]:
                    if num_items >= num_words:
                        break   
                    if witem["word"] in word_dict:
                        word_dict[witem["word"]].append(alg_id)
                    else:
                        word_dict[witem["word"]] = [alg_id]
                    num_items += 1
    return [(k,v) for k, v in word_dict.items()]

def load_concept_cid(c_id: int):
    """ Load a concept with the given ID. 
        Return concept object or None if this particular concept was already rated (in a previous session.)
    """
    csum = 0
    file_idx = -1
    while c_id >= csum :
        file_idx += 1
        csum += concept_counts[file_idx]
    infile_id = c_id - (csum-concept_counts[file_idx])
    # Check if already rated.
    rated_list = [(item["source_file"], item["source_id"]) for item in results_dict]
    if (filenames[file_idx], infile_id) in rated_list:
        return None
    my_conc =  concept_reader.read_concepts(path_prefix + filenames[file_idx], load_id=infile_id)[0]
    # Add the corresponding words.
    my_conc.wordslist = fuse_wordlists(file_idx, infile_id)
    return my_conc


def tensor_to_Qimage(img_tensor):
    """ Transform a tensor to QtGui.QImage """
    cvImg = (img_tensor.numpy()*255).astype(np.uint8)
    height, width, channel = cvImg.shape
    bytesPerLine = 3 * width
    arr2 = np.require(cvImg, np.uint8, 'C')
    qImg = QImage(arr2, width, height, bytesPerLine, QImage.Format_RGB888)
    return qImg


class ConceptIndex:
    """ Represent the state of the GUI."""
    def __init__(self):
        self.curr_ind = 0
        self.saliency = False
        self.conc = None # Concept shown. 

    def next(self, event):

        res_word = ""
        for btn in btnlistq1:
            if btn.isChecked():
                res_word = btn.text()

        if res_word == "another word:":
            res_word = word_input.toPlainText()

        res_word_algorithm = []
        # Find algorithms for text
        for k, v in self.conc.wordslist:
            if k == res_word:
                res_word_algorithm = v  
        
        res_coherences = ""
        rating_num = 0
        for btn in btnlistq2:
            if btn.isChecked():
                res_coherences = btn.text()
                break
            rating_num += 1

        if (res_word == "" or res_coherences == "") and not concept_view_mode:
            lmsg.setText("Please enter a description and name.")
            lmsg.setVisible(True)
            return

        # Find rating id.
        results_dict.append({"source_file": self.conc.source_file, "source_id": self.conc.source_id,
         "rating": res_coherences, "rating_num": rating_num, "description": res_word, "description_algorithm": res_word_algorithm})

        with open(result_file, "w") as f:
            json.dump(results_dict, f)

        if self.curr_ind == tot_count -1: # we are done.
            print("Thank you for your participation.")
            exit(0)

 
        self.update()
        for btn in btnlistq1:
            btn.setAutoExclusive(False)
            btn.setChecked(False)
            btn.setAutoExclusive(True)
        for btn in btnlistq2:
            btn.setAutoExclusive(False)
            btn.setChecked(False)
            btn.setAutoExclusive(True)


    def update(self):
        """ Update the GUI to show the next concept."""
        lmsg.setText(" ")
        word_input.setPlainText("")
        self.conc = None
        while self.conc is None:
            self.curr_ind += 1
            if self.curr_ind == tot_count: # we are done.
                print("Thank you for your participation.")
                exit(0)
            if concept_view_mode:
                c_id = self.curr_ind
            else:
                c_id = shuffled_idx[self.curr_ind].item()
            self.conc = load_concept_cid(c_id) # Returns none, if the concept is already present in the result file, indicating that it should be skipped.
            
    
        tstart = time.time_ns()
        for i in range(show_examples):
            imbox_array[i][1].setPixmap(QPixmap.fromImage(tensor_to_Qimage(self.conc[i].to_numpy_image_with_overlay())))
            imbox_array[i][0].setPixmap(QPixmap.fromImage(tensor_to_Qimage(self.conc[i].to_numpy_image())))
        tend = time.time_ns()

        # Update the word options
        random.shuffle(self.conc.wordslist)
        for i in range(len(btnlistq1)-1): # The last radio button is for the text box
            if i < len(self.conc.wordslist):
                btnlistq1[i].setText(self.conc.wordslist[i][0])
                btnlistq1[i].setVisible(True)
            else:
                btnlistq1[i].setVisible(False)
        #scrollarea.verticalScrollBar().setSliderPosition(0)
        if concept_view_mode:
            helloMsg.setText(f"Please provide an evaluation for the following concept: [ID={self.conc.source_file}-{self.conc.source_id}]")
        else:
            helloMsg.setText(f"Please provide an evaluation for the following concept ({len(results_dict)}):")
        print("Update took ", (tend-tstart)//1000000, "ms.")

app = QApplication([])

screen = app.screens()[0]
dpi = screen.physicalDotsPerInch()
print("screen dpi:", dpi)
# GUI Geometry parameters
spacing = 20 # Spacing between images
top_space = 20
left_space = 20
img_width = 200
btn_width = 150
btn_height = 40
scrollarea_sz = 2*img_width+100
zoom_img_size = 400
window = QWidget()
window_layout = QHBoxLayout(window)

window.setWindowTitle('Concept Rating GUI')

#scrollarea = QScrollArea(window)
left_area = QWidget()
left_layout = QVBoxLayout()

scrollarea_widget = QWidget()
scrollarea_container = QGridLayout(scrollarea_widget)

# left, top, width, height


helloMsg = QLabel('Please provide an evaluation for the following concept: ', parent=window)
helloMsg.move(left_space, 2)
helloMsg.setWordWrap(False)
helloMsg.setFixedHeight(20)
callback = ConceptIndex()

class ClickableLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super(ClickableLabel, self).__init__( *args, **kwargs)
        self.my_id = 0 # Id of label
        self.my_callback = None

    def resizeEvent(self, a0: QResizeEvent) -> None:
        self.setFixedWidth(self.height())
        return super().resizeEvent(a0)

# Column 0: First image, colum 1 first saliency, column 2: Spacer
imbox_array = []
for i in range(n_row):
    for j in range(n_clm):
        img_id = j*n_row + i
        pic = ClickableLabel(window)
        pic.setScaledContents(True)
        pic.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        pic.move(left_space, top_space + i*(img_width+spacing))
        #pic.setFixedSize(img_width, img_width)
        pic.setMinimumWidth(100)
        pic.my_id = img_id
        pic.my_callback = callback
        pic_sal = ClickableLabel(window)
        pic_sal.setScaledContents(True)
        pic_sal.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        pic.setMinimumWidth(100)
        #pic_sal.setFixedSize(img_width, img_width)
        pic_sal.my_id = img_id
        pic_sal.my_callback = callback
        #pic_sal.move(left_space + (img_width+spacing), top_space + i*(img_width+spacing))
        scrollarea_container.addWidget(pic, i, 3*j)
        scrollarea_container.addWidget(pic_sal, i, 3*j+1)
        imbox_array.append((pic, pic_sal))

scrollarea_container.setContentsMargins(20, 20, 20, 20)
for j in range(n_clm-1):
    scrollarea_container.setColumnMinimumWidth(3*j+2, 35) # Set width of spacer

left_layout.addWidget(helloMsg)
left_layout.addWidget(scrollarea_widget)
left_area.setLayout(left_layout)
window_layout.addWidget(left_area)

#zoom_group = QGroupBox("Close up", parent = window)
#zoom_group.resize(zoom_img_size+2*left_space, 2*zoom_img_size+3*top_space)
#zoom_group.move(scrollarea_sz+2*left_space, top_space + line_height)
#zoom_pic = QLabel(zoom_group)
#zoom_pic.setScaledContents(True)
#zoom_pic.move(left_space, top_space)
#zoom_pic.setFixedSize(zoom_img_size, zoom_img_size)
#zoom_picsal = QLabel(zoom_group)
#zoom_picsal.setScaledContents(True)
#zoom_picsal.move(left_space, top_space+zoom_img_size+ top_space)
#zoom_picsal.setFixedSize(zoom_img_size, zoom_img_size)

space_label_item = 20
space_between = 40 # Between questions

group_width = 320
group_height = 400

# Rating part
ratingwidget = QWidget()
ratinglayout = QVBoxLayout(ratingwidget)
groupq1 = QGroupBox(parent=window)
groupq2 = QGroupBox(parent=window)


groupq1.setMinimumWidth(group_width)
groupq2.setMinimumWidth(group_width)
groupq1.setMaximumWidth(2*group_width)
groupq2.setMaximumWidth(2*group_width)
test_words = ["grass", "prarie", "horse", "dragon", "tree", 
"animal", "green", "red", "blue", "option10", "optin11", "option12", "option13"] # 12+1 options.
rating_options = ["not at all", "to some extend", "mostly well", "very well"]

def add_options_to_group(option_list, group: QGroupBox, promt: str, addTextArea=False):
    """ Add the option boxes. Return list of buttons and wordinput or None"""
    options = QWidget(parent=group)
    
    options_layout = QVBoxLayout()
    lname = QLabel(promt)
    options_layout.addWidget(lname)

    ret_list: tp.List[QRadioButton] = []
    # shuffle list
    word_input = None
    for i, option in enumerate(option_list):
        rad = QRadioButton(option,parent=group)
        rad.setMinimumHeight(25)
        options_layout.addWidget(rad)
        ret_list.append(rad)
    if addTextArea:
        rad = QRadioButton("another word:", parent=group)
        word_input = QTextEdit()
        word_input.setMinimumWidth(150)
        word_input.setFixedHeight(40)
        options_layout.addWidget(rad)
        options_layout.addWidget(word_input)
        ret_list.append(rad)
        #ret_list.append(word_input)
    options.setLayout(options_layout)
    return ret_list, word_input

btnlistq1, word_input = add_options_to_group(test_words, groupq1, 'What single word is the best description for this concept?', addTextArea=True)
btnlistq2, _ = add_options_to_group(rating_options, groupq2, 'How well does the selected word fit this concept?')

bnext = QPushButton(window) 
bnext.setFont(QFont('Times', 10))
bnext.setText("Next")
bnext.setMaximumWidth(2*group_width)
bnext.resize(btn_width, btn_height)


# Message label
lmsg = QLabel('highly coherent', parent=window)
red_palette = QPalette()
red_palette.setColor(QPalette.WindowText, Qt.red)
lmsg.setPalette(red_palette)
#lmsg.setVisible(False)
lmsg.setFixedHeight(20)

ratinglayout.addWidget(groupq1)
ratinglayout.addWidget(groupq2)
ratinglayout.addWidget(bnext)
ratinglayout.addWidget(lmsg)

window_layout.addWidget(ratingwidget)


callback.update()
bnext.clicked.connect(callback.next)

window.resize(600, 600)
window.show()
app.exec()

