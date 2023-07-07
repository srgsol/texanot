from typing import Union, List, Dict

import pandas as pd
from datasets import Dataset
from IPython.display import Image, clear_output, display
from ipywidgets import Button, HBox, Output, Layout, Textarea

from txtanot.core.similarity import SimilarityEngine


class DataPoint:
    def __init__(self, data: pd.Series):
        self.data = data
        self.idx = self.data['id']
        
    def render(self):
        print('ID:', self.data['id'])
        print('Label:', self.data['label'])
        print()
        print(self.data['text'])
        print('------------------------------')

    def set_label(self, label):
        self.data['label'] = label

    @property
    def label(self):
        return self.data['label']

    def is_annotated(self):
        return not pd.isna(self.data['label'])

    def __eq__(self, other):
        return self.idx == other.idx
        
    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(data={self.data})'


class AnnotatorDataLoader:
    def __init__(self, data: Union[pd.DataFrame, List[Dict]], filter_annotated):
        if isinstance(data, pd.DataFrame):
            self.data = data.copy().to_dict('records')
        else:
            self.data = data
        self.num_rows = len(self.data)
        self.position = 0
        self.filter_annotated = filter_annotated
        self.items_in_session = []
        
    def get_item(self, position) -> DataPoint:
        data_point = DataPoint(self.data[position])
        return data_point
    
    def next_item(self) -> DataPoint:
        self.position += 1
        if self.position == self.num_rows:
            self.position = 0

        data_point = self.get_item(self.position)
        annotated = data_point.is_annotated()
        in_sess = data_point in self.items_in_session

        if self.filter_annotated and annotated and not in_sess:
            try:
                return self.next_item()
            except RecursionError:
                print('RecursionError: There are no items to be annotated. Try'
                      ' `filter_annotated=False`')
        else:
            return data_point
    
    def previous_item(self) -> DataPoint:
        self.position -= 1
        if self.position == -1:
            self.position = self.num_rows - 1

        data_point = self.get_item(self.position)
        annotated = data_point.is_annotated()
        in_sess = data_point in self.items_in_session

        if self.filter_annotated and annotated and not in_sess:
            try:
                return self.previous_item()
            except RecursionError:
                print('RecursionError: There are no items to be annotated. Try'
                      ' `filter_annotated=False`')
        else:
            return data_point

    def annotate(self, data_point):
        self.items_in_session.append(data_point)


class AnnotatorGUI:
    """Graphic User Interface"""
    def __init__(self, data: AnnotatorDataLoader, classes: list) -> None:

        # Store list of available class labels.
        self.classes = classes
        
        # Dataloader.
        self.data = data
        
        # Current data point being annotated.
        self.data_point: DataPoint = None

    def _next(self, *args) -> None:
        self.data_point = self.data.next_item()
        with self.frame:
            clear_output(wait=True)
            print(f'{self.data.position}/{self.data.num_rows}')
            self.data_point.render()

    def _go_back(self, *args) -> None:
        self.data_point = self.data.previous_item()
        with self.frame:
            clear_output(wait=True)
            print(f'{self.data.position}/{self.data.num_rows}')
            self.data_point.render()

    def _select_label(self, button: Button) -> None:
        self.data_point.set_label(button.description)
        self.data.annotate(self.data_point)
        self._next()

    def start(self) -> None:
        """Start the annotation procedure.

        Load the first item to label and set up the user interface.
        """
        self.frame = Output(layout=Layout(height="300px", max_width="600px"))
        
        self.data_point = self.data.next_item()
        with self.frame:
            self.data_point.render()

        # Navigation buttons
        backward_button = Button(description="< go back")
        backward_button.on_click(self._go_back)
        forward_button = Button(description="next >")
        forward_button.on_click(self._next)
        self.navigation_buttons = [backward_button, forward_button]

        # Class label buttons
        self.class_buttons = []
        for label in self.classes:
            label_button = Button(description=label)
            label_button.on_click(self._select_label)
            self.class_buttons.append(label_button)

        # Display contents
        display(self.frame)
        display(HBox(self.navigation_buttons))
        display(HBox(self.class_buttons))


class Annotator:
    """Facade class.


    """
    def __init__(self, data: Union[pd.DataFrame, List[Dict]], classes: list,
                 filter_annotated: bool = True, shuffle=False):

        # Data may be a DataFrame or a list of rows (dicts)
        if isinstance(data, pd.DataFrame):
            if shuffle:
                self.data = data.sample(frac=1).copy().to_dict('records')
            else:
                self.data = data.copy().to_dict('records')
        else:
            self.data = data

        self.classes = classes

        self.main_dataloader = AnnotatorDataLoader(self.data, filter_annotated=filter_annotated)
        self.sim_dataloader = None
        self.similarity_engine = None

        print(f'Data rows: {len(self.data)}')

    def build_index(self, field, checkpoint=None):
        index_dataset = Dataset.from_pandas(pd.DataFrame(self.data))
        self.similarity_engine = SimilarityEngine(checkpoint)
        self.similarity_engine.index(index_dataset, field)

    def start(self):
        gui = AnnotatorGUI(self.main_dataloader, self.classes)
        gui.start()

    def similar(self, text, n=10):
        df: pd.DataFrame = self.similarity_engine.similar(text, n)

        idxs = [DataPoint(row).idx for i, row in df.iterrows()]
        data = [row for row in self.data if DataPoint(row).idx in idxs]
        self.sim_dataloader = AnnotatorDataLoader(data, filter_annotated=True)
        gui = AnnotatorGUI(self.sim_dataloader, self.classes)
        gui.start()

    def merge_similar(self):
        sim_data = self.sim_dataloader.data

        sim_data_points = [DataPoint(item) for item in sim_data]
        data_points = [DataPoint(item) for item in self.data]
        for dp_sim in sim_data_points:
            for dp in data_points:
                if dp_sim == dp:
                    if dp_sim.is_annotated():
                        dp.set_label(dp_sim.label)
                        self.main_dataloader.annotate(dp)

    def counts(self, field):
        df = pd.DataFrame(self.data)
        print(df[field].value_counts())
