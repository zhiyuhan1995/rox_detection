import os 

from ultralytics import YOLO

import torch
from sklearn.model_selection import train_test_split
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from ..core import BaseConfig



class YoloSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        '''1: SETTINGS FOR DOE'''
        self.cfg = cfg
        self.data_cfg_file = self.cfg.yaml_cfg['data_cfg']

        #self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.getcwd()

        self.model_size = 'n' #'m'
        self.imgsz = cfg.yaml_cfg['imgsz']
        self.batch_size = cfg.yaml_cfg['batch_size'] #should use maximum possible (setting to 8 is only necessary e.g. for model m and imgsz = 1024)
        self.fraction_val_data = cfg.yaml_cfg['fraction_val_data']
        self.apply_augmentations = cfg.yaml_cfg['apply_augmentations']
        self.training_dataset = self.cfg.yaml_cfg["training_dataset"]
        self.test_dataset = self.cfg.yaml_cfg["test_dataset"]
        
        # Get the existing training_index (if any)
        self.existing_index = 0

        self.row_data = {
            ('Training Data', ''): self.training_dataset,
            ('Instance Segmentation', ''): 'No',
            ('Depth Data', ''): 'No',
        }

        self.test_data_str = ', '.join(self.test_dataset)


    '''2: TRAIN MODELS BASED ON DOE'''
    def delete_folder_contents(self, folder_path):
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

    def copy_files(self, files, src_dir, dst_dir):
        for file in files:
            shutil.copy(os.path.join(src_dir, file), os.path.join(dst_dir, file))

    def prepare_training_data(self):
        #2.1: SETUP TRAINING AND VALIDATION DATA
        target_dir = os.path.join(self.base_dir, 'data_yolo', 'dataset_temp')
        if not os.path.exists(os.path.join(target_dir, 'images/train')):
            os.makedirs(os.path.join(target_dir, 'images/train'))
        if not os.path.exists(os.path.join(target_dir, 'images/val')):
            os.makedirs(os.path.join(target_dir, 'images/val'))
        if not os.path.exists(os.path.join(target_dir, 'labels/train')):
            os.makedirs(os.path.join(target_dir, 'labels/train'))
        if not os.path.exists(os.path.join(target_dir, 'labels/val')):
            os.makedirs(os.path.join(target_dir, 'labels/val'))
        
        self.delete_folder_contents(os.path.join(target_dir, 'images/train'))
        self.delete_folder_contents(os.path.join(target_dir, 'images/val'))
        self.delete_folder_contents(os.path.join(target_dir, 'labels/train'))
        self.delete_folder_contents(os.path.join(target_dir, 'labels/val'))

        main_dir = os.path.join(self.base_dir, self.training_dataset)

        image_dir = os.path.join(main_dir, 'images')

        label_dir = os.path.join(main_dir, 'labels')

        # Get list of label files and corresponding images
        label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        image_files = [f.replace('.txt', '.png') for f in label_files]

        # Split data into train and val
        train_images, val_images, train_labels, val_labels = train_test_split(image_files, label_files, test_size=self.fraction_val_data, random_state=42)

        # Move train files
        self.copy_files(train_images, image_dir, os.path.join(target_dir, 'images/train'))
        self.copy_files(train_labels, label_dir, os.path.join(target_dir, 'labels/train'))

        # Move val files
        self.copy_files(val_images, image_dir, os.path.join(target_dir, 'images/val'))
        self.copy_files(val_labels, label_dir, os.path.join(target_dir, 'labels/val'))

    def _setup(self):
        """Avoid instantiating unnecessary classes"""
        cfg = self.cfg
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.is_resumed = False 
        
        # NOTE: Must load_tuning_state before EMA instance building
        if self.cfg.tuning:
            print(f'Tuning checkpoint from {self.cfg.tuning}')
            self.model = YOLO(self.cfg.tuning).to(device)
        elif self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.model = YOLO(self.cfg.resume).to(device)
            self.is_resumed = True
        else:
            # Load the 4-channel model configuration
            self.model = YOLO(f'yolov8{self.model_size}.pt').to(device)

    def fit(self):
        cfg = self.cfg
        cfg.output_dir = f"yolov8{self.model_size}"
        self.save_dir = os.path.join(self.base_dir, cfg.output_dir) #path where all training runs are saved
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.allfiles = os.listdir(self.save_dir)

        if self.allfiles is not None:
            #if cfg.test_only:
            files = [fname for fname in self.allfiles if 'rgb' in fname and 'test' not in fname]
            for file in files:
                curr_index = int(file.split('_')[0])
                if curr_index > self.existing_index:
                    self.existing_index = curr_index

            training_data_index = self.existing_index + 1
        else:
            training_data_index = self.existing_index

        #self.prepare_training_data()

        #2.2: SET UP YOLO MODEL WITH 4 CHANNELS
        # https://github.com/ultralytics/ultralytics/issues/3432
        # https://github.com/g-h-anna/ultralytics4channel

        self._setup()

        #2.3: MODEL TRAINING AND SAVING RESULTS
        # Specify the save directory for training runs

        input_data_type = 'rgb'

        experiment = f'{training_data_index}_{input_data_type}'
        experiment_dir = os.path.join(self.save_dir, experiment)

        if(self.apply_augmentations):
            degrees = cfg.yaml_cfg["augmentations"]["degrees"]
            flipud = cfg.yaml_cfg["augmentations"]["flipud"]
            fliplr = cfg.yaml_cfg["augmentations"]["fliplr"]
            mosaic = cfg.yaml_cfg["augmentations"]["mosaic"]
            mixup = cfg.yaml_cfg["augmentations"]["mixup"]
            copy_paste = cfg.yaml_cfg["augmentations"]["copy_paste"]
            # TODO: investigate setting copy_paste, could introduce a large difference between with/without segmentation masks
        else:
            degrees, flipud, fliplr, mosaic, mixup, copy_paste = 0, 0, 0, 0, 0, 0


        self.model.train(data=os.path.join(self.base_dir, self.data_cfg_file), epochs=cfg.yaml_cfg["epochs"], imgsz=cfg.yaml_cfg["imgsz"], batch=cfg.yaml_cfg["batch_size"],
                    project = self.save_dir, name = experiment_dir, seed=cfg.yaml_cfg["seed"], patience=cfg.yaml_cfg["patience"],

                    #https://docs.ultralytics.com/reference/data/augment/

                    # always disable augmentations that are problematic for RGB-D data
                    hsv_h = 0, hsv_s = 0, hsv_v = 0, translate= 0, scale = 0, shear = 0, perspective = 0,

                    # set parameters for "depth compatible" data augmentation        
                    degrees = degrees, flipud = flipud, fliplr = fliplr, mosaic = mosaic, mixup = mixup, copy_paste = copy_paste, 
                    
                    # set other parameters
                    save_period = cfg.yaml_cfg["save_period"], resume = self.is_resumed, amp = cfg.use_amp
                    )
        
                
        training_data_index += 1

        self.evaluate()

    def prepare_test_data(self):
        
        target_dir = os.path.join(self.base_dir, 'data_yolo', 'dataset_temp')

        if not os.path.exists(os.path.join(target_dir, 'images/test')):
            os.makedirs(os.path.join(target_dir, 'images/test'))
        if not os.path.exists(os.path.join(target_dir, 'labels/test')):
            os.makedirs(os.path.join(target_dir, 'labels/test'))

        self.delete_folder_contents(os.path.join(target_dir, 'images/test'))
        self.delete_folder_contents(os.path.join(target_dir, 'labels/test'))

        main_dir = os.path.join(self.base_dir, self.test_dataset)

        image_dir = os.path.join(main_dir, 'images')

        label_dir = os.path.join(main_dir, 'labels')

        # Get list of label files and corresponding images
        test_labels = [f for f in os.listdir(label_dir) if f.endswith('.txt')]
        test_images = [f.replace('.txt', '.png') for f in test_labels]

        # Move test files
        self.copy_files(test_images, image_dir, os.path.join(target_dir, 'images/test'))
        self.copy_files(test_labels, label_dir, os.path.join(target_dir, 'labels/test'))
    
    def report(self, metrics, data_index):

        # Initialize a list to hold all the results in the desired format
        results_data = []

        # List to hold MultiIndex column tuples
        multiindex_columns = [('Training Data', ''), ('Instance Segmentation', ''), ('Depth Data', '')]

        # Store the evaluation metrics for Box and Mask mAP as well as Box precision and recall
        mAP_50_95_object_detection = round(metrics.box.map, 3)
        box_precision = round(metrics.box.mp, 3)
        box_recall = round(metrics.box.mr, 3)
        box_f1 = round((2*(box_precision*box_recall)/(box_precision+box_recall)), 3)

        mAP_50_95_instance_segmentation = 0

        # Add mAP results for this test dataset to the row data
        self.row_data[(f'Box mAP50-95', f'{self.test_data_str}')] = mAP_50_95_object_detection
        self.row_data[(f'Mask mAP50-95', f'{self.test_data_str}')] = mAP_50_95_instance_segmentation

        # Add column headings to MultiIndex columns list if not already present
        if (f'Box mAP50-95', f'{self.test_data_str}') not in multiindex_columns:
            multiindex_columns.append((f'Box mAP50-95', f'{self.test_data_str}'))
        if (f'Mask mAP50-95', f'{self.test_data_str}') not in multiindex_columns:
            multiindex_columns.append((f'Mask mAP50-95', f'{self.test_data_str}'))

        # Add precision and recall results for object detection
        self.row_data[(f'Box Precision', f'{self.test_data_str}')] = box_precision
        self.row_data[(f'Box Recall', f'{self.test_data_str}')] = box_recall
        self.row_data[(f'Box F1', f'{self.test_data_str}')] = box_f1

        if (f'Box Precision', f'{self.test_data_str}') not in multiindex_columns:
            multiindex_columns.append((f'Box Precision', f'{self.test_data_str}'))
        if (f'Box Recall', f'{self.test_data_str}') not in multiindex_columns:
            multiindex_columns.append((f'Box Recall', f'{self.test_data_str}'))
        if (f'Box F1', f'{self.test_data_str}') not in multiindex_columns:
            multiindex_columns.append((f'Box F1', f'{self.test_data_str}'))

        # Append the results for this experiment to the list
        results_data.append(self.row_data)

        data_index += 1

        # Set the CSV index
        csv_index = data_index - 1

        # Create a DataFrame from the results with MultiIndex columns
        df = pd.DataFrame(results_data)

        # Assign the MultiIndex to the DataFrame columns
        df.columns = pd.MultiIndex.from_tuples(multiindex_columns)

        # Change numbers to use commas as decimal separators and use a semicolon as a delimiter
        df = df.applymap(lambda x: str(x).replace('.', ',') if isinstance(x, float) else x)

        # Save the DataFrame to a CSV file with semicolon separator
        csv_output_path = os.path.join(self.save_dir, f'{csv_index}_DOE_results_summary.csv')
        df.to_csv(csv_output_path, sep=';', index=False)

        print(f"Results summary saved to {csv_output_path}")

        '''4: Visualize the results'''
        # Load the CSV file
        csv_file = os.path.join(self.save_dir, f'{csv_index}_DOE_results_summary.csv')  # Replace with the path to your file
        df = pd.read_csv(csv_file, sep=';')

        # Plot the styled table
        self.plot_table(df)


    def val(self):
        cfg = self.cfg

        self._setup()
        cfg.output_dir = f"yolov8{self.model_size}"
        self.save_dir = os.path.join(self.base_dir, cfg.output_dir, 'test_results')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.allfiles = os.listdir(self.save_dir)

        if self.allfiles is not None:
            files = [fname for fname in self.allfiles if '_DOE_results_summary.csv' in fname]
            for file in files:
                curr_index = int(file.split('_')[0])
                if curr_index > self.existing_index:
                    self.existing_index = curr_index

        self.evaluate(only_test = True)

    def evaluate(self, only_test:bool = False):
        '''3: EVALUATE TRAINED MODELS AND SAVE RESULTS IN CSV'''  
        if self.allfiles is not None:
            files = [fname for fname in self.allfiles if '_DOE_results_summary.csv' in fname]
            for file in files:
                curr_index = int(file.split('_')[0])
                if curr_index > self.existing_index:
                    self.existing_index = curr_index

        if self.allfiles is not None:
            data_index = self.existing_index + 1

        else:
            data_index = self.existing_index
            
        '''3.1: RETRIEVE INDIVIDUAL EXPERIMENT'''
        input_data_type = 'rgb'

        if not only_test:
            experiment = f'{data_index}_{input_data_type}'
            print(f'Experiment: {experiment} ({self.training_dataset})')

        if only_test and self.cfg.resume:
            model_path = os.path.join(self.cfg.resume)
        else:
            model_path = os.path.join(self.save_dir, experiment, 'weights/best.pt')

        # Load the model and handle errors if the model can't be loaded
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
        
        '''3.2: SETUP TEST DATA'''
        #self.prepare_test_data()

        '''3.3: EVALUATE MODEL AND REPORT PERFORMANCE'''
        metrics = self.model.val(data=os.path.join(self.base_dir, self.data_cfg_file), split="test")
        self.report(metrics, data_index)

    
    # Function to plot the DataFrame as a table
    def plot_table(self, df):
        fig, ax = plt.subplots(figsize=(len(df.columns) * 1.5, len(df) * 0.5))  # Adjusting the figure size based on data

        # Remove the axis
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_frame_on(False)

        # Define a color map (alternating rows)
        cmap = colors.ListedColormap(['#f0f8ff', '#fafafa'])

        # Plot the table
        table = ax.table(cellText=df.values,
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        colColours=['#d1ecf1'] * len(df.columns),  # Column headers color
                        rowColours=[cmap(i % 2) for i in range(len(df))])  # Alternating row colors

        # Make the headers bold
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        # Bold headers
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # header row
                cell.set_text_props(weight='bold')

        # Adjust layout
        plt.tight_layout()
        plt.show()
    




    

