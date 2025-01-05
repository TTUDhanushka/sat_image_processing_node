#!/usr/bin/env python3
"""
Satellite image classification neural network.

The network trained in another desktop PC and get the model as *.pt
""" 
import numpy as np
import torch
from torchvision import transforms, models
from time import perf_counter
import os
from PIL import Image
from vector_export import ShapeFileGenerator
from geotypes import GeoCoordinate


class ImageClassifier:
    def __init__(self):
        self.img_height = None
        self.img_width = None
        self.channels = 3

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.class_colors = [[109, 158, 235],
                            [147, 196, 125],
                            [255, 255, 0],
                            [204, 0, 0],
                            [51, 51, 51],
                            [17, 85, 204], 
                            [193, 42, 201]]

        self.model = models.segmentation.fcn_resnet101(weights=None,
                                                        num_classes=7)
        self.shapefile_export = None

    def load_saved_model(self) -> str:
        model_name = 'sat_segmentation_model_1.pt'
        model_folder_path = '/home/scctower1/models'

        if not os.path.exists(model_folder_path):
            print(f"Model folder doesn't exist!")

            # There is no model to do inference.
            return None

        model_path = os.path.join(model_folder_path, model_name)

        print(f"Model full path {model_path}")

        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)


    def classify(self, image: np.ndarray, image_bottom_lft_coord: GeoCoordinate):

        self.img_height, self.img_width, _ = image.shape

        img_transforms = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
                                            ])

        input_image = img_transforms(image)

        input_image = torch.unsqueeze(input_image, 0)

        input_image = input_image.to(self.device).float()

        # Set model for inference
        self.model.eval()

        preds = self.model(input_image)
        class_preds = torch.argmax(preds['out'], dim=1)

        # Remove one dimension from the classification result [1, 512, 512] -> [512, 512]
        classification_output = torch.squeeze(class_preds, 0)

        # Display binary layers
        # shp_files = ShapeFileGenerator()
        # shp_files.classification_to_binary(classification_output)
        self.shapefile_export.classification_to_binary(classification_output, image_bottom_lft_coord)

        mapped_image = self.hot_decode(classification_output)
        pil_image = Image.fromarray(mapped_image.astype(np.uint8), 'RGB')

        pil_image.show()

        print(f"Classification imags shape {classification_output.shape}")


    def set_shapefile_generator(self, shapefile_obj):
        self.shapefile_export = shapefile_obj

    def hot_decode(self, image):

        result_image = np.zeros([self.img_height, self.img_width, self.channels], dtype= np.uint8)

        for ix in range(self.img_height):
            for iy in range (self.img_width):
                key = image[ix, iy].item()

                rgb_triplets = self.class_colors[key]

                result_image[ix, iy, :] = rgb_triplets
        
        return result_image



