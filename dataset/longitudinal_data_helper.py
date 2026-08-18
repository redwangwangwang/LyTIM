import os
import json
import re
import numpy as np
from PIL import Image
import torch.utils.data as data
import pandas as pd
from transformers import BertTokenizer, AutoImageProcessor


class FieldParser:
    def __init__(
            self,
            args
    ):
        super().__init__()
        self.args = args
        self.dataset = args.dataset
        self.vit_feature_extractor = AutoImageProcessor.from_pretrained(args.vision_model)


    def _parse_image(self, img):
        pixel_values = self.vit_feature_extractor(img, return_tensors="pt").pixel_values
        return pixel_values[0]

    @staticmethod
    def clean_report(report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace(':', ' :') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+()\[\]{}]', '', t.replace('"', '').replace('/', '')
                            .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def load_image(self, image_paths):
        images = []
        for image_path in image_paths:
            with Image.open(os.path.join(self.args.base_dir, 'files', image_path)) as pil:
                array = np.array(pil, dtype=np.uint8)
                if array.shape[-1] != 3 or len(array.shape) != 3:
                    array = np.array(pil.convert("RGB"), dtype=np.uint8)
                image = self._parse_image(array)
                images.append(image)
        return images

    def parse(self, features):
        prior_report = features.get("context_report", "")
        prior_report = self.clean_report(prior_report)

        current_report = features.get("report", "")
        current_report = self.clean_report(current_report)

        prior_image_paths = features.get("context_image", "")
        prior_images = self.load_image(prior_image_paths)

        current_image_paths = features.get("image_path", "")
        current_images = self.load_image(current_image_paths)

        sample = {
            'id': features['id'],
            'prev_text': prior_report,
            'curr_text': current_report,
            'prev_image': prior_images,
            'curr_image': current_images,
            'prior_image_paths': prior_image_paths,
            'current_image_paths': current_image_paths,
        }

        if "Findings" in features:
            findings = features.get("Findings", "")
            findings = self.clean_report(findings)

            changes = features.get("Changes", "")
            if changes == "":
                changes = "There is no changes between prior and current image."
            changes = self.clean_report(changes)

            sample['findings'] = findings
            sample['changes'] = changes
        return sample

    def transform_with_parse(self, inputs):
        return self.parse(inputs)


class LongitudinalMimicDataset(data.Dataset):
    def __init__(self, args, split='train'):
        self.args = args
        self.parser = FieldParser(args)
        with open(args.annotation, 'r') as f:
            self.annotations_original = json.load(f)
        self.annotations_original = self.annotations_original[split]
        self.annotations_longitudinal = []

        # sort by patient_id, time and view
        df = pd.DataFrame(self.annotations_original)
        df.sort_values(['subject_id', 'StudyDate', 'StudyTime'], inplace=True)

        for subject_id, group in df.groupby('subject_id'):
            group = group.reset_index(drop=True)

            # load single view
            for i in range(len(group)):
                if i == 0:
                    continue
                current = group.loc[i].to_dict()

                prev = group.loc[i - 1].to_dict()
                current['context_image'] = prev['image_path']
                current['context_report'] = prev['report']
                current['context_time'] = prev['StudyDate']
                self.annotations_longitudinal.append(current)

    def __len__(self):
        return len(self.annotations_longitudinal)

    def __getitem__(self, index):
        return self.parser.transform_with_parse(self.annotations_longitudinal[index])


def create_datasets(args):
    train_dataset = LongitudinalMimicDataset(args, 'train')
    dev_dataset = LongitudinalMimicDataset(args, 'val')
    test_dataset = LongitudinalMimicDataset(args, 'test')
    return train_dataset, dev_dataset, test_dataset
