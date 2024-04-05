import json
from ast import literal_eval
from pathlib import Path

class Config:
    def __init__(self, **kwargs):

        self.batch_size = kwargs.pop('batch_size', 32)
        self.max_char_len = kwargs.pop('max_char_len', 128)
        self.shuffle = kwargs.pop('shuffle', True)
        self.num_workers = kwargs.pop('num_workers', 12)
        self.lr = kwargs.pop('lr', 0.01)
        self.epoch = kwargs.pop('epoch', 1)
        self.char_embedding_dim = kwargs.pop('char_embedding_dim', 64)

        self.transformer_encoder = kwargs.pop('transformer_encoder', {})
        self.transformer_decoder = kwargs.pop('transformer_decoder', {})

        self.model_eval_epoch = kwargs.pop('model_eval_epoch', 5)
        self.eval_gen_data_length = kwargs.pop('eval_gen_data_length', 10)

    @classmethod
    def from_json_file(cls, json_file):

        config_dict = cls._dict_from_json_file(json_file)
        return cls(**config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return json.loads(text)

    def update_batch_size(self, device_count):
        if device_count:
            self.batch_size *= device_count


class OCRConfig(Config):

    DATA_DIR = 'data/nic_refined/handwriting'
    DATASET_DIR = Path(DATA_DIR)/'dataset_v2'
    VOCAB_DIR = Path(DATA_DIR)/'vocab'

    CHAR_VOCAB_FILE = Path("./")/'combined_char.txt'
    FONT_VOCAB_FILE = Path(VOCAB_DIR)/'writer.txt'

    IMAGE_DIR = Path(DATA_DIR)/'lines'
    EXP_DIR = Path('out/handwriting')

    TXT_DATASET = Path(DATASET_DIR)/'text.csv'
    IMG_DATASET = Path(DATASET_DIR)/'image.csv'
    IMG_TXT_DATASET = Path(DATASET_DIR)/'image_text.csv'

    TRAIN_DATASET = Path(DATASET_DIR)/'train.csv'
    EVAL_DATASET = Path(DATASET_DIR)/'eval.csv'
    TEST_DATASET = Path(DATASET_DIR)/'test.csv'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.mixed_precision_training = kwargs.pop('mixed_precision_training', False)
        self.random_masking = kwargs.pop('random_masking', False)
        self.distillation = kwargs.pop('distillation', False)
        self.aux_ctc = kwargs.pop('aux_ctc', False)
        self.type_bias = kwargs.pop('type_bias', False)

        self.txt_batch_size = kwargs.pop('txt_batch_size', 12)
        self.img_batch_size = kwargs.pop('img_batch_size', 12)
        self.img_txt_batch_size = kwargs.pop('img_txt_batch_size', 12)
        # self.batch_size = self.txt_batch_size + self.img_batch_size + self.img_txt_batch_size
        self.batch_size = self.img_txt_batch_size
        self.iter_dataset_index = kwargs.pop('iter_dataset_index', 0)
        self.font_embedding_dim = kwargs.pop('font_embedding_dim', 64)

        self.resnet_encoder = kwargs.pop('resnet_encoder', {})
        self.resnet_encoder['channels'] = literal_eval(
            self.resnet_encoder['channels']) if 'channels' in self.resnet_encoder else []
        self.resnet_encoder['strides'] = literal_eval(
            self.resnet_encoder['strides']) if 'strides' in self.resnet_encoder else []
        self.resnet_encoder['depths'] = literal_eval(
            self.resnet_encoder['depths']) if 'depths' in self.resnet_encoder else []

        self.resnet_decoder = kwargs.pop('resnet_decoder', {})
        self.resnet_decoder['reshape_size'] = literal_eval(
                self.resnet_decoder['reshape_size']) if 'reshape_size' in self.resnet_decoder else []
        self.resnet_decoder['channels'] = literal_eval(
            self.resnet_decoder['channels']) if 'channels' in self.resnet_decoder else []
        self.resnet_decoder['strides'] = literal_eval(
            self.resnet_decoder['strides']) if 'strides' in self.resnet_decoder else []
        self.resnet_decoder['depths'] = literal_eval(
            self.resnet_decoder['depths']) if 'depths' in self.resnet_decoder else []
        self.resnet_decoder['kernels_size'] = literal_eval(
            self.resnet_decoder['kernels_size']) if 'kernels_size' in self.resnet_decoder else []
        self.resnet_decoder['paddings'] = literal_eval(
            self.resnet_decoder['paddings']) if 'paddings' in self.resnet_decoder else []

        self.max_char_len = kwargs.pop('max_char_len', 128)
        self.bleu_score_ngram = kwargs.pop('bleu_score_ngram', 16)
        self.bleu_score_weights = [1/self.bleu_score_ngram] * self.bleu_score_ngram

    def update_batch_size(self, device_count):
        if device_count:
            self.txt_batch_size *= device_count
            self.img_batch_size *= device_count
            self.img_txt_batch_size *= device_count
            self.batch_size *= device_count
