import random

from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import torch


class CharsDataset(IterableDataset):

    def __init__(self, chars, text: str, seq_length=5):

        # represent characters by integers
        self.chars_to_int = {}
        self.int_to_chars = {}
        self.seq_length = seq_length
        for num, char in enumerate(chars):
            self.chars_to_int[char] = num
            self.int_to_chars[num] = char
        # convert text to characters encodes
        encoded_txt = []
        print("converting characters to numbers ....")
        for char in text:
            encoded_txt.append(self.chars_to_int[char])

        # more efficient in slicing when we take a slice from the array it won't take a copy from it , it will have a reference on the slice
        encoded_txt = np.array(encoded_txt)
        # divide the data into sequences

        # label of char[i] is the char[i+1]
        self.data = encoded_txt[:-1]
        self.labels = encoded_txt[1:]

        # take the integer value for length/seq length
        self.no_sequences = len(self.data) // seq_length
        int_size = self.no_sequences * seq_length
        self.data = self.data[:int_size]
        self.labels = self.labels[:int_size]

        print("Done")

    def one_hot_encode(self, char_idx):
        vector = np.zeros((len(self.chars_to_int)))
        vector[char_idx] = 1
        return vector

    def __iter__(self):
        chars_idx = 0
        seq_idx = 0
        while seq_idx < self.no_sequences:
            seq_data, seq_labels = [], []
            for i in range(self.seq_length):
                seq_data.append(self.one_hot_encode(self.data[chars_idx]))
                seq_labels.append(self.labels[chars_idx])
                chars_idx += 1
            yield torch.tensor(np.array(seq_data), dtype=torch.float32), torch.tensor(np.array(seq_labels),
                                                                                      dtype=torch.float32)
            seq_idx += 1

    def __len__(self):
        return self.no_sequences

    def __str__(self):
        return f"sequence_length:{self.seq_length} , no_sequences {self.no_sequences} , no_chars {len(self.chars_to_int)}"


def data_loader_test(characters: str, batch_size=2, seq_length=3):
    no_chars = random.randint(5, 100)
    testText = "".join(random.choices(characters, k=no_chars))

    print(f"test text no of characters={no_chars}")
    print(testText)
    characters = list(set(characters))
    dataset_test_obj = CharsDataset(characters, testText, seq_length=seq_length)
    data_loader = DataLoader(dataset_test_obj, batch_size=batch_size)

    # testing sequences creation
    assert dataset_test_obj.no_sequences == (no_chars - 1) // seq_length
    print(f"characters after clipping to fit n sequence size={dataset_test_obj.no_sequences * seq_length}")

    assert dataset_test_obj.data.size == dataset_test_obj.no_sequences * seq_length
    assert dataset_test_obj.labels.size == dataset_test_obj.no_sequences * seq_length

    print(
        f"no of batches={len(data_loader)}  , batch_size {batch_size} , seq_length={seq_length} , no of chars={len(data_loader) * batch_size * seq_length}")

    convertedTextData = ""
    convertedTextLabels = ""
    for data, labels in data_loader:
        for i in range(data.shape[0]):
            for j in range(seq_length):
                char_idx = torch.argmax(data[i][j]).item()
                char = dataset_test_obj.int_to_chars[char_idx]
                convertedTextData += char

                char = dataset_test_obj.int_to_chars[labels[i][j].item()]
                convertedTextLabels += char

    print(f"Test Text {testText}")
    print(f"Data text {convertedTextData}")
    print(f"labels text {convertedTextLabels}")
    valid_int_size = dataset_test_obj.no_sequences * seq_length
    assert convertedTextData == testText[:valid_int_size]
    assert convertedTextLabels == testText[1:valid_int_size + 1]


