import torch
from torch.utils.data import Dataset
from typing import List

class VirtualTokenDataset(Dataset):
    """A virtual dataset that generates random tensor data on-the-fly."""

    def __init__(self, num_samples: int, sequence_length: int, vocab_size: int):
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.randint(0, self.vocab_size, (self.sequence_length+1,), dtype=torch.long)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        
        return input_ids, labels

def packed_sequence_collate_fn(batch: List[tuple[torch.Tensor, torch.Tensor]]):
    """
    A collate_fn that takes a list of variable-length sequences and packs them.

    Args:
        batch: A list of tuples, where each tuple is (input_ids, labels).
               - input_ids is a 1D tensor of shape (seq_len,)
               - labels is a 1D tensor of shape (seq_len,)

    Returns:
        A dictionary containing the packed tensors required by the model.
    """
    list_input_ids, list_labels = zip(*batch)

    seqlens = torch.tensor([len(s) for s in list_input_ids], dtype=torch.int32)
    max_seqlen = seqlens.max().item()

    packed_input_ids = torch.cat(list_input_ids)
    packed_labels = torch.cat(list_labels)

    cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
    torch.cumsum(seqlens, dim=0, out=cu_seqlens[1:])

    position_ids = torch.cat([torch.arange(0, length, dtype=torch.long) for length in seqlens])

    return {
        'input_ids': packed_input_ids,
        'labels': packed_labels,
        'cu_seqlens': cu_seqlens,
        'max_seqlen': max_seqlen,
        'position_ids': position_ids,
    }