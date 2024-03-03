import jsonlines
from transformers import T5TokenizerFast
from transformers import T5ForConditionalGeneration
from transformers import AdamW
from tqdm.notebook import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import time
import copy
from transformers import (
    AutoConfig,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_pt_utils import LengthGroupedSampler

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler



def main():
    MAXLENI = 512
    MAXLENO = 128

    with jsonlines.open('../GENIE/GENIE/datasets/ToTTo/train.jsonl') as f:
        train = [i for i in f.iter()]

    with jsonlines.open('../GENIE/GENIE/datasets/ToTTo/valid.jsonl') as f:
        valid = [i for i in f.iter()]


    tokenizer = T5TokenizerFast.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained('t5-small', return_dict=True)

    # special_tokens_dict = {'pad_token': '<pad>', 'bos_token': '<bos>', 'eos_token': '<eos>',
    #                        'additional_special_tokens': []}
    #                        # 'additional_special_tokens': ['<PAGESTART>', '<PAGEEND>', '<SECTIONSTART>', '<SECTIONEND>',
    #                        #                               '<TABLESTART>', '<TABLEEND>', '<CELLSTART>', '<CELLEND>',
    #                        #                               '<COLHEADERSTART>',
    #                        #                               '<COLHEADEREND>', '<ROWHEADERSTART>', '<ROWHEADEREND>']}
    #
    # num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    #
    # print('We have added', num_added_toks, 'tokens')
    # model.encoder.resize_token_embeddings(len(tokenizer))
    # model.decoder.resize_token_embeddings(len(tokenizer))

    class tottodataset(Dataset):
        def __init__(self, df, tokenizer):
            self.sentence = [i['trg'] for i in df]
            self.table = [i['src'] for i in df]
            self.tokenizer = tokenizer

        def __len__(self):
            return len(self.sentence)

        def __getitem__(self, idx):
            inp = (self.table[idx])
                # .replace("<page_title>", "<PAGESTART>").replace("</page_title>",
                #                                                                             "<PAGEEND>") \
                # .replace("<section_title>", "<SECTIONSTART>").replace("</section_title>", "<SECTIONEND>") \
                # .replace("<table>", "<TABLESTART>").replace("</table>", "<TABLEEND>") \
                # .replace("<cell>", "<CELLSTART>").replace("</cell>", "<CELLEND>") \
                # .replace("<col_header>", "<COLHEADERSTART>").replace("</col_header>", "<COLHEADEREND>") \
                # .replace("<row_header>", "<ROWHEADERSTART>").replace("</row_header>", "<ROWHEADEREND>")
            out = self.sentence[idx]
            inp_tokens = self.tokenizer(inp, max_length=512, padding=False, truncation=True)
            out_tokens = self.tokenizer(out, max_length=128, padding=False, truncation=True)
            inp_id = inp_tokens.input_ids
            # out_id = out_tokens.input_ids
            inp_mask = inp_tokens.attention_mask
            # out_mask = out_tokens.attention_mask
            labels = out_tokens.input_ids.copy()
            labels = [-100 if x == self.tokenizer.pad_token_id else x for x in labels]

            return {
                "input_ids": inp_id,
                "attention_mask": inp_mask,
                # "decoder_input_ids": torch.tensor(out_id, dtype=torch.long),
                # "decoder_attention_mask": torch.tensor(out_mask, dtype=torch.long),
                "labels": labels
            }

    train_df = train
    val_df = valid

    train_dataset = tottodataset(train_df, tokenizer)
    val_dataset = tottodataset(val_df, tokenizer)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None
    )

    generator = torch.Generator()
    # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
    # `args.seed`) if data_seed isn't provided.
    # Further on in this method, we default to `args.seed` instead.
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator.manual_seed(seed)
    # sampler = RandomSampler(train_dataset, generator=generator)
    sampler = LengthGroupedSampler(
        8,
        dataset=train_dataset,
        lengths=None,
        model_input_name='input_ids',
        generator=generator,
    )

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=8,
                                  num_workers=0,
                                  collate_fn=data_collator,
                                  sampler=sampler
                                  # shuffle=False
                                  )


    val_dataloader = DataLoader(val_dataset,
                                batch_size=8,
                                num_workers=0,
                                collate_fn=data_collator,
                                sampler=SequentialSampler(val_dataset)
                                # shuffle=False
                                )

    dataloaders = {'train': train_dataloader, 'eval': val_dataloader}

    dataset_sizes = {'train': len(train_dataset), 'eval': len(val_dataset)}

    def train_fn(model, optimizer, num_epochs=5):
        since = time.time()
        best_wts = copy.deepcopy(model.state_dict())
        best_loss = float('inf')
        for epoch in range(num_epochs):

            print(f'Epoch:{epoch}/{num_epochs}')
            print('-' * 10)

            for mode in ['train', 'eval']:
                if mode == 'train':
                    model.train()
                elif mode == 'eval':
                    model.eval()

                running_loss = 0.0
                for istep, data in enumerate(dataloaders[mode]):
                    input_ids = data["input_ids"].to(device, dtype=torch.long)
                    labels = data['labels'].to(device, dtype=torch.long)
                    attention_mask = data['attention_mask'].to(device, dtype=torch.long)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(mode == 'train'):
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                        loss, logits = outputs[:2]


                        if mode == 'train':
                            loss.backward()

                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                1.0,
                            )

                            optimizer.step()
                            if istep % 1000 == 0:
                                print(istep, loss)
                        running_loss += loss.item()


                epoch_loss = running_loss / dataset_sizes[mode]

                print('{} Loss: {:.4f} '.format(
                    mode, epoch_loss))

                if mode == 'eval' and epoch_loss < best_loss:
                    best_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss

                print()

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val loss: {:4f}'.format(best_loss))

            model.load_state_dict(best_wts)
        return model

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-4,
                      weight_decay=0,
                      eps=1e-08)

    history = train_fn(model, optimizer, num_epochs=10)

    torch.save(model, "./T5Epoch:10")


if __name__ == '__main__':
    main()