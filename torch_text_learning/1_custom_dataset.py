# STEPS:

# 1. Specify how preprocessing should be done - Fields
# 2. Use dataset to load the data -> TabularDataset(JSON/CSV/TSV)
# 3. Construct an iterator to do batching & padding -> BucketIterator

from torchtext.data import Field,TabularDataset,BucketIterator


tokenizer = lambda x: x.split()

quote = Field(sequential=True,use_vocab=True,tokenize=tokenizer,lower=True)
score = Field(sequential=False,use_vocab=False)

fields = {"quote": ('q',quote),"score": ('s',score)}

train_data,test_data = TabularDataset.splits(
                        path="dataset",
                        train="train.json",
                        test="test.json",
                        format='json',
                        fields=fields
                    )

print(train_data[0].__dict__.keys())
print(train_data[0].__dict__.values())

quote.build_vocab(train_data,max_size=10000,min_freq=1)

train_iter,test_iter = BucketIterator.splits(
    (train_data,test_data),
    batch_size=2,
    device="cuda"
)

for batch in train_iter:
    print(batch.q)
    