# Dataset

## Usage

```py
from datasets import Dataset
```

1. The base class Dataset implements a Dataset

   ```py
   Dataset({
       features: ['text', 'label'],
       num_rows: 8530
   })
   ```

## Attributes

1. ***info*** (The information of the dataset)

   `datasets.info.DatasetInfo`

   ```py
   DatasetInfo(
       description='', citation='', homepage='', license='', 
       features={'text': Value(dtype='string', id=None), 
                 'label': ClassLabel(names=['neg', 'pos'], id=None)}, 
       post_processed=None, 
       supervised_keys=None, 
       builder_name='parquet', 
       dataset_name='rotten_tomatoes', 
       config_name='default', 
       version=0.0.0, 
       splits={'train': SplitInfo(name='train', num_bytes=1075873, num_examples=8530, shard_lengths=None, dataset_name='rotten_tomatoes'), 
               'validation': SplitInfo(name='validation', num_bytes=134809, num_examples=1066, shard_lengths=None, dataset_name='rotten_tomatoes'), 
               'test': SplitInfo(name='test', num_bytes=136102, num_examples=1066, shard_lengths=None, dataset_name='rotten_tomatoes')}
   ...)
   ```

2. ***split*** (The split of the dataset)

   `datasets.splits.NamedSplit`

   ```py
   train
   ```

3. ***[index]*** (Indexing visit of the dataset)

   >2. Visit by **[col_name] [row_index]**
   >
   >   | col_name_1               | col_name_2                   | col_name_3               | ...... | col_name_n               |
   >   | ------------------------ | ---------------------------- | ------------------------ | ------ | ------------------------ |
   >   | `dataset[col_name_1][0]` | `dataset[col_name_2][0]`     | `dataset[col_name_3][0]` | ...... | `dataset[col_name_n][0]` |
   >   | `dataset[col_name_1][1]` | ==`dataset[col_name_2][1]`== | `dataset[col_name_3][1]` | ...... | `dataset[col_name_n][1]` |
   >   | ......                   | ......                       | ......                   | ...... | ......                   |
   >   | `dataset[col_name_1][2]` | `dataset[col_name_2][2]`     | `dataset[col_name_3][2]` | ...... | `dataset[col_name_n][2]` |
   >
   >   ```py
   >   dataset[0]["text"]
   >   
   >   # Return value
   >   'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'
   >   ```

   >1. Visit by **[row_index] [col_name]**
   >
   >   | col_name_1               | col_name_2                   | col_name_3               | ...... | col_name_n               |
   >   | ------------------------ | ---------------------------- | ------------------------ | ------ | ------------------------ |
   >   | `dataset[0][col_name_1]` | `dataset[0][col_name_2]`     | `dataset[0][col_name_3]` | ...... | `dataset[0][col_name_n]` |
   >   | `dataset[1][col_name_1]` | ==`dataset[1][col_name_2]`== | `dataset[1][col_name_3]` | ...... | `dataset[1][col_name_n]` |
   >   | ......                   | ......                       | ......                   | ...... | ......                   |
   >   | `dataset[2][col_name_1]` | `dataset[2][col_name_2]`     | `dataset[2][col_name_3]` | ...... | `dataset[2][col_name_n]` |
   >
   >   ```py
   >   dataset["text"][0]
   >   
   >   # Return value
   >   'the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'
   >   ```

   >
   >
   >3. Visit by **row / row slice**
   >
   >   | col_name_1                   | col_name_2                   | col_name_3                   | ......     | col_name_n                   |
   >   | ---------------------------- | ---------------------------- | ---------------------------- | ---------- | ---------------------------- |
   >   | `dataset[0][col_name_1]`     | `dataset[0][col_name_2]`     | `dataset[0][col_name_3]`     | ......     | `dataset[0][col_name_n]`     |
   >   | ==`dataset[1][col_name_1]`== | ==`dataset[1][col_name_2]`== | ==`dataset[1][col_name_3]`== | ==......== | ==`dataset[1][col_name_n]`== |
   >   | ......                       | ......                       | ......                       | ......     | ......                       |
   >   | `dataset[2][col_name_1]`     | `dataset[2][col_name_2]`     | `dataset[2][col_name_3]`     | ......     | `dataset[2][col_name_n]`     |
   >
   >   ```py
   >   dataset[3:6]
   >   
   >   # Return value
   >   {
   >       'label': [1, 1, 1],
   >    	'text': [
   >           'if you sometimes like to go to the movies to have fun , wasabi is a good place to start .',
   >     		"emerges as something rare , an issue movie that's so honest and keenly observed that it doesn't feel like one .",
   >     		'the film provides some great insight into the neurotic mindset of all comics -- even those who have reached the absolute top of the game .'
   >       		]
   >   }
   >   ```

   > 4. Visit by **col**
   >
   >    | col_name_1                 | col_name_2                   | col_name_3               | ...... | col_name_n               |
   >    | -------------------------- | ---------------------------- | ------------------------ | ------ | ------------------------ |
   >    | `dataset[0][col_name_1]`   | ==`dataset[0][col_name_2]`== | `dataset[0][col_name_3]` | ...... | `dataset[0][col_name_n]` |
   >    | `dataset[1][col_name_1]`   | ==`dataset[1][col_name_2]`== | `dataset[1][col_name_3]` | ...... | `dataset[1][col_name_n]` |
   >    | ......                     | ==......==                   | ......                   | ...... | ......                   |
   >    | `dataset[2][col_name_1]==` | ==`dataset[2][col_name_2]`== | `dataset[2][col_name_3]` | ...... | `dataset[2][col_name_n]` |
   >
   >    ```py
   >    dataset[3: 10]['label']
   >       
   >    # Return value
   >    [1, 1, 1, 1, 1, 1, 1]
   >    ```

4. ***column_names*** (Names of the columns in the dataset)

   `List[str]`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds.column_names
   
   # value
   ['text', 'label']
   ```

## Methods

### ***DatasetObj.to_iterable_dataset()***

> Transform a `Dataset` object to its equivalent `IterableDataset` object

#### Return

`IterableDataset` 	The equivalent IterableDataset of the original dataset object

```python
dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
iterable_dataset = dataset.to_iterable_dataset()

# Retuen value
IterableDataset({
    features: ['text', 'label'],
    num_shards: 1
})
```

***

### ***DatasetObj.set_format()***

> Change the format of the content of the dataset

#### Returns 

`None`	This method will change the config inside the dataset that called it in place

#### Parameters

1. ***type*** (The output type of which the dataset is going to be transformed)

   ==`None`== | `numpy` | `pandas` | `torch`

   > Set to `numpy`

   ```py
   dataset.set_format('numpy')
   
   # The dataset inplace
   {
       'label': array([1, 1, 1, 1, 1], dtype=int64),
    	'text': array(['the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .',
          'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . tolkien\'s middle-earth .',
          'effective but too-tepid biopic',
          'if you sometimes like to go to the movies to have fun , wasabi is a good place to start .',
          "emerges as something rare , an issue movie that's so honest and keenly observed that it doesn't feel like one ."],dtype='<U226')
   }
   ```

   > Set to `pandas`

   ```py
   dataset.set_format('pandas')
   
   # The dataset inplace
                                                   text  label
   0  the rock is destined to be the 21st century's ...      1
   1  the gorgeously elaborate continuation of " the...      1
   2                     effective but too-tepid biopic      1
   3  if you sometimes like to go to the movies to h...      1
   4  emerges as something rare , an issue movie tha...      1
   ```

   > Set to `torch`

   ```py
   dataset.set_format('torch')
   
   # The dataset inplace
   {
       'label': tensor([1, 1, 1, 1, 1]),
    	'text': ['the rock is destined to be the 21st century\'s new " conan " and '
                 "that he's going to make a splash even greater than arnold "
                 'schwarzenegger , jean-claud van damme or steven segal .',
                 'the gorgeously elaborate continuation of " the lord of the rings " '
                 'trilogy is so huge that a column of words cannot adequately '
                 "describe co-writer/director peter jackson's expanded vision of j . "
                 "r . r . tolkien's middle-earth .",
                 'effective but too-tepid biopic',
                 'if you sometimes like to go to the movies to have fun , wasabi is a '
                 'good place to start .',
                 "emerges as something rare , an issue movie that's so honest and "
                 "keenly observed that it doesn't feel like one ."]
   }
   ```

***

### ***DatasetObj.from_generator()***

> Create a Dataset based on a python generator

#### Returns

`Dataset`	The dataset you create based on the generator you passed in

#### Parameters

1. ***generator*** (A generator function that yields examples)

   `Callable`

   ```py
   def create_examples():
       size = 100
       for _ in range(size):
           x = np.random.randint(high=10, low=0)
           y = np.random.randint(high=10, low=0)
           label = x + y
   
           yield {'col_1': x, 'col_2': y, 'label': label}
           
   dataset = Dataset.from_generator(create_examples)
   dataset.set_format('pandas')
   
   # Return value
      col_1  col_2  label
   0      8      9     17
   1      7      4     11
   2      4      7     11
   3      8      5     13
   4      2      5      7
   ```

2. ***num_proc*** (Number of processes when downloading and generating the dataset locally)

   ==`None`== | `int`

   ```py
   dataset = Dataset.from_generator(create_examples, num_proc=8)
   ```

***

### ***DatasetObj.from_dict()***

> Create a Dataset based on a python dictionary

#### Returns

`Dataset`	The dataset you create based on the dictionary you passed in

#### Parameters

1. ***mapping*** (The dictionary of strings to Arrays or Python lists)

   `Dict`

   ```py
   dataset = Dataset.from_dict({
       "a": [1, 2, 3],
       "b": [1, 2, 3],
       "c": [1, 2, 3],
       "d": [1, 2, 3],
   })
   ```

***

### ***DatasetObj.from_pandas()***

> Create a Dataset based on a `panads.Dataframe`

#### Returns

`Dataset`	The dataset you create based on the dataframe you passed in

#### Parameters

1. ***df*** (Dataframe that contains the dataset)

   `pandas.Dataframe`

   ```py
   df = pd.DataFrame(
       np.arange(1, 12 + 1).reshape((3, 4)),
       columns=list("abcd"),
       index=list("xyz")
   )
   dataset = Dataset.from_pandas(df, preserve_index=False)
   dataset.set_format("pandas")
   
   # Return value
      a   b   c   d
   0  1   2   3   4
   1  5   6   7   8
   2  9  10  11  12
   
   ```

2. ***preserve_index*** (Whether to store the index as an additional column in the resulting Dataset)

   ==`bool=True`==

   ```py
   df = pd.DataFrame(
       np.arange(1, 12 + 1).reshape((3, 4)),
       columns=list("abcd"),
       index=list("xyz")
   )
   dataset = Dataset.from_pandas(df, preserve_index=True)
   dataset.set_format("pandas")
   
   # Return value
      a   b   c   d __index_level_0__
   0  1   2   3   4                 x
   1  5   6   7   8                 y
   2  9  10  11  12                 z
   ```

***

### ***DatasetObj.sort()***

> Create a new dataset sorted according to a single or multiple columns

#### Returns

`Dataset`	The sorted new dataset

#### Parameters

1. ***column_names*** (Column name(s) to sort by, the priority decreases along the order of the list)

   `str` | `List[str]`

   ```py
   ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='validation')
   sorted_ds = ds.sort('label')
   
   # Return value (sorted_ds[0: 3]['label'])
   [0, 0, 0]
   ```

2. ***reverse*** (For each column(s), decide to sort by descending order rather than ascending)

   ==`bool=False`==

   ```py
   ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='validation')
   another_sorted_ds = ds.sort(['label', 'text'], reverse=[True, False])
   
   # Return value (sorted_ds[0: 3]['label'])
   [1, 1, 1]
   ```

***

### ***DatasetObj.shuffle()***

> Create a new Dataset where the rows are shuffled

#### Returns

`Dataset`	The shuffled new dataset

#### Parameters

1. ***seed*** (The random seed for shuffling)

   `int`

   ```py
   ds = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='validation')
   my_dataset = my_dataset.shuffle(seed=42)
   ```

***

### ***DatasetObj.select()***

> Create a new dataset with rows selected following the list/array of indices

#### Returns

`Dataset`	The new dataset containing only the selected rows

#### Parameters

1. ***indices*** (Range, list or 1D-array of integer indices for indexing)

   `Range` | `List` | `Iterable` | `numpy.ndarray` | `pandas.Series`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds.select(range(4))
   ```

***

### ***DatasetObj.filter()***

> Apply a filter function to all the elements in the table in batches and update the table so that the dataset only includes examples according to the filter function

#### Returns

`Dataset`	The new dataset containing only the roles satisfying the filter function

#### Parameters

1. ***function*** (Callable with default signatures)

   `Callable`

   > Default function signature: `function(row) -> bool`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds.filter(lambda x: x["label"] == 1)
   
   # Return value
   Dataset({
       features: ['text', 'label'],
       num_rows: 533
   })
   ```

2. ***with_indices*** (Pass the index with the row to the function)

   ==`bool=False`==

   > Index added function signature: `function(row, index) -> bool`

   ```py
   dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   even_dataset = dataset.filter(lambda example, idx: idx % 2 == 0, with_indices=True)
   ```

3. ***num_proc*** (Number of processes when downloading and generating the dataset locally)

   ==`None`== | `int`

   > Have to run under `if __name__ == __main__`

   ```py
   if __name__ == '__main__':
       ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
       ds.filter(lambda x: x["label"] == 1, num_proc=8)
   ```

4. ***batched*** (Provide batch of examples to `function`)

   ==`bool=False`==

5. ***batched_size*** (Number of examples per batch provided to `function` if `batched = True`)

   ==`int=1000`==

   > Batched function signature: `function(rows_in_dict) -> List[bool]`

   ```py
   def filter_batched_text_len(x):
       clip_length = 100
       content_len = list(map(lambda k: len(k) < clip_length, x['text']))
       return content_len
   
   # Batched sample (batched rows)
   {
       'text': ['the rock is destined to be the 21st century\'s new " conan " and that he\'s going to make a 				splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .', 'the 				gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of 			   words cannot adequately describe co-writer/director peter jackson\'s expanded vision of j . r . r . 			   tolkien\'s middle-earth .', 'effective but too-tepid biopic', 'if you sometimes like to go to the 			 movies to have fun , wasabi is a good place to start .'], 
    	'label': [1, 1, 1, 1]
   }
   
   dataset = load_dataset('cornell-movie-review-data/rotten_tomatoes', split='train', num_proc=8)
   dataset_filtered = dataset.filter(filter_batched_text_len, batched=True, batch_size=4)
   
   # Original dataset
   Dataset({
       features: ['text', 'label'],
       num_rows: 8530
   })
   
   # Filtered dataset
   Dataset({
       features: ['text', 'label'],
       num_rows: 3551
   })
   ```

***

### ***DatasetObj.train_test_split()***

> Return a `datasets.DatasetDict` with two random train and test `dataset` with key `"train"` and `"test"`

#### Returns

`DatasetDict`

​	`"train"`: `Dataset`	The training split as a subset `Dataset` from the total `Dataset`

​	`"test"`: `Dataset`	The testing split as a subset `Dataset` from the total `Dataset`

#### Parameters

1. ***test_size*** (Size of the test split)

   `None` | `float` | `int`

2. ***train_size*** (Size of the train split)

   `None` | `float` | `int`

   > 1. If `_size` is an instance of `float`, interpret as fractional size
   > 2. If `_size` is an instance of `int`, interpret as actual number of rows
   > 3. If `_size` is `None`, automatically set to the complement of the other `_size`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.train_test_split(test_size=0.2)
   
   # Return value
   DatasetDict({
       train: Dataset({
           features: ['text', 'label'],
           num_rows: 852
       })
       test: Dataset({
           features: ['text', 'label'],
           num_rows: 214
       })
   })
   ```

3. ***shuffle*** (Whether or not to shuffle the data before splitting)

   ==`bool=True`==

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.train_test_split(test_size=0.2, shuffle=True)
   ```

4. ***seed*** (The random seed)

   ==`None`== | `int`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.train_test_split(test_size=0.2, shuffle=True, seed=42)
   ```

***

### ***DatasetObj.shard()***

> Divide a very large dataset into a predefined number of chunks by row

#### Returns

`Dataset`	The specified chunk of `Dataset` in the divided chunks from the total `Dataset`

#### Parameters

1. ***num_shards*** (How many shards to split the dataset into)

   `int`

2. ***index*** (Which shard to select and return)

   `int`

   ```py
   dataset.shard(num_shards=4, index=1)
   
   # Return value
   Dataset({
       features: ['text', 'label'],
       num_rows: 6250
   })
   ```

***

### ***DatasetObj.rename()***

> Create a copy of the dataset with a renamed column

#### Returns

`Dataset`	The new dataset with the single column name changed

#### Parameters

1. ***original_column_name*** (Name of the column to rename)

   `str`

2. ***new_column_name*** (New name for the column)

   `str`

   ```py
   dataset = dataset.rename_column("sentence1", "sentenceA")
   dataset = dataset.rename_column("sentence2", "sentenceB")
   
   # Original dataset
   Dataset({
       features: ['sentence1', 'sentence2', 'label', 'idx'],
       num_rows: 3668
   })
   
   # Return value
   Dataset({
       features: ['sentenceA', 'sentenceB', 'label', 'idx'],
       num_rows: 3668
   })
   ```

***

### ***DatasetObj.remove()***

> Return a copy of the dataset object without the columns to remove

#### Returns

`Dataset`	The new dataset object without the specified column(s)

#### Parameters

1. ***column_names*** (Name of the column(s) to remove)

   `str` | `List[str]`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.remove_columns('label')
   
   # Return 
   Dataset({
       features: ['text'],
       num_rows: 1066
   })
   ```

***

### ***DatasetObj.select_columns()***

> Return a copy of the dataset object which only consists of selected columns

#### Returns

`Dataset`	The new dataset object with the specified column(s) only

#### Parameters

1. ***column_names*** (Name of the column(s) to remove)

   `str` | `List[str]`

   ```py
   dataset = dataset.select_columns(['sentence1', 'sentence2', 'idx'])
   
   # Return value
   Dataset({
       features: ['sentence1', 'sentence2', 'idx'],
       num_rows: 3668
   })
   ```

***

### ***DatasetObj.cast()***

> Return a copy of the dataset with casted features

#### Returns

`Dataset`	The new dataset object with all features casted

#### Parameters

1. ***features*** (New features to cast the dataset to)

   `Features`

   > - The name of the fields in the features must match the current column names
   > - The type of the data must also be convertible from one type to the other

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   new_features = Features({'text': Value('large_string'), 'label': ClassLabel(names=['bad', 'good'])})
   ds = ds.cast(new_features)
   
   # Return value
   Dataset({
       features: ['text', 'label'],
       num_rows: 1066
   })
   ```

2. ***batch_size*** (Number of examples per batch provided to cast)

   ==`int=1000`==

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   new_features = Features({'text': Value('large_string'), 'label': ClassLabel(names=['bad', 'good'])})
   ds = ds.cast(new_features, batch_size=128)
   
   # Return value
   Dataset({
       features: ['text', 'label'],
       num_rows: 1066
   })
   ```

3. ***num_proc*** ()

   `int`

   > Have to run under `if __name__ == __main__`

   ```py
   if __name__ == '__main__':
       ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
       new_features = Features({'text': Value('large_string'), 'label': ClassLabel(names=['bad', 'good'])})
       ds = ds.cast(new_features, batch_size=128, num_proc=4)
      
   # Return value
   Dataset({
       features: ['text', 'label'],
       num_rows: 1066
   })
   ```

***

### ***DatasetObj.cast_column()***

> Return a copy of the dataset with casted feature for the specific column

#### Returns

`Dataset`	The new dataset with casted feature for the specific column

#### Parameters

1. ***column*** (Column name)

   `str`

2. ***feature*** (Target feature)

   `Features`

   ```py
   dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
   ```

***

### ***DatasetObj.flatten()***

> Return a copy of the dataset with all column with nested structure Features expanded into new columns

#### Returns

`Dataset`	The new dataset with the expanded columns

```py
dataset = load_dataset("rajpurkar/squad", split="train")
flat_dataset = dataset.flatten()

# Original dataset feature (answers is a Sequence type)
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers'],
    num_rows: 87599
})

# Return value feature
Dataset({
    features: ['id', 'title', 'context', 'question', 'answers.text', 'answers.answer_start'],
 num_rows: 87599
})
```

***

### ***DatasetObj.map()***

> - Apply a function to all the examples in the dataset (individually or in batches) and update the dataset
> - If your function returns a column that already exists, then it overwrites it

#### Returns

`Dataset`	The new dataset with all elements updated according to the function

#### Parameters

1. ***function*** (Callable with default signatures)

   `Callable`

   > Default function signature: `function(row: Dict[str, Any]) -> row: Dict[str, Any]`

   ```py
   def add_prefix(example):
       example["sentence1"] = 'My sentence: ' + example["sentence1"]
       return example
   updated_dataset = small_dataset.map(add_prefix)
   
   # Return value (updated_dataset["sentence1"][:5])
   [
       'My sentence: Amrozi accused his brother , whom he called " the witness " , of deliberately distorting his 	   evidence .',
   	"My sentence: Yucaipa owned Dominick 's before selling the chain to Safeway in 1998 for $ 2.5 billion .",
   	'My sentence: They had published an advertisement on the Internet on June 10 , offering the cargo for sale 	   , he added .',
   	'My sentence: Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a 	record high of A $ 4.57 .',
   ]
   ```

2. ***with_indices*** (Pass the index with the row to the function)

   ==`bool=False`==

   > Index added signature `function(row: Dict[str, Any], index: int) -> row: Dict[str, Any]`

   ```py
   def funct(row, index):
       row['text'] = f'Trail #{index + 1}: ' + row['text']
       return row
   
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.map(funct, with_indices=True)
   ds.set_format("pandas")
   
   # Return value
                                                   text  label
   0  Trail #1: compassionately explores the seeming...      1
   1  Trail #2: the soundtrack alone is worth the pr...      1
   2  Trail #3: rodriguez does a splendid job of rac...      1
   ```

3. ***batched*** (Provide batch of examples to `function`)

   ==`bool=False`==

4. ***batched_size*** (Number of examples per batch provided to `function` if `batched = True`)

   ==`int=1000`==

   > Batched function signature: `function(row: Dict[str, List]) -> row: Dict[str, List]`

   ```py
   tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
   def funct(rows):
       rows['text'] = tokenizer(rows['text'])['input_ids']
       pprint(rows)
       return rows
   
   # Batched sample (batched rows)
   {
       'text': ['compassionately explores the seemingly irreconcilable situation between conservative christian 			 parents and their estranged gay and lesbian children .', 
                'the soundtrack alone is worth the price of admission .', 
                'rodriguez does a splendid job of racial profiling hollywood style--casting excellent latin actors 			of all ages--a trend long overdue .', 
                "beneath the film's obvious determination to shock at any cost lies considerable skill and 				determination , backed by sheer nerve ."], 
       'label': [1, 1, 1, 1]
   }
   
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.map(funct, batched=True, batch_size=4)
   ds.set_format("pandas")
   
   # Original dataset
                                         text  label
   0  compassionately explores the seeming...      1
   1  the soundtrack alone is worth the pr...      1
   2  rodriguez does a splendid job of rac...      1
   
   # Return value
                                                   text  label
   0  [101, 29353, 2135, 15102, 1996, 9428, 20868, 2...      1
   1  [101, 1996, 6050, 2894, 2003, 4276, 1996, 3976...      1
   2  [101, 9172, 2515, 1037, 21459, 3105, 1997, 576...      1
   ```

   > Batch-Index function signature: `function(row: Dict[str, List], index: List[int]) -> row: Dict[str, List]`

   ```py
   def funct(rows, idx):
       batch_num = (idx[0] // 4) + 1
       rows['text'] = list(map(lambda x: f"Batch #{batch_num}: " + x, rows['text']))
       return rows
   
   # Batched sample (batched rows & index)
   {
       'text': ['compassionately explores the seemingly irreconcilable situation between conservative christian 			 parents and their estranged gay and lesbian children .', 
                'the soundtrack alone is worth the price of admission .', 
                'rodriguez does a splendid job of racial profiling hollywood style--casting excellent latin actors 			of all ages--a trend long overdue .', 
                "beneath the film's obvious determination to shock at any cost lies considerable skill and 				determination , backed by sheer nerve ."], 
       'label': [1, 1, 1, 1]
   }
   [0, 1, 2, 3]
   
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.map(funct, with_indices=True, batched=True, batch_size=4)
   ds.set_format("pandas")
   
   # Original dataset
                                         text  label
   0  compassionately explores the seeming...      1
   1  the soundtrack alone is worth the pr...      1
   2  rodriguez does a splendid job of rac...      1
   
   # Return value
                                                   text  label
   0  Batch #1: compassionately explores the seeming...      1
   1  Batch #1: the soundtrack alone is worth the pr...      1
   2  Batch #1: rodriguez does a splendid job of rac...      1
   ```

5. ***num_proc*** (Max number of processes apply in map function)

   `int`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds = ds.map(funct, batched=True, batch_size=4, num_proc=4)
   ```

6. ***remove_columns*** (Remove a selection of columns while doing the mapping)

   `str` | `List[str]`

   > Columns will be removed before updating the examples with the output of `function`, so repetitive column names are allowed

   ```py
   updated_dataset = dataset.map(
       lambda example: {"sentence1": "<bos>" + example["sentence1"] + "<eos>"}, 
       remove_columns=["sentence1"]
   )
   ```

***

### ***DatasetObj.batch()***

> returns a new Dataset where each item is a batch of multiple samples from the original dataset

#### Returns

`Dataset`	The new dataset with each batch_size rows squeezed into 1 row

#### Parameters

1. ***batch_size*** (The number of samples in each batch)

   `int`

   ```py
   dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
   batched_dataset = dataset.batch(batch_size=4)
   
   # Return value (batched_dataset[0])
   {
    'text': ['the rock is destined to be the 21st century\'s new " conan " and '
             "that he's going to make a splash even greater than arnold "
             'schwarzenegger , jean-claud van damme or steven segal .',
             'the gorgeously elaborate continuation of " the lord of the rings " '
             'trilogy is so huge that a column of words cannot adequately '
             "describe co-writer/director peter jackson's expanded vision of j . "
             "r . r . tolkien's middle-earth .",
             'effective but too-tepid biopic',
             'if you sometimes like to go to the movies to have fun , wasabi is a '
             'good place to start .']
    'label': [1, 1, 1, 1],
   }
   ```

2. ***drop_last_batch*** (Whether to drop the last incomplete batch if the dataset size is not divisible by the batch size)

   ==`bool=False`==

   ```py
   dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
   batched_dataset = dataset.batch(batch_size=4, drop_last_batch=True)
   ```

3. ***num_proc*** (The number of processes to use for multiprocessing)

   ==`None`== | `int`

   > Have to run under `if __name__ == __main__`

   ```py
   if __name__ == '__main__':
   	dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
       batched_dataset = dataset.batch(batch_size=4, num_proc=4)
   ```

***

### ***DatasetObj.with_format()***

> - Return a new dataset object with the format of column(s) / entire dataset changed to be compatible with some common data formats
> - The difference between `.with_format()` and `.set_format()` is that `.with_format()` will return a new dataset, whereas `.set_format()` changes in place

#### Returns

`Dataset`	The new dataset with its column(s) / entire dataset changed into the specified type

#### Parameters

1. ***type*** (String representation of the desired type to be changed into)

   `str='numpy' | 'torch' | 'pandas'`

   ```py
   ds = ds.with_format("torch")
   ```

2. ***columns*** (Columns to format in the output)

   ==`None`== | `List[str]`

   > `None` means all columns (the entire dataset) 's type will be changed

   ```py
   dataset = dataset.with_format(type="torch", columns=['label'])
   ```

***

### ***DatasetObj.save_to_disk()***

> Save the dataset locally

#### Parameters

1. ***dataset_path*** (the saving path or remote URL)

   `str` | `os.Pathlike`

   ```py
   encoded_dataset.save_to_disk("path/of/my/dataset/directory")
   ```

***

### DatasetObj.to_csv()

> Save the dataset locally as a csv file

#### Parameters

1. ***path_or_buf*** (The saving path or remote URL)

   `str` | `os.Pathlike`

   ```py
   encoded_dataset.to_csv("path/of/my/dataset.csv")
   ```

***

### DatasetObj.to_json()

> Save the dataset locally as a json file

#### Parameters

1. ***path_or_buf*** (The saving path or remote URL)

   `str` | `os.Pathlike`

   ```py
   ds.to_json("path/to/dataset/directory/filename.jsonl")
   ```

***




# IterableDataset

## Usage

```python
from datasets import IterableDataset
```

1. Dataset class that read data by steaming, either locally or via online connection

   ```py
   IterableDataset({
       features: ['text', 'label'],
       num_shards: 1
   })
   ```

## Attributes

1. ***info*** (The information of the dataset)

   `datasets.info.DatasetInfo`

   ```py
   DatasetInfo(
       description='', citation='', homepage='', license='', 
       features={'text': Value(dtype='string', id=None), 
                 'label': ClassLabel(names=['neg', 'pos'], id=None)}, 
       post_processed=None, 
       supervised_keys=None, 
       builder_name='parquet', 
       dataset_name='rotten_tomatoes', 
       config_name='default', 
       version=0.0.0, 
       splits={'train': SplitInfo(name='train', num_bytes=1075873, num_examples=8530, shard_lengths=None, dataset_name='rotten_tomatoes'), 
               'validation': SplitInfo(name='validation', num_bytes=134809, num_examples=1066, shard_lengths=None, dataset_name='rotten_tomatoes'), 
               'test': SplitInfo(name='test', num_bytes=136102, num_examples=1066, shard_lengths=None, dataset_name='rotten_tomatoes')}
   ...)
   ```

2. ***iter()*** (Visit by treating it as a python `generator` object)

   > Transform explicitly it into an iterable object

   ```python
   next(iter(iterable_dataset))
   
   # Return value
   {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x7F0681F59B50>,
    'label': 6}
   ```

   ```python
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train", streaming=True)
   list(ds)
   
   #Return value
   [{'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x7F7479DEE9D0>,
     'label': 6},
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x512 at 0x7F7479DE8190>,
     'label': 6},
    {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=512x383 at 0x7F7479DE8310>,
     'label': 6}]
   ```

   > Transform implicitly it into an iterable object

   ```python
   for example in iterable_dataset:
       print(example)
       break
       
   # Return value
   {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=384x512 at 0x7F7479DE82B0>, 'label': 6}
   ```

## Methods

### ***IterableDatasetObj.take()***

> Create a new IterableDataset with only the first n elements

#### Returns

`IterableDataset` 	The smaller IterableDataset object containing the first n elements

#### Parameters

1. ***n*** (Number of elements to take)

   `int`

   ```python
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train", streaming=True)
   small_ds = ds.take(2)
   list(small_ds)
   
   # Retuen value
   [{'label': 1,
    'text': 'the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .'},
    {'label': 1,
    'text': 'the gorgeously elaborate continuation of " the lord of the rings " trilogy is so huge that a column of words cannot adequately describe co-writer/director peter jackson's expanded vision of j . r . r . tolkien's middle-earth .'}]
   ```



# DatasetDict

## Usage

```py
from datasets import DatasetDict
```

1. A dictionary (dict of str: datasets.Dataset) with dataset transforms methods (map, filter, etc.)

   ```py
   DatasetDict({
       train: Dataset({
           features: ['text', 'label'],
           num_rows: 8530
       })
       validation: Dataset({
           features: ['text', 'label'],
           num_rows: 1066
       })
       test: Dataset({
           features: ['text', 'label'],
           num_rows: 1066
       })
   })
   ```

## Attribute

1. ***column_names*** (Names of the columns in each split of the dataset)

   `Dict[str, List[str]]`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
   ds.column_names
   
   # values
   {
       'test': ['text', 'label'],
    	'train': ['text', 'label'],
    	'validation': ['text', 'label']
   }
   ```

## Methods

### ***DatasetDictObj.map()***

> - The transformation is applied to all the datasets of the dataset dictionary
> - Apply a function to all the examples in the dataset (individually or in batches) and update the dataset
> - If your function returns a column that already exists, then it overwrites it

#### Returns

`DataDict`	The new datadict object with all datasets in it updated according to the function

#### Parameters

1. ***function*** (Callable with default signatures)

   `Callable`

   > Default function signature: `function(row: Dict[str, Any]) -> row: Dict[str, Any]`

   ```py
   def add_prefix(row):
       row["text"] = "Review: " + row["text"]
       return row
   
   # row
   {'text': "10 minutes into the film you'll be white-knuckled and unable to look away .", 'label': 1}
   
   datadict = load_dataset("cornell-movie-review-data/rotten_tomatoes")
   datadict = datadict.map(add_prefix)
   ```

2. ***with_indices*** (Pass the index with the row to the function)

   ==`bool=False`==

   > Index added signature `function(row: Dict[str, Any], index: int) -> row: Dict[str, Any]`

   ```py
   def funct(row, index):
       row['text'] = f'Trail #{index + 1}: ' + row['text']
       return row
   
   # row, index
   {
       'text': "one of the smarter offerings the horror genre has produced in recent memory , even if it's far 		   tamer than advertised .", 
       'label': 1
   }
   3365
   
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
   ds = ds.map(funct, with_indices=True)
   ds.set_format("pandas")
   ```

3. ***batched*** (Provide batch of examples to `function`)

   ==`bool=False`==

4. ***batched_size*** (Number of examples per batch provided to `function` if `batched = True`)

   ==`int=1000`==

   > Batched function signature: `function(row: Dict[str, List]) -> row: Dict[str, List]`

   ```py
   tokenizer = AutoTokenizer.from_pretrained("gpt2")
   def funct(rows):
       rows['sentence1'] = tokenizer(rows['sentence1'])['input_ids']
       rows['sentence2'] = tokenizer(rows['sentence2'])['input_ids']
       return rows
   
   # Batched sample (batched rows)
   {
       'sentence1': [
           'Some 175 million shares traded on the Big Board , a 7 percent increase from the same time a week ago 		  .',
           'The government defeated the rebel motion by 297 votes to 117 in the 659-seat House of Commons .',
           'Peter Smith , a planetary scientist at the University of Arizona , leads the $ 325 million Phoenix 		mission .',
           'The new Army Commander is the Masaka Armoured Brigade commanding officer , Brigadier Aronda Nyakairima 		who is now promoted to major general .'
       ],
       'sentence2': [
           'Some 1.6 billion shares traded on the Big Board , a 17 percent increase over the three-month daily 		average .',
           'It was defeated by 297 votes to 117 , a Government majority of 180 .',
           'TEGA is the product of University of Arizona planetary scientist William Boynton , co-investigator on 		   the Phoenix mission .',
           'PRESIDENT Yoweri Museveni has promoted Brigadier Aronda Nyakairima to Major General and named him Army 		Commander .'
       ],
       'label': [0, 0, 0, 1],
       'idx': [2362, 2363, 2364, 2365]
   }
   
   datadict = load_dataset('nyu-mll/glue', 'mrpc')
   datadict = datadict.map(funct, batched=True, batch_size=4)
   ```

   > Batch-Index function signature: `function(row: Dict[str, List], index: List[int]) -> row: Dict[str, List]`

   ```py
   def funct(rows, idx):
       batch_num = (idx[0] // 4) + 1
       rows['sentence1'] = list(map(lambda x: f"Batch #{batch_num}: " + x, rows['sentence1']))
       return rows
   
   # Batched sample (batched rows, indexes)
   {
       'sentence1': [
           'Some 175 million shares traded on the Big Board , a 7 percent increase from the same time a week ago 		  .',
           'The government defeated the rebel motion by 297 votes to 117 in the 659-seat House of Commons .',
           'Peter Smith , a planetary scientist at the University of Arizona , leads the $ 325 million Phoenix 		mission .',
           'The new Army Commander is the Masaka Armoured Brigade commanding officer , Brigadier Aronda Nyakairima 		who is now promoted to major general .'
       ],
       'sentence2': [
           'Some 1.6 billion shares traded on the Big Board , a 17 percent increase over the three-month daily 		average .',
           'It was defeated by 297 votes to 117 , a Government majority of 180 .',
           'TEGA is the product of University of Arizona planetary scientist William Boynton , co-investigator on 		   the Phoenix mission .',
           'PRESIDENT Yoweri Museveni has promoted Brigadier Aronda Nyakairima to Major General and named him Army 		Commander .'
       ],
       'label': [0, 0, 0, 1],
       'idx': [2362, 2363, 2364, 2365]
   }
   [2362, 2363, 2364, 2365]
   
   datadict = load_dataset('nyu-mll/glue', 'mrpc')
   datadict = datadict.map(funct, batched=True, batch_size=4, with_indices=True)
   
   ```

5. ***num_proc*** (Max number of processes apply in map function)

   `int`

   ```py
   ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
   ds = ds.map(funct, batched=True, batch_size=4, num_proc=4)
   ```

6. ***remove_columns*** (Remove a selection of columns while doing the mapping)

   `str` | `List[str]`

   > Columns will be removed before updating the examples with the output of `function`, so repetitive column names are allowed

   ```py
   updated_datadict = datadict.map(
       lambda example: {"sentence1": "<bos>" + example["sentence1"] + "<eos>"}, 
       remove_columns=["sentence1"]
   )
   ```

***





# DatasetInfo

## Usage

```py
from datasets import DatasetInfo
```

1. The class for restoring information about a dataset

## Attributes

1. ***description*** (A description of the dataset)

   `str`

   ```py
   '''
   Movie Review Dataset. This is a dataset of containing 5,331 positive and 5,331 negative processed sentences from Rotten Tomatoes movie reviews. This data was first used in Bo Pang and Lillian Lee, ``Seeing stars: Exploiting class relationships for sentiment categorization with respect to rating scales.'', Proceedings of the ACL, 2005.
   '''
   ```

2. ***features*** (The features used to specify the dataset’s column types)

   `Features`

   ```py
   {'text': Value(dtype='string', id=None), 'label': ClassLabel(names=['neg', 'pos'], id=None)}
   ```





# Features

## Usage

```py
from datasets import Features
```

1. A dictionary defining the type of all columns respectively for a `Dataset`

## Methods

### ***Features.____init____()***

> Initialize a new Features object

#### Returns

`Features`	The features object initialized

#### Parameters

1. ***dict*** (The configuration dict representing the Features object)

   `Dict[str: Value | ClassLabel | List | Dict | ndarray]`

   ```py
   features = Features({
       'text': Value(dtype="string"), label': ClassLabel(num_classes=3, names=['bad', 'ok', 'good'])
   })
   ```

***

### FeaturesObj.copy()

> Copy an existing Features object

#### Returns

`Features`	A copy of the feature

```py
new_features = dataset.features.copy()
new_features["label"] = ClassLabel(names=["negative", "positive"])
new_features["idx"] = Value("int64")
```

***





# Value

## Usage

```py
from datasets import Value
```

1. An object representing single scalar feature value of a particular data type

## Methods

### ***Value.____init____()***

> Initialize a new Value object

#### Returns

`Value`	The new Value object initialized

#### Parameters

1. ***dtype*** (The string representation of the datatype)

   `"null"` | 

   `"bool"` | 

   `"int8"` | `"int16"` | `"int32"` | `"int64"` | 

   `"float16"` | `"float32"` | `"float64"` | `"float32"` | `"float64"` | 

   `"time32"` | `"time64"` | `"timestamp"` | `"date32"` | `"date64"` | 

   `"string"`  

   ```py
   value = Value(dtype='int32')
   ```

***





# ClassLabel

## Usage

```py
from datasets import ClassLabel
```

1. Feature type for integer class labels
2. Under the hood the labels are stored as integers
3. Negative integers represent unknown/missing labels

## Methods

### ***ClassLabel.____init____()***

> Initialize a new ClassLabel object

#### Returns

`ClassLabel`	The ClassLabel object initialized

#### Parameters

1. ***num_classes*** (Number of different classes)

   `int`

   > The integer labels have to be created from 0 to (num_classes-1) labels

   ```py
   cl = ClassLabel(num_classes=3)
   ```

2. ***names*** (String names for the integer classes. The order in which the names are provided corresponds to the integers)

   `List[str]`

   ```py
   cl = ClassLabel(names=['bad', 'ok', 'good'])
   ```

***

### ***ClassLabelObj.int2str()***

> Convert `integer` => class name `string`

#### Returns

`str` | `List[str]`	The str(s) corresponding to the passed value(s)

#### Parameters

1. ***values*** ()

   `int` | `Iterable[int]`

   ```py
   cl = ClassLabel(names=["good", "normal", "bad"])
   cl.int2str([0, 0, 2])
   
   # Return value
   ['good', 'good', 'bad']
   ```

***

### ***ClassLabelObj.str2int()***

> Convert class name `string` => `integer`

#### Returns

`int` | `List[int]`	The int(s) corresponding to the passed value(s)

#### Parameters

1. ***values*** ()

   `str` | `Iterable[str]`

   ```py
   cl = ClassLabel(names=["good", "normal", "bad"])
   cl.str2int(["good", "normal", "bad"])
   
   # Return value
   [0, 1, 2]
   ```

***





# get_dataset_split_names()

## Usage

```py
from datasets import get_dataset_split_names
```

1. Get the split the dataset have

## Call

### Returns

`List`	The available split name in strings for the dataset 

### Parameters

1. ***path*** (Path or name of the dataset)

   str

   ```py
   get_dataset_split_names("cornell-movie-review-data/rotten_tomatoes")
   
   # Return value
   ['train', 'validation', 'test']
   ```

***



# load_dataset()

## Usage

```python
from datasets import load_dataset
```

1. Load a dataset builder according to the specified dataset type (JSON/CSV/Web dataset/Folder)
2. Run the dataset builder
   - If not steaming: Download and cache the dataset
   - If steaming: Lazily steaming the dataset when iterating on it
3. Return a dataset built from the requested splits in `split`

## Call

### Returns

`Dataset`	The specific split of the dataset you load

`DatasetDict`	All split of the dataset you load

### Parameters

1. ***path*** (Path or name of the dataset)

   `str`

   > If `path` is a dataset repository on the HF hub
   >
   > If `path` is a local directory

   ```py
   dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes")
   
   # Return values
   DatasetDict({
       train: Dataset({
           features: ['text', 'label'],
           num_rows: 8530
       })
       validation: Dataset({
           features: ['text', 'label'],
           num_rows: 1066
       })
       test: Dataset({
           features: ['text', 'label'],
           num_rows: 1066
       })
   })
   ```

   > If `path` is a name of a dataset builder (e.g. "csv", "json", "text"...) && `data_files` and `data_dir` is specified
>
   > 1. Handling **CSV** files
   >
   >    | col_0           | col_1           | col_2           | ...... | col_n           |
   >    | --------------- | --------------- | --------------- | ------ | --------------- |
   >    | `dataset[0][0]` | `dataset[0][1]` | `dataset[0][2]` | ...... | `dataset[0][n]` |
   >    | `dataset[1][0]` | `dataset[1][1]` | `dataset[1][2]` | ...... | `dataset[1][n]` |
   >    | ......          | ......          | ......          | ...... | ......          |
   >    | `dataset[m][0]` | `dataset[m][1]` | `dataset[m][2]` | ...... | `dataset[m][n]` |
   >
   > ```py
   > dataset = load_dataset("csv", data_files=r"C:/Users/Mty/Desktop/my_data.csv", split="train")
   > dataset.set_format("pandas")
   > 
   > # Each row is a row in the dataset && each col is a col in the dataset
   >    col_0  col_1  col_2  col_3  col_4  col_5  col_6  col_7  col_8  col_9  col_10
   > 0      0      1      2      3      4      5      6      7      8      9      10
   > 1      1      2      3      4      5      6      7      8      9     10      11
   > 2      2      3      4      5      6      7      8      9     10     11      12
   > 3      3      4      5      6      7      8      9     10     11     12      13
   > 4      4      5      6      7      8      9     10     11     12     13      14
   > ```

   > 2. Handling **JSON** files
   >
   >    `[{"a": 1, "b": 2.0, "c": "foo", "d": False}`
   >    `{"a": 4, "b": -5.5, "c": None, "d": True}`
   >    `{"a": 4, "b": -6.5, "c": None, "d": True}`
   >    `{"a": 4, "b": 8.3, "c": "item", "d": False}]`
   >
   > ```py
   > dataset = load_dataset("json", data_files="C:/Users/Mty/Desktop/data.json", split="train")
   > dataset.set_format("pandas")
   > 
   > # Each row is a row in the dataset && each col is a col in the dataset
   >    a    b     c      d
   > 0  1  2.0   foo  False
   > 1  4 -5.5  None   True
   > 2  4 -6.5  None   True
   > 3  4  8.3  item  False
   > ```
   >
   > ​	`{`
   >
   > `		"version": "0.1.0", `
   >
   > `		"data": [{"a": 1, "b": 2.0, "c": "foo", "d": False},{"a": 4, "b": -5.5, "c": None, "d": True}]`
   >
   > ​	`}`
   >
   > ```py
   > dataset = load_dataset("json", data_files="my_file.json", split="train", field="data")
   > dataset.set_format("pandas")
   > 
   > # Each row is a row in the dataset && each col is a col in the dataset
   >    a    b     c      d
   > 0  1  2.0   foo  False
   > 1  4 -5.5  None   True
   > ```

2. ***name*** (Defining the name of the dataset configuration)

   `str`

   > For example some datasets have smaller sub-datasets in it which you have to specify

   ```py
   mindsFR = load_dataset("PolyAI/minds14", name="fr-FR", split="train")
   ```

3. ***data_dir*** (The `data_dir` of the dataset configuration)

   ==`None`== | `str`

   ![img](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/folder-based-builder.png)

   ```py
   # Full path: /path/to/folder -> /train/grass/...
       #                                 /fire/...
       #                                 /water/...
       #						  /validation/grass/...
       #						             /fire/...
       #						             /water/...
       #						  /test/grass/...
       #						       /fire/...
       #						       /water/...
   dataset = load_dataset("imagefolder", data_dir="/path/to/folder")
   ```

4. ***data_files*** (Path(s) to source data file(s))

   ==`None`== | `str` | `Sequence` | `Mapping`

   > If `data_files` is an instance of `str`

   ```py
   # By default loads all the data into the train split
   datadict = load_dataset("csv", data_files="/path/to/my_file.csv")
   ```

   > If `data_files` is an instance of `Mapping`

   ```py
   # Load the DatasetDict with specified splits (Locally)
   datadict = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
   
   # Load the DatasetDict with specified splits (From Hub)
   datadict = load_dataset(
       "allenai/c4", data_files={"validation": "en/c4-validation.*.json.gz"}, split="validation"
   )
   ```

5. ***field*** (specify the key to retrieve the data list in the json str)

   `str`

   > Can only be used when `path="json"`

   ```py
   dataset = load_dataset("json", data_files="my_file.json", split="train", field="data")
   ```

6. ***split*** (Which split of the data to load)

   ==`None="all"`== | `str` | `datasets.ReadInstruction`

   > If `split = None`, the return value would be a `DatasetDict`

   ```py
   dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split=None)
   
   # Return values
   DatasetDict({
       train: Dataset({
           features: ['text', 'label'],
           num_rows: 8530
       })
       validation: Dataset({
           features: ['text', 'label'],
           num_rows: 1066
       })
       test: Dataset({
           features: ['text', 'label'],
           num_rows: 1066
       })
   })
   
   ```

   > If `split = str` || `split = datasets.ReadInstruction`, the return value would be a `Dataset`
   >
   > 1. If `split = str`
   >
   >    ```py
   >    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
   >    
   >    # Return values
   >    Dataset({
   >        features: ['text', 'label'],
   >        num_rows: 8530
   >    })
   >    ```
   >
   > 2. If `split = datasets.ReadInstruction`
   >
   >    ```py
   >    # 1. Combining different splits
   >    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train+test")
   >    # 2. Regular index slices
   >    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train[10:1000]")
   >    # 3. Percentage slices
   >    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train[30%:70%]")
   >    # 4. Negative Index/Percentage slices
   >    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train[-70%:]")
   >    # 5. Changing rounding rule for Percentage slicing from *rounding* to *truncating*
   >    dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train[-70%:](pct1_dropremainder)")
   >                         
   >    # Return values
   >    Dataset({
   >        features: ['text', 'label'],
   >        num_rows: 9596
   >    })
   >    ```
   >
   >    ```py
   >    # 5. When split is a list of datasets.ReadInstruction, it will return a list of Dataset
   >    dataset = load_dataset(
   >        "cornell-movie-review-data/rotten_tomatoes",
   >        split=[f"train[{k * 10}%:{(k + 1) * 10}%]+test[{k * 10}%:{(k + 1) * 10}%]" for k in range(0, 3 + 1)]
   >    )
   >                         
   >    # Return values
   >    [Dataset({
   >        features: ['text', 'label'],
   >        num_rows: 960
   >    }), Dataset({
   >        features: ['text', 'label'],
   >        num_rows: 959
   >    }), Dataset({
   >        features: ['text', 'label'],
   >        num_rows: 960
   >    }), Dataset({
   >        features: ['text', 'label'],
   >        num_rows: 959
   >    })]
   >    ```

7. ***num_proc*** (Number of processes when downloading and generating the dataset locally)

   ==`None`== | int

   ```py
   dataset = load_dataset("cornell-movie-review-data/rotten_tomatoes", num_proc=4)
   ```

8. ***trust_remote_code*** (Whether or not to allow for datasets defined on the Hub using a dataset script)

   ==`bool=False`==

   > Some dataset does not have physical storage in the hub, they are downloaded / streamed by running a generation script 

   ```py
   c4 = load_dataset("c4", "en", split="train", trust_remote_code=True)
   ```

***





# load_dataset_builder()

## Usage

```py
from datasets import load_dataset_builder
```

1. Load a dataset builder
   - Inspect general information that is required to build a dataset (cache directory, config, dataset info, features, data files, etc.)

## Call

### Returns

`DatasetBuilder`	The dataset builder that contains useful information about the dataset's configuration

### Parameters

1. ***path*** (Path or name of the dataset)

   str

   ```py
   ds_builder = load_dataset_builder("cornell-movie-review-data/rotten_tomatoes")
   ```


***





# concatenate_datasets()

## Usage

```py
from datasets import concatenate_datasets
```

1. Concatenate a list of datasets vertically or horizontally

## Call

### Returns

`Dataset`	The concatenated dataset

### Parameters

1. ***dsets*** (List of Datasets to be concatenated)

   `List[Dataset]`

   ```py
   ds1 = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
   ds1 = ds1.select_columns("text")
   ds2 = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds2 = ds2.select_columns("text")
   
   assert ds1.features.type == ds2.features.type
   dataset = concatenate_datasets([ds1, ds2])
   
   # Return value
   Dataset({
       features: ['text'],
       num_rows: 9596
   })
   ```

2. ***axis*** (Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns (horizontally))

   ==`int=0`==

   > - If `axis = 0`, the `column_names` and `features` should be the same for all datasets to be concatenated
   > - If `axis = 0`, the return value from `__len__` should be the same for all datasets to be concatenated

   ```py
   ds1 = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="train")
   ds1 = ds1.select_columns("text").rename_column(original_column_name="text", new_column_name="sentence1").select(range(100))
   ds2 = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="validation")
   ds2 = ds2.select_columns("text").rename_column(original_column_name="text", new_column_name="sentence2").select(range(100))
   
   dataset = concatenate_datasets([ds1, ds2], axis=1)
   
   # Return value
   Dataset({
       features: ['sentence1', 'sentence2'],
       num_rows: 100
   })
   ```



# interleave_datasets()

## Usage

```py
from datasets import interleave_datasets
```

1. Interleave several datasets (sources) into a single dataset. The new dataset is constructed by alternating between the sources to get the examples
2. The alternative sampling can be probabilistic or deterministic

## Call

### Returns

`Dataset`	the new dataset after the alterative sampling if the input is a list of `Dataset`

`IterableDataset`	the new iterable dataset after the alterative sampling if the input is a list of `IterableDataset`

### Parameters

1. ***datasets*** (List of datasets to interleave)

   `List[Dataset]` | `List[IterableDataset]`

   ```py
   d1 = Dataset.from_dict({"a": [0, 1, 2]})
   d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
   d3 = Dataset.from_dict({"a": [20, 21, 22]})
   
   dataset = interleave_datasets([d1, d2, d3])
   dataset.set_format("pandas")
   
   # Return value (dataset[:])
       a
   0   0
   1  10
   2  20
   3   1
   4  11
   5  21
   6   2
   7  12
   8  22
   ```

2. ***probabilities*** (Sampling examples from one source at a time according to these probabilities)

   `List[float]`

   ```py
   d1 = Dataset.from_dict({"a": [0, 1, 2]})
   d2 = Dataset.from_dict({"a": [10, 11, 12, 13]})
   d3 = Dataset.from_dict({"a": [20, 21, 22]})
   
   dataset = interleave_datasets([d1, d2, d3], probabilities=[0.3, 0.5, 0.2])
   dataset.set_format("pandas")
   
   # Return value (dataset[:])
       a
   0  10
   1   0
   2  20
   3  11
   4   1
   5  21
   6   2
   ```

3. ***seed*** (The random seed used to choose a source for each example)

   `int`

   > Should be used with `probabilities`

   ```py
   dataset = interleave_datasets([d1, d2, d3], probabilities=[0.3, 0.5, 0.2], seed=42)
   ```

4. ***stopping_strategy*** (Two mode for sampling data)

   ==`str="first_exhausted"`== | `"all_exhausted"`

   > `"first_exhausted"`: Keep sampling examples (rows) until all examples in one dataset runs out of samples
   >
   >  `"all_exhausted"`: Keep sampling examples (rows) until all examples in all dataset runs out of samples

   ```py
   dataset = interleave_datasets([d1, d2, d3], stopping_strategy="all_exhausted")
   ```

   

# load_from_disk()

## Usage

```py
from datasets import load_from_disk
```

1. Reload the dataset locally

## Call

### Returns

`Dataset`	The dataset loaded form your local path

### Parameters

1. ***dataset_path*** (the loading path or remote URL)

   `str` | `os.Pathlike`

   ```py
   reloaded_dataset = load_from_disk("path/of/my/dataset/directory")
   ```



