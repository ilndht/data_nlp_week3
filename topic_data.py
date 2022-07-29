import json 
import datasets
from importlib_metadata import version

class TopicConfig(datasets.BuilderConfig): 
    def __init__(self, **kwargs): 
        """
        Args :
            **kwargs: keyword arguements forwarded to super.
        """
        super(TopicConfig,self).__init__(**kwargs)



class TopicCLS(datasets.GeneratorBasedBuilder): 
    BUILDER_CONFIGS=[
        TopicConfig(
            name= "plain_text", 
            version = datasets.Version("0.0.1",""),
            description= "Plain text",
        ),
    ]

    def _info(self): 
        return datasets.DatasetInfo(
            description="topic classification", 
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "text": datasets.Value("string"),
                    "labels": datasets.Value("int16"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": "/content/data/clean/train.json"}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": "/content/data/clean/valid.json"}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": "/content/data/clean/test.json"}),
        ]
    
    def _generate_examples(self, filepath):
        #"""This function returns the examples in the raw (text) form."""
        key = 0 
        with open(filepath,encoding='utf-8') as f:
            data = json.load(f)
            for data_id in data: 
                yield key, {
                    "id": data_id,
                    "text": data[data_id]['text'],
                    "labels":data[data_id]["labels"],
                }
                key += 1