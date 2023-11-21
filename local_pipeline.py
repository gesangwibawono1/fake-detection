#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[1]:


import os
import pandas as pd
from typing import Text

from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner


# # Prepocessing

# In[2]:


df = pd.read_csv("news_articles.csv")
df = df.query("text_without_stopwords == text_without_stopwords")
df = df.groupby('label').apply(lambda s: s.sample(500)).reset_index(drop=True)
df.text = df["text_without_stopwords"]
df.label = pd.factorize(df.label)[0]
df = df[["text", "label"]]
df.to_csv("data/news_articles.csv",index=False)


# # Set Variabel

# In[3]:


PIPELINE_NAME = "gesang_wibawono-pipeline"

DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/fake_detection_transform.py"
TRAINER_MODULE_FILE = "modules/fake_detection_trainer.py"

OUTPUT_BASE = "output"
serving_model_dir = os.path.join(OUTPUT_BASE, 'serving_model')
pipeline_root = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
metadata_path = os.path.join(pipeline_root, "metadata.sqlite")


# # Pipeline

# In[4]:


def init_local_pipeline(
    components, pipeline_root: Text
) -> pipeline.Pipeline:
    """
    Main
    """

    logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_args = [
        "--direct_running_mode=multi_processing"
        "----direct_num_workers=0"
    ]

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root,
        components=components,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            metadata_path
        ),
        eam_pipeline_args=beam_args
    )


# In[5]:


logging.set_verbosity(logging.INFO)

from modules.components import init_components

components = init_components(
    DATA_ROOT,
    transform_module=TRANSFORM_MODULE_FILE,
    trainer_module=TRAINER_MODULE_FILE,
    training_steps=20,
    eval_steps=10,
    serving_model_dir=serving_model_dir,
)

pipeline = init_local_pipeline(components, pipeline_root)
BeamDagRunner().run(pipeline=pipeline)


# In[6]:


get_ipython().system('pip freeze >> requirements.txt')


# In[ ]:




