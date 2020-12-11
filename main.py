import os.path as osp
import os
from dotenv import load_dotenv
load_dotenv()

from neuralogic.Settings import Settings
from neuralogic.Sources import Sources
from neuralogic.Pipeline import Pipeline
from neuralogic import initialize


import neuralogic.PytorchLayer as torchlayer


def train(neuralogic_path, pytorch_path):
    initialize(os.environ["CLASSPATH"])

    settings = Settings()
    sources = Sources(settings, neuralogic_path)

    pipeline = Pipeline(settings, sources)
    r, s = pipeline.execute(sources)

    dataset = torchlayer.PytorchDataset(pytorch_path)
    device = torchlayer.get_device()
    model = torchlayer.get_model(dataset, device)
    optimizer = torchlayer.get_optimizer(model)

    torchlayer.eval(dataset, model, optimizer, device)


train("./dataset/molecules/mutagenesis", osp.join(osp.dirname(osp.realpath(__file__)), "..", "dataset", "pytorch_mutagenesis"))
