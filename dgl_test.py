import os
from dotenv import load_dotenv
from neuralogic.settings import Settings
from neuralogic.sources import Sources
from neuralogic.builder import Model
from neuralogic.preprocessing import to_processed_samples
from neuralogic import initialize
from neuralogic.dgl import NeuraLogicLayer
import torch.nn.functional as F
import torch

load_dotenv()


def train(neuralogic_path):
    initialize(os.environ["CLASSPATH"])

    settings = Settings()

    sources = Sources.from_dir(neuralogic_path, settings)
    neuralogic_model = Model.from_neuralogic(settings, sources)

    model = NeuraLogicLayer(neuralogic_model.weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = 0
    seen_instances = 0
    printouts = 10

    samples = to_processed_samples(neuralogic_model.samples)

    for iter in range(800):
        if iter > 0 and iter % printouts == 0:
            print(iter, " average loss is:", total_loss / seen_instances)
        seen_instances = 0
        total_loss = 0

        for sample in samples:
            model.train()

            label = torch.tensor([sample.target])

            optimizer.zero_grad()
            out = model(sample)

            loss = F.mse_loss(out, label)
            loss.backward()
            optimizer.step()

            seen_instances += 1
            total_loss += float(loss)

    for sample in samples:
        output = model(sample)[0]
        label = sample.target
        print(f"label: {label}, output: {output}")


train("./dataset/simple/xor/naive")
